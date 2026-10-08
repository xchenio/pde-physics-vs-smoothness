import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.physics_equations_v1 import equation_residual,dealiased_cube
from src.time_fno import coefficient_energy
from scripts.physics_stress_core import train_one
from scripts.run_physics_stress import observations
from scripts.allen_cahn_trajectory_v1 import trajectories
from scripts.allen_cahn_data import reaction


class PhysicsStressTests(unittest.TestCase):
    def test_allen_cahn_sign_length_nonlinearity_and_dealiasing(self):
        n=64;x=torch.arange(n,dtype=torch.float64)*2/n;u=torch.sin(torch.pi*x)[None]
        c=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=5.,resolved_max_mode=16)
        ut=-.01*torch.pi**2*u+5*u*(1-u*u)
        self.assertLess(float(coefficient_energy(equation_residual(u,ut,c))),1e-24)
        constant=torch.full_like(u,.3)
        self.assertLess(float(coefficient_energy(equation_residual(constant,5*constant*(1-constant**2),c))),1e-24)
        high=torch.cos(2*torch.pi*25*torch.arange(n,dtype=torch.float64)/n)[None]
        cube=dealiased_cube(high)
        self.assertLess(float(cube[:,11].abs().max()),1e-14)
        torch.testing.assert_close(cube[:,25].real,torch.tensor([.375],dtype=torch.float64))

    def test_sparse_observation_isolation_and_noise_repeatability(self):
        rng=np.random.default_rng(5);y=rng.normal(size=(4,5,32));x=y[:,0]
        clean=dict(train_x=x,train_y=y,val_x=x,val_y=y,times=[0.,.1,.25,.5,1.],grid256=np.arange(32)/32)
        for condition in ['sparse','noisy_sparse','noisy_dense']:
            a,meta=observations(clean,condition);b,m=observations(clean,condition)
            self.assertEqual(meta,m);np.testing.assert_array_equal(a['train_y'],b['train_y'])
            np.testing.assert_array_equal(a['train_y'][:,0],x)
            if condition!='noisy_dense':
                changed=y.copy();changed[:,1:4]+=1000
                b,m=observations(dict(clean,train_y=changed),condition)
                self.assertEqual(meta,m);np.testing.assert_array_equal(a['train_y'],b['train_y'])
        const=np.full((2,32),.25);t=[0.,.05,.125,.25,.5]
        result=trajectories(const,t)
        for j,time in enumerate(t):np.testing.assert_allclose(result[:,j],reaction(const,time),rtol=1e-11,atol=1e-12)

    def test_both_pdes_cpu_cuda_resume_and_fork_pairing(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(5);x=rng.normal(size=(8,32)).astype('float32')
        a=dict(train_x=x,train_y=np.stack([x,.5*x],1),val_x=x[:2],val_y=np.stack([x[:2],.8*x[:2],.5*x[:2]],1),
               times=[0.,1.],val_times=[0.,.5,1.],grid256=np.arange(32)/32)
        base=dict(seed=0,steps=4,width=4,modes=4,layers=2,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,
                  eval_every=2,checkpoint_every=2,resolved_max_mode=4,time_lower=.1,time_upper=1.,evaluation_columns=[1])
        for pde in ['burgers','allen_cahn']:
            base.update(pde=pde,length=1. if pde=='burgers' else 2.,diffusion=.001/np.pi if pde=='burgers' else .01,reaction=0. if pde=='burgers' else 5.)
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                with tempfile.TemporaryDirectory() as d:
                    root=Path(d);c=dict(base,kind='warm',strength=0.)
                    r=train_one(c,a,{},root/'warm',torch.device(device))
                    train_one(c,a,{},root/'warm_resume',torch.device(device),stop_after=2)
                    rr=train_one(c,a,{},root/'warm_resume',torch.device(device));self.assertEqual(r['history'],rr['history'])
                    warm=root/'warm/checkpoints/latest.pt';pair=[]
                    for kind in ['none','physics','curvature']:
                        c=dict(base,kind=kind,strength=0. if kind=='none' else .3,lr=.0003)
                        first,second=root/(kind+'a'),root/(kind+'b')
                        r=train_one(c,a,{},first,torch.device(device),warm_path=warm)
                        train_one(c,a,{},second,torch.device(device),warm_path=warm,stop_after=2)
                        rr=train_one(c,a,{},second,torch.device(device),warm_path=warm)
                        self.assertEqual(r['history'],rr['history']);self.assertEqual(r['batch_hash'],rr['batch_hash']);self.assertEqual(r['collocation_hash'],rr['collocation_hash'])
                        if kind!='none':self.assertAlmostEqual(r['history'][0]['effective_gradient_ratio'],.3,places=6)
                        ca=torch.load(first/'checkpoints/final.pt',weights_only=False);cb=torch.load(second/'checkpoints/final.pt',weights_only=False)
                        for name in ca['model']:self.assertTrue(torch.equal(ca['model'][name],cb['model'][name]))
                        pair.append((r['initial'],r['batch_hash'],r['collocation_hash'],r['calibration']))
                    self.assertTrue(all(v==pair[0] for v in pair))


if __name__=='__main__':unittest.main()
