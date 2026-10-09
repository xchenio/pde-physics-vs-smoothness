import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.time_fno import TimeFNO1d,spectral_derivative,time_fields
from src.periodic_position_fno import PositionTimeFNO1d
from src.physics_equations_v1 import penalties as old_penalties
from scripts.periodic_curvature_core import train_one,curvature_penalties


class PeriodicCurvatureTests(unittest.TestCase):
    def test_capacity_state_linear_equivalence_and_periodicity(self):
        torch.manual_seed(17);a=TimeFNO1d(4,4,2)
        torch.manual_seed(17);b=PositionTimeFNO1d(4,4,2,position='linear')
        torch.manual_seed(17);p=PositionTimeFNO1d(4,4,2,position='periodic')
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key],b.state_dict()[key]));self.assertTrue(torch.equal(a.state_dict()[key],p.state_dict()[key]))
        z=torch.linspace(0,1,17,dtype=torch.float64)
        torch.testing.assert_close(p.encode_position(z),p.encode_position(z+1),rtol=1e-13,atol=1e-13)
        torch.nn.init.normal_(a.project[-1].weight,std=.2);b.load_state_dict(a.state_dict());p.load_state_dict(a.state_dict())
        x=torch.randn(2,32);t=torch.tensor([.2,.7]);coords=torch.arange(32)/32
        self.assertTrue(torch.equal(a(x,t,coords),b(x,t,coords)))
        torch.testing.assert_close(p(x,t,coords),p(x,t,coords+1),rtol=1e-5,atol=1e-6)
        count=lambda m:sum(v.numel()*(2 if v.is_complex() else 1) for v in m.parameters())
        self.assertEqual(count(a),count(p));self.assertEqual(count(PositionTimeFNO1d()),34209)

    def test_low_exact_equivalence_and_full_parseval_nyquist(self):
        torch.set_num_threads(2)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            torch.manual_seed(5);m=PositionTimeFNO1d(4,4,2).to(device);torch.nn.init.normal_(m.project[-1].weight,std=.2)
            x=torch.randn(2,32,device=device);t=torch.tensor([.2,.7],device=device);coords=torch.arange(32,device=device)/32;off=x.mean(-1)
            c=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=5.,resolved_max_mode=4)
            ps=curvature_penalties(m,x,t,coords,2.,off,c)
            torch.testing.assert_close(ps['low'],old_penalties(m,x,t,coords,2.,off,c)['curvature'],rtol=0,atol=0)
            u,_=time_fields(m,x,t,coords,2.,off)
            torch.testing.assert_close(ps['full'],spectral_derivative(u,2,2.).square().mean()/4,rtol=1e-6,atol=1e-7)
        class Fixed:
            def __call__(self,x,t,coords):return x
        x=((-1.)**torch.arange(32))[None];c=dict(length=1.,resolved_max_mode=4)
        ps=curvature_penalties(Fixed(),x,torch.tensor([.2]),torch.arange(32)/32,1.,torch.tensor([0.]),c)
        self.assertAlmostEqual(float(ps['low']),0.,places=9)
        self.assertAlmostEqual(float(ps['full'])/(2*np.pi*16)**4,1.,places=6)

    def test_cpu_cuda_exact_resume_pairing_and_adapter_restore(self):
        import scripts.physics_stress_core as engine
        original_factory,original_penalty=engine.TimeFNO1d,engine.penalties
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(5);x=rng.normal(size=(8,32)).astype('float32')
        a=dict(train_x=x,train_y=np.stack([x,.8*x,.5*x],1),val_x=x[:2],val_y=np.stack([x[:2],.8*x[:2],.5*x[:2]],1),
               times=[0.,.5,1.],val_times=[0.,.5,1.],grid256=np.arange(32)/32)
        base=dict(seed=0,steps=4,width=4,modes=4,layers=2,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,eval_every=2,
                  checkpoint_every=2,resolved_max_mode=4,time_lower=.1,time_upper=1.,evaluation_columns=[1,2],length=1.,position_center=.5,position_scale=.4)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            initial=[];streams=[]
            with tempfile.TemporaryDirectory() as d:
                for position in ['linear','periodic']:
                    root=Path(d)/position;base['position']=position
                    warm=train_one(dict(base,kind='warm',strength=0.),a,{},root/'warm',torch.device(device));initial.append(warm['initial'])
                    pair=[]
                    for kind in ['none','low','full']:
                        c=dict(base,kind=kind,strength=0. if kind=='none' else .3,lr=.0003)
                        first,second=root/(kind+'a'),root/(kind+'b');warm_path=root/'warm/checkpoints/latest.pt'
                        r=train_one(c,a,{},first,torch.device(device),warm_path=warm_path)
                        train_one(c,a,{},second,torch.device(device),warm_path=warm_path,stop_after=2)
                        s=train_one(c,a,{},second,torch.device(device),warm_path=warm_path)
                        self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                        self.assertEqual(r['collocation_hash'],s['collocation_hash'])
                        ca=torch.load(first/'checkpoints/final.pt',weights_only=False);cb=torch.load(second/'checkpoints/final.pt',weights_only=False)
                        for key in ca['model']:self.assertTrue(torch.equal(ca['model'][key],cb['model'][key]))
                        if kind!='none':self.assertAlmostEqual(r['history'][0]['effective_gradient_ratio'],.3,places=6)
                        pair.append((r['initial'],r['batch_hash'],r['collocation_hash'],r['calibration']));streams.append((r['batch_hash'],r['collocation_hash']))
                    self.assertTrue(all(v==pair[0] for v in pair))
                self.assertEqual(initial[0],initial[1]);self.assertTrue(all(v==streams[0] for v in streams))
            self.assertIs(engine.TimeFNO1d,original_factory);self.assertIs(engine.penalties,original_penalty)


if __name__=='__main__':unittest.main()
