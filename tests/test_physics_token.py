import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.physics_token_core import physical_fields,residual,train_one


class AnalyticTrunk:
    rank=1
    def trunk_features(self,c):return (torch.sin(2*torch.pi*c[:,0])*torch.exp(-c[:,1]))[:,None]


class PhysicsTests(unittest.TestCase):
    def test_physical_derivatives_and_residual(self):
        coords=torch.tensor([[.23,.2],[.91,.6]],dtype=torch.float64);b=torch.tensor([[.7]],dtype=torch.float64,requires_grad=True)
        u,ux,ut,uxx=physical_fields(AnalyticTrunk(),b,coords,2.,torch.tensor([[.3]],dtype=torch.float64))
        v=1.4*torch.sin(2*torch.pi*coords[:,0])*torch.exp(-coords[:,1])
        torch.testing.assert_close(u[0],v+.3);torch.testing.assert_close(ux[0],1.4*2*torch.pi*torch.cos(2*torch.pi*coords[:,0])*torch.exp(-coords[:,1]))
        torch.testing.assert_close(ut[0],-v);torch.testing.assert_close(uxx[0],-(2*torch.pi)**2*v)
        r=residual(u,ux,ut,uxx,.001/np.pi);r.square().sum().backward();self.assertGreater(float(b.grad.abs().sum()),0.)

    def test_cpu_cuda_resume(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(4);x=rng.normal(size=(8,256)).astype('float32');y=np.stack([x,.8*x,.5*x],1)
        arrays=dict(train_x=x,train_y=y,val_x=x[:2],val_y=y[:2],times=[0.,.5,1.],grid256=(np.arange(256)*4+.5)/1024)
        c=dict(kind='attention',physics_weight=.01,seed=0,steps=4,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,collocation=4,eval_every=2,checkpoint_every=2,diffusion=.001/np.pi)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b';r=train_one(c,arrays,{},a,torch.device(device))
                self.assertIsNone(train_one(c,arrays,{},b,torch.device(device),stop_after=2))
                s=train_one(c,arrays,{},b,torch.device(device));self.assertEqual(r['history'],s['history']);self.assertEqual(r['collocation_hash'],s['collocation_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))


if __name__=='__main__':unittest.main()
