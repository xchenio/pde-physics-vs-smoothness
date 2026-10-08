import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.spatial_deeponet import SpatialDeepONet,coordinates,train_one


class SpatialTests(unittest.TestCase):
    def test_smooth_physical_coordinates(self):
        torch.manual_seed(3);m=SpatialDeepONet(length=2.,final_time=.5).double()
        c=torch.tensor([[.231,.2],[.913,.37]],dtype=torch.float64,requires_grad=True)
        v=m.trunk_features(c);g=torch.autograd.grad(v.sum(),c,create_graph=True)[0]
        gxx=torch.autograd.grad(g[:,0].sum(),c)[0][:,0]
        h=1e-5;d=torch.zeros_like(c);d[:,0]=h
        finite=(m.trunk_features(c+d).sum(1)-m.trunk_features(c-d).sum(1))/(2*h)
        finite2=(m.trunk_features(c+d).sum(1)-2*v.sum(1)+m.trunk_features(c-d).sum(1))/h**2
        torch.testing.assert_close(g[:,0],finite,rtol=1e-5,atol=1e-6);torch.testing.assert_close(gxx,finite2,rtol=1e-4,atol=1e-4)
        shifted=c.detach().clone();shifted[:,0]+=2.;torch.testing.assert_close(v,m.trunk_features(shifted),rtol=1e-12,atol=1e-12)
        grid=coordinates('burgers_original',256,torch.device('cpu'));self.assertAlmostEqual(float(grid[0,0]),.5/1024)
        self.assertEqual(m.encoder(torch.randn(3,1,256,dtype=torch.float64)).numel(),3*64)

    def test_exact_resume(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(2);x=rng.normal(size=(10,256));y=.2*x+.3
        arrays=dict(train_x=x,train_y=y,val_x=x[:3],val_y=y[:3]);q=np.eye(256)[:,:64]
        anchor=dict(mean=0.,std=1.,q=q)
        c=dict(rank=64,seed=0,task='burgers_original',steps=4,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,eval_every=2,checkpoint_every=2,length=1.,final_time=1.)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b';r=train_one(c,arrays,anchor,{},a,torch.device(device))
                self.assertIsNone(train_one(c,arrays,anchor,{},b,torch.device(device),stop_after=2))
                s=train_one(c,arrays,anchor,{},b,torch.device(device));self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))
                with self.assertRaises(ValueError):train_one(dict(c,steps=5),arrays,anchor,{},b,torch.device(device))


if __name__=='__main__':unittest.main()
