import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.manifold_reconstruction import Autoencoder,derivative,errors,metric_floors,train_one


class ManifoldTests(unittest.TestCase):
    def test_shapes_pairing_and_no_skip(self):
        for rank in (16,32,64):
            models=[]
            for kind in ('local','scrambled','global'):
                torch.manual_seed(7);m=Autoencoder(kind,rank);x=torch.randn(3,256)
                self.assertEqual(m.encode(x).numel(),3*rank);self.assertEqual(m(x).shape,x.shape)
                y=m.decode(torch.zeros_like(m.encode(x)))
                self.assertEqual(y.shape,x.shape);models.append(m)
            for a,b in zip(models[0].parameters(),models[1].parameters()):self.assertTrue(torch.equal(a,b))
            self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))

    def test_spectral_derivatives_and_bands(self):
        x=np.arange(256)/256;y=np.sin(2*np.pi*3*x)[None]
        np.testing.assert_allclose(derivative(y,1.,1),6*np.pi*np.cos(6*np.pi*x)[None],atol=1e-10)
        np.testing.assert_allclose(derivative(y,1.,2),-(6*np.pi)**2*y,atol=1e-8)
        floors=metric_floors(y,1.);e=errors(np.zeros_like(y),y,1.,floors)
        self.assertAlmostEqual(e['field'][0],1.);self.assertAlmostEqual(e['gradient'][0],1.)
        self.assertAlmostEqual(sum(e[k][0]**2 for k in e if k.startswith('band')),e['field'][0]**2)
        z=np.zeros_like(y);self.assertTrue(all(np.isfinite(v).all() for v in errors(z,z,1.,metric_floors(z,1.)).values()))

    def test_exact_resume_and_identity(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        train=np.random.default_rng(2).normal(size=(12,256));val=train[:4]+.1
        for device,kind in [(device,kind) for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']) for kind in ('global','local','scrambled')]:
            c=dict(kind=kind,rank=16,seed=0,steps=4,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,eval_every=2,checkpoint_every=2,length=1.)
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b';r=train_one(c,train,val,{},a,torch.device(device))
                self.assertIsNone(train_one(c,train,val,{},b,torch.device(device),stop_after=2))
                s=train_one(c,train,val,{},b,torch.device(device))
                self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))
                with self.assertRaises(ValueError):train_one(dict(c,steps=5),train,val,{},b,torch.device(device))


if __name__=='__main__':unittest.main()
