import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.fno_projection_core import projected_loss,train_one
from scripts.architecture_core import train_one as old_train


class FNOProjectionTests(unittest.TestCase):
    def test_projection_gradient(self):
        torch.set_num_threads(4)
        q=torch.linalg.qr(torch.randn(32,8,dtype=torch.float64))[0]
        z=torch.randn(7,32,dtype=torch.float64,requires_grad=True);y=torch.randn_like(z);w=torch.rand(7,dtype=torch.float64)
        a=projected_loss(z,y@q,w,q,8,32)
        b=(w*(z@q@q.T-y).square().sum(1)).mean()/32
        ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-12,rtol=0))
        self.assertTrue(torch.allclose(ga-ga@q@q.T,torch.zeros_like(ga),atol=1e-12,rtol=0))
        self.assertAlmostEqual((b-a).item(),(w*(y-y@q@q.T).square().sum(1)).mean().item()/32,places=12)

    def test_pairing_resume_and_legacy_full(self):
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(4)
        arrays={f'{s}_{v}':rng.normal(size=(8,32)) for s in ('train','val') for v in ('x','y')}
        anchor=dict(q=np.linalg.qr(rng.normal(size=(32,16)))[0],mean=np.zeros(32),std=np.ones(32),weights=np.zeros((32,16)),intercept=np.zeros(16))
        with tempfile.TemporaryDirectory() as tmp:
            for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                pair=None
                for rank in (0,4,8,16):
                    with self.subTest(device=dev,rank=rank):
                        c=dict(seed=0,output_rank=rank,evaluation_ranks=[4,8,16],steps=6,batch_size=4,lr=.001,lr_min=.00001,weight_decay=.0001,eval_every=2,checkpoint_every=2)
                        root=Path(tmp)/dev/str(rank);device=torch.device(dev)
                        full=train_one(c,arrays,anchor,{},root/'full',device)
                        self.assertIsNone(train_one(c,arrays,anchor,{},root/'resume',device,stop_after=3))
                        resumed=train_one(c,arrays,anchor,{},root/'resume',device)
                        self.assertEqual(full['history'],resumed['history']);self.assertEqual(full['batch_sha256'],resumed['batch_sha256'])
                        now=(full['initial_sha256'],full['batch_sha256'],full['parameters_real'])
                        if pair is None:pair=now
                        self.assertEqual(pair,now)
                        one=torch.load(root/'full/checkpoints/final.pt',weights_only=False)['model']
                        two=torch.load(root/'resume/checkpoints/final.pt',weights_only=False)['model']
                        self.assertTrue(all(torch.equal(one[k],two[k]) for k in one))
                        with self.assertRaises(ValueError):train_one(dict(c,output_rank=1),arrays,anchor,{},root/'resume',device)
                        if rank==0:
                            old=dict(c,architecture='fno',mapping='direct',penalty=0.,rank=16)
                            result=old_train(old,arrays,anchor,{},root/'legacy',device)
                            previous=torch.load(root/'legacy/checkpoints/final.pt',weights_only=False)['model']
                            self.assertTrue(all(torch.equal(one[k],previous[k]) for k in one))
                            for s in ('train','val'):self.assertEqual(full['metrics'][s]['prediction'],result['metrics'][s]['prediction'])


if __name__=='__main__':unittest.main()
