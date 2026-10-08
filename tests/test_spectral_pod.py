import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.spectral_pod_models import make_model
from scripts.spectral_pod_core import train_one
from scripts.run_spectral_pod import nested_basis


class SpectralPODTests(unittest.TestCase):
    def test_matching_and_zero(self):
        for rank in (16,32,64):
            models=[make_model(dict(architecture=a,rank=rank,seed=0,mapping='direct',penalty=0),256,torch.device('cpu')) for a in ('spectral_pod','local_pod')]
            counts=[sum(v.numel()*(2 if v.is_complex() else 1) for v in m.parameters()) for m in models]
            self.assertEqual(counts[0],counts[1])
            for m in models:self.assertEqual(torch.count_nonzero(m(torch.ones(2,256))).item(),0)

    def test_nested_basis(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(90,80));y=rng.normal(size=(90,80))
        old=np.linalg.svd(y-x.mean(1,keepdims=True),full_matrices=False)[2].T[:,:16]
        q=nested_basis(x,y,old)
        np.testing.assert_array_equal(q[:,:16],old)
        for r in (16,32,64):np.testing.assert_allclose(q[:,:r].T@q[:,:r],np.eye(r),atol=1e-12,rtol=0)
        errors=[np.linalg.norm(y-y@q[:,:r]@q[:,:r].T) for r in (16,32,64)]
        self.assertTrue(errors[0]>=errors[1]>=errors[2])

    def test_resume(self):
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(5)
        arrays={f'{s}_{v}':rng.normal(size=(8,64)) for s in ('train','val') for v in ('x','y')}
        anchor=dict(q=np.linalg.qr(rng.normal(size=(64,4)))[0],mean=np.zeros(64),std=np.ones(64),weights=np.zeros((64,4)),intercept=np.zeros(4))
        with tempfile.TemporaryDirectory() as tmp:
            for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                for a in ('spectral_pod','local_pod'):
                    with self.subTest(device=dev,architecture=a):
                        c=dict(architecture=a,rank=4,seed=0,mapping='direct',penalty=0.,steps=6,batch_size=4,lr=.001,lr_min=.00001,weight_decay=.0001,eval_every=2,checkpoint_every=2)
                        root=Path(tmp)/dev/a;device=torch.device(dev)
                        full=train_one(c,arrays,anchor,{},root/'full',device)
                        self.assertIsNone(train_one(c,arrays,anchor,{},root/'resume',device,stop_after=3))
                        resumed=train_one(c,arrays,anchor,{},root/'resume',device)
                        self.assertEqual(full['history'],resumed['history'])
                        one=torch.load(root/'full/checkpoints/final.pt',weights_only=False)['model']
                        two=torch.load(root/'resume/checkpoints/final.pt',weights_only=False)['model']
                        self.assertTrue(all(torch.equal(one[k],two[k]) for k in one))
                        with self.assertRaises(ValueError):train_one(dict(c,rank=5),arrays,anchor,{},root/'resume',device)


if __name__=='__main__':unittest.main()
