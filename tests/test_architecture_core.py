import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.architecture_core import DeterministicPool, make_model, output, train_one


class ArchitectureTests(unittest.TestCase):
    def test_pool_equivalence(self):
        x=torch.randn(3,16,256,dtype=torch.float64,requires_grad=True)
        a=DeterministicPool()(x)
        b=torch.nn.functional.adaptive_avg_pool1d(x,24)
        self.assertTrue(torch.allclose(a,b,atol=1e-14,rtol=0))
        ga=torch.autograd.grad(a.square().sum(),x,retain_graph=True)[0]
        gb=torch.autograd.grad(b.square().sum(),x)[0]
        self.assertTrue(torch.allclose(ga,gb,atol=1e-14,rtol=0))

    def test_field_coefficient_gradient_equivalence(self):
        q=torch.linalg.qr(torch.randn(32,4,dtype=torch.float64))[0]
        y=torch.randn(7,32,dtype=torch.float64)
        c=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
        w=torch.rand(7,dtype=torch.float64)
        field=(w*(c@q.T-y).square().sum(1)).mean()
        coefficient=(w*(c-y@q).square().sum(1)).mean()
        a=torch.autograd.grad(field,c,retain_graph=True)[0]
        b=torch.autograd.grad(coefficient,c)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0))

    def test_resume_and_zero_initialization(self):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(9)
        q=np.linalg.qr(rng.normal(size=(32,4)))[0]
        arrays={f'{s}_{v}':rng.normal(size=(12,32)) for s in ('train','val') for v in ('x','y')}
        anchor=dict(q=q,mean=np.zeros(32),std=np.ones(32),weights=rng.normal(size=(32,4))*.01,intercept=np.zeros(4))
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        with tempfile.TemporaryDirectory() as tmp:
            for dev in devices:
                for architecture in ('mlp','cnn','fno'):
                    for mapping in ('direct','residual'):
                        with self.subTest(device=dev,architecture=architecture,mapping=mapping):
                            c=dict(architecture=architecture,mapping=mapping,rank=4,seed=0,lr=.001,lr_min=.00001,
                                   weight_decay=.0001,batch_size=4,steps=6,eval_every=2,checkpoint_every=2,penalty=.1 if mapping=='residual' else 0.)
                            device=torch.device(dev)
                            model=make_model(c,32,device)
                            self.assertEqual(torch.count_nonzero(output(model,torch.ones(2,32,device=device),architecture)).item(),0)
                            root=Path(tmp)/dev/architecture/mapping
                            full=train_one(c,arrays,anchor,{},root/'full',device)
                            self.assertIsNone(train_one(c,arrays,anchor,{},root/'resume',device,stop_after=3))
                            resumed=train_one(c,arrays,anchor,{},root/'resume',device)
                            self.assertEqual(full['history'],resumed['history'])
                            a=torch.load(root/'full/checkpoints/final.pt',weights_only=False)['model']
                            b=torch.load(root/'resume/checkpoints/final.pt',weights_only=False)['model']
                            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
                            with self.assertRaises(ValueError):train_one(dict(c,penalty=2.),arrays,anchor,{},root/'resume',device)


if __name__=='__main__':unittest.main()
