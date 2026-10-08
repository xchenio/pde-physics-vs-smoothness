import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.token_operator import TokenDeepONet,train_one


class TokenTests(unittest.TestCase):
    def test_exact_active_parameter_pairing(self):
        torch.manual_seed(5);a=TokenDeepONet('gate');torch.manual_seed(5);b=TokenDeepONet('attention')
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))
        x=torch.randn(2,256);self.assertEqual(a.tokens(x).numel(),2*64)
        for model in (a,b):
            # Nonzero head lets every block parameter demonstrate its active gradient path.
            torch.nn.init.normal_(model.branch_head.weight,std=.1)
            y=model(x,torch.tensor([[.1,.5],[.3,1.]]));y.square().sum().backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name);self.assertGreater(float(p.grad.abs().sum()),0.,name)

    def test_resume(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(2);x=rng.normal(size=(10,256));y=.2*x+.3
        arrays=dict(train_x=x,train_y=y,val_x=x[:3],val_y=y[:3]);anchor=dict(mean=0.,std=1.,q=np.eye(256)[:,:64])
        c=dict(kind='attention',rank=64,seed=0,task='burgers_original',steps=4,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,eval_every=2,checkpoint_every=2,length=1.,final_time=1.)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b';r=train_one(c,arrays,anchor,{},a,torch.device(device))
                self.assertIsNone(train_one(c,arrays,anchor,{},b,torch.device(device),stop_after=2))
                s=train_one(c,arrays,anchor,{},b,torch.device(device));self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))


if __name__=='__main__':unittest.main()
