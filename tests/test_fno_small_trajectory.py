import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.fno_small_trajectory import make_model,train_one


class FnoTrajectoryTests(unittest.TestCase):
    def test_heads_and_resume(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(4);x=rng.normal(size=(8,256)).astype('float32');y=np.stack([x,.9*x,.8*x,.7*x,.5*x],1)
        arrays=dict(train_x=x,train_y=y,val_x=x[:2],val_y=y[:2],grid256=(np.arange(256)*4+.5)/1024)
        c=dict(width=16,seed=0,steps=4,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,collocation=4,eval_every=2,checkpoint_every=2)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            model=make_model(16,0,torch.device(device));self.assertEqual(tuple(model(torch.tensor(x,device=device)).shape),(8,5,256))
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b';r=train_one(c,arrays,{},a,torch.device(device))
                self.assertIsNone(train_one(c,arrays,{},b,torch.device(device),stop_after=2));s=train_one(c,arrays,{},b,torch.device(device))
                self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))


if __name__=='__main__':unittest.main()
