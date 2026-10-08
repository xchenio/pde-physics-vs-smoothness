import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.time_fno_baseline import train_one


class TimeFNOBaselineTests(unittest.TestCase):
    def test_exact_resume_cpu_cuda_and_identity_rejection(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(4);x=rng.normal(size=(8,32)).astype('float32')
        arrays=dict(train_x=x,train_y=np.stack([x,.8*x,.5*x],1),val_x=x[:2],
                    val_y=np.stack([x[:2],.8*x[:2],.5*x[:2]],1),times=[0.,.5,1.],grid256=(np.arange(32)+.5)/32)
        c=dict(seed=0,steps=4,width=4,modes=4,layers=2,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,
               eval_every=2,checkpoint_every=2,diffusion=.001/np.pi,resolved_max_mode=4)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with tempfile.TemporaryDirectory() as d:
                a,b=Path(d)/'a',Path(d)/'b'
                r=train_one(c,arrays,{},a,torch.device(device))
                self.assertIsNone(train_one(c,arrays,{},b,torch.device(device),stop_after=2))
                s=train_one(c,arrays,{},b,torch.device(device))
                self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    train_one(dict(c,lr=.002),arrays,{},b,torch.device(device))


if __name__=='__main__':unittest.main()
