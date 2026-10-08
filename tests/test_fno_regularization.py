import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.time_fno_baseline import train_one as warm_train
from scripts.fno_regularization_core import train_one


class RegularizationTests(unittest.TestCase):
    def test_pairing_calibration_and_exact_resume(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        rng=np.random.default_rng(5);x=rng.normal(size=(8,32)).astype('float32')
        arrays=dict(train_x=x,train_y=np.stack([x,.8*x,.5*x],1),val_x=x[:2],
                    val_y=np.stack([x[:2],.8*x[:2],.5*x[:2]],1),times=[0.,.5,1.],grid256=(np.arange(32)+.5)/32)
        base=dict(seed=0,steps=4,width=4,modes=4,layers=2,lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=4,
                  eval_every=2,checkpoint_every=2,diffusion=.001/np.pi,resolved_max_mode=4)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);warm_train(base,arrays,{},root/'warm',torch.device(device));warm=root/'warm/checkpoints/latest.pt'
                values=[]
                for kind in ['none','physics','gradient','curvature']:
                    c=dict(base,kind=kind,strength=0. if kind=='none' else .1,lr=.0003)
                    a,b=root/(kind+'a'),root/(kind+'b')
                    r=train_one(c,arrays,{},warm,a,torch.device(device))
                    self.assertIsNone(train_one(c,arrays,{},warm,b,torch.device(device),stop_after=2))
                    s=train_one(c,arrays,{},warm,b,torch.device(device))
                    self.assertEqual(r['history'],s['history']);self.assertEqual(r['batch_hash'],s['batch_hash'])
                    self.assertEqual(r['collocation_hash'],s['collocation_hash'])
                    if kind!='none':self.assertAlmostEqual(r['history'][0]['effective_gradient_ratio'],.1,places=6)
                    ca=torch.load(a/'checkpoints/final.pt',weights_only=False);cb=torch.load(b/'checkpoints/final.pt',weights_only=False)
                    for k in ca['model']:self.assertTrue(torch.equal(ca['model'][k],cb['model'][k]))
                    values.append((r['initial'],r['batch_hash'],r['collocation_hash'],r['calibration']))
                self.assertTrue(all(v==values[0] for v in values))


if __name__=='__main__':unittest.main()
