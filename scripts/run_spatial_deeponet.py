"""Stage2 CNN/smooth-trunk baseline, compared contextually with retained full FNO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.spatial_deeponet import train_one
from scripts.run_spectral_pod import plan as parent_plan,task_data
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/spatial_deeponet_v1'


def plan():
    old=parent_plan();files=['scripts/run_spatial_deeponet.py','scripts/spatial_deeponet.py','scripts/manifold_reconstruction.py']
    p=dict(parent=digest(old),source=digest([file_hash(ROOT/f) for f in files]),tasks=old['tasks'],seeds=[0,1,2],rank=64,
           steps=old['steps'],lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=32,eval_every=250,checkpoint_every=50,
           encoder='periodic CNN32 stride4/4,16positions x4channels=64 scalars, linear64->64 branch head',
           trunk='physical x,t; periodic sin/cos frequencies1..16 plus normalized time; tanh33->64->64->64; inner product/sqrt64',
           data='historical correction600 train, development200; same normalization, input sample mean offset and guarded weighted field loss as FNO',
           selection='rank64 locked before compression study finished; fixed final step, all9 fits reported; no per-model tuning',
           grid='Burgers true retained cell centers(4i+0.5)/1024,t1; Allen-Cahn2i/256,t0.5',
           comparison='retained full-field FNO136481 params, same seeds/batches/loss/steps; different params and readout, contextual baseline only',
           limits='Fixed-time data-only training: supplied t is constant within each task, temporal dependence unvalidated. No physics, attention, cross-resolution or fresh confirmation yet. No future-field encoder at inference.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def run(p,device):
    old=parent_plan()
    for task in p['tasks']:
        arrays,anchors,context=task_data(old,task);context=dict(parent=context,plan=digest(p),task=task)
        runs=[]
        for seed in p['seeds']:
            c={k:p[k] for k in ('rank','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every')}
            c.update(seed=seed,task=task,steps=p['steps'][task],length=2. if task=='allen_cahn' else 1.,final_time=.5 if task=='allen_cahn' else 1.)
            runs.append(train_one(c,arrays,anchors[64],context,OUT/task/f'cnn_trunk_s{seed}',device))
        oldsummary=json.loads((ROOT/'results/public_fno_projection_v1'/task/'summary.json').read_text())
        summary=dict(context=context,parameters=runs[0]['parameters'],completed_runs=len(runs),
                     metrics={s:{k:dict(mean=float(np.mean([r['metrics'][s]['prediction'][k] for r in runs])),seed_sd=float(np.std([r['metrics'][s]['prediction'][k] for r in runs],ddof=1))) for k in runs[0]['metrics'][s]['prediction']} for s in ('train','val')},
                     retained_fno=oldsummary['groups']['full'])
        atomic_json(OUT/task/'summary.json',summary);print('TASK COMPLETE',task,summary['metrics']['val'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))


if __name__=='__main__':main()
