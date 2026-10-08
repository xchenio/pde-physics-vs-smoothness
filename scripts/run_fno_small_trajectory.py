"""Register FNO16/32 controls before any new confirmation target is opened."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_token import plan as physics_plan,load_data
from scripts.fno_small_trajectory import train_one
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/fno_small_trajectory_v1'


def plan():
    old=physics_plan();p=dict(parent=digest(old),source=digest([old['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/fno_small_trajectory.py'),file_hash(ROOT/'src/fno.py')]),
           widths=[16,32],seeds=old['seeds'],steps=old['steps'],lr=old['lr'],lr_min=old['lr_min'],weight_decay=old['weight_decay'],
           batch_size=old['batch_size'],collocation=old['collocation'],eval_every=old['eval_every'],checkpoint_every=old['checkpoint_every'],
           data='exact same100 trajectories,5 supervised time slices,train-only scalar normalization/scale/weights and minibatch sequence as physics factorial',
           architecture='FNO16modes4layers,width16 or32,pointwise head5 known-time channels,zero initialized; output256 only',
           selection='all6 fits reported,fixed final;source/widths locked before confirmation;no model tuning',
           limits='width16 is a rough capacity bracket,not exact parameter matching. Same labels/steps but FNO does not compute PDE derivatives,so unequal FLOPs/time. Discrete time heads cannot extrapolate/query512 without another policy. Descriptive strong controls,not additional registered primary hypothesis tests.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def run(p,device):
    arrays,context=load_data(physics_plan());context=dict(parent=context,plan=digest(p));groups={}
    for width in p['widths']:
        runs=[]
        for seed in p['seeds']:
            c={k:p[k] for k in ('steps','lr','lr_min','weight_decay','batch_size','collocation','eval_every','checkpoint_every')};c.update(width=width,seed=seed)
            runs.append(train_one(c,arrays,context,OUT/f'w{width}_s{seed}',device))
        groups[str(width)]=dict(parameters_real=runs[0]['parameters_real'],metrics={s:dict(mean=float(np.mean([r['metrics'][s]['primary'] for r in runs])),seed_sd=float(np.std([r['metrics'][s]['primary'] for r in runs],ddof=1))) for s in ('train','val')})
        atomic_json(OUT/'progress.json',dict(completed_runs=len(groups)*3,total_runs=6,groups=groups))
    atomic_json(OUT/'summary.json',dict(context=context,groups=groups,completed_runs=6));print('FNO100 COMPLETE',groups,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))
