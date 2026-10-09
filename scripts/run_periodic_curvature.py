"""Paired2 positions x3 smoothing modes x3 seeds x2 periodic nonlinear PDEs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_stress import clean_data,observations
from scripts.periodic_curvature_core import train_one
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/periodic_curvature_v1'


def plan():
    parent=ROOT/'results/physics_stress_v1/plan.json';old=json.loads(parent.read_text())
    files=[Path(__file__),ROOT/'src/periodic_position_fno.py',ROOT/'scripts/periodic_curvature_core.py',ROOT/'scripts/physics_stress_core.py',
           ROOT/'src/time_fno.py',ROOT/'src/fno.py',ROOT/'scripts/run_physics_stress.py']
    p=dict(parent_plan_sha256=file_hash(parent),diagnostic_summary_sha256=file_hash(ROOT/'results/physics_diagnosis_v2/summary.json'),
           source={f.relative_to(ROOT).as_posix():file_hash(f) for f in files},tasks=old['tasks'],
           positions=['linear','periodic'],kinds=['none','low','full'],seeds=[0,1,2],strength=.3,warm_steps=1500,continuation_steps=750,
           width=16,modes=16,layers=4,weight_decay=.0001,batch_size=16,eval_every=250,checkpoint_every=50,resolved_max_mode=16,
           observation='Both PDEs dense CLEAN same100 initial trajectories and5 times, reused development200. No new data/confirmation. Ignore parent stress conditions: dense_clean only.',
           position='Only replace the EXISTING scalar coordinate with mean+sqrt2*std*sin(2*pi*x/L), constants from native grid only. Same3 lift channels and34209 real parameters; no modules/extra channels. A single periodic feature is NOT injective fullsin/cos encoding, nor a hard guarantee of continuous periodic output.',
           pairing='Same random initial tensors and minibatch/query RNG across positions; EACH position has its OWN1500-step warm fit, so input intervention is not applied to pretrained linear weights. Then shared model/optimizer/RNG forks750 steps for none/low/full within position.',
           smoothing='Low modes0..16; full modes0..128 with proper single Nyquist energy. Training-only gradient calibration at warm start to ratio0.3, weights then fixed. Both penalties computed by every continuation; backward work differs.',
           evaluation='Final2250-step models only, no seed/checkpoint selection. Mean3 seeds: mean relativeL2 over4 noninitial times primary; full-grid gradient/curvature, high-band energy, edge8 concentration and seam-error secondary.',
           screen='Exploratory reporting gate: vs same-task linear_none, field mean<=1.01*baseline AND curvature mean<=0.5*baseline. No significance/independent confirmation claim or winner export.',
           historical_control='Linear warm/none/low must reproduce existing corresponding dense-clean models tensor-for-tensor where matching budgets exist; compare old checkpoints read-only.',
           limits='Two PDE development diagnosis ablation, not novel architecture, improved time dynamics or independent verification. Both previous500 Burgers sets consumed; reserved Allen500 not generated.')
    locked_json(OUT/'plan.json',p);return p


def run(p):
    old=json.loads((ROOT/'results/physics_stress_v1/plan.json').read_text());groups={};completed=0;initial={};streams={}
    device=torch.device('cuda')
    for task,params in p['tasks'].items():
        clean,data=clean_data(old,task);a,obs=observations(clean,'dense_clean')
        context=dict(plan=digest(p),task=task,data=data,observations=obs)
        for position in p['positions']:
            base={k:p[k] for k in ['width','modes','layers','weight_decay','batch_size','eval_every','checkpoint_every','resolved_max_mode']}
            base.update({k:params[k] for k in ['pde','length','diffusion','reaction','time_lower','time_upper']})
            base.update(position=position,position_center=float(np.mean(a['grid256'])),position_scale=float(np.std(a['grid256'])*np.sqrt(2)),evaluation_columns=[1,2,3,4])
            root=OUT/task/position;warm_paths={};pair={}
            for seed in p['seeds']:
                c=dict(base,seed=seed,kind='warm',strength=0.,steps=p['warm_steps'],lr=.001,lr_min=.00001)
                r=train_one(c,a,context,root/f'warm_s{seed}',device);completed+=1;warm_paths[seed]=root/f'warm_s{seed}/checkpoints/latest.pt'
                key=(task,seed)
                if key in initial and initial[key]!=r['initial']:raise ValueError('encoding initial tensors differ')
                initial[key]=r['initial']
            for kind in p['kinds']:
                rows=[]
                for seed in p['seeds']:
                    c=dict(base,seed=seed,kind=kind,strength=0. if kind=='none' else p['strength'],steps=p['continuation_steps'],lr=.0003,lr_min=.00001)
                    r=train_one(c,a,context,root/f'{kind}_s{seed}',device,warm_path=warm_paths[seed]);rows.append(r);completed+=1
                    v=(r['initial'],r['batch_hash'],r['collocation_hash'],r['calibration'],r['parameters_real'])
                    if seed in pair and pair[seed]!=v:raise ValueError('smoothing fork mismatch')
                    pair[seed]=v;key=(task,seed);stream=(r['batch_hash'],r['collocation_hash'],r['parameters_real'])
                    if key in streams and streams[key]!=stream:raise ValueError('encoding streams/capacity differ')
                    streams[key]=stream
                name=f'{task}/{position}/{kind}'
                groups[name]=dict(task=task,position=position,kind=kind,field_mean=float(np.mean([r['metrics']['val']['primary'] for r in rows])),
                                  seed_sd=float(np.std([r['metrics']['val']['primary'] for r in rows],ddof=1)),per_seed=[r['metrics']['val']['primary'] for r in rows])
                atomic_json(OUT/'progress.json',dict(completed_runs=completed,total_runs=48,groups=groups,confirmation_opened=False))
    atomic_json(OUT/'summary.json',dict(plan=digest(p),completed_runs=completed,total_runs=48,groups=groups,confirmation_opened=False))
    print('PAIRED48 COMPLETE',json.dumps(groups),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
