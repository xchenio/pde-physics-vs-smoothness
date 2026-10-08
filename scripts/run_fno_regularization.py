"""Locked development grid: PDE versus matched resolved smoothness penalties."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_token import load_data
from scripts.fno_regularization_core import train_one
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/fno_regularization_v1'


def plan():
    base=ROOT/'results/time_fno_baseline_v1'
    if not json.loads((base/'summary.json').read_text())['gate_passed']:raise ValueError('time architecture gate failed')
    p=dict(parent_plan_sha256=file_hash(base/'plan.json'),parent_summary_sha256=file_hash(base/'summary.json'),
           source={f.relative_to(ROOT).as_posix():file_hash(f) for f in [Path(__file__),ROOT/'scripts/fno_regularization_core.py',ROOT/'scripts/time_fno_baseline.py',ROOT/'src/time_fno.py',ROOT/'src/fno.py']},
           warm_checkpoints={str(seed):file_hash(base/f's{seed}/checkpoints/latest.pt') for seed in [0,1,2]},
           seeds=[0,1,2],kinds=['physics','gradient','curvature'],strengths=[.03,.1,.3],
           steps=750,width=16,modes=16,layers=4,lr=.0003,lr_min=.00001,weight_decay=.0001,
           batch_size=16,eval_every=250,checkpoint_every=50,diffusion=.001/float(np.pi),resolved_max_mode=16,
           pairing='Each seed forks its SAME1500-step data-only model+optimizer+minibatch RNG; all arms including no-penalty control continue750 steps with common cosine LR restart. Separate shared collocation RNG, time uniform0.1..1 for labeled batch trajectories. All compute same penalty forwards, but backward cost differs.',
           calibration='Training-only first16 IDs, fixed linspace0.1..1; each penalty weight=strength*||grad Ldata||/||grad Lpenalty|| at warm start, held fixed. Complex parameters counted by real components. Effective later ratios diagnostic only.',
           controls='Resolved modes0..16 for PDE, gradient and curvature alike; this is a matched-band smoothness comparison, not optimally tuned full-band regularization. Losses are in physical units divided by target scale squared.',
           primary='Same reused development200, relativeL2 trajectory avg4 noninitial times, average3 seeds. Every grid result retained, fixed final2250 total steps, no validation checkpoint selection.',
           selection='Within each penalty family choose lowest three-seed development mean, tie by smallest strength. Compare selected physics to none and both selected smoothness controls; development selection is exploratory, not confirmation.',
           confirmation='No fresh IDs selected/opened here. Select/seal all rules/models before a separate fresh-ID confirmation protocol; previous500 already consumed.',
           limitation='Same100 trajectories x5 fitting labels; development reused in many prior designs. No noise/extrapolation/second-PDE claim. Time-conditioned architecture differs from historical five-head FNO.')
    locked_json(OUT/'plan.json',p);return p


def run(p):
    parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text());arrays,data=load_data(parent)
    context=dict(plan=digest(p),data=data);groups={};pair={}
    arms=[('none',0.)]+[(k,s) for k in p['kinds'] for s in p['strengths']]
    for kind,strength in arms:
        name=f'{kind}_r{strength:g}';runs=[]
        for seed in p['seeds']:
            c={k:p[k] for k in ['steps','width','modes','layers','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every','diffusion','resolved_max_mode']}
            c.update(kind=kind,strength=strength,seed=seed)
            warm=ROOT/f'results/time_fno_baseline_v1/s{seed}/checkpoints/latest.pt'
            if file_hash(warm)!=p['warm_checkpoints'][str(seed)]:raise ValueError('warm checkpoint changed')
            r=train_one(c,arrays,context,warm,OUT/f'{name}_s{seed}',torch.device('cuda'));runs.append(r)
            value=dict(initial=r['initial'],batch_hash=r['batch_hash'],collocation_hash=r['collocation_hash'],parameters=r['parameters_real'],calibration=r['calibration'])
            if seed in pair and pair[seed]!=value:raise ValueError('paired continuation mismatch')
            pair[seed]=value
        groups[name]=dict(kind=kind,strength=strength,development_mean=float(np.mean([r['metrics']['val']['primary'] for r in runs])),
                          development_seed_sd=float(np.std([r['metrics']['val']['primary'] for r in runs],ddof=1)),
                          per_seed=[r['metrics']['val']['primary'] for r in runs],train_mean=float(np.mean([r['metrics']['train']['primary'] for r in runs])),
                          training_probe_penalties={k:float(np.mean([r['history'][-1]['training_probe_penalties'][k] for r in runs])) for k in p['kinds']})
        atomic_json(OUT/'progress.json',dict(completed_runs=len(groups)*3,total_runs=30,groups=groups))
    selected={k:min([n for n,v in groups.items() if v['kind']==k],key=lambda n:(groups[n]['development_mean'],groups[n]['strength'])) for k in p['kinds']}
    summary=dict(plan=digest(p),completed_runs=30,groups=groups,selected=selected,pairing={str(k):v for k,v in pair.items()},confirmation_opened=False)
    atomic_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
