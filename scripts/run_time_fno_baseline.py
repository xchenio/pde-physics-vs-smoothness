"""Registered time-conditioned data-only control; no confirmation access."""
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
from scripts.time_fno_baseline import train_one
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/time_fno_baseline_v1'


def plan():
    parent_path=ROOT/'results/physics_token_v1/plan.json';parent=json.loads(parent_path.read_text())
    audit_path=ROOT/'results/fno_physics_resolution_v1/audit.json'
    if not json.loads(audit_path.read_text())['feasibility_gate_passed']:raise ValueError('physics feasibility gate failed')
    p=dict(parent_sha256=file_hash(parent_path),resolution_audit_sha256=file_hash(audit_path),
           source={f.relative_to(ROOT).as_posix():file_hash(f) for f in [Path(__file__),ROOT/'scripts/time_fno_baseline.py',ROOT/'src/time_fno.py',ROOT/'src/fno.py']},
           train_ids=parent['train_ids'],val_ids=parent['val_ids'],times=parent['times'],
           seeds=[0,1,2],steps=1500,width=16,modes=16,layers=4,lr=.001,lr_min=.00001,
           weight_decay=.0001,batch_size=16,eval_every=250,checkpoint_every=50,
           diffusion=parent['diffusion'],resolved_max_mode=16,
           fitting='Same100 trajectories, all5 registered labels per minibatch; train-only scalar normalization, guarded normalized squared field loss, zero final head, fixed1500 steps. No extra adjacent time label fitting.',
           architecture='Query time broadcast as third lift input; same16wide4-layer FNO backbone, single pointwise output channel. Known initial sample mean added in physical units.',
           primary='Development200 mean trajectory relativeL2 over four noninitial times, three seeds averaged. No checkpoint selection.',
           next_stage_gate='Three-seed development mean <=1.10 times historical same100-shot discrete-head FNO16 development mean. Contextual architecture feasibility, not independent confirmation.',
           confirmation='NONE opened or selected in this prerequisite. Previous500 consumed.')
    locked_json(OUT/'plan.json',p);return p


def run(p):
    parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text())
    arrays,old_context=load_data(parent);context=dict(plan=digest(p),data=old_context)
    results=[]
    for seed in p['seeds']:
        c={k:p[k] for k in ['steps','width','modes','layers','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every','diffusion','resolved_max_mode']}
        c['seed']=seed
        results.append(train_one(c,arrays,context,OUT/f's{seed}',torch.device('cuda')))
        atomic_json(OUT/'progress.json',dict(completed_runs=len(results),total_runs=len(p['seeds'])))
    old=json.loads((ROOT/'results/fno_small_trajectory_v1/summary.json').read_text())
    # Read the recorded comparator structure without altering its protocol or labels.
    old_mean=old['groups']['16']['metrics']['val']['mean']
    mean=float(np.mean([r['metrics']['val']['primary'] for r in results]))
    summary=dict(plan=digest(p),completed_runs=len(results),parameters_real=results[0]['parameters_real'],
                 development_mean=mean,development_seed_sd=float(np.std([r['metrics']['val']['primary'] for r in results],ddof=1)),
                 historical_discrete_head_development_mean=old_mean,gate_passed=mean<=1.10*old_mean,
                 per_seed=[dict(seed=i,primary=r['metrics']['val']['primary'],train=r['metrics']['train']['primary'],training_probe_resolved_pde_rms=r['training_probe_resolved_pde_rms'],batch_hash=r['batch_hash']) for i,r in zip(p['seeds'],results)],
                 confirmation_opened=False)
    atomic_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
