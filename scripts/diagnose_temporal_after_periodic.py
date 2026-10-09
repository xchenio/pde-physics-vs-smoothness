"""Resumable per-model frozen time-dynamics audit after seam intervention."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_stress import clean_data
from scripts.periodic_curvature_core import make_model
from scripts.time_fno_baseline import predict
from scripts.temporal_after_periodic_core import analyze,temporal_score,aggregate
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from src.time_fno import time_fields
from src.public_benchmark import atomic_json,file_hash
OUT=ROOT/'results/temporal_after_periodic_v1';PARENT=ROOT/'results/periodic_curvature_v1'


def plan():
    audit=json.loads((PARENT/'audit.json').read_text())
    if not audit['complete'] or audit['historical_linear_tensor_matches']!=18:raise ValueError('complete parent audit required')
    old=json.loads((ROOT/'results/physics_diagnosis_v2/plan.json').read_text())
    models={}
    for f in sorted(PARENT.rglob('result.json')):
        r=json.loads(f.read_text())
        if r['identity']['config']['kind']=='warm':continue
        models[f.parent.relative_to(PARENT).as_posix()]=dict(path=(f.parent/'checkpoints/final.pt').relative_to(ROOT).as_posix(),sha256=r['final_sha256'],result_sha256=file_hash(f))
    refs={task:dict(path=f'external_data/physics_diagnosis_v2/{task}_reference.pt',sha256=file_hash(ROOT/f'external_data/physics_diagnosis_v2/{task}_reference.pt')) for task in old['dev_ids']}
    files=[Path(__file__),ROOT/'scripts/temporal_after_periodic_core.py',ROOT/'scripts/periodic_curvature_core.py',ROOT/'src/periodic_position_fno.py',ROOT/'src/time_fno.py',ROOT/'src/fno.py',ROOT/'scripts/field_derivative_diagnostics.py',ROOT/'scripts/run_physics_stress.py',ROOT/'scripts/time_fno_baseline.py']
    p=dict(parent_plan_sha256=file_hash(PARENT/'plan.json'),parent_audit_sha256=file_hash(PARENT/'audit.json'),
        reference_plan_sha256=file_hash(ROOT/'results/physics_diagnosis_v2/plan.json'),reference_plan_digest=digest(old),
        models=models,references=refs,dev_ids=old['dev_ids'],model_fd_half_widths=[.002,.001,.0005],
        source={f.relative_to(ROOT).as_posix():file_hash(f) for f in files},
        policy='All36 final models, existing first32 development IDs/four observed future times, reused hashed native caches only. No new IDs/times/labels/training/modules/confirmation.',
        references_policy='Allen primary native semidiscrete RHS checked against two FD widths; Burgers FDh.01 with .02 sensitivity, NOT exact truth. Resolved0..16 native Fourier projection BEFORE subsampling; full sampled to256 explicitly has aliasing/reference limitations.',
        decision='For Allen periodic/none, all3seeds: resolved error>=.1, >=10x max native FDwidth/FDvsRHS uncertainty, time error energy>rhs-field error energy in common native-ut norm, oracle time reduces residual. Descriptive hypothesis screen only; does not prove causal field bottleneck.',
        limitations='Observed times32 reused cases, not unseen-time interpolation, all-dev estimate, inferential significance or unique causation. Oracle derivative replacements are diagnostics, not implementable methods. No constraint trained in this family.')
    locked_json(OUT/'plan.json',p);return p


def load_task(p,task):
    entry=p['references'][task];path=ROOT/entry['path']
    if file_hash(path)!=entry['sha256']:raise ValueError('reference changed')
    ref=torch.load(path,weights_only=False)
    if ref['plan']!=p['reference_plan_digest'] or ref['ids']!=p['dev_ids'][task]:raise ValueError('reference identity/IDs changed')
    old=json.loads((ROOT/'results/physics_stress_v1/plan.json').read_text());a,_=clean_data(old,task)
    if old['tasks'][task]['val_ids'][:32]!=ref['ids']:raise ValueError('development ordering changed')
    np.testing.assert_allclose(ref['fields'][256],a['val_y'][:32,1:],rtol=1e-5,atol=1e-6)
    return a,ref


def model_inputs(p,name,a,device):
    entry=p['models'][name];path=ROOT/entry['path']
    if file_hash(path)!=entry['sha256']:raise ValueError('model changed')
    result=path.parent.parent/'result.json'
    if file_hash(result)!=entry['result_sha256']:raise ValueError('parent result changed')
    cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['identity']['config']
    model=make_model(c,c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
    raw=np.asarray(a['val_x'][:32],dtype=float);norm=cp['normalization']
    x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
    coords=torch.tensor(a['grid256'],dtype=torch.float32,device=device)
    times=torch.tensor(a['times'][1:],dtype=torch.float32,device=device)
    b=x.repeat_interleave(4,0);tt=times.repeat(32)
    off=torch.tensor(raw.mean(1),dtype=torch.float32,device=device).repeat_interleave(4)
    return model,c,cp,x,coords,times,b,tt,off


def run(p):
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed source changed')
    cache={};rows={};device=torch.device('cuda')
    for name,entry in p['models'].items():
        task=name.split('/')[0]
        if task not in cache:cache[task]=load_task(p,task)
        a,ref=cache[task];path=OUT/'runs'/f'{name.replace("/","__")}.json'
        identity=dict(plan=digest(p),model=entry['sha256'],reference=p['references'][task]['sha256'])
        if path.exists():
            r=json.loads(path.read_text())
            if r['identity']!=identity:raise ValueError('result changed')
            rows[name]=r;continue
        model,c,cp,x,coords,times,b,tt,off=model_inputs(p,name,a,device)
        _,ut=time_fields(model,b,tt,coords,cp['scale'],off,create_graph=False)
        # Match parent inference batches and add the known mean in float64, not
        # inside a float32 JVP forward (which would round the constant offset).
        with torch.no_grad():
            all_times=torch.tensor(a['times'],dtype=torch.float32,device=device)
            pred=torch.cat([predict(model,q,all_times,coords) for q in x.split(16)]).cpu().numpy().astype(float)
        pred=(pred*cp['scale']+np.asarray(a['val_x'][:32],dtype=float).mean(1)[:,None,None])[:,1:]
        ut=ut.detach().cpu().numpy().astype(float).reshape(32,4,256)
        own={}
        with torch.no_grad():
            for h in p['model_fd_half_widths']:
                fd=((model(b,tt+h,coords)-model(b,tt-h,coords))*cp['scale']/(2*h)).cpu().numpy().astype(float).reshape(32,4,256)
                own[f'{h:g}']=temporal_score(fd,ut)
        field=temporal_score(pred,np.asarray(a['val_y'][:32,1:],dtype=float))
        parent=json.loads((ROOT/entry['path']).parent.parent.joinpath('result.json').read_text())
        np.testing.assert_allclose(field['per_sample'],parent['metrics']['val']['per_sample'][:32],rtol=1e-10,atol=1e-11)
        r=dict(identity=identity,field32=field,temporal=analyze(pred,ut,ref,c),own_fd=own,git_commit=git_commit())
        atomic_json(path,r);rows[name]=r
        atomic_json(OUT/'progress.json',dict(completed=len(rows),total=36,training_performed=False,confirmation_opened=False))
        print('TIME_DIAG',name,'resolved',r['temporal']['truth_resolved_native']['mean'],flush=True)
    groups,decision=aggregate(rows)
    atomic_json(OUT/'summary.json',dict(plan=digest(p),checked_models=len(rows),groups=groups,decision=decision,training_performed=False,confirmation_opened=False))
    print('TEMPORAL36 COMPLETE',json.dumps(groups),json.dumps(decision),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
