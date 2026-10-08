"""Registered fixed-architecture stress pilot then nonlinear Allen-Cahn control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save
from scripts.run_public_ridge_residual import digest,locked_json
from scripts.run_physics_token import load_data
from scripts.physics_stress_core import train_one
from scripts.allen_cahn_data import initial
from scripts.allen_cahn_trajectory_v1 import trajectories,qa
OUT=ROOT/'results/physics_stress_v1';DATA=ROOT/'external_data/physics_stress_v1'


def observations(clean,condition,noise_fraction=.1,noise_seed=2026100810):
    columns=[0,len(clean['times'])-1] if condition in ('sparse','noisy_sparse') else list(range(len(clean['times'])))
    # Corruption amplitude uses ONLY selected observed training times, never omitted labels.
    noisy=np.array(clean['train_y'][:,columns],copy=True,dtype=float)
    sigma=noise_fraction*float(np.sqrt(np.mean(noisy[:,1:]**2)))
    noise=np.random.default_rng(noise_seed).normal(0.,sigma,size=noisy[:,1:].shape)
    if condition in ('noisy_dense','noisy_sparse'):noisy[:,1:]+=noise
    a=dict(train_x=clean['train_x'],train_y=noisy,val_x=clean['val_x'],val_y=clean['val_y'],
           times=[clean['times'][i] for i in columns],val_times=clean['times'],grid256=clean['grid256'])
    return a,dict(observation_columns=columns,noise_sigma=sigma if condition.startswith('noisy') else 0.,
                  noise_seed=noise_seed,input_noise=False,initial_label_noise=False,
                  observation_sha256=digest(a['train_y'].tolist()))


def plan():
    parent=ROOT/'results/fno_regularization_confirmation_v1/confirmation.json'
    old=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text())
    source_files=[Path(__file__),ROOT/'scripts/physics_stress_core.py',ROOT/'src/physics_equations_v1.py',
                  ROOT/'scripts/allen_cahn_trajectory_v1.py',ROOT/'scripts/allen_cahn_data.py',ROOT/'src/time_fno.py',
                  ROOT/'src/fno.py',ROOT/'scripts/time_fno_baseline.py',ROOT/'scripts/fno_regularization_core.py']
    p=dict(parent_confirmation_sha256=file_hash(parent),source={f.relative_to(ROOT).as_posix():file_hash(f) for f in source_files},
           seeds=[0,1,2],kinds=['none','physics','curvature'],strength=.3,warm_steps=1500,continuation_steps=750,
           width=16,modes=16,layers=4,weight_decay=.0001,batch_size=16,eval_every=250,checkpoint_every=50,resolved_max_mode=16,
           noise_fraction=.1,noise_seed=2026100810,
           tasks=dict(burgers=dict(pde='burgers',length=1.,diffusion=.001/float(np.pi),reaction=0.,time_lower=.1,time_upper=1.,
                                   train_ids=old['train_ids'],val_ids=old['val_ids'],conditions=['sparse','noisy_dense','noisy_sparse']),
                      allen_cahn=dict(pde='allen_cahn',length=2.,diffusion=.01,reaction=5.,time_lower=.05,time_upper=.5,
                                      train_ids=list(range(200000,200100)),val_ids=list(range(200100,200300)),
                                      reserved_confirmation=list(range(200300,200800)),conditions=['dense_clean'],times=[0.,.05,.125,.25,.5])),
           architecture='Unchanged TimeFNO1d16wide16modes4layers34209 real params; zero final head at observation-only warm start. Known initial mean offset; NOT a conservation constraint for Allen-Cahn.',
           budget='Each condition fits its OWN1500-step observation-only warm start (no previously dense clean weights), then same optimizer/weights/RNG fork750 steps. No penalty/physics/curvature share all observations and batches.',
           regularization='Transfer previously selected strength0.3, no condition-specific strength search. Training-observation gradient calibration at frozen warm start only. Fixed modes0..16 for PDE and curvature; all continuation penalty forwards, backward costs differ.',
           noise='Fixed Gaussian FUTURE LABEL noise0.1 times global RMS of SELECTED observed clean training future labels; omitted time labels never affect noise amplitude or fitting. Same fixed corruption within each condition across all seeds/arms. Branch initial field and IC labels clean. Not noisy-input robustness.',
           primary='Sparse conditions: mean relativeL2 at OMITTED intermediate times (columns1/2/3), final observed time secondary. Dense conditions: mean at all4 noninitial clean development labels. Average3 fits. These development trajectories and Burgers query times were used in prior design, not independent validation.',
           physics='Burgers conservative dealiased square; Allen-Cahn ut=0.01uxx+5u(1-u^2), dealiased cube4x FFT and native length2. Temporal derivative JVP, uniform noninitial query time, no collocation target labels.',
           sequencing='Burgers36 fits first; then new synthetic nonlinear Allen-Cahn12 fits, after new solver/residual QA passes. Historical Allen-Cahn final-time tests are not physics validation.',
           confirmation='None opened by this driver. Both prior500 sets consumed. Register independent stress confirmation only after all development decisions sealed. New Allen-Cahn reserved500 not generated here.',
           limitation='Pilot at one transferred penalty strength, not fully tuned optimal regularization or proof of architecture novelty. Noise uncertainty conditional on one predefined corruption; three model seeds not three independent noise realizations.')
    locked_json(OUT/'plan.json',p);return p


def clean_data(p,task):
    if task=='burgers':
        parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text());a,context=load_data(parent)
        return a,context
    params=p['tasks'][task];qpath=OUT/'allen_cahn_solver_qa.json'
    if qpath.exists():q=json.loads(qpath.read_text())
    else:q=qa();atomic_json(qpath,q)
    if not q['passed']:raise ValueError('new Allen-Cahn trajectory/residual QA failed')
    values={};hashes={}
    for split,key in [('train','train_ids'),('val','val_ids')]:
        chunks=[];ids=params[key]
        for start in range(0,len(ids),50):
            part=ids[start:start+50];path=DATA/f'allen_{split}_{start:04d}.pt'
            if path.exists():
                obj=torch.load(path,weights_only=False)
                if obj['plan']!=digest(p) or obj['ids']!=part:raise ValueError('synthetic cache mismatch')
            else:
                x=initial(part);y=trajectories(x,params['times']);obj=dict(plan=digest(p),ids=part,x=x,y=y);atomic_save(obj,path)
            chunks.append(obj);hashes[path.name]=file_hash(path)
            atomic_json(OUT/'allen_data_progress.json',dict(split=split,completed=start+len(part),total=len(ids),confirmation_generated=False))
        values[split+'_x']=np.concatenate([v['x'] for v in chunks]);values[split+'_y']=np.concatenate([v['y'] for v in chunks])
    values.update(times=params['times'],grid256=np.arange(256)*2/256)
    return values,dict(plan=digest(p),chunks=hashes,qa_sha256=file_hash(qpath))


def run(p):
    groups={};total=48;completed=0;device=torch.device('cuda')
    for task,params in p['tasks'].items():
        clean,data=clean_data(p,task)
        for condition in params['conditions']:
            a,obs=observations(clean,condition,p['noise_fraction'],p['noise_seed'])
            context=dict(plan=digest(p),task=task,condition=condition,data=data,observations=obs)
            cols=[1,2,3] if condition in ('sparse','noisy_sparse') else [1,2,3,4]
            base={k:p[k] for k in ['width','modes','layers','weight_decay','batch_size','eval_every','checkpoint_every','resolved_max_mode']}
            base.update({k:params[k] for k in ['pde','length','diffusion','reaction','time_lower','time_upper']});base['evaluation_columns']=cols
            root=OUT/task/condition;pair={};warm_paths={}
            for seed in p['seeds']:
                c=dict(base,seed=seed,kind='warm',strength=0.,steps=p['warm_steps'],lr=.001,lr_min=.00001)
                train_one(c,a,context,root/f'warm_s{seed}',device);warm_paths[seed]=root/f'warm_s{seed}/checkpoints/latest.pt';completed+=1
            for kind in p['kinds']:
                rows=[]
                for seed in p['seeds']:
                    c=dict(base,seed=seed,kind=kind,strength=0. if kind=='none' else p['strength'],steps=p['continuation_steps'],lr=.0003,lr_min=.00001)
                    r=train_one(c,a,context,root/f'{kind}_s{seed}',device,warm_path=warm_paths[seed]);rows.append(r);completed+=1
                    value=(r['initial'],r['batch_hash'],r['collocation_hash'],r['calibration'],r['parameters_real'])
                    if seed in pair and pair[seed]!=value:raise ValueError('stress fork pairing mismatch')
                    pair[seed]=value
                name=f'{task}/{condition}/{kind}'
                groups[name]=dict(task=task,condition=condition,kind=kind,primary_columns=cols,development_mean=float(np.mean([r['metrics']['val']['primary'] for r in rows])),
                                  seed_sd=float(np.std([r['metrics']['val']['primary'] for r in rows],ddof=1)),per_seed=[r['metrics']['val']['primary'] for r in rows],
                                  by_time=np.mean([r['metrics']['val']['by_time'] for r in rows],axis=0).tolist())
                atomic_json(OUT/'progress.json',dict(completed_runs=completed,total_runs=total,groups=groups,confirmation_opened=False))
    summary=dict(plan=digest(p),completed_runs=completed,total_runs=total,groups=groups,confirmation_opened=False)
    atomic_json(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
