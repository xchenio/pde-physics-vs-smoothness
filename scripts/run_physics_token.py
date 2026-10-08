"""Registered Burgers100-shot trajectory pilot; unused confirmation IDs reserved."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_field_modal import plan as field_plan
from scripts.run_token_operator import plan as token_plan
from scripts.run_public_ridge_residual import digest,locked_json
from scripts.physics_token_core import train_one
from scripts.manifold_reconstruction import array_digest
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save
OUT=ROOT/'results/physics_token_v1';DATA=ROOT/'external_data/physics_token_v1'


def historical_ids():
    path=OUT/'historical_id_audit.json'
    if path.exists():
        old=json.loads(path.read_text())
        for name,h in old['artifacts'].items():
            if file_hash(ROOT/name)!=h:raise ValueError('historical ID registry changed')
        return old
    used=set(range(128));artifacts={}
    keys={'train','val','test','reserved','reserved_confirmation','confirmation_ids','train_ids','val_ids','test_ids'}
    def visit(value):
        if not isinstance(value,dict):
            if isinstance(value,list):
                for v in value:visit(v)
            return
        for k,v in value.items():
            if k in keys and isinstance(v,list) and all(isinstance(i,int) for i in v):used.update(v)
            visit(v)
    paths=sorted((ROOT/'results').rglob('plan*.json'))+[ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json']
    for f in paths:
        if OUT in f.parents:continue
        visit(json.loads(f.read_text()));artifacts[f.relative_to(ROOT).as_posix()]=file_hash(f)
    old=dict(used_ids=sorted(i for i in used if 0<=i<10000),artifacts=artifacts,
             policy='conservative union of every registered integer trajectory-ID list, including synthetic IDs; generic smoke prefix128 excluded')
    locked_json(path,old);return old


def plan():
    OUT.mkdir(parents=True,exist_ok=True);old=field_plan();parent=token_plan();s=old['parent']['tasks']['burgers_original'];history=historical_ids()
    available=sorted(set(range(10000))-set(history['used_ids']))
    if len(available)<500:raise ValueError('insufficient genuinely unused trajectories')
    confirm=np.random.default_rng(2026100805).permutation(available)[:500].tolist()
    p=dict(source=digest([parent['source'],file_hash(Path(__file__)),file_hash(ROOT/'scripts/physics_token_core.py')]),parent=digest(parent),
           dataset_source=s['source'],dataset_sha256=s['sha256'],historical_id_audit_sha256=file_hash(OUT/'historical_id_audit.json'),
           train_ids=[s['train'][i] for i in old['correction_rows'][:100]],val_ids=s['val'],confirmation_ids=confirm,
           time_indices=[0,10,25,50,100],times=[0.,.1,.25,.5,1.],train_grid=256,secondary_output_grid=512,
           kinds=['gate','attention'],physics_weights=[0.,.01],seeds=[0,1,2],steps=1500,
           lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=16,collocation=32,eval_every=250,checkpoint_every=50,
           diffusion=.001/float(np.pi),normalization='train100-only scalar input mean/std and trajectory target std; no representation400 target or future-field branch input',
           loss='per-time energy-guarded normalized field squared loss at all5 times including IC; +lambda sample-weighted physical residual squared / target_scale^2. Periodic smooth trunk hard-encodes BC.',
           pairing='all12 fits have identical initial tensors, labeled minibatches and collocation points; off-arm also computes PDE for diagnostic compute fairness',
           primary='mean per-trajectory relativeL2 averaged over noninitial times0.1/0.25/0.5/1 on output256, three seeds averaged; small-data stress only',
           secondary='same256 input sensors, query512 native retained cell centers/labels; output-grid transfer only, not input-resolution transfer; per-time and PDE values descriptive',
           confirmation='all four locked arms,500 unused original-PDE trajectories;4 primary contrasts: attention0-gate0,gateP-gate0,attentionP-attention0,attentionP-gate0;paired trajectory bootstrap10000,Bonferroni4 two-sided CI,>=1%gain AND upper<0;seed2026100806',
           limits='one Burgers PDE100-shot pilot, not second-PDE physics validation or noise/temporal extrapolation.512 contains256 training grid positions; extra query-point errors reported separately in confirmation. No superiority over FNO inferred without a matching100-shot control.')
    locked_json(OUT/'plan.json',p);return p


def load_data(p):
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/'development.pt'
    if path.exists():
        obj=torch.load(path,weights_only=False)
        if obj['plan']!=digest(p):raise ValueError('data cache mismatch')
        return obj['arrays'],dict(plan=digest(p),data_sha256=file_hash(path))
    if file_hash(Path(p['dataset_source']))!=p['dataset_sha256']:raise ValueError('source checksum mismatch')
    with h5py.File(p['dataset_source'],'r') as f:
        if f['tensor'].shape!=(10000,201,1024) or abs(float(f.attrs['Nu'])-.001)>1e-12:raise ValueError('source physics/layout mismatch')
        np.testing.assert_allclose(f['t-coordinate'][p['time_indices']],p['times'],atol=1e-7)
        arrays=dict(times=p['times'],grid256=f['x-coordinate'][::4],grid512=f['x-coordinate'][::2])
        for split,ids in (('train',p['train_ids']),('val',p['val_ids'])):
            # Each whole trajectory is read once, then registered times/resolutions selected.
            values=np.stack([f['tensor'][int(i),p['time_indices'],:] for i in ids])
            arrays[split+'_x']=values[:,0,::4].copy();arrays[split+'_y']=values[:,:,::4].copy()
            if split=='val':arrays['val_y512']=values[:,:,::2].copy()
    atomic_save(dict(plan=digest(p),arrays=arrays),path)
    return arrays,dict(plan=digest(p),data_sha256=file_hash(path))


def run(p,device):
    arrays,context=load_data(p);groups={};pair={}
    for kind in p['kinds']:
        for weight in p['physics_weights']:
            arm=f'{kind}_p{weight:g}';runs=[]
            for seed in p['seeds']:
                c={k:p[k] for k in ('steps','lr','lr_min','weight_decay','batch_size','collocation','eval_every','checkpoint_every','diffusion')}
                c.update(kind=kind,physics_weight=weight,seed=seed)
                r=train_one(c,arrays,context,OUT/f'{arm}_s{seed}',device);runs.append(r)
                value=(r['initial'],r['batch_hash'],r['collocation_hash'],r['parameters'])
                if seed in pair and pair[seed]!=value:raise ValueError('factorial pairing mismatch')
                pair[seed]=value
            groups[arm]=dict(parameters=runs[0]['parameters'],metrics={s:dict(mean=float(np.mean([r['metrics'][s]['primary'] for r in runs])),seed_sd=float(np.std([r['metrics'][s]['primary'] for r in runs],ddof=1))) for s in ('train','val','val512')})
            atomic_json(OUT/'progress.json',dict(completed_runs=len(groups)*3,total_runs=12,groups=groups))
    atomic_json(OUT/'summary.json',dict(context=context,groups=groups,completed_runs=12,pairing={str(s):v for s,v in pair.items()}));print('PILOT COMPLETE',groups,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))
