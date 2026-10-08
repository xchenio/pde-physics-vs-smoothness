"""Fresh500 confirmation after committing model/rule/ID seals; no tuning."""
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
from src.time_fno import TimeFNO1d
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save
from scripts.time_fno_baseline import predict
from scripts.physics_token_core import trajectory_errors
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from scripts.run_incremental_modal import committed
OUT=ROOT/'results/fno_regularization_confirmation_v1'
DATA=ROOT/'external_data/fno_regularization_confirmation_v1'
DEV=ROOT/'results/fno_regularization_v1'
CONTRASTS=[('physics_vs_none','physics','none'),('gradient_vs_none','gradient','none'),
           ('curvature_vs_none','curvature','none'),('physics_vs_gradient','physics','gradient'),
           ('physics_vs_curvature','physics','curvature')]


def historical_registry():
    used=set(range(128));artifacts={}
    keys={'train','val','test','reserved','reserved_confirmation','confirmation_ids','train_ids','val_ids','test_ids'}
    def visit(value):
        if isinstance(value,list):
            for v in value:visit(v)
        if isinstance(value,dict):
            for k,v in value.items():
                if k in keys and isinstance(v,list) and all(type(i) is int for i in v):used.update(v)
                visit(v)
    for f in sorted((ROOT/'results').rglob('plan*.json'))+[ROOT/'external_data/pdebench/fixed_t1_v1/protocol.json']:
        if OUT in f.parents:continue
        visit(json.loads(f.read_text()));artifacts[f.relative_to(ROOT).as_posix()]=file_hash(f)
    return dict(used_ids=sorted(i for i in used if 0<=i<10000),artifacts=artifacts,
                policy='Conservative union of ALL current registered trajectory ID lists, including synthetic IDs and the consumed previous500; smoke0..127 excluded.')


def seal():
    for name in ['plan.json','summary.json','checkpoint_audit.json']:committed(DEV/name)
    summary=json.loads((DEV/'summary.json').read_text());audit=json.loads((DEV/'checkpoint_audit.json').read_text())
    if not audit['complete'] or audit['completed_runs']!=30:raise ValueError('full audit required')
    history=historical_registry();locked_json(OUT/'historical_id_audit.json',history)
    available=sorted(set(range(10000))-set(history['used_ids']))
    if len(available)<500:raise ValueError('insufficient unused trajectories')
    ids=np.random.default_rng(2026100808).permutation(available)[:500].tolist()
    selected=dict(none='none_r0',**summary['selected']);models={}
    for family,arm in selected.items():
        models[family]=[]
        for seed in [0,1,2]:
            folder=DEV/f'{arm}_s{seed}';rpath=folder/'result.json';committed(rpath)
            r=json.loads(rpath.read_text());cp=folder/'checkpoints/final.pt'
            if file_hash(cp)!=r['final_sha256']:raise ValueError('model hash mismatch')
            models[family].append(dict(seed=seed,arm=arm,path=cp.relative_to(ROOT).as_posix(),sha256=file_hash(cp),result_sha256=file_hash(rpath)))
    parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text())
    lock=dict(models=models,confirmation_ids=ids,historical_id_audit_sha256=file_hash(OUT/'historical_id_audit.json'),
              development_plan_sha256=file_hash(DEV/'plan.json'),development_summary_sha256=file_hash(DEV/'summary.json'),
              development_audit_sha256=file_hash(DEV/'checkpoint_audit.json'),
              dataset_source=parent['dataset_source'],dataset_sha256=parent['dataset_sha256'],
              time_indices=parent['time_indices'],times=parent['times'],grid=256,
              source={f.relative_to(ROOT).as_posix():file_hash(f) for f in [Path(__file__),ROOT/'src/time_fno.py',ROOT/'src/fno.py',ROOT/'scripts/time_fno_baseline.py',ROOT/'scripts/physics_token_core.py']},
              contrasts=CONTRASTS,bootstrap_replicates=10000,bootstrap_seed=2026100809,multiplicity=5,alpha=.05,min_relative_gain=.01,
              primary='Per-trajectory relativeL2 mean four noninitial times, average3 fitted seeds; five paired trajectory bootstrap contrasts, Bonferroni5 two-sided CI, gain>=1% and upper<0. Conditional on fitted seeds, not seed uncertainty.',
              selection='All four frozen families; per-penalty strength selected on reused development mean, all choose0.3 (upper grid boundary). No further search before this confirmation; not fully tuned optimal regularization.',
              limitations='One Burgers PDE, fixed256 sensors/seen times, same100x5 fitting labels,2250 total steps. No second PDE/time extrapolation/input-grid transfer claim.',preseal_commit=git_commit())
    locked_json(OUT/'plan.json',lock)
    print(f'SEALED12 models; used{len(history["used_ids"])}, available{len(available)}, fresh500. Commit plan/registry before run.',flush=True)


def contrast(a,b,replicates=10000,seed=2026100809,multiplicity=5):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2:raise ValueError('paired trajectories required')
    delta=a-b;rng=np.random.default_rng(seed)
    draws=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(replicates)])
    lo,hi=np.quantile(draws,[.05/(2*multiplicity),1-.05/(2*multiplicity)])
    gain=float(-delta.mean()/b.mean())
    return dict(difference=float(delta.mean()),relative_gain=gain,interval=[float(lo),float(hi)],success=bool(gain>=.01 and hi<0))


def load_arrays(p):
    locked_json(OUT/'confirmation_opened.json',dict(plan=digest(p),confirmation_ids=p['confirmation_ids'],consumed=True,
               note='Consumed from first extraction. Never call these trajectories unseen again.'))
    path=DATA/'confirmation.pt'
    if path.exists():
        obj=torch.load(path,weights_only=False)
        if obj['plan']!=digest(p):raise ValueError('cache mismatch')
        return obj['arrays'],file_hash(path)
    if file_hash(Path(p['dataset_source']))!=p['dataset_sha256']:raise ValueError('dataset changed')
    with h5py.File(p['dataset_source'],'r') as f:
        if f['tensor'].shape!=(10000,201,1024):raise ValueError('layout mismatch')
        np.testing.assert_allclose(f['t-coordinate'][p['time_indices']],p['times'],atol=1e-7)
        values=np.stack([f['tensor'][int(i),p['time_indices'],::4] for i in p['confirmation_ids']])
        arrays=dict(x=values[:,0,:].copy(),y=values,grid=f['x-coordinate'][::4],times=p['times'])
    atomic_save(dict(plan=digest(p),arrays=arrays),path)
    return arrays,file_hash(path)


def run(replay=False):
    p=json.loads((OUT/'plan.json').read_text());committed(OUT/'plan.json');committed(OUT/'historical_id_audit.json')
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed source changed: '+name)
    history=json.loads((OUT/'historical_id_audit.json').read_text())
    if file_hash(OUT/'historical_id_audit.json')!=p['historical_id_audit_sha256']:raise ValueError('registry changed')
    for name,h in history['artifacts'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('historical plan changed')
    if set(p['confirmation_ids'])&set(history['used_ids']) or len(set(p['confirmation_ids']))!=500:raise ValueError('ID independence failed')
    arrays,dh=load_arrays(p);device=torch.device('cuda');rows={};means={}
    coords=torch.tensor(arrays['grid'],dtype=torch.float32,device=device);times=torch.tensor(arrays['times'],dtype=torch.float32,device=device)
    raw=np.asarray(arrays['x'],dtype=float)
    for family,entries in p['models'].items():
        records=[]
        for entry in entries:
            cp_path=ROOT/entry['path']
            if file_hash(cp_path)!=entry['sha256']:raise ValueError('model changed')
            path=OUT/'runs'/f"{family}_s{entry['seed']}.json"
            identity=dict(plan=digest(p),model=entry['sha256'],data_sha256=dh)
            if path.exists() and not replay:
                row=json.loads(path.read_text())
                if row['identity']!=identity:raise ValueError('result mismatch')
            else:
                cp=torch.load(cp_path,map_location='cpu',weights_only=False);c=cp['identity']['config']
                model=TimeFNO1d(c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
                norm=cp['normalization'];x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
                with torch.no_grad():pred=torch.cat([predict(model,b,times,coords) for b in x.split(16)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1)[:,None,None]
                e=trajectory_errors(pred,arrays['y']);row=dict(identity=identity,seed=entry['seed'],primary=float(e[:,1:].mean()),per_sample=e[:,1:].mean(1).tolist(),by_time=e.mean(0).tolist())
                if replay:
                    old=json.loads(path.read_text())
                    if old['identity']!=identity:raise ValueError('replay identity mismatch')
                    for k in ['primary','per_sample','by_time']:np.testing.assert_allclose(row[k],old[k],rtol=1e-11,atol=1e-12)
                else:atomic_json(path,row)
            records.append(row);print('REPLAY' if replay else 'CONFIRM',family,entry['seed'],row['primary'],flush=True)
        rows[family]=dict(mean=float(np.mean([r['primary'] for r in records])),seed_sd=float(np.std([r['primary'] for r in records],ddof=1)),per_seed=[r['primary'] for r in records])
        means[family]=np.mean([r['per_sample'] for r in records],axis=0)
    comparisons={name:contrast(means[a],means[b],p['bootstrap_replicates'],p['bootstrap_seed'],p['multiplicity']) for name,a,b in p['contrasts']}
    result=dict(plan=digest(p),data_sha256=dh,trajectories=500,groups=rows,comparisons=comparisons,confirmation_consumed=True)
    if replay:
        if result!=json.loads((OUT/'confirmation.json').read_text()):raise ValueError('summary/decision replay mismatch')
        atomic_json(OUT/'confirmation_audit.json',dict(plan=digest(p),checked_models=12,checked_decisions=5,predictions_replayed=True,confirmation_consumed=True))
    else:atomic_json(OUT/'confirmation.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','run','audit']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='seal':seal()
    else:run(replay=args.stage=='audit')
