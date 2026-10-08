"""Checkpoint/observation/time-label/batch replay without fresh confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_stress import OUT,clean_data,observations
from scripts.time_fno_baseline import predict
from scripts.physics_token_core import trajectory_errors
from src.time_fno import TimeFNO1d,spectral_derivative
from src.public_benchmark import file_hash,atomic_json


def run(require_complete):
    p=json.loads((OUT/'plan.json').read_text());paths=sorted(OUT.rglob('result.json'));pair={};evidence=[];cache={}
    if require_complete and len(paths)!=48:raise ValueError('incomplete')
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('source changed')
    device=torch.device('cuda');groups={}
    for path in paths:
        r=json.loads(path.read_text());c=r['identity']['config'];ctx=r['identity']['context'];seed=c['seed'];task=ctx['task'];condition=ctx['condition']
        key=(task,condition)
        if key not in cache:
            clean,_=clean_data(p,task);a,obs=observations(clean,condition,p['noise_fraction'],p['noise_seed']);cache[key]=(a,obs)
        a,obs=cache[key]
        if obs!=ctx['observations']:raise ValueError('observation changed')
        cp_path=path.parent/'checkpoints/final.pt'
        if file_hash(cp_path)!=r['final_sha256']:raise ValueError('checkpoint changed')
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=r['identity'] or cp['step']!=c['steps']:raise ValueError('checkpoint identity/stage mismatch')
        model=TimeFNO1d(c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
        norm=cp['normalization'];raw=np.asarray(a['val_x'],dtype=float);x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
        t=torch.tensor(a['val_times'],dtype=torch.float32,device=device);coords=torch.tensor(a['grid256'],dtype=torch.float32,device=device)
        with torch.no_grad():pred=torch.cat([predict(model,b,t,coords) for b in x.split(16)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1)[:,None,None]
        e=trajectory_errors(pred,a['val_y']);cols=c['evaluation_columns']
        np.testing.assert_allclose(e[:,cols].mean(1),r['metrics']['val']['per_sample'],rtol=1e-11,atol=1e-12)
        rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest()
        steps=c['steps']+(p['warm_steps'] if c['kind']!='warm' else 0)
        for _ in range(steps):
            idx=rng.choice(len(a['train_x']),min(c['batch_size'],len(a['train_x'])),replace=False);rng.uniform(0,1,size=(32,2))
            bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
        if bh!=r['batch_hash']:raise ValueError('batch replay mismatch')
        crng=np.random.default_rng(seed+4000);ch=hashlib.sha256(b'').hexdigest()
        for _ in range(c['steps']):
            qt=crng.uniform(c['time_lower'],c['time_upper'],min(c['batch_size'],len(a['train_x']))).astype('<f4')
            ch=hashlib.sha256(bytes.fromhex(ch)+qt.tobytes()).hexdigest()
        if ch!=r['collocation_hash']:raise ValueError('query-time replay mismatch')
        if c['kind']!='warm':
            warm=path.parent.parent/f'warm_s{seed}/checkpoints/latest.pt'
            if file_hash(warm)!=r['identity']['warm_sha256']:raise ValueError('warm changed')
            w=torch.load(warm,map_location='cpu',weights_only=False)
            initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in w['model'].values())).hexdigest()
            if initial!=r['initial']:raise ValueError('warm initial mismatch')
            pairkey=(task,condition,seed);value=(initial,bh,ch,r['calibration'],r['parameters_real'])
            if pairkey in pair and pair[pairkey]!=value:raise ValueError('paired fork mismatch')
            pair[pairkey]=value
        derivative={}
        for order,name in [(1,'gradient'),(2,'curvature')]:
            def deriv(v):
                return spectral_derivative(torch.tensor(v,dtype=torch.float64),order,c['length']).numpy()
            reference=deriv(a['val_y']);estimate=deriv(pred);initial_derivative=deriv(a['train_x'])
            floor=max(1e-12,.001*np.sqrt(np.mean(initial_derivative**2))*np.sqrt(reference.shape[-1]))
            error=np.linalg.norm(estimate-reference,axis=-1)/np.maximum(np.linalg.norm(reference,axis=-1),floor)
            derivative[name]=dict(primary_times_mean=float(error[:,cols].mean()),by_time=error.mean(0).tolist(),
                                  denominator_floor=floor,policy='max(true derivative norm,0.001 clean TRAIN INITIAL derivative RMS*sqrtN,1e-12); descriptive secondary')
        if c['kind']!='warm':groups.setdefault(f'{task}/{condition}/{c["kind"]}',[]).append((seed,r['metrics']['val']['primary']))
        evidence.append(dict(run=path.parent.relative_to(OUT).as_posix(),predictions_replayed=True,batches_replayed=True,observations_verified=True,derivative_diagnostics=derivative))
    if len(paths)==48:
        summary=json.loads((OUT/'summary.json').read_text())
        for name,rows in groups.items():
            rows.sort();values=[v for _,v in rows]
            np.testing.assert_allclose(summary['groups'][name]['per_seed'],values,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(summary['groups'][name]['development_mean'],np.mean(values),rtol=1e-11,atol=1e-12)
    atomic_json(OUT/'checkpoint_audit.json',dict(checked_runs=len(paths),expected_runs=48,complete=len(paths)==48,evidence=evidence,confirmation_opened=False))
    print(f'AUDIT {len(paths)}/48 passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    run(args.require_complete)
