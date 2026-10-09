"""Independent paired prediction/RNG/history replay and full-grid diagnostics."""
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
from scripts.run_periodic_curvature import OUT
from scripts.run_physics_stress import clean_data,observations
from scripts.periodic_curvature_core import make_model
from scripts.time_fno_baseline import predict
from scripts.physics_token_core import trajectory_errors
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from src.public_benchmark import file_hash,atomic_json
from scripts.run_public_ridge_residual import digest


def historical_path(task,kind,seed):
    if task=='allen_cahn':
        oldkind='curvature' if kind=='low' else kind
        return ROOT/f'results/physics_stress_v1/allen_cahn/dense_clean/{oldkind}_s{seed}/checkpoints/final.pt'
    if kind=='warm':return ROOT/f'results/time_fno_baseline_v1/s{seed}/checkpoints/final.pt'
    arm='none_r0' if kind=='none' else 'curvature_r0.3'
    return ROOT/f'results/fno_regularization_v1/{arm}_s{seed}/checkpoints/final.pt'


def run(require_complete):
    p=json.loads((OUT/'plan.json').read_text());old=json.loads((ROOT/'results/physics_stress_v1/plan.json').read_text())
    paths=sorted(OUT.rglob('result.json'));device=torch.device('cuda');cache={};initial={};pair={};streams={};evidence=[];groups={};matches=0
    if require_complete and len(paths)!=48:raise ValueError('incomplete')
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed source changed')
    for path in paths:
        r=json.loads(path.read_text());c=r['identity']['config'];ctx=r['identity']['context'];task=ctx['task'];seed=c['seed'];position=c['position'];kind=c['kind']
        if task not in cache:
            clean,_=clean_data(old,task);a,obs=observations(clean,'dense_clean');cache[task]=(a,obs)
        a,obs=cache[task]
        if ctx['observations']!=obs or ctx['plan']!=digest(p):raise ValueError('observation/plan mismatch')
        cp_path=path.parent/'checkpoints/final.pt'
        if file_hash(cp_path)!=r['final_sha256']:raise ValueError('checkpoint changed')
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=r['identity'] or cp['step']!=c['steps']:raise ValueError('checkpoint stage/identity mismatch')
        rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest();total=c['steps']+(p['warm_steps'] if kind!='warm' else 0)
        for _ in range(total):
            idx=rng.choice(len(a['train_x']),min(c['batch_size'],len(a['train_x'])),replace=False);rng.uniform(0,1,size=(32,2))
            bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
        crng=np.random.default_rng(seed+4000);ch=hashlib.sha256(b'').hexdigest()
        for _ in range(c['steps']):
            qt=crng.uniform(c['time_lower'],c['time_upper'],min(c['batch_size'],len(a['train_x']))).astype('<f4')
            ch=hashlib.sha256(bytes.fromhex(ch)+qt.tobytes()).hexdigest()
        if (bh,ch)!=(r['batch_hash'],r['collocation_hash']):raise ValueError('RNG replay mismatch')
        if position=='linear' and kind in ('warm','none','low'):
            hpath=historical_path(task,kind,seed);historical=torch.load(hpath,map_location='cpu',weights_only=False)
            if not all(torch.equal(v,historical['model'][k]) for k,v in cp['model'].items()):raise ValueError('historical linear control tensor mismatch: '+str(path))
            matches+=1
        if kind=='warm':
            key=(task,seed)
            if key in initial and initial[key]!=r['initial']:raise ValueError('position init mismatch')
            initial[key]=r['initial']
        else:
            warm=path.parent.parent/f'warm_s{seed}/checkpoints/latest.pt'
            if file_hash(warm)!=r['identity']['warm_sha256']:raise ValueError('warm changed')
            w=torch.load(warm,map_location='cpu',weights_only=False);ih=hashlib.sha256(b''.join(v.numpy().tobytes() for v in w['model'].values())).hexdigest()
            if ih!=r['initial']:raise ValueError('fork state mismatch')
            key=(task,position,seed);value=(ih,bh,ch,r['calibration'])
            if key in pair and pair[key]!=value:raise ValueError('smoothing pairing mismatch')
            pair[key]=value;key=(task,seed);value=(bh,ch,r['parameters_real'])
            if key in streams and streams[key]!=value:raise ValueError('cross-position stream/capacity mismatch')
            streams[key]=value
        model=make_model(c,c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
        norm=cp['normalization'];raw=np.asarray(a['val_x'],dtype=float);x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
        times=torch.tensor(a['val_times'],dtype=torch.float32,device=device);coords=torch.tensor(a['grid256'],dtype=torch.float32,device=device)
        with torch.no_grad():pred=torch.cat([predict(model,b,times,coords) for b in x.split(16)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1)[:,None,None]
        error=trajectory_errors(pred,a['val_y']);np.testing.assert_allclose(error[:,1:].mean(1),r['metrics']['val']['per_sample'],rtol=1e-11,atol=1e-12)
        scores=safe_band_errors(pred,a['val_y'],a['train_x'],c['length'],[1,2,3,4])
        if kind!='warm':
            name=f'{task}/{position}/{kind}'
            groups.setdefault(name,[]).append(dict(seed=seed,field=r['metrics']['val']['primary'],diagnostics=scores))
        evidence.append(dict(run=path.parent.relative_to(OUT).as_posix(),prediction_replayed=True,rng_replayed=True,diagnostics=scores))
    report={}
    for name,rows in groups.items():
        rows.sort(key=lambda v:v['seed'])
        report[name]=dict(per_seed_field=[r['field'] for r in rows],field=float(np.mean([r['field'] for r in rows])),
                         gradient=float(np.mean([r['diagnostics']['gradient']['relative_mean'] for r in rows])),
                         curvature=float(np.mean([r['diagnostics']['curvature']['relative_mean'] for r in rows])),
                         edge_fraction=float(np.mean([r['diagnostics']['curvature']['boundary8_energy_fraction'] for r in rows])),
                         high_band_share=float(np.mean([r['diagnostics']['curvature']['error_energy_shares']['65_128'] for r in rows])))
    if len(paths)==48:
        summary=json.loads((OUT/'summary.json').read_text())
        for name,v in report.items():
            np.testing.assert_allclose(summary['groups'][name]['field_mean'],v['field'],rtol=1e-11,atol=1e-12)
            base=report[name.split('/')[0]+'/linear/none']
            v['exploratory_screen_passed']=bool(v['field']<=1.01*base['field'] and v['curvature']<=.5*base['curvature'])
        if matches!=18:raise ValueError('historical control count mismatch')
    atomic_json(OUT/'audit.json',dict(plan=digest(p),audit_source_sha256=file_hash(Path(__file__)),checked_runs=len(paths),complete=len(paths)==48,
                                    historical_linear_tensor_matches=matches,evidence=evidence,groups=report,confirmation_opened=False))
    print('AUDIT',len(paths),'/48; historical tensor matches',matches,json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    run(args.require_complete)
