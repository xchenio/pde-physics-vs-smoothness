"""Independent checkpoint prediction and minibatch replay for new FNO families."""
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
from src.time_fno import TimeFNO1d
from src.public_benchmark import atomic_json,file_hash
from scripts.run_physics_token import load_data
from scripts.time_fno_baseline import predict
from scripts.physics_token_core import trajectory_errors


def run(family,require_complete):
    out=ROOT/'results'/family;plan=json.loads((out/'plan.json').read_text())
    for name,h in plan['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('source changed: '+name)
    parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text());arrays,_=load_data(parent)
    continuation=family=='fno_regularization_v1';expected=30 if continuation else 3
    paths=sorted(out.glob('*/result.json'))
    if require_complete and len(paths)!=expected:raise ValueError('incomplete')
    device=torch.device('cuda');pair={};evidence=[]
    for path in paths:
        result=json.loads(path.read_text());c=result['identity']['config'];seed=c['seed'];folder=path.parent
        final=folder/'checkpoints/final.pt'
        if file_hash(final)!=result['final_sha256']:raise ValueError('checkpoint hash changed')
        cp=torch.load(final,map_location='cpu',weights_only=False)
        if cp['identity']!=result['identity'] or cp['step']!=c['steps']:raise ValueError('checkpoint identity/stage mismatch')
        model=TimeFNO1d(c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
        coords=torch.tensor(arrays['grid256'],dtype=torch.float32,device=device);times=torch.tensor(arrays['times'],dtype=torch.float32,device=device)
        for split in ('train','val'):
            raw=np.asarray(arrays[split+'_x'],dtype=float);norm=cp['normalization']
            x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
            with torch.no_grad():pred=torch.cat([predict(model,b,times,coords) for b in x.split(16)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1)[:,None,None]
            e=trajectory_errors(pred,arrays[split+'_y'])
            np.testing.assert_allclose(e[:,1:].mean(1),result['metrics'][split]['per_sample'],rtol=1e-11,atol=1e-12)
        rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest()
        if continuation:
            warm=ROOT/f'results/time_fno_baseline_v1/s{seed}/checkpoints/latest.pt'
            if file_hash(warm)!=result['identity']['warm_sha256']:raise ValueError('warm changed')
            w=torch.load(warm,map_location='cpu',weights_only=False);rng.bit_generator.state=w['rng']['generator'];bh=w['batch_hash']
            initial=hashlib.sha256(b''.join(v.numpy().tobytes() for v in w['model'].values())).hexdigest()
            if initial!=result['initial']:raise ValueError('warm initial mismatch')
        for _ in range(c['steps']):
            idx=rng.choice(len(arrays['train_x']),min(c['batch_size'],len(arrays['train_x'])),replace=False);rng.uniform(0,1,size=(32,2))
            bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
        if bh!=result['batch_hash']:raise ValueError('batch replay mismatch')
        if continuation:
            crng=np.random.default_rng(seed+4000);ch=hashlib.sha256(b'').hexdigest()
            for _ in range(c['steps']):
                qt=crng.uniform(.1,1.,min(c['batch_size'],len(arrays['train_x']))).astype('<f4')
                ch=hashlib.sha256(bytes.fromhex(ch)+qt.tobytes()).hexdigest()
            if ch!=result['collocation_hash']:raise ValueError('collocation replay mismatch')
            value=(result['initial'],bh,ch,result['calibration'])
            if seed in pair and pair[seed]!=value:raise ValueError('arm pairing mismatch')
            pair[seed]=value
        else:
            old=json.loads((ROOT/f'results/fno_small_trajectory_v1/w16_s{seed}/result.json').read_text())
            if old['batch_hash']!=bh:raise ValueError('historical pairing mismatch')
        evidence.append(dict(run=folder.name,final_sha256=result['final_sha256'],batch_replayed=True,predictions_replayed=True))
    if len(paths)==expected:
        summary=json.loads((out/'summary.json').read_text())
        if continuation:
            groups={}
            for path in paths:
                r=json.loads(path.read_text());c=r['identity']['config']
                name=f"{c['kind']}_r{c['strength']:g}"
                groups.setdefault(name,[]).append((c['seed'],r['metrics']['val']['primary']))
            for name,rows in groups.items():
                rows.sort();values=[v for _,v in rows]
                np.testing.assert_allclose(summary['groups'][name]['per_seed'],values,rtol=1e-11,atol=1e-12)
                np.testing.assert_allclose(summary['groups'][name]['development_mean'],np.mean(values),rtol=1e-11,atol=1e-12)
                np.testing.assert_allclose(summary['groups'][name]['development_seed_sd'],np.std(values,ddof=1),rtol=1e-11,atol=1e-12)
            selected={k:min([n for n in groups if summary['groups'][n]['kind']==k],key=lambda n:(summary['groups'][n]['development_mean'],summary['groups'][n]['strength'])) for k in plan['kinds']}
            if selected!=summary['selected']:raise ValueError('selection replay mismatch')
        else:
            mean=np.mean([json.loads(p.read_text())['metrics']['val']['primary'] for p in paths])
            np.testing.assert_allclose(summary['development_mean'],mean,rtol=1e-11,atol=1e-12)
            if summary['gate_passed']!=(mean<=1.1*summary['historical_discrete_head_development_mean']):raise ValueError('gate replay mismatch')
    atomic_json(out/'checkpoint_audit.json',dict(completed_runs=len(paths),expected_runs=expected,complete=len(paths)==expected,evidence=evidence,confirmation_opened=False))
    print(f'AUDIT {family}: {len(paths)}/{expected} prediction and batch replays passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('family',choices=['time_fno_baseline_v1','fno_regularization_v1']);parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    run(args.family,args.require_complete)
