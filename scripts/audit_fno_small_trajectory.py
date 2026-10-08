"""Replay all same100-shot known-time FNO heads before confirmation sealing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_fno_small_trajectory import plan,OUT
from scripts.run_physics_token import plan as physics_plan,load_data,OUT as PHYSICS
from scripts.fno_small_trajectory import make_model
from scripts.physics_token_core import trajectory_errors
from scripts.audit_manifold_reconstruction import close
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash


def audit(device):
    p=plan();arrays,parent=load_data(physics_plan());context=dict(parent=parent,plan=digest(p));groups={};checked=0
    raw=np.asarray(arrays['train_x'],dtype=float);truth=np.asarray(arrays['train_y'],dtype=float)
    norm=dict(mean=float(raw.mean()),std=float(raw.std()));scale=float(truth.std());coords=torch.tensor(arrays['grid256'],dtype=torch.float32,device=device)
    for width in p['widths']:
        rows=[]
        for seed in p['seeds']:
            folder=OUT/f'w{width}_s{seed}';r=json.loads((folder/'result.json').read_text());path=folder/'checkpoints/final.pt'
            if r['identity']['context']!=context or file_hash(path)!=r['final_sha256']:raise ValueError('artifact/context mismatch')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp['identity']!=r['identity'] or cp['step']!=p['steps']:raise ValueError('checkpoint mismatch')
            close(cp['normalization'],norm);close(cp['scale'],scale);model=make_model(width,seed,device)
            initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            if initial!=r['initial'] or cp['initial']!=initial:raise ValueError('initial mismatch')
            bh=hashlib.sha256(b'').hexdigest();rng=np.random.default_rng(seed+3000)
            for _ in range(p['steps']):
                idx=rng.choice(100,16,replace=False);rng.uniform(0.,1.,size=(32,2));bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
            token=json.loads((PHYSICS/f'gate_p0_s{seed}'/'result.json').read_text())
            if bh!=r['batch_hash'] or bh!=cp['batch_hash'] or bh!=token['batch_hash']:raise ValueError('FNO/factorial batch pairing failed')
            count=sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters())
            if count!=r['parameters_real']:raise ValueError('parameter mismatch')
            model.load_state_dict(cp['model']);model.eval();scores={}
            with torch.no_grad():
                for split in ('train','val'):
                    raw=np.asarray(arrays[split+'_x'],dtype=float);x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
                    pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*scale+raw.mean(1)[:,None,None]
                    e=trajectory_errors(pred,arrays[split+'_y']);scores[split]=dict(primary=float(e[:,1:].mean()),per_sample=e[:,1:].mean(1).tolist(),by_time=e.mean(0).tolist())
            close(scores,r['metrics']);rows.append(dict(seed=seed,metrics=scores,parameters_real=count,seconds=r['seconds']));checked+=1
        groups[str(width)]=rows
    summary=json.loads((OUT/'summary.json').read_text())
    for w,rows in groups.items():
        for split in ('train','val'):
            v=[r['metrics'][split]['primary'] for r in rows];close(dict(mean=float(np.mean(v)),seed_sd=float(np.std(v,ddof=1))),summary['groups'][w]['metrics'][split])
    result=dict(plan=digest(p),checked_runs=checked,records=groups,summary=summary,status='6 same100-shot/time/batch FNO fits replayed; no fresh confirmation')
    atomic_json(OUT/'audit.json',result);print('AUDITED FNO ALL6',flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    print(audit(torch.device('cuda'))['summary']['groups'])
