"""Replay small-data factorial and independently regenerate both sampling streams."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_token import plan,load_data,OUT
from scripts.token_operator import TokenDeepONet
from scripts.physics_token_core import evaluate,physical_fields,residual
from scripts.audit_manifold_reconstruction import close
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash
from src.utils import set_seed


def audit(device):
    p=plan();arrays,context=load_data(p);records={};pairs={};checked=0
    norm=dict(mean=float(np.asarray(arrays['train_x'],dtype=float).mean()),std=float(np.asarray(arrays['train_x'],dtype=float).std()))
    scale=float(np.asarray(arrays['train_y'],dtype=float).std());probe=np.random.default_rng(2026100807).uniform(0.,1.,size=(128,2)).astype('float32')
    for kind in p['kinds']:
        for weight in p['physics_weights']:
            arm=f'{kind}_p{weight:g}';runs=[]
            for seed in p['seeds']:
                folder=OUT/f'{arm}_s{seed}';r=json.loads((folder/'result.json').read_text());path=folder/'checkpoints/final.pt'
                if file_hash(path)!=r['final_sha256'] or r['identity']['context']!=context:raise ValueError('artifact/context mismatch')
                cp=torch.load(path,map_location='cpu',weights_only=False)
                if cp['identity']!=r['identity'] or cp['step']!=p['steps']:raise ValueError('checkpoint mismatch')
                close(cp['normalization'],norm);close(cp['scale'],scale)
                set_seed(seed+2000);model=TokenDeepONet(kind,64,1.,1.).to(device)
                initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                if initial!=r['initial'] or cp['initial']!=initial:raise ValueError('initial mismatch')
                bh=hashlib.sha256(b'').hexdigest();ch=bh;rng=np.random.default_rng(seed+3000)
                for _ in range(p['steps']):
                    idx=rng.choice(100,16,replace=False);points=rng.uniform(0.,1.,size=(32,2)).astype('<f4')
                    bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest();ch=hashlib.sha256(bytes.fromhex(ch)+points.tobytes()).hexdigest()
                count=sum(v.numel() for v in model.parameters())
                if (bh,ch,count)!=(r['batch_hash'],r['collocation_hash'],r['parameters']) or (bh,ch)!=(cp['batch_hash'],cp['collocation_hash']):raise ValueError('sampling/count mismatch')
                pair=(initial,bh,ch,count)
                if seed in pairs and pairs[seed]!=pair:raise ValueError('factorial pairing mismatch')
                pairs[seed]=pair;model.load_state_dict(cp['model']);scores=evaluate(model,arrays,norm,scale,device);close(scores,r['metrics'])
                # Shared fixed development collocation diagnostic, secondary only.
                with torch.no_grad():
                    raw=np.asarray(arrays['val_x'],dtype=float);x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
                    off=torch.tensor(raw.mean(1,keepdims=True),dtype=torch.float32,device=device)
                    b=model.branch(x);values=physical_fields(model,b,torch.tensor(probe,device=device),scale,off)
                    resid=residual(*values,p['diffusion']);pde=float(torch.sqrt(resid.square().mean()).cpu())
                runs.append(dict(seed=seed,metrics=scores,parameters=count,seconds=r['seconds'],fixed_probe_pde_rms=pde,
                                 peak_allocated_bytes_last_segment=r['peak_allocated_bytes_last_segment']));checked+=1
            records[arm]=runs
    summary=json.loads((OUT/'summary.json').read_text())
    for arm,runs in records.items():
        for split in ('train','val','val512'):
            v=[r['metrics'][split]['primary'] for r in runs]
            close(dict(mean=float(np.mean(v)),seed_sd=float(np.std(v,ddof=1))),summary['groups'][arm]['metrics'][split])
    out=dict(plan=digest(p),checked_runs=checked,records=records,summary=summary,fixed_probe_seed=2026100807,
             status='12 completed/replayed; all4 factorial arms paired; development only; no confirmation yet')
    atomic_json(OUT/'audit.json',out);print('AUDITED ALL12',flush=True);return out


def report(a):
    lines=['# Burgers100-shot physics/token trajectory pilot, 2026-10-08','',
           '12 fits;100 labeled trajectories,5 times, three seeds; existing200 development.',
           'Factorial gate/attention x physics0/.01, same initial tensors/batches/collocation/steps.',
           'Primary averages per-trajectory relativeL2 over0.1/0.25/0.5/1.0; no time extrapolation.',
           '512 is native output-query transfer with fixed256 input sensors, not full sensor transfer.',
           'Shared probe PDE RMS is a secondary development diagnostic; never replaces field metric.',
           '','| Arm | Params | Train | Dev256 | Dev512 | PDE RMS | Training seconds mean |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,g in a['summary']['groups'].items():
        r=a['records'][arm];m=g['metrics']
        lines.append(f"| {arm} | {g['parameters']} | {m['train']['mean']:.6f} | {m['val']['mean']:.6f} | {m['val512']['mean']:.6f} | {np.mean([v['fixed_probe_pde_rms'] for v in r]):.6f} | {np.mean([v['seconds'] for v in r]):.2f} |")
    lines.extend(['','Only one Burgers PDE; no Allen-Cahn physics validation. No weight or checkpoint tuning.',
                  'Retained FNO600 is a different sample/time budget and contextual only.',
                  'All model/data/source hashes must be committed and sealed before fresh500 targets.',
                  'Checkpoint/runtime logs and per-sample metrics retained in the versioned family.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    print(report(audit(torch.device('cuda'))))
