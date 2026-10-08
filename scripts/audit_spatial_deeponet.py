"""Replay stage2 CNN DeepONet states and retained FNO contextual metrics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_spatial_deeponet import plan,OUT
from scripts.run_spectral_pod import plan as parent_plan,task_data
from scripts.spatial_deeponet import SpatialDeepONet,coordinates
from scripts.run_public_pod_readout import prepare
from scripts.run_public_benchmark import metrics
from scripts.manifold_reconstruction import metric_floors,score
from scripts.audit_manifold_reconstruction import close
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash
from src.utils import set_seed


def audit(device):
    p=plan();old=parent_plan();tasks={};checked=0
    for task in p['tasks']:
        arrays,anchors,parent=task_data(old,task);prepared,scale=prepare(arrays,anchors[64]);records=[]
        context=dict(parent=parent,plan=digest(p),task=task);length=2. if task=='allen_cahn' else 1.;final_time=.5 if task=='allen_cahn' else 1.
        grid=coordinates(task,256,device);floors=metric_floors(arrays['train_y'],length)
        for seed in p['seeds']:
            folder=OUT/task/f'cnn_trunk_s{seed}';r=json.loads((folder/'result.json').read_text());path=folder/'checkpoints/final.pt'
            if file_hash(path)!=r['final_sha256'] or r['identity']['context']!=context:raise ValueError('artifact/context mismatch')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            if cp['identity']!=r['identity'] or cp['step']!=p['steps'][task]:raise ValueError('checkpoint mismatch')
            close(cp['scale'],scale)
            for k in anchors[64]:close(cp['anchor'][k],anchors[64][k])
            set_seed(seed+2000);model=SpatialDeepONet(64,length,final_time).to(device)
            initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            if initial!=r['initial'] or cp['initial']!=initial:raise ValueError('initial mismatch')
            rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest()
            for _ in range(p['steps'][task]):
                idx=rng.choice(600,32,replace=False);bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
            if bh!=r['batch_hash'] or bh!=cp['batch_hash']:raise ValueError('batch mismatch')
            oldr=json.loads((ROOT/'results/public_fno_projection_v1'/task/f'full_s{seed}'/'result.json').read_text())
            if bh!=oldr['batch_sha256']:raise ValueError('FNO batch pairing mismatch')
            model.load_state_dict(cp['model']);model.eval();scores={};per_sample={}
            with torch.no_grad():
                for split,v in prepared.items():
                    x=torch.tensor(v['x'],dtype=torch.float32,device=device)
                    pred=torch.cat([model(b,grid) for b in x.split(128)]).cpu().numpy().astype(float)*scale+v['offset']
                    scores[split]=dict(prediction=metrics(pred,v['y']),structure=score(pred,v['y'],length,floors))
                    per_sample[split]=(np.linalg.norm(pred-v['y'],axis=1)/np.maximum(np.linalg.norm(v['y'],axis=1),1e-12)).tolist()
            close(scores,r['metrics']);records.append(dict(seed=seed,metrics=scores,per_sample=per_sample,
                                                           parameters=r['parameters'],seconds=r['optimization_seconds'],fno=oldr['metrics']))
            checked+=1
        summary=json.loads((OUT/task/'summary.json').read_text())
        for split in ('train','val'):
            for key in records[0]['metrics'][split]['prediction']:
                v=[r['metrics'][split]['prediction'][key] for r in records]
                close(dict(mean=float(np.mean(v)),seed_sd=float(np.std(v,ddof=1))),summary['metrics'][split][key])
        tasks[task]=dict(summary=summary,runs=records)
        print('AUDITED',task,flush=True)
    out=dict(plan=digest(p),checked_runs=checked,tasks=tasks,status='development-only fixed-time baseline; FNO pairing checked; no physics or attention')
    atomic_json(OUT/'audit.json',out);return out


def report(a):
    lines=['# Spatial CNN / smooth-trunk DeepONet baseline, 2026-10-08','',
           '9 fits, rank64, three seeds; historical correction600 fitting and development200.',
           'Same guarded field objective, minibatches and step budgets as retained FNO.',
           'CNN/trunk capacity differs from FNO, so comparison is contextual, not encoder-isolated.',
           'No physics/attention, future-label branch input, fresh confirmation, or temporal claims.',
           '','| Task | Model | Params | Train relative L2 | Dev relative L2 | Dev seed SD |',
           '|---|---|---:|---:|---:|---:|']
    for task,t in a['tasks'].items():
        s=t['summary'];m=s['metrics']['val']['mean_relative_l2']
        lines.append(f"| {task} | CNN DeepONet64 | {s['parameters']} | {s['metrics']['train']['mean_relative_l2']['mean']:.6f} | {m['mean']:.6f} | {m['seed_sd']:.6f} |")
        f=s['retained_fno'];m=f['metrics']['val']['mean_relative_l2']
        lines.append(f"| {task} | Retained full FNO | 136481 | {f['metrics']['train']['mean_relative_l2']['mean']:.6f} | {m['mean']:.6f} | {m['std']:.6f} |")
    lines.extend(['','Full per-seed and per-sample replay: results/spatial_deeponet_v1/audit.json.',
                  'Coordinates are native retained cell centers; physical x/time inputs have tested',
                  'first/second derivative chain factors. Constant training time does not constrain',
                  'u_t or certify extrapolation. The next physics stage needs trajectory data first.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    result=audit(torch.device('cuda'));print(report(result))
