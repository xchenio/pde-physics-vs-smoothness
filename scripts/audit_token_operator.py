"""Replay matched gate/attention fixed-time ablation, without opening holdouts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_token_operator import plan,OUT
from scripts.run_spectral_pod import plan as parent_plan,task_data
from scripts.token_operator import TokenDeepONet
from scripts.spatial_deeponet import coordinates
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
        arrays,anchors,parent=task_data(old,task);prepared,scale=prepare(arrays,anchors[64]);context=dict(parent=parent,plan=digest(p),task=task)
        length=2. if task=='allen_cahn' else 1.;time=.5 if task=='allen_cahn' else 1.;grid=coordinates(task,256,device)
        floors=metric_floors(arrays['train_y'],length);groups={};pair={}
        for kind in p['arms']:
            records=[]
            for seed in p['seeds']:
                folder=OUT/task/f'{kind}_s{seed}';r=json.loads((folder/'result.json').read_text());path=folder/'checkpoints/final.pt'
                if file_hash(path)!=r['final_sha256'] or r['identity']['context']!=context:raise ValueError('artifact/context mismatch')
                cp=torch.load(path,map_location='cpu',weights_only=False)
                if cp['identity']!=r['identity'] or cp['step']!=p['steps'][task]:raise ValueError('checkpoint mismatch')
                close(cp['scale'],scale)
                for k in anchors[64]:close(cp['anchor'][k],anchors[64][k])
                set_seed(seed+2000);model=TokenDeepONet(kind,64,length,time).to(device)
                initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                if initial!=r['initial'] or cp['initial']!=initial:raise ValueError('initial mismatch')
                rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest()
                for _ in range(p['steps'][task]):
                    idx=rng.choice(600,32,replace=False);bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
                count=sum(v.numel() for v in model.parameters())
                if bh!=r['batch_hash'] or bh!=cp['batch_hash'] or count!=r['parameters']:raise ValueError('batch/count mismatch')
                value=(initial,bh,count)
                if seed in pair and pair[seed]!=value:raise ValueError('paired arms mismatch')
                pair[seed]=value;model.load_state_dict(cp['model']);model.eval();scores={};per_sample={}
                with torch.no_grad():
                    for split,v in prepared.items():
                        x=torch.tensor(v['x'],dtype=torch.float32,device=device)
                        pred=torch.cat([model(b,grid) for b in x.split(128)]).cpu().numpy().astype(float)*scale+v['offset']
                        scores[split]=dict(prediction=metrics(pred,v['y']),structure=score(pred,v['y'],length,floors))
                        per_sample[split]=(np.linalg.norm(pred-v['y'],axis=1)/np.maximum(np.linalg.norm(v['y'],axis=1),1e-12)).tolist()
                close(scores,r['metrics']);records.append(dict(seed=seed,metrics=scores,per_sample=per_sample,parameters=count,seconds=r['optimization_seconds']));checked+=1
            groups[kind]=records
        summary=json.loads((OUT/task/'summary.json').read_text())
        for kind,records in groups.items():
            for split in ('train','val'):
                for key in records[0]['metrics'][split]['prediction']:
                    v=[r['metrics'][split]['prediction'][key] for r in records]
                    close(dict(mean=float(np.mean(v)),seed_sd=float(np.std(v,ddof=1))),summary['groups'][kind]['metrics'][split][key])
        tasks[task]=dict(summary=summary,runs=groups,paired_dev_differences=[groups['attention'][s]['metrics']['val']['prediction']['mean_relative_l2']-groups['gate'][s]['metrics']['val']['prediction']['mean_relative_l2'] for s in p['seeds']])
        print('AUDITED',task,flush=True)
    out=dict(plan=digest(p),checked_runs=checked,tasks=tasks,status='development-only;18 exact-parameter/init/batch pairs replayed; no physics/holdout')
    atomic_json(OUT/'audit.json',out);return out


def report(a):
    lines=['# Matched token attention ablation, 2026-10-08','',
           '18 fits, same CNN/trunk,64-scalar bottleneck, fixed-time600 fitting /200 development.',
           'All gate/attention parameters, initial tensors and minibatch sequences match exactly.',
           'Pointwise QKV gate versus four-head attention; final branch head mixes globally in both.',
           'No physics or unused confirmation in this study. Reused development, three seeds.',
           '','| Task | Arm | Params | Train L2 | Dev L2 | Seed SD |','|---|---|---:|---:|---:|---:|']
    for task,t in a['tasks'].items():
        for kind,g in t['summary']['groups'].items():
            m=g['metrics']['val']['mean_relative_l2'];lines.append(f"| {task} | {kind} | {g['parameters']} | {g['metrics']['train']['mean_relative_l2']['mean']:.6f} | {m['mean']:.6f} | {m['seed_sd']:.6f} |")
    lines.extend(['','Per-seed paired attention-minus-gate differences and per-sample errors in audit.json.',
                  'Equal parameters/steps do not imply equal time/FLOPs. This isolates the registered',
                  'attention-vs-gate interaction, not all possible nonattention/global mixers.',''])
    return '\n'.join(lines)


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    print(report(audit(torch.device('cuda'))))
