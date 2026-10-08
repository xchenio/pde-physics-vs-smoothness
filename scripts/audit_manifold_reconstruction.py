"""Replay completed compressor outputs, POD controls, batch and parameter pairing."""
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
from scripts.run_manifold_reconstruction import OUT,plan,task_arrays
from scripts.manifold_reconstruction import Autoencoder,errors,score,metric_floors
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import file_hash,atomic_json
from src.utils import set_seed


def close(a,b):
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('metric keys differ')
        for k in a:close(a[k],b[k])
    else:np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)


def audit(device):
    p=plan();records={};comparisons={};checked=0
    for task in p['tasks']:
        train,val,context=task_arrays(p,task);floors=metric_floors(train,p['lengths'][task])
        pod=json.loads((OUT/task/'pod.json').read_text());cp=torch.load(OUT/task/'checkpoints/pod.pt',weights_only=False)
        if cp['context']!=context or pod['context']!=context:raise ValueError('POD identity mismatch')
        np.testing.assert_allclose(cp['q'].T@cp['q'],np.eye(64),atol=1e-12)
        close(cp['mean'],float(train.mean()))
        for rank in p['ranks']:
            q=cp['q'][:,:rank];pred=(val-cp['mean'])@q@q.T+cp['mean'];ref=pod['ranks'][str(rank)]
            close(score(pred,val,p['lengths'][task],floors),ref['metrics']['val'])
            close(errors(pred,val,p['lengths'][task],floors),ref['per_sample'])
        summary=json.loads((OUT/task/'summary.json').read_text());runs={}
        for rank in p['ranks']:
            for arm in p['arms']:
                label=f'{arm}_r{rank}';group=[]
                for seed in p['seeds']:
                    folder=OUT/task/f'{label}_s{seed}';r=json.loads((folder/'result.json').read_text())
                    cp_path=folder/'checkpoints/final.pt'
                    if file_hash(cp_path)!=r['final_sha256'] or r['identity']['context']!=context:raise ValueError('artifact/context mismatch')
                    cp=torch.load(cp_path,map_location='cpu',weights_only=False);c=r['identity']['config']
                    if cp['identity']!=r['identity'] or cp['step']!=p['steps']:raise ValueError('checkpoint identity mismatch')
                    set_seed(seed+2000);model=Autoencoder(arm,rank).to(device)
                    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
                    if initial!=r['initial'] or cp['initial']!=initial:raise ValueError('initialization mismatch')
                    rng=np.random.default_rng(seed+3000);bh=hashlib.sha256(b'').hexdigest()
                    for _ in range(p['steps']):
                        idx=rng.choice(len(train),p['batch_size'],replace=False)
                        bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
                    if bh!=r['batch_hash'] or cp['batch_hash']!=bh:raise ValueError('batch hash mismatch')
                    if sum(v.numel() for v in model.parameters())!=r['parameters']:raise ValueError('parameter count mismatch')
                    close(cp['mean'],float(train.mean()));close(cp['scale'],float(train.std()))
                    model.load_state_dict(cp['model']);model.eval();metrics={}
                    with torch.no_grad():
                        for split,y in (('train',train),('val',val)):
                            x=torch.tensor((y-cp['mean'])/cp['scale'],dtype=torch.float32,device=device)
                            pred=torch.cat([model(b) for b in x.split(128)]).cpu().numpy().astype(float)*cp['scale']+cp['mean']
                            metrics[split]=score(pred,y,p['lengths'][task],floors)
                            if split=='val':
                                err=errors(pred,y,p['lengths'][task],floors);close(err,r['per_sample'])
                                np.testing.assert_allclose(sum(v*v for k,v in err.items() if k.startswith('band')),err['field']**2,rtol=1e-12,atol=1e-12)
                    close(metrics,r['metrics']);group.append(r);checked+=1
                for split in ('train','val'):
                    for k in group[0]['metrics'][split]:
                        v=[g['metrics'][split][k] for g in group]
                        close(dict(mean=float(np.mean(v)),seed_sd=float(np.std(v,ddof=1))),summary['groups'][label]['metrics'][split][k])
                runs[label]=group
            for seed in p['seeds']:
                set_seed(seed+2000);a=Autoencoder('local',rank)
                set_seed(seed+2000);b=Autoencoder('scrambled',rank)
                if not all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())):raise ValueError('trainable initialization pairing failed')
            comparisons[f'{task}_r{rank}']={}
            for control in ('global','scrambled'):
                # Descriptive paired seed differences; no confirmation intervals from reused development.
                comparisons[f'{task}_r{rank}'][f'local_minus_{control}']={k:[runs[f'local_r{rank}'][s]['metrics']['val'][k]-runs[f'{control}_r{rank}'][s]['metrics']['val'][k] for s in p['seeds']] for k in ('field','gradient','curvature')}
        records[task]=dict(pod={r:v['metrics']['val'] for r,v in pod['ranks'].items()},groups=summary['groups'])
        print('AUDITED',task,flush=True)
    out=dict(plan_sha256=digest(p),checked_runs=checked,pod_checks=9,comparisons=comparisons,tasks=records,
             status='completed; development only; no physics/attention/operator prediction inference')
    atomic_json(OUT/'audit.json',out);return out


def report(a):
    lines=['# Compression reconstruction results, 2026-10-08','',
           '81 completed neural fits, independently replayed; POD train-only controls at each rank.',
           'Reused development200. Means over three seeds; no unseen confirmation or operator accuracy claim.',
           'Identical scalar bottlenecks. Local/scrambled exactly matched capacity; global capacity differs.',
           'New POD uses train-only scalar centering, not the historical input-mean-offset POD.',
           'Derivative metrics are finite-grid spectral diagnostics with train-only denominator guards.',
           '','| Task | Rank | Model | Parameters | Field | Gradient | Curvature |','|---|---:|---|---:|---:|---:|---:|']
    for task,t in a['tasks'].items():
        for rank in (16,32,64):
            m=t['pod'][str(rank)];lines.append(f"| {task} | {rank} | POD | basis | {m['field']:.6f} | {m['gradient']:.6f} | {m['curvature']:.6f} |")
            for arm in ('global','local','scrambled'):
                g=t['groups'][f'{arm}_r{rank}'];m=g['metrics']['val']
                lines.append(f"| {task} | {rank} | {arm} | {g['parameters']} | {m['field']['mean']:.6f} | {m['gradient']['mean']:.6f} | {m['curvature']['mean']:.6f} |")
    lines.extend(['','## Frequency-band errors','',
                  'Band error / total field norm (not relative band energy); mean across samples/seeds.',
                  'Per sample, sum of squared band errors equals squared guarded relative field error.',
                  '','| Task | Rank | Model | 0–4 | 5–16 | 17–64 | 65–128 |','|---|---:|---|---:|---:|---:|---:|'])
    for task,t in a['tasks'].items():
        for rank in (16,32,64):
            for arm in ('POD','global','local','scrambled'):
                m=t['pod'][str(rank)] if arm=='POD' else {k:v['mean'] for k,v in t['groups'][f'{arm}_r{rank}']['metrics']['val'].items()}
                values=' | '.join(f"{m[k]:.6f}" for k in ('band_0_4','band_5_16','band_17_64','band_65_128'))
                lines.append(f'| {task} | {rank} | {arm} | {values} |')
    lines.extend(['','Machine-readable per-seed pairing differences: results/manifold_reconstruction_v1/audit.json.',
                  'Full per-sample errors, training curves, time, peak last-segment GPU allocation and hashes',
                  'remain in result.json per run. Fixed1500-step MSE training; derivative errors not optimized.',
                  'All reconstruction encoders see the true output snapshot; this is a representation',
                  'diagnostic and does not show initial-field-to-future-field prediction performance.',''])
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    a=audit(torch.device(args.device));print(report(a))


if __name__=='__main__':main()
