"""POD/global/local/permuted-local reconstruction at identical scalar bottlenecks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.manifold_reconstruction import train_one,score,errors,metric_floors,array_digest,Autoencoder
from scripts.run_uncertainty_modal import data_and_root
from scripts.run_field_modal import plan as field_plan
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save
OUT=ROOT/'results/manifold_reconstruction_v1'


def plan():
    parent=field_plan()
    files=['scripts/run_manifold_reconstruction.py','scripts/manifold_reconstruction.py','src/resume.py','src/utils.py']
    p=dict(version=1,parent=digest(parent),source=digest([file_hash(ROOT/f) for f in files]),
           tasks=['burgers_original','burgers_independent','allen_cahn'],ranks=[16,32,64],seeds=[0,1,2],
           arms=['global','local','scrambled'],steps=1500,lr=.001,lr_min=.00001,weight_decay=.0001,
           batch_size=32,eval_every=250,checkpoint_every=50,
           representation_rows=parent['representation_rows'],lengths=dict(burgers_original=1.,burgers_independent=1.,allen_cahn=2.),
           data='output snapshots: representation400 train, historical development200 evaluation; no correction600 or confirmation labels',
           normalization='one train-only scalar mean/std shared by all arms; no per-sample mean side channel; POD linear orthogonal basis on same centered training snapshots',
           compression='POD r; global MLP r scalars; local/permuted CNN16 positions x r/16 channels, no skip connections',
           objective='normalized field MSE only; fixed1500 steps, no checkpoint selection; same batches per seed across arms/ranks',
           metrics='per-sample relative field/first/second spectral derivative L2; denominator protected by max(1e-12,0.001 training RMS); spectral bands0-4/5-16/17-64/65-128 error divided by full field norm',
           inference='all ranks/seeds/tasks reported; local/scrambled exact parameter+initialization matched; global not parameter matched; seed differences descriptive; reused development, no confirmation',
           limitations='Reconstruction gives encoder access to true output, not operator prediction; finite-grid spectral derivatives not PDE residual or certified continuum derivatives. Fixed training budget, no universal architecture optimality. No physics loss, attention or cross-resolution evaluation yet.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def task_arrays(p,task):
    arrays,_=data_and_root(task)
    train=np.asarray(arrays['train_y'][p['representation_rows']],dtype=float);val=np.asarray(arrays['val_y'],dtype=float)
    context=dict(plan=digest(p),task=task,train_sha256=array_digest(train),val_sha256=array_digest(val))
    return train,val,context


def pod_scores(p,task,train,val,context):
    mean=float(train.mean());_,_,vt=np.linalg.svd(train-mean,full_matrices=False);floors=metric_floors(train,p['lengths'][task]);out={}
    atomic_save(dict(context=context,mean=mean,q=vt[:64].T),OUT/task/'checkpoints/pod.pt')
    for r in p['ranks']:
        q=vt[:r].T;pred=(val-mean)@q@q.T+mean
        out[str(r)]=dict(metrics={s:score((y-mean)@q@q.T+mean,y,p['lengths'][task],floors) for s,y in (('train',train),('val',val))},
                         per_sample={k:v.tolist() for k,v in errors(pred,val,p['lengths'][task],floors).items()},latent_scalars=r,floors=floors)
    atomic_json(OUT/task/'pod.json',dict(context=context,ranks=out));return out


def run(p,device):
    for task in p['tasks']:
        folder=OUT/task;folder.mkdir(parents=True,exist_ok=True);train,val,context=task_arrays(p,task)
        pod=pod_scores(p,task,train,val,context);groups={}
        for r in p['ranks']:
            paired={}
            for arm in p['arms']:
                runs=[]
                for seed in p['seeds']:
                    c={k:p[k] for k in ('steps','lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every')}
                    c.update(kind=arm,rank=r,seed=seed,length=p['lengths'][task])
                    runs.append(train_one(c,train,val,context,folder/f'{arm}_r{r}_s{seed}',device))
                groups[f'{arm}_r{r}']=dict(parameters=runs[0]['parameters'],metrics={s:{k:dict(mean=float(np.mean([x['metrics'][s][k] for x in runs])),seed_sd=float(np.std([x['metrics'][s][k] for x in runs],ddof=1))) for k in runs[0]['metrics'][s]} for s in ('train','val')})
                paired[arm]=runs
                atomic_json(folder/'progress.json',dict(completed_runs=len(groups)*3,total_runs=27,groups=groups))
            for a,b in zip(paired['local'],paired['scrambled']):
                # Permutation buffers differ; trainable initialization is separately checked by tests/audit.
                if a['parameters']!=b['parameters'] or a['batch_hash']!=b['batch_hash']:raise ValueError('pairing mismatch')
        atomic_json(folder/'summary.json',dict(context=context,pod=pod,groups=groups,completed_runs=27))
        print('TASK COMPLETE',task,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p,torch.device(args.device))


if __name__=='__main__':main()
