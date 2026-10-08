"""Development-only rank and exactly parameter-matched encoder study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.spectral_pod_core import train_one
from scripts.run_architecture_stages import plan as old_plan,group
from scripts.run_uncertainty_modal import data_and_root
from scripts.run_field_modal import plan as field_plan
from scripts.run_public_ridge_residual import digest,locked_json
from src.public_benchmark import file_hash,atomic_json
from src.resume import atomic_save
OUT=ROOT/'results/public_spectral_pod_v1'


def plan():
    old=old_plan()
    sources=['scripts/run_spectral_pod.py','scripts/spectral_pod_core.py','scripts/spectral_pod_models.py','src/fno.py']
    p=dict(parent=digest(old),source=digest([old['source']]+[file_hash(ROOT/f) for f in sources]),
           tasks=old['tasks'],seeds=[0,1,2],ranks=[16,32,64],architectures=['spectral_pod','local_pod'],
           steps=dict(burgers_original=1500,burgers_independent=1500,allen_cahn=6000),
           lr=.001,lr_min=.00001,weight_decay=.0001,batch_size=32,eval_every=250,checkpoint_every=50,
           mapping='direct',penalty=0.,
           comparison='four width32 GELU blocks; spectral16 complex modes + local1 vs periodic real kernel33; identical lift,24-bin pooling,zero POD head; exactly equal real parameter count per rank',
           basis='representation400 only; preserve historical q16 exactly; nested orthogonal complement from representation POD; shared historical input normalization; correction600 fitting,development200',
           selection='report all54 fits; fixed final step; select rank by mean development relativeL2 within encoder only; ties lower rank; no new confirmation',
           limits='Repeated development reuse; three seeds; equal parameter count is not equal compute, initialization or receptive field. Budgets transferred from prior direct MLP choices, no per-architecture tuning. No novelty claim.')
    OUT.mkdir(parents=True,exist_ok=True);locked_json(OUT/'plan.json',p);return p


def nested_basis(x,y,q16):
    residual=np.asarray(y,dtype=float)-np.asarray(x,dtype=float).mean(1,keepdims=True)
    raw=np.linalg.svd(residual,full_matrices=False)[2].T[:,:64]
    if not np.allclose(raw[:,:16]@raw[:,:16].T,q16@q16.T,atol=1e-10,rtol=0):
        raise ValueError('historical POD subspace does not match representation rows')
    upper=raw[:,16:64]-q16@(q16.T@raw[:,16:64])
    upper=np.linalg.qr(upper)[0]
    q=np.concatenate([q16,upper],1)
    np.testing.assert_allclose(q.T@q,np.eye(64),atol=1e-12,rtol=0)
    return q


def task_data(p,task):
    arrays,oldroot=data_and_root(task);fp=field_plan()
    oldpath=oldroot/'linear_final/checkpoints/model.pt'
    old=torch.load(oldpath,weights_only=False)['model']
    identity=dict(plan=digest(p),old_anchor_sha256=file_hash(oldpath),task=task)
    path=OUT/task/'checkpoints/basis64.pt'
    if path.exists():
        obj=torch.load(path,weights_only=False)
        if obj['identity']!=identity:raise ValueError('basis identity mismatch')
        q=obj['q']
    else:
        rep=fp['representation_rows']
        q=nested_basis(arrays['train_x'][rep],arrays['train_y'][rep],old['q'])
        atomic_save(dict(identity=identity,q=q),path)
    rows=fp['correction_rows']
    arrays=dict(arrays,train_x=arrays['train_x'][rows],train_y=arrays['train_y'][rows])
    anchors={r:dict(mean=old['mean'],std=old['std'],q=q[:,:r].copy(),
                   weights=np.zeros((q.shape[0],r)),intercept=np.zeros(r)) for r in p['ranks']}
    return arrays,anchors,dict(identity,basis_sha256=file_hash(path))


def config(p,architecture,rank,seed,task):
    return dict({k:p[k] for k in ('lr','lr_min','weight_decay','batch_size','eval_every','checkpoint_every','mapping','penalty')},
                architecture=architecture,rank=rank,seed=seed,steps=p['steps'][task])


def run(p,device):
    for task in p['tasks']:
        arrays,anchors,context=task_data(p,task);groups={}
        for rank in p['ranks']:
            counts=[]
            for architecture in p['architectures']:
                label=f'{architecture}_r{rank}'
                runs=[train_one(config(p,architecture,rank,s,task),arrays,anchors[rank],context,OUT/task/f'{label}_s{s}',device) for s in p['seeds']]
                groups[label]=dict(rank=rank,architecture=architecture,oracle=runs[0]['metrics']['val']['oracle'],**group(runs))
                counts.append(groups[label]['parameters_real'])
                atomic_json(OUT/task/'progress.json',dict(completed_runs=len(groups)*3,total_runs=18,groups=groups))
            if counts[0]!=counts[1]:raise ValueError('parameter matching failed')
        selected={a:min([g for g in groups.values() if g['architecture']==a],key=lambda g:g['metrics']['val']['mean_relative_l2']['mean'])['rank'] for a in p['architectures']}
        atomic_json(OUT/task/'summary.json',dict(context=context,groups=groups,selected_ranks=selected,completed_runs=18))
        print('TASK COMPLETE',task,selected,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['plan','run']);args=a.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;p=plan()
    if args.stage=='run':run(p,torch.device('cuda'))


if __name__=='__main__':main()
