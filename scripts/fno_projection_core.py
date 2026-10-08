"""Paired FNO optimization, differing only in output projection during training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.architecture_core import make_model as old_make_model,output as old_output
from scripts.run_public_pod_readout import prepare,learning_rate
from scripts.objective_modal_core import sample_weights
from scripts.run_public_ridge_residual import git_commit,event,score_coefficients
from scripts.run_public_benchmark import metrics
from src.utils import env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash


def make_model(c,device):
    return old_make_model(dict(architecture='fno',rank=64,seed=c['seed']),256,device)


def output(model,x):return old_output(model,x,'fno')


def state_digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        h.update(name.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def projected_loss(z,target,weights,q,rank,n):
    estimate=z if rank==0 else z@q[:,:rank]
    return (weights*(estimate-target).square().sum(1)).mean()/n


def evaluate_fields(z,v,q,ranks,rank):
    raw=metrics(z+v['offset'],v['y'])
    projections={str(r):score_coefficients(z@q[:,:r],dict(v,coef=v['coef'][:,:r],
        oracle=metrics(v['coef'][:,:r]@q[:,:r].T+v['offset'],v['y'])),q[:,:r]) for r in ranks}
    return dict(prediction=raw if rank==0 else projections[str(rank)]['prediction'],
                raw_output=raw,projections=projections)


def train_one(c,arrays,anchor,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=dict(config=c,context=context,environment=env_fingerprint(device));manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r['identity']!=identity:raise ValueError('result identity mismatch')
        print('[skip]',folder.name,flush=True);return r
    atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()))
    prepared,scale=prepare(arrays,anchor);q=anchor['q'];rank=c['output_rank'];n=arrays['train_y'].shape[1]
    if rank not in [0]+c['evaluation_ranks'] or max(c['evaluation_ranks'])>q.shape[1]:raise ValueError('invalid rank')
    xs={s:torch.tensor(v['x'],dtype=torch.float32,device=device) for s,v in prepared.items()}
    values=prepared['train']['y']-prepared['train']['offset'] if rank==0 else prepared['train']['coef'][:,:rank]
    target=torch.tensor(values/scale,dtype=torch.float32,device=device)
    qt=torch.tensor(q,dtype=torch.float32,device=device)
    w,floor=sample_weights(arrays['train_y'],'normalized');weights=torch.tensor(w,dtype=torch.float32,device=device)
    model=make_model(c,device);initial=state_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[]
    batch_digest=hashlib.sha256(b'').hexdigest();cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['initial_sha256']!=initial:raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step=cp['step'];seconds=cp['optimization_seconds'];history=cp['history'];batch_digest=cp['batch_sha256']
    event(folder,'resume' if step else 'start',step=step,git_commit=git_commit())

    def evaluate():
        model.eval();scores={}
        with torch.no_grad():
            for split,v in prepared.items():
                z=torch.cat([output(model,b) for b in xs[split].split(128)]).cpu().numpy().astype(float)*scale
                scores[split]=evaluate_fields(z,v,q,c['evaluation_ranks'],rank)
        model.train();return dict(step=step,metrics=scores)

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),
                         initial_sha256=initial,batch_sha256=batch_digest,step=step,history=history,
                         optimization_seconds=seconds,anchor=anchor,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],status='running',optimization_seconds=seconds,last_evaluation=history[-1]))

    if not history:history.append(evaluate());save()
    end=min(c['steps'],stop_after) if stop_after is not None else c['steps']
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(target),min(c['batch_size'],len(target)),replace=False)
        batch_digest=hashlib.sha256(bytes.fromhex(batch_digest)+idx.astype('<i8').tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        z=output(model,xs['train'][idx]);loss=projected_loss(z,target[idx],weights[idx],qt,rank,n)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            history.append(evaluate());event(folder,'evaluation',step=step,val=history[-1]['metrics']['val']['prediction']['mean_relative_l2'])
            print(folder.name,step,'train',round(history[-1]['metrics']['train']['prediction']['mean_relative_l2'],6),
                  'val',round(history[-1]['metrics']['val']['prediction']['mean_relative_l2'],6),flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt'
    atomic_save(dict(identity=identity,model=model.state_dict(),step=step,anchor=anchor,scale=scale,
                     initial_sha256=initial,batch_sha256=batch_digest),final)
    result=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],
                initial_sha256=initial,batch_sha256=batch_digest,optimization_seconds=seconds,
                parameters_real=sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters()),
                selected_sha256=file_hash(final),denominator_floor=floor,selection='fixed final; no checkpoint picking')
    atomic_json(folder/'result.json',result);atomic_json(manifest,dict(identity=identity,status='completed',git_commit=git_commit()))
    atomic_json(folder/'progress.json',dict(status='completed',step=step,optimization_seconds=seconds));return result
