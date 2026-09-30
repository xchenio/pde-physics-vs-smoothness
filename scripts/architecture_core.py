"""Common, resumable optimization for POD readouts and full-field FNO controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.models import MLP
from src.fno import FNO1d
from src.utils import set_seed,env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash
from scripts.run_public_pod_readout import prepare,learning_rate
from scripts.objective_modal_core import sample_weights
from scripts.run_public_benchmark import metrics
from scripts.run_public_ridge_residual import git_commit,event,score_coefficients


class DeterministicPool(torch.nn.Module):
    """Adaptive average bins expressed as deterministic matrix multiplication."""
    def forward(self,x):
        n=x.shape[-1]
        matrix=x.new_zeros(n,24)
        for i in range(24):
            lo=i*n//24;hi=((i+1)*n+23)//24
            matrix[lo:hi,i]=1/(hi-lo)
        return x@matrix


class SpatialBranch(torch.nn.Module):
    def __init__(self,rank):
        super().__init__()
        self.features=torch.nn.Sequential(
            torch.nn.Conv1d(1,16,5,padding=2,padding_mode='circular'),torch.nn.Tanh(),
            torch.nn.Conv1d(16,16,5,padding=2,padding_mode='circular'),torch.nn.Tanh(),
            torch.nn.Conv1d(16,16,5,padding=2,padding_mode='circular'),torch.nn.Tanh(),
            DeterministicPool())
        self.head=torch.nn.Linear(16*24,rank)
    def forward(self,x):return self.head(self.features(x[:,None]).flatten(1))


def make_model(c,in_dim,device):
    set_seed(c['seed']+2000)
    if c['architecture']=='mlp':model=MLP(in_dim,c['rank'],32,1);last=model.net[-1]
    elif c['architecture']=='cnn':model=SpatialBranch(c['rank']);last=model.head
    elif c['architecture']=='fno':model=FNO1d(width=32,modes=16,layers=4);last=model.project[-1]
    else:raise ValueError('unknown architecture')
    torch.nn.init.zeros_(last.weight);torch.nn.init.zeros_(last.bias)
    return model.to(device)


def output(model,x,architecture):
    if architecture=='fno':
        coordinates=(torch.arange(x.shape[1],device=x.device,dtype=x.dtype)+.5)/x.shape[1]
        return model(x,coordinates)
    return model(x)


def train_one(c,arrays,anchor,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=dict(config=c,context=context,environment=env_fingerprint(device));manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result['identity']!=identity:raise ValueError('result mismatch')
        print('[skip]',folder.name,flush=True);return result
    atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()))
    prepared,scale=prepare(arrays,anchor);q=anchor['q'];n=arrays['train_y'].shape[1]
    base={s:v['x']@anchor['weights']+anchor['intercept'] for s,v in prepared.items()}
    field=c['architecture']=='fno';residual=c['mapping']=='residual'
    xs={s:torch.tensor(v['x'],dtype=torch.float32,device=device) for s,v in prepared.items()}
    targets={s:(v['y']-v['offset'] if field else v['coef']) for s,v in prepared.items()}
    baselines={s:(base[s]@q.T if field else base[s]) if residual else np.zeros_like(targets[s]) for s in prepared}
    target=torch.tensor((targets['train']-baselines['train'])/scale,dtype=torch.float32,device=device)
    w,floor=sample_weights(arrays['train_y'],'normalized');weights=torch.tensor(w,dtype=torch.float32,device=device)
    model=make_model(c,xs['train'].shape[1],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[];cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity:raise ValueError('checkpoint mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step=cp['step'];seconds=cp['optimization_seconds'];history=cp['history']
    event(folder,'resume' if history else 'start',step=step,git_commit=git_commit())

    def evaluate():
        model.eval();scores={}
        with torch.no_grad():
            for s,v in prepared.items():
                z=torch.cat([output(model,b,c['architecture']) for b in xs[s].split(128)]).cpu().numpy().astype(float)*scale
                estimate=baselines[s]+z
                if field:
                    pred=estimate+v['offset'];scores[s]=dict(prediction=metrics(pred,v['y']),pod_oracle_reference=v['oracle'])
                else:scores[s]=score_coefficients(estimate,v,q)
                # Oracle minimizes the penalized squared objective pointwise; not a predictor.
                optimal=baselines[s]+(targets[s]-baselines[s])/(1+c['penalty'])
                oracle=(optimal if field else optimal@q.T)+v['offset']
                scores[s]['penalized_free_output_oracle']=metrics(oracle,v['y'])
        model.train();return dict(step=step,metrics=scores)

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=random_state(rng),step=step,history=history,optimization_seconds=seconds,anchor=anchor,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],optimization_seconds=seconds,status='running',last_evaluation=history[-1]))

    if not history:history.append(evaluate());save()
    end=min(c['steps'],stop_after) if stop_after is not None else c['steps']
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(target),min(c['batch_size'],len(target)),replace=False)
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        z=output(model,xs['train'][idx],c['architecture'])
        loss=(weights[idx]*((z-target[idx]).square().sum(1)+c['penalty']*z.square().sum(1))).mean()/n
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
    atomic_save(dict(identity=identity,model=model.state_dict(),step=step,anchor=anchor,scale=scale),final)
    result=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],
        optimization_seconds=seconds,parameters_real=sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters()),
        selected_sha256=file_hash(final),denominator_floor=floor,selection='fixed final step; all histories retained, no checkpoint picking')
    atomic_json(folder/'result.json',result);atomic_json(manifest,dict(identity=identity,status='completed',git_commit=git_commit()))
    atomic_json(folder/'progress.json',dict(status='completed',step=step,optimization_seconds=seconds));return result
