"""Minimal spatial branch and smooth, periodic-coordinate DeepONet trunk."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.run_public_pod_readout import prepare,learning_rate
from scripts.objective_modal_core import sample_weights
from scripts.run_public_benchmark import metrics
from scripts.manifold_reconstruction import score,metric_floors
from scripts.run_public_ridge_residual import event,git_commit
from src.utils import set_seed,env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash


class SpatialDeepONet(nn.Module):
    def __init__(self,rank=64,length=1.,final_time=1.):
        super().__init__()
        if rank%16 or length<=0 or final_time<=0:raise ValueError('invalid rank/physical scales')
        self.rank,self.length,self.final_time=rank,length,final_time
        self.encoder=nn.Sequential(nn.Conv1d(1,32,9,stride=4,padding=4,padding_mode='circular'),nn.GELU(),
                                   nn.Conv1d(32,32,9,stride=4,padding=4,padding_mode='circular'),nn.GELU(),nn.Conv1d(32,rank//16,1))
        self.branch_head=nn.Linear(rank,rank)
        self.trunk=nn.Sequential(nn.Linear(33,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,rank))
        self.register_buffer('frequencies',torch.arange(1,17,dtype=torch.float32))
        nn.init.zeros_(self.branch_head.weight);nn.init.zeros_(self.branch_head.bias)

    def branch(self,x):return self.branch_head(self.encoder(x[:,None]).flatten(1))

    def trunk_features(self,coords):
        phase=2*torch.pi*coords[:,0:1]*self.frequencies/self.length
        return self.trunk(torch.cat([phase.sin(),phase.cos(),coords[:,1:2]/self.final_time],1))

    def forward(self,x,coords):return self.branch(x)@self.trunk_features(coords).T/np.sqrt(self.rank)


def coordinates(task,n,device):
    if task.startswith('burgers'):
        # Actual retained centers, not recentered coarse-grid cell centers.
        x=(torch.arange(n,device=device,dtype=torch.float32)*4+.5)/1024;time=1.
    elif task=='allen_cahn':x=torch.arange(n,device=device,dtype=torch.float32)*2/n;time=.5
    else:raise ValueError('unknown grid')
    return torch.stack([x,torch.full_like(x,time)],1)


def train_one(c,arrays,anchor,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);identity=dict(config=c,context=context,environment=env_fingerprint(device))
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r['identity']!=identity:raise ValueError('result identity mismatch')
        print('[skip]',folder.name,flush=True);return r
    prepared,scale=prepare(arrays,anchor);xs={s:torch.tensor(v['x'],dtype=torch.float32,device=device) for s,v in prepared.items()}
    target=torch.tensor((prepared['train']['y']-prepared['train']['offset'])/scale,dtype=torch.float32,device=device)
    weights,floor=sample_weights(arrays['train_y'],'normalized');weights=torch.tensor(weights,dtype=torch.float32,device=device)
    set_seed(c['seed']+2000);model=SpatialDeepONet(c['rank'],c['length'],c['final_time']).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    grid=coordinates(c['task'],target.shape[1],device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[];bh=hashlib.sha256(b'').hexdigest();cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['initial']!=initial:raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step,seconds,history,bh=cp['step'],cp['seconds'],cp['history'],cp['batch_hash']
    atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()));event(folder,'resume' if step else 'start',step=step)
    floors=metric_floors(arrays['train_y'],c['length'])

    def evaluate():
        model.eval();scores={}
        with torch.no_grad():
            for s,v in prepared.items():
                pred=torch.cat([model(b,grid) for b in xs[s].split(128)]).cpu().numpy().astype(float)*scale+v['offset']
                scores[s]=dict(prediction=metrics(pred,v['y']),structure=score(pred,v['y'],c['length'],floors))
        model.train();return dict(step=step,metrics=scores)

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),initial=initial,
                         batch_hash=bh,step=step,seconds=seconds,history=history,anchor=anchor,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],seconds=seconds,status='running',last=history[-1]))

    if not history:history.append(evaluate());save()
    end=c['steps'] if stop_after is None else min(c['steps'],stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(target),min(c['batch_size'],len(target)),replace=False)
        bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        loss=(weights[idx]*(model(xs['train'][idx],grid)-target[idx]).square().sum(1)).mean()/target.shape[1]
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            history.append(evaluate());event(folder,'evaluation',step=step,metrics=history[-1]['metrics'])
            print(folder.name,step,'train',history[-1]['metrics']['train']['prediction']['mean_relative_l2'],'val',history[-1]['metrics']['val']['prediction']['mean_relative_l2'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt';atomic_save(dict(identity=identity,model=model.state_dict(),step=step,anchor=anchor,scale=scale,initial=initial,batch_hash=bh),final)
    r=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],parameters=sum(v.numel() for v in model.parameters()),
           optimization_seconds=seconds,initial=initial,batch_hash=bh,final_sha256=file_hash(final),denominator_floor=floor,
           peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
           selection='fixed final step; fixed rank64 declared before stage1 results')
    atomic_json(folder/'result.json',r);atomic_json(manifest,dict(identity=identity,status='completed'));atomic_json(folder/'progress.json',dict(status='completed',step=step))
    return r
