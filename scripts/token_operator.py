"""Matched token interaction: attention versus a pointwise gated control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts.spatial_deeponet import SpatialDeepONet,coordinates
from scripts.run_public_pod_readout import prepare,learning_rate
from scripts.objective_modal_core import sample_weights
from scripts.run_public_benchmark import metrics
from scripts.manifold_reconstruction import score,metric_floors
from scripts.run_public_ridge_residual import event,git_commit
from src.utils import set_seed,env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash


class TokenDeepONet(SpatialDeepONet):
    def __init__(self,kind,rank=64,length=1.,final_time=1.):
        super().__init__(rank,length,final_time)
        if kind not in ('gate','attention'):raise ValueError('unknown token interaction')
        self.kind=kind
        self.lift=nn.Linear(rank//16,32);self.position=nn.Linear(2,32,bias=False)
        self.norm1=nn.LayerNorm(32);self.qkv=nn.Linear(32,96);self.mix=nn.Linear(32,32)
        self.norm2=nn.LayerNorm(32);self.ffn=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
        self.lower=nn.Linear(32,rank//16)
        phase=2*torch.pi*torch.arange(16)/16
        self.register_buffer('positions',torch.stack([phase.sin(),phase.cos()],1))

    def tokens(self,x):
        z=self.encoder(x[:,None]).transpose(1,2)
        h=self.lift(z)+self.position(self.positions)[None]
        q,k,v=self.qkv(self.norm1(h)).chunk(3,dim=-1)
        if self.kind=='attention':
            # Four heads, no dropout or fused backend variation.
            reshape=lambda a:a.reshape(a.shape[0],16,4,8).transpose(1,2)
            q,k,v=map(reshape,(q,k,v))
            z=(torch.softmax(q@k.transpose(-2,-1)/np.sqrt(8),dim=-1)@v).transpose(1,2).reshape(h.shape)
        else:z=v*torch.sigmoid(q+k)
        h=h+self.mix(z);h=h+self.ffn(self.norm2(h))
        return self.lower(h).transpose(1,2)

    def branch(self,x):return self.branch_head(self.tokens(x).flatten(1))


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
    set_seed(c['seed']+2000);model=TokenDeepONet(c['kind'],c['rank'],c['length'],c['final_time']).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    grid=coordinates(c['task'],target.shape[1],device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[];bh=hashlib.sha256(b'').hexdigest();cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['initial']!=initial:raise ValueError('checkpoint mismatch')
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
            print(folder.name,step,'val',history[-1]['metrics']['val']['prediction']['mean_relative_l2'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt';atomic_save(dict(identity=identity,model=model.state_dict(),step=step,anchor=anchor,scale=scale,initial=initial,batch_hash=bh),final)
    r=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],parameters=sum(v.numel() for v in model.parameters()),
           optimization_seconds=seconds,initial=initial,batch_hash=bh,final_sha256=file_hash(final),denominator_floor=floor,
           peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
           selection='fixed final; all arms reported; no confirmation')
    atomic_json(folder/'result.json',r);atomic_json(manifest,dict(identity=identity,status='completed'));atomic_json(folder/'progress.json',dict(status='completed',step=step))
    return r
