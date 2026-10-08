"""Same100 labels/times/batches FNO capacity brackets, data-only known-time heads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.fno import FNO1d
from src.utils import set_seed,env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash
from scripts.objective_modal_core import sample_weights
from scripts.run_public_pod_readout import learning_rate
from scripts.run_public_ridge_residual import event,git_commit
from scripts.physics_token_core import trajectory_errors


def make_model(width,seed,device):
    set_seed(seed+2000);model=FNO1d(width=width,modes=16,layers=4)
    model.project[-1]=nn.Conv1d(width,5,1);nn.init.zeros_(model.project[-1].weight);nn.init.zeros_(model.project[-1].bias)
    return model.to(device)


def train_one(c,arrays,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);identity=dict(config=c,context=context,environment=env_fingerprint(device));manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r['identity']!=identity:raise ValueError('result mismatch')
        print('[skip]',folder.name,flush=True);return r
    raw=np.asarray(arrays['train_x'],dtype=float);truth=np.asarray(arrays['train_y'],dtype=float)
    norm=dict(mean=float(raw.mean()),std=float(raw.std()));scale=float(truth.std())
    xs={s:torch.tensor((np.asarray(arrays[s+'_x'],dtype=float)-norm['mean'])/norm['std'],dtype=torch.float32,device=device) for s in ('train','val')}
    target=torch.tensor((truth-raw.mean(1)[:,None,None])/scale,dtype=torch.float32,device=device)
    w,floor=sample_weights(truth.reshape(-1,256),'normalized');weights=torch.tensor(w.reshape(len(raw),5,1),dtype=torch.float32,device=device)
    coords=torch.tensor(arrays['grid256'],dtype=torch.float32,device=device);model=make_model(c['width'],c['seed'],device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);rng=np.random.default_rng(c['seed']+3000)
    step=0;seconds=0.;history=[];bh=hashlib.sha256(b'').hexdigest();cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['initial']!=initial:raise ValueError('checkpoint mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step,seconds,history,bh=cp['step'],cp['seconds'],cp['history'],cp['batch_hash']
    atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()));event(folder,'resume' if step else 'start',step=step)

    def evaluate():
        model.eval();scores={}
        with torch.no_grad():
            for split in ('train','val'):
                pred=torch.cat([model(b,coords) for b in xs[split].split(64)]).cpu().numpy().astype(float)*scale+np.asarray(arrays[split+'_x'],dtype=float).mean(1)[:,None,None]
                e=trajectory_errors(pred,arrays[split+'_y']);scores[split]=dict(primary=float(e[:,1:].mean()),per_sample=e[:,1:].mean(1).tolist(),by_time=e.mean(0).tolist())
        model.train();return dict(step=step,metrics=scores)

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),initial=initial,step=step,
                         seconds=seconds,history=history,batch_hash=bh,normalization=norm,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],status='running',last=history[-1]))

    if not history:history.append(evaluate());save()
    end=c['steps'] if stop_after is None else min(c['steps'],stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(raw),min(c['batch_size'],len(raw)),replace=False)
        rng.uniform(0.,1.,size=(c['collocation'],2)) # Match factorial batch RNG; no PDE objective in this control.
        bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        loss=(weights[idx]*(model(xs['train'][idx],coords)-target[idx]).square()).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            history.append(evaluate());event(folder,'evaluation',**history[-1]);print(folder.name,step,history[-1]['metrics']['val']['primary'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt';atomic_save(dict(identity=identity,model=model.state_dict(),normalization=norm,scale=scale,step=step,initial=initial,batch_hash=bh),final)
    r=dict(identity=identity,history=history,metrics=history[-1]['metrics'],seconds=seconds,initial=initial,batch_hash=bh,
           parameters_real=sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters()),final_sha256=file_hash(final),
           denominator_floor=floor,git_commit=git_commit(),peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0)
    atomic_json(folder/'result.json',r);atomic_json(manifest,dict(identity=identity,status='completed'));atomic_json(folder/'progress.json',dict(status='completed',step=step));return r
