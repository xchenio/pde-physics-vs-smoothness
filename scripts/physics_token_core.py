"""Small-data trajectory factorial with differentiable physical-coordinate residuals."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp
from scripts.token_operator import TokenDeepONet
from scripts.run_public_pod_readout import learning_rate
from scripts.objective_modal_core import sample_weights
from scripts.run_public_ridge_residual import event,git_commit
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save,random_state,restore_random
from src.utils import set_seed,env_fingerprint


def physical_fields(model,coeff,coords,scale,offset):
    ex=torch.zeros_like(coords);ex[:,0]=1.;et=torch.zeros_like(coords);et[:,1]=1.
    phi,px=jvp(model.trunk_features,(coords,),(ex,))
    _,pt=jvp(model.trunk_features,(coords,),(et,))
    _,pxx=jvp(lambda z:jvp(model.trunk_features,(z,),(ex,))[1],(coords,),(ex,))
    factor=scale/np.sqrt(model.rank)
    return (coeff@phi.T*factor+offset,coeff@px.T*factor,coeff@pt.T*factor,coeff@pxx.T*factor)


def residual(u,ux,ut,uxx,diffusion):return ut+u*ux-diffusion*uxx


def grid(x,times,device):
    x=torch.tensor(x,dtype=torch.float32,device=device);t=torch.tensor(times,dtype=torch.float32,device=device)
    tt,xx=torch.meshgrid(t,x,indexing='ij');return torch.stack([xx.flatten(),tt.flatten()],1)


def trajectory_errors(pred,true):
    return np.linalg.norm(pred-true,axis=2)/np.maximum(np.linalg.norm(true,axis=2),1e-12)


def evaluate(model,arrays,normalization,scale,device):
    model.eval();result={}
    with torch.no_grad():
        for split in ('train','val'):
            raw=np.asarray(arrays[split+'_x'],dtype=float);x=torch.tensor((raw-normalization['mean'])/normalization['std'],dtype=torch.float32,device=device)
            off=raw.mean(1,keepdims=True);coords=grid(arrays['grid256'],arrays['times'],device)
            pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*scale+off
            pred=pred.reshape(len(raw),len(arrays['times']),256);e=trajectory_errors(pred,arrays[split+'_y'])
            result[split]=dict(primary=float(e[:,1:].mean()),by_time=e.mean(0).tolist(),per_sample=e[:,1:].mean(1).tolist(),
                               final_time=float(e[:,-1].mean()),initial_time=float(e[:,0].mean()))
            if split=='val' and 'val_y512' in arrays:
                coords=grid(arrays['grid512'],arrays['times'],device)
                pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*scale+off
                pred=pred.reshape(len(raw),len(arrays['times']),512);e=trajectory_errors(pred,arrays['val_y512'])
                result['val512']=dict(primary=float(e[:,1:].mean()),by_time=e.mean(0).tolist(),per_sample=e[:,1:].mean(1).tolist(),final_time=float(e[:,-1].mean()))
    model.train();return result


def train_one(c,arrays,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);identity=dict(config=c,context=context,environment=env_fingerprint(device))
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r['identity']!=identity:raise ValueError('result identity mismatch')
        print('[skip]',folder.name,flush=True);return r
    raw=np.asarray(arrays['train_x'],dtype=float);truth=np.asarray(arrays['train_y'],dtype=float)
    norm=dict(mean=float(raw.mean()),std=float(raw.std()));scale=float(truth.std())
    if norm['std']<=0 or scale<=0:raise ValueError('zero scale')
    x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device);offset=torch.tensor(raw.mean(1,keepdims=True),dtype=torch.float32,device=device)
    target=torch.tensor((truth.reshape(len(raw),-1)-raw.mean(1,keepdims=True))/scale,dtype=torch.float32,device=device)
    # Per-time energy protection on exactly the100 labeled training trajectories.
    w,floor=sample_weights(truth.reshape(-1,256),'normalized');weights=torch.tensor(w.reshape(len(raw),len(arrays['times']),1),dtype=torch.float32,device=device)
    physics_weights=weights[:,1:].mean(1)
    coords=grid(arrays['grid256'],arrays['times'],device)
    set_seed(c['seed']+2000);model=TokenDeepONet(c['kind'],64,1.,1.).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[];bh=hashlib.sha256(b'').hexdigest();ch=bh;cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['initial']!=initial:raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step,seconds,history,bh,ch=cp['step'],cp['seconds'],cp['history'],cp['batch_hash'],cp['collocation_hash']
    event(folder,'resume' if step else 'start',step=step);atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()))

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),initial=initial,
                         step=step,seconds=seconds,history=history,batch_hash=bh,collocation_hash=ch,normalization=norm,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(status='running',step=step,total=c['steps'],seconds=seconds,last=history[-1]))

    if not history:history.append(dict(step=0,metrics=evaluate(model,arrays,norm,scale,device)));save()
    end=c['steps'] if stop_after is None else min(c['steps'],stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(raw),min(c['batch_size'],len(raw)),replace=False)
        points=rng.uniform(0.,1.,size=(c['collocation'],2)).astype('<f4');pc=torch.tensor(points,device=device)
        bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest();ch=hashlib.sha256(bytes.fromhex(ch)+points.tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        b=model.branch(x[idx]);z=b@model.trunk_features(coords).T/np.sqrt(64)
        dl=(weights[idx]*(z-target[idx]).reshape(len(idx),len(arrays['times']),256).square()).mean()
        # BOTH physics-on/off compute collocation derivatives for matched diagnostic cost.
        u,ux,ut,uxx=physical_fields(model,b,pc,scale,offset[idx]);r=residual(u,ux,ut,uxx,c['diffusion'])
        pl=(physics_weights[idx]*(r/scale).square()).mean();loss=dl+c['physics_weight']*pl
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        gn=None
        if (step+1)%c['eval_every']==0 or step+1==c['steps']:
            params=list(model.parameters());gd=torch.autograd.grad(dl,params,retain_graph=True,allow_unused=True)
            gp=torch.autograd.grad(pl,params,retain_graph=True,allow_unused=True)
            norm_grad=lambda values:float(torch.sqrt(sum(g.square().sum() for g in values if g is not None)).detach())
            gn=dict(data=norm_grad(gd),physics=norm_grad(gp))
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            row=dict(step=step,metrics=evaluate(model,arrays,norm,scale,device),data_loss=float(dl.detach()),physics_loss=float(pl.detach()),gradient_norms=gn)
            history.append(row);event(folder,'evaluation',**row);print(folder.name,step,'dev',row['metrics']['val']['primary'],'PDE',row['physics_loss'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt';atomic_save(dict(identity=identity,model=model.state_dict(),step=step,normalization=norm,scale=scale,initial=initial,batch_hash=bh,collocation_hash=ch),final)
    result=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],initial=initial,batch_hash=bh,collocation_hash=ch,
                parameters=sum(v.numel() for v in model.parameters()),seconds=seconds,final_sha256=file_hash(final),denominator_floor=floor,
                peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
                selection='fixed final, all four factorial arms; no weight tuning')
    atomic_json(folder/'result.json',result);atomic_json(manifest,dict(identity=identity,status='completed'));atomic_json(folder/'progress.json',dict(status='completed',step=step));return result
