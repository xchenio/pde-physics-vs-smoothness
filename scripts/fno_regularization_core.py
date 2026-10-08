"""Paired frozen-warm-start FNO continuation with train-gradient calibration."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.time_fno import time_fields,conservative_residual,coefficient_energy,TimeFNO1d
from src.utils import set_seed,env_fingerprint
from src.resume import atomic_save,random_state,restore_random
from src.public_benchmark import atomic_json,file_hash
from scripts.run_public_ridge_residual import event,git_commit
from scripts.run_public_pod_readout import learning_rate
from scripts.time_fno_baseline import predict
from scripts.objective_modal_core import sample_weights
from scripts.physics_token_core import trajectory_errors


def penalties(model,x,t,coords,scale,offset,diffusion,band,create_graph=True):
    u,ut=time_fields(model,x,t,coords,scale,offset,create_graph=create_graph)
    uf=torch.fft.rfft(u,norm='forward')[...,:band+1]
    k=torch.arange(band+1,device=u.device,dtype=u.dtype)*2*torch.pi
    return dict(physics=coefficient_energy(conservative_residual(u,ut,diffusion,band)).mean()/scale**2,
                gradient=coefficient_energy(1j*k*uf).mean()/scale**2,
                curvature=coefficient_energy(-k.square()*uf).mean()/scale**2)


def gradient_norm(loss,model):
    values=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    return float(torch.sqrt(sum(v.abs().square().sum() for v in values if v is not None)).detach())


def train_one(c,arrays,context,warm_path,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=dict(config=c,context=context,warm_sha256=file_hash(warm_path),environment=env_fingerprint(device))
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        r=json.loads((folder/'result.json').read_text())
        if r['identity']!=identity or file_hash(folder/'checkpoints/final.pt')!=r['final_sha256']:raise ValueError('result mismatch')
        print('[skip]',folder.name,flush=True);return r
    warm=torch.load(warm_path,map_location='cpu',weights_only=False)
    norm=warm['normalization'];scale=warm['scale'];raw=np.asarray(arrays['train_x'],dtype=float);truth=np.asarray(arrays['train_y'],dtype=float)
    xs={s:torch.tensor((np.asarray(arrays[s+'_x'],dtype=float)-norm['mean'])/norm['std'],dtype=torch.float32,device=device) for s in ('train','val')}
    n=raw.shape[-1];nt=truth.shape[1]
    target=torch.tensor((truth-raw.mean(1)[:,None,None])/scale,dtype=torch.float32,device=device)
    w,_=sample_weights(truth.reshape(-1,n),'normalized');weights=torch.tensor(w.reshape(len(raw),nt,1),dtype=torch.float32,device=device)
    offset=torch.tensor(raw.mean(1),dtype=torch.float32,device=device)
    times=torch.tensor(arrays['times'],dtype=torch.float32,device=device);coords=torch.tensor(arrays['grid256'],dtype=torch.float32,device=device)
    set_seed(c['seed']+2000);model=TimeFNO1d(c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(warm['model'])
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);optimizer.load_state_dict(warm['optimizer'])
    rng=np.random.default_rng();restore_random(warm['rng'],rng)
    crng=np.random.default_rng(c['seed']+4000)
    # Same first16 training IDs and fixed interleaved times in every arm.
    probe=np.arange(min(16,len(raw)));pt=torch.linspace(.1,1.,len(probe),device=device)
    data_loss=(weights[probe]*(predict(model,xs['train'][probe],times,coords)-target[probe]).square()).mean()
    ps=penalties(model,xs['train'][probe],pt,coords,scale,offset[probe],c['diffusion'],c['resolved_max_mode'])
    gdata=gradient_norm(data_loss,model)
    calibration=dict(data_gradient_norm=gdata,penalty_gradient_norms={k:gradient_norm(v,model) for k,v in ps.items()})
    if min([gdata]+list(calibration['penalty_gradient_norms'].values()))<=1e-12:raise ValueError('degenerate gradient calibration')
    calibration['unit_weights']={k:gdata/v for k,v in calibration['penalty_gradient_norms'].items()}
    lam=0. if c['kind']=='none' else c['strength']*calibration['unit_weights'][c['kind']]
    del data_loss,ps
    step=0;seconds=0.;history=[];bh=warm['batch_hash'];ch=hashlib.sha256(b'').hexdigest();cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity']!=identity or cp['calibration']!=calibration:raise ValueError('checkpoint/calibration mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng);crng.bit_generator.state=cp['collocation_rng']
        step,seconds,history,bh,ch=cp['step'],cp['seconds'],cp['history'],cp['batch_hash'],cp['collocation_hash']
    atomic_json(manifest,dict(identity=identity,status='running',git_commit=git_commit()));event(folder,'resume' if step else 'start',step=step)

    def evaluate():
        model.eval();scores={}
        with torch.no_grad():
            for split in ('train','val'):
                pred=torch.cat([predict(model,b,times,coords) for b in xs[split].split(16)]).cpu().numpy().astype(float)*scale+np.asarray(arrays[split+'_x'],dtype=float).mean(1)[:,None,None]
                e=trajectory_errors(pred,arrays[split+'_y']);scores[split]=dict(primary=float(e[:,1:].mean()),per_sample=e[:,1:].mean(1).tolist(),by_time=e.mean(0).tolist())
        probe_data=(weights[probe]*(predict(model,xs['train'][probe],times,coords)-target[probe]).square()).mean()
        probe_ps=penalties(model,xs['train'][probe],pt,coords,scale,offset[probe],c['diffusion'],c['resolved_max_mode'])
        gn=gradient_norm(probe_data,model)
        ratio=0. if c['kind']=='none' else lam*gradient_norm(probe_ps[c['kind']],model)/max(gn,1e-12)
        diag={k:float(v.detach()) for k,v in probe_ps.items()}
        model.train();return dict(step=step,metrics=scores,training_probe_penalties=diag,effective_gradient_ratio=ratio)

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),collocation_rng=crng.bit_generator.state,
                         initial=initial,step=step,seconds=seconds,history=history,batch_hash=bh,collocation_hash=ch,normalization=norm,scale=scale,calibration=calibration),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],status='running',last=history[-1]))
    if not history:history.append(evaluate());save()
    end=c['steps'] if stop_after is None else min(c['steps'],stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(raw),min(c['batch_size'],len(raw)),replace=False);rng.uniform(0.,1.,size=(32,2))
        qt=crng.uniform(.1,1.,len(idx)).astype('<f4')
        bh=hashlib.sha256(bytes.fromhex(bh)+idx.astype('<i8').tobytes()).hexdigest();ch=hashlib.sha256(bytes.fromhex(ch)+qt.tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        dl=(weights[idx]*(predict(model,xs['train'][idx],times,coords)-target[idx]).square()).mean()
        # Every arm computes all diagnostics; only the registered term affects gradients.
        ps=penalties(model,xs['train'][idx],torch.tensor(qt,device=device),coords,scale,offset[idx],c['diffusion'],c['resolved_max_mode'])
        loss=dl if c['kind']=='none' else dl+lam*ps[c['kind']]
        if not torch.isfinite(loss) or not all(torch.isfinite(v) for v in ps.values()):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            history.append(evaluate());event(folder,'evaluation',**history[-1]);print(folder.name,step,history[-1]['metrics']['val']['primary'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt';atomic_save(dict(identity=identity,model=model.state_dict(),normalization=norm,scale=scale,step=step,initial=initial,batch_hash=bh,collocation_hash=ch),final)
    r=dict(identity=identity,history=history,metrics=history[-1]['metrics'],seconds=seconds,initial=initial,batch_hash=bh,collocation_hash=ch,calibration=calibration,
           penalty_weight=lam,parameters_real=sum(v.numel()*(2 if v.is_complex() else 1) for v in model.parameters()),final_sha256=file_hash(final),git_commit=git_commit(),
           peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0)
    atomic_json(folder/'result.json',r);atomic_json(manifest,dict(identity=identity,status='completed'));atomic_json(folder/'progress.json',dict(status='completed',step=step));return r
