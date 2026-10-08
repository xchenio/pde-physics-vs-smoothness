"""Stage-one compression diagnostic; no input-to-solution prediction or holdouts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.resume import atomic_save, random_state, restore_random
from src.public_benchmark import atomic_json, file_hash
from src.utils import set_seed, env_fingerprint
from scripts.run_public_ridge_residual import digest, event, git_commit
from scripts.run_public_pod_readout import learning_rate


class Autoencoder(nn.Module):
    def __init__(self, kind, rank, n=256):
        super().__init__()
        if kind not in ('global', 'local', 'scrambled') or rank % 16 or n != 256:
            raise ValueError('invalid autoencoder shape/kind')
        self.kind, self.rank = kind, rank
        permutation = np.random.default_rng(2026100801).permutation(n) if kind == 'scrambled' else np.arange(n)
        self.register_buffer('permutation', torch.tensor(permutation, dtype=torch.long))
        self.register_buffer('inverse', torch.tensor(np.argsort(permutation), dtype=torch.long))
        if kind == 'global':
            self.encoder = nn.Sequential(nn.Linear(n,64), nn.GELU(), nn.Linear(64,64), nn.GELU(), nn.Linear(64,rank))
            self.decoder = nn.Sequential(nn.Linear(rank,64), nn.GELU(), nn.Linear(64,64), nn.GELU(), nn.Linear(64,n))
        else:
            conv = lambda i,o,s=1: nn.Conv1d(i,o,9,stride=s,padding=4,padding_mode='circular')
            self.encoder = nn.Sequential(conv(1,32,4),nn.GELU(),conv(32,32,4),nn.GELU(),nn.Conv1d(32,rank//16,1))
            self.decoder = nn.Sequential(nn.Conv1d(rank//16,32,1),nn.GELU(),nn.Upsample(scale_factor=4,mode='nearest'),
                                         conv(32,32),nn.GELU(),nn.Upsample(scale_factor=4,mode='nearest'),conv(32,1))

    def encode(self,x):
        return self.encoder(x if self.kind == 'global' else x[:,self.permutation,None].transpose(1,2))

    def decode(self,z):
        if self.kind == 'global': return self.decoder(z)
        return self.decoder(z).squeeze(1)[:,self.inverse]

    def forward(self,x): return self.decode(self.encode(x))


def derivative(y, length, order):
    y=np.asarray(y,dtype=float); n=y.shape[1]
    k=2*np.pi*np.fft.rfftfreq(n,d=length/n); multiplier=(1j*k)**order
    if order % 2 and n % 2 == 0: multiplier[-1]=0
    return np.fft.irfft(np.fft.rfft(y,axis=1)*multiplier,n=n,axis=1)


def metric_floors(train,length):
    return {str(o):max(1e-12,1e-3*float(np.sqrt(np.mean((train if o==0 else derivative(train,length,o))**2)))) for o in (0,1,2)}


def errors(pred,true,length,floors):
    pred,true=np.asarray(pred,dtype=float),np.asarray(true,dtype=float); out={}
    for o,key in ((0,'field'),(1,'gradient'),(2,'curvature')):
        a,b=(pred,true) if o==0 else (derivative(pred,length,o),derivative(true,length,o))
        out[key]=np.sqrt(np.mean((a-b)**2,axis=1))/np.maximum(np.sqrt(np.mean(b*b,axis=1)),floors[str(o)])
    # Parseval weighting: bands are errors divided by TOTAL field norm, not tiny band energies.
    a,b=np.fft.rfft(pred,axis=1),np.fft.rfft(true,axis=1); n=true.shape[1]
    weight=np.full(a.shape[1],2.);weight[0]=1.;weight[-1]=1.
    den=np.maximum(np.sqrt(np.mean(true*true,axis=1)),floors['0'])
    for lo,hi in ((0,4),(5,16),(17,64),(65,128)):
        out[f'band_{lo}_{hi}']=np.sqrt(np.sum(weight[lo:hi+1]*np.abs(a[:,lo:hi+1]-b[:,lo:hi+1])**2,axis=1))/n/den
    return out


def score(pred,true,length,floors):
    return {k:float(v.mean()) for k,v in errors(pred,true,length,floors).items()}


def array_digest(x):
    a=np.ascontiguousarray(x); return digest([str(a.dtype),list(a.shape),hashlib.sha256(a.tobytes()).hexdigest()])


def train_one(c,train,val,context,folder,device,stop_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=dict(config=c,context=context,environment=env_fingerprint(device))
    manifest=folder/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity: raise ValueError('identity mismatch')
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result['identity'] != identity: raise ValueError('result identity mismatch')
        print('[skip]',folder.name,flush=True);return result
    set_seed(c['seed']+2000);model=Autoencoder(c['kind'],c['rank']).to(device)
    initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    mean,scale=float(train.mean()),float(train.std())
    if scale<=0:raise ValueError('zero scale')
    xs={k:torch.tensor((v-mean)/scale,device=device,dtype=torch.float32) for k,v in (('train',train),('val',val))}
    floors=metric_floors(train,c['length']);optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+3000);step=0;seconds=0.;history=[];batch_hash=hashlib.sha256(b'').hexdigest()
    cp_path=folder/'checkpoints/latest.pt'
    if cp_path.exists():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        if cp['identity'] != identity or cp['initial'] != initial:raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);restore_random(cp['rng'],rng)
        step,seconds,history,batch_hash=cp['step'],cp['seconds'],cp['history'],cp['batch_hash']
    event(folder,'resume' if step else 'start',step=step,git_commit=git_commit())

    def predictions(split):
        model.eval()
        with torch.no_grad(): pred=torch.cat([model(b) for b in xs[split].split(128)]).cpu().numpy().astype(float)*scale+mean
        model.train();return pred

    def evaluate():
        return dict(step=step,metrics={s:score(predictions(s),y,c['length'],floors) for s,y in (('train',train),('val',val))})

    def save():
        atomic_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=random_state(rng),
                         initial=initial,step=step,seconds=seconds,history=history,batch_hash=batch_hash,mean=mean,scale=scale),cp_path)
        atomic_json(folder/'progress.json',dict(step=step,total=c['steps'],seconds=seconds,last=history[-1],status='running'))

    if not history:history.append(evaluate());save()
    end=c['steps'] if stop_after is None else min(c['steps'],stop_after)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    while step<end:
        if device.type=='cuda':torch.cuda.synchronize()
        begin=time.perf_counter();idx=rng.choice(len(train),min(c['batch_size'],len(train)),replace=False)
        batch_hash=hashlib.sha256(bytes.fromhex(batch_hash)+idx.astype('<i8').tobytes()).hexdigest()
        optimizer.param_groups[0]['lr']=learning_rate(c,step);optimizer.zero_grad(set_to_none=True)
        loss=(model(xs['train'][idx])-xs['train'][idx]).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        seconds+=time.perf_counter()-begin;step+=1
        if step%c['eval_every']==0 or step==c['steps']:
            history.append(evaluate());event(folder,'evaluation',step=step,metrics=history[-1]['metrics'])
            print(folder.name,step,'val',history[-1]['metrics']['val'],flush=True)
        if step%c['checkpoint_every']==0 or step==end:save()
    if step<c['steps']:return None
    final=folder/'checkpoints/final.pt'
    atomic_save(dict(identity=identity,model=model.state_dict(),step=step,mean=mean,scale=scale,initial=initial,batch_hash=batch_hash),final)
    pred=predictions('val');per_sample={k:v.tolist() for k,v in errors(pred,val,c['length'],floors).items()}
    result=dict(identity=identity,git_commit=git_commit(),history=history,metrics=history[-1]['metrics'],per_sample=per_sample,
                parameters=sum(v.numel() for v in model.parameters()),seconds=seconds,initial=initial,batch_hash=batch_hash,
                final_sha256=file_hash(final),floors=floors,peak_allocated_bytes_last_segment=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
                selection='fixed final step; all arms reported',latent_scalars=c['rank'])
    atomic_json(folder/'result.json',result);atomic_json(manifest,dict(identity=identity,status='completed'))
    atomic_json(folder/'progress.json',dict(status='completed',step=step,seconds=seconds));return result
