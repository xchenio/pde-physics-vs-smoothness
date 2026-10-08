"""Seal all4 factorial arms before fresh500 extraction, then evaluate without tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import time
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_token import plan,load_data,OUT,DATA
from scripts.token_operator import TokenDeepONet
from scripts.physics_token_core import grid,trajectory_errors,physical_fields,residual
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
from scripts.run_incremental_modal import committed
from scripts.manifold_reconstruction import score,metric_floors
from scripts.run_public_benchmark import metrics
from scripts.fno_projection_core import make_model as fno_model,output as fno_output
from src.public_benchmark import atomic_json,file_hash
from src.resume import atomic_save


def seal(p):
    committed(OUT/'plan.json');committed(OUT/'historical_id_audit.json')
    a=json.loads((OUT/'audit.json').read_text())
    if a['checked_runs']!=12 or a['plan']!=digest(p):raise ValueError('complete current audit required')
    artifacts={};models={}
    for kind in p['kinds']:
        for weight in p['physics_weights']:
            arm=f'{kind}_p{weight:g}';models[arm]=[]
            for seed in p['seeds']:
                folder=OUT/f'{arm}_s{seed}';r=json.loads((folder/'result.json').read_text());cp=folder/'checkpoints/final.pt'
                if r['final_sha256']!=file_hash(cp) or r['identity']['context']['plan']!=digest(p):raise ValueError('model/context mismatch')
                models[arm].append(dict(seed=seed,path=cp.relative_to(ROOT).as_posix(),sha256=file_hash(cp)))
                artifacts[(folder/'result.json').relative_to(ROOT).as_posix()]=file_hash(folder/'result.json')
    fno_models=[]
    for seed in p['seeds']:
        folder=ROOT/'results/public_fno_projection_v1/burgers_original'/f'full_s{seed}'
        path=folder/'checkpoints/final.pt';r=json.loads((folder/'result.json').read_text())
        if file_hash(path)!=r['selected_sha256']:raise ValueError('historical FNO changed')
        fno_models.append(dict(seed=seed,path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path)))
    locked_json(OUT/'selection_lock.json',dict(plan=digest(p),models=models,fno_models=fno_models,artifacts=artifacts,
                audit_sha256=file_hash(OUT/'audit.json'),summary_sha256=file_hash(OUT/'summary.json'),
                confirm_source_sha256=file_hash(Path(__file__)),preseal_commit=git_commit(),
                policy='all4 arms, no adaptive model/weight/seed selection;4 registered primary contrasts; bootstrap conditional on3 fitted seeds'))
    print('SEALED ALL12; confirm requires committing selection_lock first',flush=True)


def contrast(a,b,seed=2026100806,replicates=10000):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);delta=a-b;rng=np.random.default_rng(seed)
    draws=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(replicates)])
    lo,hi=np.quantile(draws,[.05/8,1-.05/8]);gain=float(-delta.mean()/b.mean())
    return dict(difference=float(delta.mean()),relative_gain=gain,interval=[float(lo),float(hi)],success=bool(gain>=.01 and hi<0))


def confirmation_arrays(p,lockhash):
    marker=OUT/'confirmation_opened.json';path=DATA/'confirmation.pt'
    locked_json(marker,dict(lock=lockhash,ids=p['confirmation_ids'],consumed=True,
                          note='consumed from first target extraction; never relabel as untouched'))
    if path.exists():
        obj=torch.load(path,weights_only=False)
        if obj['lock']!=lockhash or obj['ids']!=p['confirmation_ids']:raise ValueError('confirmation cache mismatch')
        return obj['arrays']
    if file_hash(Path(p['dataset_source']))!=p['dataset_sha256']:raise ValueError('source mismatch')
    with h5py.File(p['dataset_source'],'r') as f:
        vals=np.stack([f['tensor'][int(i),p['time_indices'],:] for i in p['confirmation_ids']])
        arrays=dict(x=vals[:,0,::4].copy(),y256=vals[:,:,::4].copy(),y512=vals[:,:,::2].copy(),
                    grid256=f['x-coordinate'][::4],grid512=f['x-coordinate'][::2],times=p['times'])
    atomic_save(dict(lock=lockhash,ids=p['confirmation_ids'],arrays=arrays),path);return arrays


def load_model(entry,kind,device):
    path=ROOT/entry['path']
    if file_hash(path)!=entry['sha256']:raise ValueError('sealed model changed')
    cp=torch.load(path,map_location='cpu',weights_only=False);model=TokenDeepONet(kind,64,1.,1.).to(device)
    model.load_state_dict(cp['model']);model.eval();return model,cp


def profile(model,x,coords,device):
    """Host-inclusive synchronized timings, same batch64/query grid; no concurrent trainer."""
    with torch.inference_mode():
        for _ in range(10):model(x,coords)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        durations=[]
        for _ in range(50):
            begin=time.perf_counter();model(x,coords)
            if device.type=='cuda':torch.cuda.synchronize()
            durations.append(time.perf_counter()-begin)
    return dict(batch=len(x),query_points=len(coords),warmup=10,repeats=50,mean_ms=float(np.mean(durations)*1000),
                median_ms=float(np.median(durations)*1000),peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
                note='host-inclusive synchronized wall latency; inclusive live model/input/output allocation, not reserved VRAM')


def confirm(p,device):
    lockpath=OUT/'selection_lock.json';committed(lockpath);committed(OUT/'plan.json');lock=json.loads(lockpath.read_text());lockhash=file_hash(lockpath)
    if lock['plan']!=digest(p) or lock['confirm_source_sha256']!=file_hash(Path(__file__)):raise ValueError('plan/confirm source changed')
    if lock['audit_sha256']!=file_hash(OUT/'audit.json') or lock['summary_sha256']!=file_hash(OUT/'summary.json'):raise ValueError('development audit changed')
    for name,h in lock['artifacts'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed result changed')
    for entry in [e for entries in lock['models'].values() for e in entries]+lock['fno_models']:
        if file_hash(ROOT/entry['path'])!=entry['sha256']:raise ValueError('sealed checkpoint changed')
    resultpath=OUT/'confirmation.json'
    if resultpath.exists():
        r=json.loads(resultpath.read_text())
        if r['lock']!=lockhash:raise ValueError('confirmation identity changed')
        print('[skip completed confirmation]',flush=True);return r
    a=confirmation_arrays(p,lockhash);development,_=load_data(p);floors=metric_floors(development['train_y'][:,-1],1.);records={};profiles={}
    probe=torch.tensor(np.random.default_rng(2026100807).uniform(0.,1.,size=(128,2)).astype('float32'),device=device)
    for arm,entries in lock['models'].items():
        kind='attention' if arm.startswith('attention') else 'gate';rows=[]
        for entry in entries:
            path=OUT/'confirmation_runs'/f"{arm}_s{entry['seed']}.json";identity=dict(lock=lockhash,model=entry['sha256'],data_sha256=file_hash(DATA/'confirmation.pt'))
            if path.exists():
                row=json.loads(path.read_text())
                if row['identity']!=identity:raise ValueError('partial confirmation changed')
                rows.append(row);continue
            model,cp=load_model(entry,kind,device);norm=cp['normalization'];scale=cp['scale'];raw=np.asarray(a['x'],dtype=float)
            x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device);offset=raw.mean(1,keepdims=True);scores={};timing={}
            with torch.no_grad():
                for resolution in (256,512):
                    coords=grid(a[f'grid{resolution}'],a['times'],device)
                    pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*scale+offset
                    pred=pred.reshape(len(raw),5,resolution);e=trajectory_errors(pred,a[f'y{resolution}'])
                    scores[str(resolution)]=dict(primary=float(e[:,1:].mean()),per_sample=e[:,1:].mean(1).tolist(),by_time=e.mean(0).tolist(),
                                                final_structure=score(pred[:,-1],a[f'y{resolution}'][:,-1],1.,floors))
                    if resolution==512:
                        extra=trajectory_errors(pred[:,:,1::2],a['y512'][:,:,1::2]);scores['512_new_queries']=dict(primary=float(extra[:,1:].mean()),per_sample=extra[:,1:].mean(1).tolist())
                b=model.branch(x);values=physical_fields(model,b,probe,scale,torch.tensor(offset,dtype=torch.float32,device=device))
                pde=float(torch.sqrt(residual(*values,p['diffusion']).square().mean()).cpu())
            for resolution in (256,512):timing[str(resolution)]=profile(model,x[:64],grid(a[f'grid{resolution}'],a['times'],device),device)
            row=dict(identity=identity,seed=entry['seed'],scores=scores,pde_rms=pde,profiles=timing)
            atomic_json(path,row);rows.append(row);print('CONFIRMED',arm,entry['seed'],scores['256']['primary'],flush=True)
        records[arm]=rows
    means={arm:np.mean([r['scores']['256']['per_sample'] for r in rows],axis=0) for arm,rows in records.items()}
    comparisons={name:contrast(means[left],means[right]) for name,left,right in (
        ('attention_without_physics','attention_p0','gate_p0'),('physics_without_attention','gate_p0.01','gate_p0'),
        ('physics_with_attention','attention_p0.01','attention_p0'),('combination_vs_gate','attention_p0.01','gate_p0'))}
    # Strong historical control:600-shot fixed-final-time FNO, not the100-shot multitime primary contrast.
    fnopath=OUT/'confirmation_fno_context.json'
    if fnopath.exists():
        fno=json.loads(fnopath.read_text())
        if fno['lock']!=lockhash:raise ValueError('FNO context changed')
    else:
        fno_rows=[]
        for entry in lock['fno_models']:
            seed=entry['seed'];path=ROOT/entry['path']
            cp=torch.load(path,map_location='cpu',weights_only=False);model=fno_model(dict(seed=seed),device);model.load_state_dict(cp['model']);model.eval()
            raw=np.asarray(a['x'],dtype=float);anchor=cp['anchor'];x=torch.tensor((raw-anchor['mean'])/anchor['std'],dtype=torch.float32,device=device)
            with torch.no_grad():pred=torch.cat([fno_output(model,b) for b in x.split(128)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1,keepdims=True)
            fno_rows.append(dict(seed=seed,checkpoint_sha256=file_hash(path),metrics=metrics(pred,a['y256'][:,-1])))
        fno=dict(lock=lockhash,runs=fno_rows,note='600-shot retained FNO;final-time only;contextual, not matched100-shot/time supervision')
        atomic_json(fnopath,fno)
    summary={arm:{str(res):dict(mean=float(np.mean([r['scores'][str(res)]['primary'] for r in rows])),seed_sd=float(np.std([r['scores'][str(res)]['primary'] for r in rows],ddof=1))) for res in (256,512)} for arm,rows in records.items()}
    interaction=float(np.mean((means['attention_p0.01']-means['attention_p0'])-(means['gate_p0.01']-means['gate_p0'])))
    r=dict(lock=lockhash,plan=digest(p),consumed=True,confirmation_ids=p['confirmation_ids'],records=records,summary=summary,comparisons=comparisons,
           interaction_difference_descriptive=interaction,fno_context=fno,bootstrap='10k trajectories, Bonferroni4; conditional on3 fixed fits, no seed uncertainty bootstrap',
           limitations='one PDE100-shot;all evaluation times supervised;output query transfer with fixed input256;no noise/time extrapolation/full sensor transfer/universal superiority claim')
    atomic_json(resultpath,r);print('CONFIRMATION COMPLETE',summary,comparisons,flush=True);return r


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','confirm']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='seal':seal(p)
    else:confirm(p,torch.device('cuda'))
