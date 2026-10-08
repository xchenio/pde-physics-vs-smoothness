"""Versioned development-only frequency, temporal and discretization diagnosis."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_stress import clean_data
from scripts.allen_cahn_data import initial
from scripts.allen_cahn_trajectory_v1 import trajectories
from scripts.time_fno_baseline import predict
from scripts.field_derivative_diagnostics import (band_errors,coefficients,derivative,lowpass,
    residual_summary,pde_terms,resolved_energy)
from src.time_fno import TimeFNO1d,time_fields
from src.public_benchmark import file_hash,atomic_json
from src.resume import atomic_save
from scripts.run_public_ridge_residual import digest,locked_json
OUT=ROOT/'results/physics_diagnosis_v1';DATA=ROOT/'external_data/physics_diagnosis_v1';PARENT=ROOT/'results/physics_stress_v1'


def plan():
    old=json.loads((PARENT/'plan.json').read_text());audit=json.loads((PARENT/'checkpoint_audit.json').read_text())
    if not audit['complete']:raise ValueError('completed parent audit required')
    models={}
    for f in sorted(PARENT.rglob('result.json')):
        r=json.loads(f.read_text())
        if r['identity']['config']['kind']=='warm':continue
        models[f.parent.relative_to(PARENT).as_posix()]=dict(path=(f.parent/'checkpoints/final.pt').relative_to(ROOT).as_posix(),sha256=r['final_sha256'])
    p=dict(parent_plan_sha256=file_hash(PARENT/'plan.json'),parent_audit_sha256=file_hash(PARENT/'checkpoint_audit.json'),models=models,
           dev_ids={task:v['val_ids'][:32] for task,v in old['tasks'].items()},space_grids=[1024,512,256],
           time_half_widths=dict(burgers=[.01,.02],allen_cahn=[.0005,.001]),bands=[[0,4],[5,16],[17,64],[65,128]],
           counterfactual_cutoffs=[16,32,64],model_fd_half_widths=[.001,.0005],
           source={f.relative_to(ROOT).as_posix():file_hash(f) for f in [Path(__file__),ROOT/'scripts/field_derivative_diagnostics.py',ROOT/'src/time_fno.py',ROOT/'src/fno.py',ROOT/'scripts/allen_cahn_trajectory_v1.py']},
           policy='Existing development only: all200 for spectrum and32 fixed first IDs for time/discretization; extra adjacent time snapshots disclosed as NEW diagnostic development access, never training targets. No confirmation/new IDs/models/training.',
           analysis='Parseval-closed actual Fourier error bands; boundary8 of256 concentration; fixed-cutoff posthoc projection diagnostic only, no winner selection. JVP versus own FD and native truth central FD; all three frozen arms/three seeds/four conditions retained.',
           limits='Exploratory diagnosis, not independent verification or causal proof. Small FD probes just outside final supervised time are numerical differentiation checks, not extrapolation validation. No corrective output is exported as a selected model.')
    locked_json(OUT/'plan.json',p);return p


def reference(p,task,clean):
    path=DATA/f'{task}_reference.pt';old=json.loads((PARENT/'plan.json').read_text());c=old['tasks'][task];ids=p['dev_ids'][task]
    if path.exists():
        obj=torch.load(path,weights_only=False)
        if obj['plan']!=digest(p):raise ValueError('reference identity changed')
        return obj,file_hash(path)
    centers=np.array(clean['times'][1:]);widths=p['time_half_widths'][task]
    if task=='burgers':
        parent=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text())
        if file_hash(Path(parent['dataset_source']))!=parent['dataset_sha256']:raise ValueError('source changed')
        indices=sorted({int(round((t+d)/.01)) for t in centers for d in [-.02,-.01,0.,.01,.02]})
        with h5py.File(parent['dataset_source'],'r') as f:
            values=np.stack([f['tensor'][int(i),indices,:] for i in ids]).astype(float);times=np.asarray(f['t-coordinate'][indices],dtype=float)
        grid_values={n:values[...,::1024//n] for n in p['space_grids']}
    else:
        times=np.array(sorted({0.}|{float(t+d) for t in centers for d in [-.001,-.0005,0.,.0005,.001]}))
        grid_values={n:trajectories(initial(ids,n),times,dt=.00025) for n in p['space_grids']}
    centered={};uts={}
    for n,values in grid_values.items():
        centered[n]=values[:,[int(np.argmin(abs(times-t))) for t in centers]]
        for h in widths:
            uts[(n,h)]=np.stack([(values[:,int(np.argmin(abs(times-(t+h))))]-values[:,int(np.argmin(abs(times-(t-h))))])/
                                (times[int(np.argmin(abs(times-(t+h))))]-times[int(np.argmin(abs(times-(t-h))))]) for t in centers],axis=1)
    np.testing.assert_allclose(centered[256],clean['val_y'][:32,1:],rtol=1e-5,atol=1e-6)
    obj=dict(plan=digest(p),ids=ids,times=centers,fields=centered,ut=uts,extra_development_time_snapshots=times.tolist())
    atomic_save(obj,path);return obj,file_hash(path)


def relative_temporal(estimate,true,columns,band=None):
    if band is None:
        energy=lambda v:np.mean(v*v,axis=-1)
    else:
        energy=lambda v:resolved_energy(coefficients(v)[...,:band+1])
    denom=energy(true);floor=max(1e-24,.000001*float(denom.mean()))
    values=np.sqrt(energy(estimate-true)/np.maximum(denom,floor))
    return dict(mean=float(values[:,columns].mean()),by_time=values.mean(0).tolist(),floor_active_fraction=float((denom[:,columns]<floor).mean()))


def discretization(p,task,ref,c):
    rows=[]
    for n in p['space_grids']:
        for h in p['time_half_widths'][task]:
            u=ref['fields'][n];ut=ref['ut'][(n,h)]
            rows.append(dict(grid=n,half_width=h,resolved_by_time=[residual_summary(u[:,j],ut[:,j],c,16) for j in range(4)],
                             full_by_time=[residual_summary(u[:,j],ut[:,j],c,n//2-1) for j in range(4)]))
    fine=ref['fields'][1024];h=p['time_half_widths'][task][0];fine_ut=ref['ut'][(1024,h)]
    filtered=lowpass(fine);filtered_ut=lowpass(fine_ut)
    native_terms=pde_terms(fine,fine_ut,c,16);filtered_terms=pde_terms(filtered,filtered_ut,c,16)
    closure=filtered_terms[2]-native_terms[2]
    np.testing.assert_allclose(sum(filtered_terms)-sum(native_terms),closure,rtol=1e-9,atol=1e-10)
    result=dict(records=rows,filtered256_resolved=[residual_summary(filtered[:,j],filtered_ut[:,j],c,16) for j in range(4)],
                unresolved_nonlinear_term_rms=float(np.sqrt(resolved_energy(closure).mean())),
                native_resolved_residual_rms=float(np.sqrt(resolved_energy(sum(native_terms)).mean())),
                temporal_fd_width_disagreement=relative_temporal(ref['ut'][(1024,p['time_half_widths'][task][1])],fine_ut,list(range(4))))
    if task=='allen_cahn':
        rhs=c['diffusion']*derivative(fine,2,c['length'])+c['reaction']*fine*(1-fine*fine)
        result['native_fd_vs_semidiscrete_rhs']=relative_temporal(fine_ut,rhs,list(range(4)))
    return result


def run(p):
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('diagnostic source changed')
    old=json.loads((PARENT/'plan.json').read_text());cache={};refs={};discrete={};rows={};device=torch.device('cuda')
    for task in old['tasks']:
        cache[task],_=clean_data(old,task);refs[task],_=reference(p,task,cache[task]);discrete[task]=discretization(p,task,refs[task],old['tasks'][task])
    atomic_json(OUT/'discretization.json',discrete)
    for name,entry in p['models'].items():
        if file_hash(ROOT/entry['path'])!=entry['sha256']:raise ValueError('model changed')
        task=name.split('/')[0];clean=cache[task];ref=refs[task];path=OUT/'runs'/f'{name.replace("/","__")}.json'
        identity=dict(plan=digest(p),model=entry['sha256'])
        if path.exists():
            r=json.loads(path.read_text())
            if r['identity']!=identity:raise ValueError('result changed')
            rows[name]=r;continue
        cp=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False);c=cp['identity']['config'];cols=c['evaluation_columns'];tcols=[i-1 for i in cols]
        model=TimeFNO1d(c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
        raw=np.asarray(clean['val_x'],dtype=float);norm=cp['normalization'];scale=cp['scale']
        x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device);t=torch.tensor(clean['times'],dtype=torch.float32,device=device)
        coords=torch.tensor(clean['grid256'],dtype=torch.float32,device=device)
        with torch.no_grad():pred=torch.cat([predict(model,b,t,coords) for b in x.split(16)]).cpu().numpy().astype(float)*scale+raw.mean(1)[:,None,None]
        spectrum=band_errors(pred,clean['val_y'],clean['train_x'],c['length'],cols)
        filters={}
        for cutoff in p['counterfactual_cutoffs']:
            cf=coefficients(pred);cf[...,cutoff+1:]=0
            filtered=np.fft.irfft(cf,n=256,norm='forward',axis=-1)
            filters[str(cutoff)]=band_errors(filtered,clean['val_y'],clean['train_x'],c['length'],cols)
        # Time JVP at fixed initial field/grid; no differentiation in sensor coordinates.
        b=x[:32].repeat_interleave(4,0);tt=t[1:].repeat(32)
        off=torch.tensor(raw[:32].mean(1),dtype=torch.float32,device=device).repeat_interleave(4)
        _,ut=time_fields(model,b,tt,coords,scale,off,create_graph=False);ut=ut.detach().cpu().numpy().astype(float).reshape(32,4,256)
        temporal={};true_ut=ref['ut'][(1024,p['time_half_widths'][task][0])][...,::4]
        for band in [None,16]:temporal['truth_full' if band is None else 'truth_resolved']=relative_temporal(ut,true_ut,tcols,band)
        with torch.no_grad():
            for h in p['model_fd_half_widths']:
                fd=((model(b,tt+h,coords)-model(b,tt-h,coords))*scale/(2*h)).cpu().numpy().astype(float).reshape(32,4,256)
                temporal[f'own_fd_{h:g}']=relative_temporal(ut,fd,tcols)
        temporal['predicted_residual']=residual_summary(pred[:32,1:],ut,c,16)
        r=dict(identity=identity,spectrum=spectrum,filter_diagnostics=filters,temporal=temporal,primary_columns=cols)
        atomic_json(path,r);rows[name]=r;print('DIAGNOSED',name,'ut',temporal['truth_full']['mean'],'edgeCurv',spectrum['curvature']['boundary8_energy_fraction'],flush=True)
    atomic_json(OUT/'summary.json',dict(plan=digest(p),checked_models=len(rows),models=rows,discretization=discrete,confirmation_opened=False,training_performed=False))
    print('DIAGNOSIS COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
