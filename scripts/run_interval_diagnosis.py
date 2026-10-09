"""Frozen midpoint/interval diagnosis with chunked, hashed numerical references."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.interval_reference_v1 import build_reference
from scripts.interval_diagnostics_core import indices,interval_metrics,aggregate
from scripts.temporal_after_periodic_core import score_coefficients,rhs_coefficients
from scripts.field_derivative_diagnostics import coefficients
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from scripts.run_physics_stress import clean_data
from scripts.periodic_curvature_core import make_model
from scripts.time_fno_baseline import predict
from src.time_fno import time_fields
from src.public_benchmark import file_hash,atomic_json
from scripts.run_public_ridge_residual import digest,locked_json,git_commit
OUT=ROOT/'results/interval_diagnosis_v1'


def plan():
    parent=ROOT/'results/temporal_after_periodic_v2';previous=json.loads((parent/'plan.json').read_text())
    audit=json.loads((parent/'audit_v3.json').read_text())
    if not audit['complete'] or audit['checked_models']!=36:raise ValueError('complete parent required')
    stress=json.loads((ROOT/'results/physics_stress_v1/plan.json').read_text());data=json.loads((ROOT/'results/physics_token_v1/plan.json').read_text())
    tasks={}
    for task,centers,hs in [('allen_cahn',[.025,.0875,.1875,.375,.475],[.005,.01]),('burgers',[.05,.18,.38,.75,.95],[.02,.04])]:
        windows=[]
        for center in centers:
            for h in hs:
                count=17 if task=='allen_cahn' else int(round(2*h/.005))+1
                rtcount=17 if task=='allen_cahn' else int(round(2*h/.01))+1
                windows.append(dict(center=center,half_width=h,secondary_late=center==centers[-1],
                    model_times=np.round(np.linspace(center-h,center+h,count),12).tolist(),
                    reference_times=np.round(np.linspace(center-h,center+h,rtcount),12).tolist()))
        tasks[task]=dict(centers=centers,half_widths=hs,windows=windows,label_times=stress['tasks'][task].get('times',[0.,.1,.25,.5,1.]))
    files=['scripts/run_interval_diagnosis.py','scripts/interval_reference_v1.py','scripts/interval_diagnostics_core.py','scripts/temporal_after_periodic_core.py',
           'scripts/field_derivative_diagnostics.py','scripts/allen_cahn_trajectory_v1.py','scripts/allen_cahn_data.py',
           'scripts/periodic_curvature_core.py','scripts/diagnose_physics_stress_v2.py','scripts/run_physics_stress.py','scripts/time_fno_baseline.py',
           'src/time_fno.py','src/fno.py','src/periodic_position_fno.py']
    p=dict(parent_plan_sha256=file_hash(parent/'plan.json'),parent_audit_sha256=file_hash(parent/'audit_v3.json'),
        models=previous['models'],dev_ids=previous['dev_ids'],task_spec=stress['tasks'],tasks=tasks,
        dataset=dict(path=data['dataset_source'],sha256=data['dataset_sha256']),resolved_max_mode=16,
        source={name:file_hash(ROOT/name) for name in files},
        policy='36 frozen models,32 EXISTING dev IDs per task; NEW midpoint/interval development snapshots diagnostic-only, never training labels/stats/calibration. No confirmation/new IDs/training/modules/loss.',
        references_policy='Allen1024 dt.0000625 vs.000125 and512 finegrid, resumable chunks8. Burgers original native1024 dt.01 frames, approximate midpoints.18/.38 are+.005 offsets, no invented finer truth.',
        quadrature='Allen Simpson17 vs9; Burgers native dt.01 vs.02, model dt.005 vs.01. Same native increment normalization, all signed cross/oracle terms. No FD-derived fullband truth.',
        decisions='Primary4centers x2windows x3seeds ALL must have incrementerror>=.1, >=10x max nativeclosure/quadrature/solver/spacediscrepancy, modelquadrature<=.1*incrementerror, floors0 for descriptive pilot eligibility. Secondary latepoint reported separately, no center selection.',
        repair='Fixed full low-mode endpoint integral repair once (gamma1), no true future values used; report uncorrected+corrected field for EVERY arm/window. Diagnostic counterfactual only, no model export/tuning/independent gain.',
        innovation='Future hypothesis only: TRAIN-internal numerical-uncertainty continuous interval/frequency weighting versus uniform integral and curvature controls. Weak/integral and midpoint ideas have prior art; no novelty/benefit claimed.')
    locked_json(OUT/'plan.json',p);return p


def frozen_predictions(p,name,a,times,device):
    entry=p['models'][name];path=ROOT/entry['path']
    if file_hash(path)!=entry['sha256']:raise ValueError('model changed')
    if file_hash(path.parent.parent/'result.json')!=entry['result_sha256']:raise ValueError('model record changed')
    cp=torch.load(path,map_location='cpu',weights_only=False);c=cp['identity']['config']
    model=make_model(c,c['width'],c['modes'],c['layers']).to(device);model.load_state_dict(cp['model']);model.eval()
    raw=np.asarray(a['val_x'][:32],dtype=float);norm=cp['normalization']
    x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
    coords=torch.tensor(a['grid256'],dtype=torch.float32,device=device);ts=torch.tensor(times,dtype=torch.float32,device=device)
    with torch.no_grad():pred=torch.cat([predict(model,q,ts,coords) for q in x.split(1)]).cpu().numpy().astype(float)
    return model,c,cp,x,coords,pred*cp['scale']+raw.mean(1)[:,None,None]


def run(p):
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed source changed')
    cache={};rows={};device=torch.device('cuda');stress=json.loads((ROOT/'results/physics_stress_v1/plan.json').read_text())
    for name,entry in p['models'].items():
        task=name.split('/')[0]
        if task not in cache:
            a,_=clean_data(stress,task);ref,h=build_reference(p,task);cache[task]=(a,ref,h)
            if ref['ids']!=p['dev_ids'][task] or stress['tasks'][task]['val_ids'][:32]!=ref['ids']:raise ValueError('case ordering changed')
            np.testing.assert_allclose(ref['fields'][256][:,indices(ref['times'],a['times'])],a['val_y'][:32],rtol=1e-5,atol=1e-6)
        a,ref,rhash=cache[task];path=OUT/'runs'/f'{name.replace("/","__")}.json';identity=dict(plan=digest(p),model=entry['sha256'],reference=rhash)
        if path.exists():
            r=json.loads(path.read_text())
            if r['identity']!=identity:raise ValueError('result identity changed')
            rows[name]=r;continue
        taskplan=p['tasks'][task];times=sorted({t for w in taskplan['windows'] for t in w['model_times']})
        model,c,cp,x,coords,pred=frozen_predictions(p,name,a,times,device)
        centers=taskplan['centers'];ix=indices(times,centers);ri=indices(ref['times'],centers)
        truth=ref['fields'][1024][:,ri];pc=pred[:,ix]
        tt=torch.tensor(centers,dtype=torch.float32,device=device).repeat(32);b=x.repeat_interleave(len(centers),0)
        off=torch.tensor(np.asarray(a['val_x'][:32],dtype=float).mean(1),dtype=torch.float32,device=device).repeat_interleave(len(centers))
        _,ut=time_fields(model,b,tt,coords,cp['scale'],off,create_graph=False)
        ut=ut.detach().cpu().numpy().astype(float).reshape(32,len(centers),256)
        if task=='allen_cahn':truecf=rhs_coefficients(truth,c,16)
        else:
            plus=ref['fields'][1024][:,indices(ref['times'],np.array(centers)+.01)]
            minus=ref['fields'][1024][:,indices(ref['times'],np.array(centers)-.01)]
            truecf=coefficients((plus-minus)/.02)[...,:17]
        w=np.r_[1.,np.full(16,2.)];ucf=coefficients(ut)[...,:17]
        intervals=[]
        for window in taskplan['windows']:
            result=interval_metrics(pred,times,ref,c,window);result['secondary_late']=window['secondary_late'];intervals.append(result)
        r=dict(identity=identity,midpoint_field=safe_band_errors(pc,truth[...,::4],a['train_x'],c['length'],[0,1,2,3]),
            late_field=safe_band_errors(pc,truth[...,::4],a['train_x'],c['length'],[4]),
            midpoint_time_resolved=score_coefficients(ucf[:,:4],truecf[:,:4],w),late_time_resolved=score_coefficients(ucf[:,4:],truecf[:,4:],w),
            intervals=intervals,git_commit=git_commit())
        atomic_json(path,r);rows[name]=r
        atomic_json(OUT/'progress.json',dict(completed=len(rows),total=36,confirmation_opened=False,training_performed=False))
        print('INTERVAL',name,'increment',np.mean([r['increment']['mean'] for r in intervals[:8]]),flush=True)
    groups,decision=aggregate(rows)
    atomic_json(OUT/'summary.json',dict(plan=digest(p),checked_models=len(rows),groups=groups,decision=decision,confirmation_opened=False,training_performed=False))
    print('INTERVAL36 COMPLETE',json.dumps(groups),json.dumps(decision),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    p=plan()
    if args.stage=='run':run(p)
