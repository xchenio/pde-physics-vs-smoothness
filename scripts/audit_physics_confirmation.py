"""Independent prediction/metric replay of consumed, sealed500 confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_physics_token import OUT,DATA,plan
from scripts.token_operator import TokenDeepONet
from scripts.fno_small_trajectory import make_model
from scripts.physics_token_core import grid,trajectory_errors
from scripts.confirm_physics_token import contrast
from scripts.audit_manifold_reconstruction import close
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json,file_hash


def audit(device):
    p=plan();c=json.loads((OUT/'confirmation.json').read_text());lockpath=OUT/'selection_lock.json';lock=json.loads(lockpath.read_text())
    marker=json.loads((OUT/'confirmation_opened.json').read_text());history=json.loads((OUT/'historical_id_audit.json').read_text())
    if c['lock']!=file_hash(lockpath) or c['plan']!=digest(p) or not c['consumed'] or not marker['consumed'] or marker['lock']!=c['lock'] or marker['ids']!=p['confirmation_ids']:raise ValueError('confirmation marker/identity mismatch')
    if c['confirmation_ids']!=p['confirmation_ids'] or len(set(p['confirmation_ids']))!=500 or set(p['confirmation_ids'])&set(history['used_ids']):raise ValueError('IDs overlap/mismatch')
    cache=torch.load(DATA/'confirmation.pt',weights_only=False)
    if cache['lock']!=c['lock'] or cache['ids']!=p['confirmation_ids']:raise ValueError('cache mismatch')
    a=cache['arrays'];np.testing.assert_array_equal(a['y256'],a['y512'][:,:,::2]);np.testing.assert_array_equal(a['grid256'],a['grid512'][::2])
    raw=np.asarray(a['x'],dtype=float);checked=0;vectors={}
    for arm,entries in lock['models'].items():
        kind='attention' if arm.startswith('attention') else 'gate';vv=[]
        for entry,row in zip(entries,c['records'][arm]):
            path=ROOT/entry['path']
            if file_hash(path)!=entry['sha256']:raise ValueError('sealed model mismatch')
            cp=torch.load(path,map_location='cpu',weights_only=False);model=TokenDeepONet(kind,64,1.,1.).to(device);model.load_state_dict(cp['model']);model.eval()
            norm=cp['normalization'];x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device)
            with torch.no_grad():
                for n in (256,512):
                    coords=grid(a[f'grid{n}'],a['times'],device)
                    pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1,keepdims=True)
                    pred=pred.reshape(500,5,n);e=trajectory_errors(pred,a[f'y{n}']);ref=row['scores'][str(n)]
                    close(float(e[:,1:].mean()),ref['primary']);close(e[:,1:].mean(1),ref['per_sample']);close(e.mean(0),ref['by_time'])
                    if n==256:vv.append(e[:,1:].mean(1))
                    else:
                        extra=trajectory_errors(pred[:,:,1::2],a['y512'][:,:,1::2]);close(extra[:,1:].mean(1),row['scores']['512_new_queries']['per_sample'])
            for n in ('256','512'):
                prof=row['profiles'][n]
                if prof['batch']!=64 or prof['query_points']!=5*int(n) or prof['mean_ms']<=0 or prof['repeats']!=50:raise ValueError('profile contract mismatch')
            checked+=1
        vectors[arm]=np.mean(vv,axis=0)
        for n in ('256','512'):
            vals=[r['scores'][n]['primary'] for r in c['records'][arm]];close(dict(mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))),c['summary'][arm][n])
    for entry in lock['small_fno_models']:
        path=ROOT/entry['path']
        if file_hash(path)!=entry['sha256']:raise ValueError('FNO100 sealed model mismatch')
        row=next(r for r in c['small_fno_records'][str(entry['width'])] if r['seed']==entry['seed'])
        cp=torch.load(path,map_location='cpu',weights_only=False);model=make_model(entry['width'],entry['seed'],device);model.load_state_dict(cp['model']);model.eval();norm=cp['normalization']
        x=torch.tensor((raw-norm['mean'])/norm['std'],dtype=torch.float32,device=device);coords=torch.tensor(a['grid256'],dtype=torch.float32,device=device)
        with torch.no_grad():pred=torch.cat([model(b,coords) for b in x.split(64)]).cpu().numpy().astype(float)*cp['scale']+raw.mean(1)[:,None,None]
        e=trajectory_errors(pred,a['y256']);close(e[:,1:].mean(1),row['per_sample']);close(float(e[:,1:].mean()),row['primary']);checked+=1
    for width,rows in c['small_fno_records'].items():
        vals=[r['primary'] for r in rows];close(dict(mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1))),c['small_fno_summary'][width])
    if checked!=18:raise ValueError('incomplete confirmation replay')
    for name,left,right in (('attention_without_physics','attention_p0','gate_p0'),('physics_without_attention','gate_p0.01','gate_p0'),
                            ('physics_with_attention','attention_p0.01','attention_p0'),('combination_vs_gate','attention_p0.01','gate_p0')):
        r=contrast(vectors[left],vectors[right]);ref=c['comparisons'][name]
        for key in ('difference','relative_gain','interval'):close(r[key],ref[key])
        if r['success']!=ref['success']:raise ValueError('success decision mismatch')
    result=dict(lock=c['lock'],confirmation_sha256=file_hash(OUT/'confirmation.json'),checked_models=checked,checked_contrasts=4,
                ids=500,historical_overlap=0,consumed=True,status='replayed all18 same100-shot models,metrics,odd512 query errors,summary and adjusted contrast decisions')
    atomic_json(OUT/'confirmation_audit.json',result);print(result,flush=True);return result


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    audit(torch.device('cuda'))
