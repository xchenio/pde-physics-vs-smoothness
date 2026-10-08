"""Training-only prerequisite for FNO physics versus smoothness experiments.

Never opens development or confirmation trajectory labels. Extra time snapshots
are diagnostic-only and are explicitly recorded, not additional fitting labels.
"""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.time_fno import dealiased_flux, conservative_residual, coefficient_energy
from src.public_benchmark import atomic_json, file_hash
from scripts.run_public_ridge_residual import digest, locked_json

OUT = ROOT/'results/fno_physics_resolution_v1'


def plan():
    parent_path = ROOT/'results/physics_token_v1/plan.json'
    parent = json.loads(parent_path.read_text())
    p = dict(parent_sha256=file_hash(parent_path), dataset_source=parent['dataset_source'],
             dataset_sha256=parent['dataset_sha256'], train_ids=parent['train_ids'],
             centers=[10, 25, 50, 100], half_widths=[1, 2], grids=[1024, 512, 256],
             resolved_max_mode=16, diffusion=parent['diffusion'],
             snapshot_indices=sorted({c+d for c in [10,25,50,100] for d in [-2,-1,0,1,2]}),
             source={str(f.relative_to(ROOT).as_posix()):file_hash(f) for f in
                     [Path(__file__), ROOT/'src/time_fno.py', ROOT/'src/fno.py']},
             policy='Only the previous100 training trajectory IDs; adjacent time labels diagnostic-only. No dev or confirmation opened. No loss weight or band selection from outcomes.',
             formula='Central time differences dt0.01/0.02; periodic Fourier conservative residual, dealiased u^2/2; Nu/pi. Resolved modes0..16 fixed before execution.',
             diagnostic='RMS residual divided by root sum of squared RMS time, conservative flux and diffusion terms; full band excludes ambiguous even-grid Nyquist mode.',
             gate='Proceed to resolved-physics comparison only if median resolved relative residual on raw256 dt0.01 <=0.1 at every registered time; this is a feasibility gate, not proof of consistency.')
    locked_json(OUT/'plan.json', p)
    return p


def statistics(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v,.9)), max=float(v.max()))


def terms(u, ut, diffusion, band):
    k = torch.arange(band+1, dtype=u.dtype)*(2*torch.pi)
    a = torch.fft.rfft(ut, norm='forward')[..., :band+1]
    b = 1j*k*dealiased_flux(u)[..., :band+1]
    c = diffusion*k.square()*torch.fft.rfft(u,norm='forward')[..., :band+1]
    r = conservative_residual(u,ut,diffusion,band)
    rms = coefficient_energy(r).sqrt()
    denominator = (coefficient_energy(a)+coefficient_energy(b)+coefficient_energy(c)).sqrt().clamp_min(1e-12)
    return dict(relative_residual=statistics((rms/denominator).numpy()), residual_rms=statistics(rms.numpy()),
                mass_derivative_rms=statistics(r[...,0].abs().numpy()),
                term_rms={name:statistics(coefficient_energy(v).sqrt().numpy()) for name,v in [('time',a),('flux',b),('diffusion',c)]})


def run(p):
    # Hashing bytes does not expose trajectory labels to fitting/selection.
    if file_hash(Path(p['dataset_source'])) != p['dataset_sha256']:
        raise ValueError('source checksum mismatch')
    snapshots=p['snapshot_indices']; records=[]
    with h5py.File(p['dataset_source'],'r') as f:
        if f['tensor'].shape != (10000,201,1024):raise ValueError('layout mismatch')
        values=np.stack([f['tensor'][int(i),snapshots,:] for i in p['train_ids']])
        times=np.asarray(f['t-coordinate'][snapshots],dtype=float)
        x=np.asarray(f['x-coordinate'][:],dtype=float)
    np.testing.assert_allclose(x,(np.arange(1024)+.5)/1024,atol=1e-8)
    for grid in p['grids']:
        v=torch.tensor(values[...,::1024//grid],dtype=torch.float64)
        for center in p['centers']:
            for half in p['half_widths']:
                left,right=snapshots.index(center-half),snapshots.index(center+half)
                u=v[:,snapshots.index(center)]
                ut=(v[:,right]-v[:,left])/(times[right]-times[left])
                record=dict(grid=grid,time_index=center,time=float(times[snapshots.index(center)]),half_width=half,
                            resolved=terms(u,ut,p['diffusion'],p['resolved_max_mode']),
                            full=terms(u,ut,p['diffusion'],grid//2-1))
                records.append(record)
    gate=all(r['resolved']['relative_residual']['median']<=.1 for r in records if r['grid']==256 and r['half_width']==1)
    result=dict(plan_sha256=file_hash(OUT/'plan.json'),data_values_sha256=digest(values.tolist()),
                trajectories=len(p['train_ids']),snapshot_count=len(snapshots),records=records,
                feasibility_gate_passed=gate,extra_training_snapshots_diagnostic_only=True,
                development_opened=False,confirmation_opened=False)
    atomic_json(OUT/'audit.json',result)
    print(json.dumps(dict(feasibility_gate_passed=gate,raw256_dt01=[r for r in records if r['grid']==256 and r['half_width']==1]),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','run']);args=parser.parse_args()
    torch.set_num_threads(4)
    p=plan()
    if args.stage=='run':run(p)
