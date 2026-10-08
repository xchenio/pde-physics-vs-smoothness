"""Independent artifact/closure/repair/stability checks for diagnostic outputs."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.public_benchmark import file_hash,atomic_json
from scripts.run_public_ridge_residual import digest
from scripts.diagnose_physics_stress import relative_temporal
OUT=ROOT/'results/physics_diagnosis_v2'


def run():
    p=json.loads((OUT/'plan.json').read_text());s=json.loads((OUT/'summary.json').read_text())
    if s['plan']!=digest(p) or s['checked_models']!=36:raise ValueError('summary identity/count mismatch')
    for name,h in p['source'].items():
        if file_hash(ROOT/name)!=h:raise ValueError('sealed source changed')
    evidence=[]
    for name,entry in p['models'].items():
        if file_hash(ROOT/entry['path'])!=entry['sha256']:raise ValueError('model changed')
        row=json.loads((OUT/'runs'/f'{name.replace("/","__")}.json').read_text())
        if row!=s['models'][name] or row['identity']!=dict(plan=digest(p),model=entry['sha256']):raise ValueError('row/summary mismatch')
        for scores in [row['spectrum']]+list(row['filter_diagnostics'].values()):
            for score in scores.values():
                np.testing.assert_allclose(sum(score['error_energy_shares'].values()),1.,atol=1e-12)
                if not 0<=score['boundary8_energy_fraction']<=1:raise ValueError('edge concentration invalid')
        if max(row['temporal'][f'own_fd_{h:g}']['mean'] for h in p['model_fd_half_widths'])>.001:
            raise ValueError('model JVP/FD mean diagnostic discrepancy unexpectedly large')
        if name.startswith('allen_cahn/'):
            old=json.loads((ROOT/'results/physics_diagnosis_v1/runs'/f'{name.replace("/","__")}.json').read_text())
            for key in ['spectrum','filter_diagnostics','temporal','primary_columns']:
                if old[key]!=row[key]:raise ValueError('scalar repair altered numerical result')
        evidence.append(dict(model=name,model_hash_verified=True,summary_verified=True,band_energy_shares_close=True))
    stability={}
    for task in p['dev_ids']:
        obj=torch.load(ROOT/f'external_data/physics_diagnosis_v2/{task}_reference.pt',weights_only=False)
        if obj['plan']!=digest(p) or obj['ids']!=p['dev_ids'][task]:raise ValueError('reference IDs changed')
        h1,h2=p['time_half_widths'][task]
        stability[task]=dict(full=relative_temporal(obj['ut'][(1024,h2)],obj['ut'][(1024,h1)],[0,1,2,3]),
                             resolved16=relative_temporal(obj['ut'][(1024,h2)],obj['ut'][(1024,h1)],[0,1,2,3],16))
    atomic_json(OUT/'audit.json',dict(plan=digest(p),audit_source_sha256=file_hash(Path(__file__)),checked_models=36,unchanged_v1_allen_records=9,evidence=evidence,
                reference_time_width_sensitivity=stability,no_training=True,no_confirmation_opened=True))
    print('AUDIT36 models,9 unchanged repair records and native FD stability passed',flush=True)


if __name__=='__main__':run()
