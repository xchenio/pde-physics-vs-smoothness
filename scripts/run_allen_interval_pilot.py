"""Registered small Allen-only exploratory joint-interval continuation pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_physics_stress import clean_data, observations
from scripts.allen_interval_pilot_core import train_one
from scripts.allen_interval_reference import build_reference
from scripts.run_public_ridge_residual import digest, locked_json
from src.public_benchmark import atomic_json, file_hash

OUT = ROOT / 'results/allen_interval_pilot_v1'
REFERENCE = ROOT / 'external_data/allen_interval_pilot_v1/train_reference'


def plan():
    parent_names = ['results/periodic_curvature_v1/plan.json', 'results/periodic_curvature_v1/audit.json',
                    'results/interval_diagnosis_v1/plan.json', 'results/interval_diagnosis_v1/audit.json',
                    'results/physics_stress_v1/plan.json']
    stress = json.loads((ROOT / parent_names[-1]).read_text())
    parents = {name: file_hash(ROOT / name) for name in parent_names}
    if not json.loads((ROOT / parent_names[1]).read_text())['complete']:
        raise ValueError('complete periodic parent required')
    if not json.loads((ROOT / parent_names[3]).read_text())['complete']:
        raise ValueError('complete frozen interval diagnosis required')
    names = ['scripts/run_allen_interval_pilot.py', 'scripts/allen_interval_pilot_core.py',
             'scripts/allen_interval_reference.py', 'scripts/audit_allen_interval_pilot.py',
             'scripts/physics_stress_core.py', 'scripts/periodic_curvature_core.py',
             'scripts/run_physics_stress.py', 'scripts/allen_cahn_data.py', 'scripts/allen_cahn_trajectory_v1.py',
             'scripts/time_fno_baseline.py', 'scripts/fno_regularization_core.py', 'scripts/objective_modal_core.py',
             'scripts/physics_token_core.py', 'scripts/interval_diagnostics_core.py',
             'scripts/temporal_after_periodic_core.py', 'scripts/field_derivative_diagnostics.py',
             'scripts/audit_interval_diagnosis.py', 'scripts/audit_temporal_after_periodic_v3.py',
             'scripts/run_public_pod_readout.py', 'scripts/run_public_ridge_residual.py',
             'scripts/diagnose_physics_stress_v2.py', 'src/physics_equations_v1.py', 'src/periodic_position_fno.py',
             'src/time_fno.py', 'src/fno.py', 'src/resume.py', 'src/utils.py', 'src/public_benchmark.py',
             'tests/test_allen_interval_pilot.py', 'tests/test_allen_interval_reference.py',
             'docs/ALLEN_INTERVAL_PILOT_V1.md']
    lineage = {}
    for seed in [0, 1, 2]:
        root = ROOT / f'results/periodic_curvature_v1/allen_cahn/periodic/warm_s{seed}/checkpoints'
        latest, final = root / 'latest.pt', root / 'final.pt'
        a = torch.load(latest, map_location='cpu', weights_only=False)
        b = torch.load(final, map_location='cpu', weights_only=False)
        receipt = json.loads((root.parent / 'result.json').read_text())
        if a['identity'] != receipt['identity'] or b['identity'] != receipt['identity'] or file_hash(final) != receipt['final_sha256']:
            raise ValueError('warm parent receipt changed')
        if a['step'] != 1500 or b['step'] != 1500 or not all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()):
            raise ValueError('warm final/latest mismatch')
        if any(k not in a for k in ['optimizer', 'rng', 'collocation_rng', 'batch_hash']) or a['batch_hash'] != b['batch_hash']:
            raise ValueError('warm full continuation state missing')
        if not a['optimizer']['state'] or any(int(v['step']) != 1500 for v in a['optimizer']['state'].values()):
            raise ValueError('warm optimizer step mismatch')
        lineage[str(seed)] = dict(latest_path=latest.relative_to(ROOT).as_posix(), latest_sha256=file_hash(latest),
                                  final_path=final.relative_to(ROOT).as_posix(), final_sha256=file_hash(final))
    params = stress['tasks']['allen_cahn']
    p = dict(version='allen_interval_pilot_v1', exploratory=True, parents=parents,
             source={name: file_hash(ROOT / name) for name in names}, warm_lineage=lineage,
             seeds=[0, 1, 2], kinds=['none', 'low', 'full', 'interval_uniform', 'interval_reliable'],
             warm_steps=1500, continuation_steps=750, strength=.3,
             width=16, modes=16, layers=4, weight_decay=.0001, batch_size=16,
             eval_every=250, checkpoint_every=50, resolved_max_mode=16,
             task_spec={k: params[k] for k in ['pde', 'length', 'diffusion', 'reaction', 'time_lower', 'time_upper']},
             train_ids=params['train_ids'], val_ids=params['val_ids'], weight_train_ids=params['train_ids'][:16],
             centers=[.025, .0875, .1875, .375, .475], interval_half_widths=[.005, .01], interval_nodes=5,
             architecture='Same periodic single-coordinate FNO34209 real parameters. No new modules, inputs or target labels.',
             observations='Original dense-clean100 training trajectories x5 times; reused200 development. First16 TRAIN IDs only for additional numerical reliability solves, never fitted midpoint labels. No reserve generation/access.',
             joint_constraint='Both predicted endpoints and predicted dealiased Allen RHS trained jointly. Simpson5 interval-average RHS; residual RATE multiplied by T/observed_training_target_std. Low modes0..16, correct DC/conjugate Parseval weights.',
             windows='Keep original uniform query RNG [0.05,0.5]. Derive fraction=(q-.05)/.45, alternate h=.005/.01 by batch slot, center=h+(T-2h)*fraction. No additional RNG and no deleting early/late windows. Linear weight interpolation on fixed reference centers; clamp edges.',
             reliability='Percenter/h/mode max of TRAIN16 complex RMS numerical RATE discrepancies: timestep, space, quadrature, native closure and actual256-grid RHS consistency. w=tau^2/(tau^2+epsilon^2), tau=.001*training_target_std/T. Fixed absolute tolerance policy, not calibrated uncertainty probability. No rank/median amplification or hard deletion.',
             normalization='Uniform/reliable differ ONLY by fixed w. Gradient calibration at same warm TRAIN probe to initial ratio.3, frozen thereafter. This cancels a globally constant weight: only relative weight shape can differ. Archive float64 and effective float32 contrast, including any all-ones degeneracy.',
             budget='Reuse full-state1500 warm latest.pt for all arms; same750 continuation steps, AdamW restart schedule .0003 to .00001, B16. Final2250 only, no early stopping/checkpoint/seed selection or hyperparameter search. Same penalty forwards, different backward FLOPs. Include warm reuse and reference cost separately.',
             primary='Report mean3seeds development relative field L2 over4 original noninitial times, all perseed paired changes. Field/gradient/curvature must be considered together; no significance claim.',
             secondary='Reuse already-consumed frozen32dev midpoint/interval references, all5centers and2widths; primary4 and secondarylate separately. Report increment and joint closure; lowering residual alone is not prediction improvement. Timing/peak allocated VRAM/inference cost.',
             historical_control='New none/low/full periodic Allen finals must reproduce all9 old tensor states exactly; new15 streams/init/calibration must match within seed. Full-state interrupt/resume CPU/CUDA tests before launch.',
             interpretation='Prior strict frozen eligibility screen failed; this newly authorized study remains exploratory. Weak/integral consistency has prior art (VPINN1912.00873 and later temporal/integral methods). Numerical-reliability weighting novelty and benefit unverified. Tiny Allen uncertainty may make weights indistinguishable from uniform: keep/report that null, never force contrast.',
             confirmation='No new or reopened confirmation. Old32dev intervals and200dev fields remain development, not unseen. No Burgers training. No winner export or deployment.')
    locked_json(OUT / 'plan.json', p)
    return p


def run(p):
    for name, expected in {**p['source'], **p['parents']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent changed: ' + name)
    old = json.loads((ROOT / 'results/physics_stress_v1/plan.json').read_text())
    clean, data = clean_data(old, 'allen_cahn')
    a, obs = observations(clean, 'dense_clean')
    start = time.perf_counter()
    reference = build_reference(REFERENCE, p['weight_train_ids'], float(np.asarray(a['train_y'], dtype=float).std()), dict(plan=digest(p)))
    map_json = json.loads((REFERENCE / 'weight_map.json').read_text())
    locked_json(OUT / 'weight_map.json', map_json)
    reference_identity = dict(map_sha256=file_hash(REFERENCE / 'weight_map.pt'), map_json_sha256=file_hash(OUT / 'weight_map.json'))
    locked_json(OUT / 'reference_manifest.json', dict(artifacts=reference_identity, reference_path=REFERENCE.relative_to(ROOT).as_posix(),
                                                    train_ids=p['weight_train_ids'], context=dict(plan=digest(p))))
    print('TRAIN-ONLY REFERENCE READY', reference.get('diagnostics', {}).get('weight32_statistics', {}),
          'lookup_or_build_seconds', time.perf_counter() - start, flush=True)
    context = dict(plan=digest(p), task='allen_cahn', data=data, observations=obs, training_reference=reference_identity)
    base = {k: p[k] for k in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every', 'checkpoint_every', 'resolved_max_mode', 'interval_half_widths', 'interval_nodes']}
    base.update(p['task_spec'])
    base.update(position='periodic', position_center=float(np.mean(a['grid256'])), position_scale=float(np.std(a['grid256']) * np.sqrt(2)),
                evaluation_columns=[1, 2, 3, 4], reliability_map=dict(centers=reference['centers'], half_widths=reference['half_widths'], weights32=np.asarray(reference['weights32']).tolist()))
    groups, pair = {}, {}
    completed = 0
    for kind in p['kinds']:
        rows = []
        for seed in p['seeds']:
            c = dict(base, seed=seed, kind=kind, strength=0. if kind == 'none' else p['strength'], steps=p['continuation_steps'], lr=.0003, lr_min=.00001)
            warm = ROOT / p['warm_lineage'][str(seed)]['latest_path']
            if file_hash(warm) != p['warm_lineage'][str(seed)]['latest_sha256']:
                raise ValueError('warm changed')
            r = train_one(c, a, context, OUT / f'{kind}_s{seed}', torch.device('cuda'), warm_path=warm)
            value = (r['initial'], r['batch_hash'], r['collocation_hash'], r['calibration'], r['parameters_real'])
            if seed in pair and pair[seed] != value:
                raise ValueError('paired fork mismatch')
            pair[seed] = value
            rows.append(r)
            completed += 1
            atomic_json(OUT / 'progress.json', dict(completed_runs=completed, total_runs=15, current=f'{kind}_s{seed}', groups=groups, confirmation_opened=False, exploratory=True))
        groups[kind] = dict(field_mean=float(np.mean([r['metrics']['val']['primary'] for r in rows])),
                            seed_sd=float(np.std([r['metrics']['val']['primary'] for r in rows], ddof=1)),
                            per_seed=[r['metrics']['val']['primary'] for r in rows], training_seconds=[r['seconds'] for r in rows])
    atomic_json(OUT / 'summary.json', dict(plan=digest(p), completed_runs=completed, total_runs=15, groups=groups, confirmation_opened=False, exploratory=True))
    atomic_json(OUT / 'progress.json', dict(completed_runs=completed, total_runs=15, status='completed', groups=groups, confirmation_opened=False, exploratory=True))
    print('EXPLORATORY ALLEN15 COMPLETE', json.dumps(groups), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'run'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    p = plan()
    if args.stage == 'run':
        run(p)
