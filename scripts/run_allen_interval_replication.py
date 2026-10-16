"""Fixed-uniform Allen replication: one reshuffled development split x supervision.

No new/reserved IDs, numerical reference solver, reliability map or parameter
search. Each condition trains its own fresh warm model on selected labels only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.allen_interval_replication_data import split_ids, load_cached_split, observations
from scripts.allen_interval_replication_core import train_one
from scripts.run_public_ridge_residual import digest, locked_json
from src.public_benchmark import atomic_json, file_hash

OUT = ROOT / 'results/allen_interval_replication_v1'


def plan():
    parent_names = ['results/allen_interval_pilot_v1/plan.json', 'results/allen_interval_pilot_v1/audit.json',
                    'results/allen_interval_pilot_v1/summary.json', 'results/physics_stress_v1/plan.json']
    previous = json.loads((ROOT / parent_names[0]).read_text())
    audit = json.loads((ROOT / parent_names[1]).read_text())
    if not audit['complete'] or audit['checked_runs'] != 15 or audit['historical_control_tensor_matches'] != 9:
        raise ValueError('complete pilot required')
    ids = split_ids(2026101601)
    files = ['scripts/run_allen_interval_replication.py', 'scripts/allen_interval_replication_data.py',
             'scripts/allen_interval_replication_core.py', 'scripts/audit_allen_interval_replication.py',
             'scripts/allen_interval_pilot_core.py', 'scripts/periodic_curvature_core.py', 'scripts/physics_stress_core.py',
             'scripts/time_fno_baseline.py', 'scripts/fno_regularization_core.py', 'scripts/objective_modal_core.py',
             'scripts/physics_token_core.py', 'scripts/diagnose_physics_stress_v2.py',
             'scripts/field_derivative_diagnostics.py', 'scripts/run_public_pod_readout.py',
             'scripts/run_public_ridge_residual.py', 'src/physics_equations_v1.py', 'src/periodic_position_fno.py',
             'src/time_fno.py', 'src/fno.py', 'src/resume.py', 'src/utils.py', 'src/public_benchmark.py',
             'tests/test_allen_interval_replication_data.py', 'tests/test_allen_interval_replication_core.py',
             'docs/ALLEN_INTERVAL_REPLICATION_V1.md']
    chunks = [f'external_data/physics_stress_v1/allen_{split}_{start:04d}.pt'
              for split, total in [('train', 100), ('val', 200)] for start in range(0, total, 50)]
    old_train = set(range(200000, 200100))
    p = dict(version='allen_interval_replication_v1', exploratory=True,
             parents={name: file_hash(ROOT / name) for name in parent_names},
             source={name: file_hash(ROOT / name) for name in files}, data_chunks={name: file_hash(ROOT / name) for name in chunks},
             split_seed=2026101601, parent_ids_pool=list(range(200000, 200300)), **ids,
             split_overlap=dict(new_train_from_old_train=len(set(ids['train_ids']) & old_train),
                                new_val_from_old_train=len(set(ids['val_ids']) & old_train)),
             conditions=['dense', 'sparse'], kinds=['none', 'full', 'interval_uniform'], seeds=[0, 1, 2],
             warm_steps=1500, continuation_steps=750, strength=.3,
             width=16, modes=16, layers=4, weight_decay=.0001, batch_size=16,
             eval_every=250, checkpoint_every=50, resolved_max_mode=16,
             task_spec=previous['task_spec'], interval_half_widths=[.005, .01], interval_nodes=5,
             observed_columns=dict(dense=[0, 1, 2, 3, 4], sparse=[0, 4]),
             evaluation_columns=dict(dense=[1, 2, 3, 4], sparse=[1, 2, 3]), common_columns=[1, 2, 3],
             architecture='Unchanged periodic single-coordinate FNO16wide16modes4layers34209 real parameters, zero last head at fresh initialization. No new modules or input channels.',
             data_policy='Deterministic permutation of PREVIOUSLY CONSUMED300 Allen IDs; first100 then sort TRAIN, remaining200 then sort DEV. No new IDs or reserves200300+. Existing cache contains all5 frames; sparse training engine receives ONLY selected2. Development is not unseen confirmation.',
             fresh_warm='All6 warm fits FROM SCRATCH, no old checkpoint/optimizer/normalizer. Each seed has common initial tensors and batches across dense/sparse, but fitted warm states differ. Each condition/seed forks its own full-state1500 warm into3x750 continuations.',
             observations='Dense sees[0,.05,.125,.25,.5]; sparse sees[0,.5]. Input normalization from new training initials; target scale, energy-protected data weights and gradient calibration from OBSERVED TRAIN labels only. Omitted TRAIN frames are loaded in old cache but never used for fitting/statistics/calibration/selection.',
             fixed_constraint='Reuse sealed uniform joint RATE residual*T/observed_train_std, Allen4x-dealiased RHS, modes0..16, Simpson5, h=.005/.01 alternating slots and same original query-to-center mapping. Fixed formula and gradient-ratio rule, not old data-dependent numeric normalizers. No reliability map, reference solves or tau adjustment.',
             optimization='Fresh warm1500 AdamW lr.001->.00001; forks750 lr.0003->.00001, B16 wd.0001; initial constraint/data gradient ratio.3 on same16 TRAIN probe then fixed. All continuation penalty forwards shared, chosen backward differs. Fixed final model, no early stopping/seed/hyperparameter selection.',
             primary='Dense:200dev mean relativeL2 at4 noninitial times. Sparse:200dev at3 OMITTED intermediate times. Report all3seed paired changes within each condition. These different primaries must NOT be compared to infer sparse-value interaction.',
             common_evaluation='Both conditions additionally report SAME3 intermediate times and SAME4 future times on SAME200dev. Descriptive sparse-minus-dense uniform-vs-none relative gain interaction uses COMMON3 times, with perseed differences; no significance, independent dataset or causal claim.',
             descriptive_screen='For each condition, label stronger directional replication ONLY if uniform-vs-none mean PRIMARY relative decrease>=1% AND all3 seed primary decreases>0. Fixed arbitrary descriptive threshold informed by pilot, not statistical significance or demonstrated application relevance. Report all results regardless of pass; no threshold/condition search.',
             secondary='Full-grid gradient/curvature on primary and common times, all5 field-time errors including IC and endpoint, strong-full-curvature tradeoffs, persisted training seconds/warm costs/final-segment memory and direct inference timings. No new midpoint truth/reference or model export.',
             audit='Replay24 completed records, all streams/initialization/warm/fork/calibration, observed-only normalizers, final predictions200dev; test dense/sparse CPU+CUDA exactresume and omitted-label poison invariance before formal run. No historical tensor equality expected because training partition changed.',
             limitations='One extra development partition,3 model seeds,one fixed transferred strength. Consumed pool and paired overlap not independent holdout replication. No matched pointwise-PDE comparator, no noise/crossresolution/PDE claim, no reliability-weight benefit or architecture novelty.',
             confirmation='Never read/generate reserves or choose a winner. Independent confirmation requires later sealed rules and explicit new authorization. This driver cannot open confirmation data.')
    locked_json(OUT / 'plan.json', p)
    return p


def summarize(rows, p, condition):
    by_time = np.mean([r['metrics']['val']['by_time'] for r in rows], axis=0)
    values = [r['metrics']['val']['primary'] for r in rows]
    return dict(primary_mean=float(np.mean(values)), per_seed=values, seed_sd=float(np.std(values, ddof=1)),
                common_interior_mean=float(by_time[p['common_columns']].mean()), all_future_mean=float(by_time[1:].mean()),
                by_time=by_time.tolist(), training_seconds=[r['seconds'] for r in rows], primary_columns=p['evaluation_columns'][condition])


def run(p):
    for name, expected in {**p['source'], **p['parents'], **p['data_chunks']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent/data changed: ' + name)
    clean, data = load_cached_split(p)
    groups, warms, initial, streams = {}, {}, {}, {}
    completed = 0

    def progress(current):
        atomic_json(OUT / 'progress.json', dict(completed_runs=completed, total_runs=24, current=current,
                    groups=groups, warm_runs=warms, confirmation_opened=False, exploratory=True))

    for condition in p['conditions']:
        a, obs = observations(clean, condition)
        context = dict(plan=digest(p), task='allen_cahn', condition=condition, data=data, observations=obs)
        base = {k: p[k] for k in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every', 'checkpoint_every', 'resolved_max_mode', 'interval_half_widths', 'interval_nodes']}
        base.update(p['task_spec'])
        base.update(position='periodic', position_center=float(np.mean(a['grid256'])),
                    position_scale=float(np.std(a['grid256']) * np.sqrt(2)), evaluation_columns=p['evaluation_columns'][condition])
        root = OUT / condition
        for seed in p['seeds']:
            c = dict(base, seed=seed, kind='warm', strength=0., steps=p['warm_steps'], lr=.001, lr_min=.00001)
            r = train_one(c, a, context, root / f'warm_s{seed}', torch.device('cuda'))
            if seed in initial and initial[seed] != r['initial']:
                raise ValueError('cross-condition initialization mismatch')
            initial[seed] = r['initial']
            value = (r['batch_hash'], r['collocation_hash'])
            if ('warm', seed) in streams and streams[('warm', seed)] != value:
                raise ValueError('cross-condition warm random streams mismatch')
            streams[('warm', seed)] = value
            warms[f'{condition}/warm_s{seed}'] = dict(primary=r['metrics']['val']['primary'], seconds=r['seconds'],
                                                       final_sha256=r['final_sha256'], initial=r['initial'])
            completed += 1
            progress(f'{condition}/warm_s{seed}')
        pair = {}
        for kind in p['kinds']:
            rows = []
            for seed in p['seeds']:
                c = dict(base, seed=seed, kind=kind, strength=0. if kind == 'none' else p['strength'],
                         steps=p['continuation_steps'], lr=.0003, lr_min=.00001)
                warm = root / f'warm_s{seed}/checkpoints/latest.pt'
                r = train_one(c, a, context, root / f'{kind}_s{seed}', torch.device('cuda'), warm_path=warm)
                value = (r['initial'], r['batch_hash'], r['collocation_hash'], r['calibration'], r['parameters_real'])
                if seed in pair and pair[seed] != value:
                    raise ValueError('within-condition paired fork mismatch')
                pair[seed] = value
                value = (r['batch_hash'], r['collocation_hash'])
                if ('continuation', seed) in streams and streams[('continuation', seed)] != value:
                    raise ValueError('cross-condition continuation random streams mismatch')
                streams[('continuation', seed)] = value
                rows.append(r)
                completed += 1
                progress(f'{condition}/{kind}_s{seed}')
            groups[f'{condition}/{kind}'] = summarize(rows, p, condition)
    summary = dict(plan=digest(p), completed_runs=completed, total_runs=24, continuation_runs=18, warm_count=6,
                   groups=groups, warm_runs=warms, exploratory=True, confirmation_opened=False)
    atomic_json(OUT / 'summary.json', summary)
    atomic_json(OUT / 'progress.json', dict(summary, status='completed'))
    print('ALLEN REPLICATION24 COMPLETE', json.dumps(groups), flush=True)


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
