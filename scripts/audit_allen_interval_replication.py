"""Independent split/state/metric audit of the fixed Allen replication.

Read the sealed plan, six old caches and 24 completed fits only. No planner,
solver, reliability map, new midpoint reference, or trainer is invoked. Loss
and Fourier helper replays are not represented as independent PDE derivations.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.allen_interval_replication_core import replication_penalties
from scripts.allen_interval_replication_data import load_cached_split, observations
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from scripts.objective_modal_core import sample_weights
from scripts.physics_token_core import trajectory_errors
from scripts.run_public_pod_readout import learning_rate
from scripts.run_public_ridge_residual import digest
from scripts.time_fno_baseline import predict
from src.periodic_position_fno import PositionTimeFNO1d
from src.public_benchmark import atomic_json, file_hash
from src.resume import random_state
from src.utils import set_seed

OUT = ROOT / 'results/allen_interval_replication_v1'
CONDITIONS = ['dense', 'sparse']
KINDS = ['none', 'full', 'interval_uniform']
RTOL, ATOL = 1e-10, 1e-11


def read_json(path):
    return json.loads(Path(path).read_text())


def strict(actual, expected, label):
    """Check all JSON metric leaves without an old auditor dependency."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('metric dictionary mismatch: ' + label)
        for key in expected:
            strict(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('metric sequence mismatch: ' + label)
        for index, value in enumerate(expected):
            strict(actual[index], value, label + '/' + str(index))
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('metric literal mismatch: ' + label)
    elif isinstance(expected, (int, float)) and not isinstance(actual, bool):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL,
                                   equal_nan=False, err_msg=label)
    else:
        raise TypeError('unsupported metric leaf: ' + label)


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def model_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def same_state(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError('tensor state mismatch: ' + label)
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError('array state mismatch: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('state dictionary mismatch: ' + label)
        for key in expected:
            same_state(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('state sequence mismatch: ' + label)
        for i, value in enumerate(expected):
            same_state(actual[i], value, label + '/' + str(i))
    elif actual != expected:
        raise ValueError('state leaf mismatch: ' + label)


def make_model(c, device, state=None):
    model = PositionTimeFNO1d(c['width'], c['modes'], c['layers'],
        position=c['position'], length=c['length'], position_center=c['position_center'],
        position_scale=c['position_scale']).to(device)
    if state is not None:
        model.load_state_dict(state)
    if sum(v.numel() * (2 if v.is_complex() else 1) for v in model.parameters()) != 34209:
        raise ValueError('fixed FNO real parameter count changed')
    model.eval()
    return model


def verify_split_and_cache(p):
    """Derive the split independently and reindex every raw cache by actual ID."""
    pool = list(range(200000, 200300))
    if p['split_seed'] != 2026101601 or p['parent_ids_pool'] != pool:
        raise ValueError('fixed previously consumed development pool/seed changed')
    order = np.random.default_rng(2026101601).permutation(np.asarray(pool))
    train, val = np.sort(order[:100]).tolist(), np.sort(order[100:]).tolist()
    if p['train_ids'] != train or p['val_ids'] != val or set(train) & set(val):
        raise ValueError('new100/200 development split does not reproduce')
    if set(train) | set(val) != set(pool) or any(i >= 200300 for i in train + val):
        raise ValueError('reserved/new cases are outside the authorized reused pool')
    old = read_json(ROOT / 'results/physics_stress_v1/plan.json')
    expected_paths = {f'external_data/physics_stress_v1/allen_{split}_{start:04d}.pt'
        for split, count in [('train', 100), ('val', 200)] for start in range(0, count, 50)}
    if set(p['data_chunks']) != expected_paths:
        raise ValueError('all and only six old caches required')
    by_id = {}
    for name, expected_hash in p['data_chunks'].items():
        path = ROOT / name
        if file_hash(path) != expected_hash:
            raise ValueError('old cache SHA changed: ' + name)
        obj = torch.load(path, map_location='cpu', weights_only=False)
        if obj['plan'] != digest(old):
            raise ValueError('old cache plan digest changed')
        if len(obj['ids']) != len(obj['x']) or obj['y'].shape != (len(obj['ids']), 5, 256):
            raise ValueError('cache ID/field layout changed')
        for i, case in enumerate(obj['ids']):
            if case in by_id or case not in pool:
                raise ValueError('duplicate or reserved raw cache ID')
            by_id[case] = (np.asarray(obj['x'][i], dtype=float), np.asarray(obj['y'][i], dtype=float))
    if set(by_id) != set(pool):
        raise ValueError('raw cache pool incomplete')
    clean, context = load_cached_split(p)
    for split, ids in [('train', train), ('val', val)]:
        np.testing.assert_array_equal(clean[split + '_x'], np.stack([by_id[i][0] for i in ids]))
        np.testing.assert_array_equal(clean[split + '_y'], np.stack([by_id[i][1] for i in ids]))
        if clean[split + '_ids'] != ids:
            raise ValueError('clean split ordering differs from independently reindexed IDs')
    if context['chunks'] != p['data_chunks'] or context['new_solver_performed'] is not False or context['new_cases_accessed'] is not False:
        raise ValueError('read-only cache lineage changed')
    if clean['times'] != [0., .05, .125, .25, .5]:
        raise ValueError('original five-time labels changed')
    np.testing.assert_array_equal(clean['grid256'], np.arange(256, dtype=float) * 2 / 256)
    return clean, context


def verify_observations(clean, condition):
    a, obs = observations(clean, condition)
    columns = [0, 1, 2, 3, 4] if condition == 'dense' else [0, 4]
    raw = np.asarray(clean['train_x'], dtype=float)
    selected = np.asarray(clean['train_y'][:, columns], dtype=float)
    np.testing.assert_array_equal(a['train_x'], raw)
    np.testing.assert_array_equal(a['train_y'], selected)
    np.testing.assert_array_equal(a['val_y'], clean['val_y'])
    norm = dict(mean=float(raw.mean()), std=float(raw.std()))
    scale = float(selected.std())
    proof = dict(input_normalization=norm, selected_target_scale=scale,
        training_inputs_sha256=array_hash(raw), selected_targets_sha256=array_hash(selected),
        normalized_inputs_sha256=array_hash((raw - norm['mean']) / norm['std']),
        normalized_targets_sha256=array_hash((selected - raw.mean(1)[:, None, None]) / scale))
    if (obs['normalization_proof'] != proof or obs['normalization_sha256'] != digest(proof) or
            obs['observation_sha256'] != digest(selected.tolist()) or obs['observation_columns'] != columns or
            obs['selected_target_scale'] != scale or obs['input_normalization'] != norm or
            obs['heldout_training_labels_accessed_for_stats'] is not False or
            obs['development_labels_accessed_for_training_stats'] is not False):
        raise ValueError('observation-only normalization/hash proof does not reproduce')
    if a['times'] != [clean['times'][i] for i in columns] or a['val_times'] != clean['times']:
        raise ValueError('selected train/full development time axes changed')
    return a, obs, norm, scale


def fresh_initial(c, device):
    set_seed(c['seed'] + 2000)
    model = make_model(c, device)
    # Match the initialization path, including the optimizer construction, but
    # do not take any update or restore any previously fitted checkpoint.
    torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    initial = model_hash(model.state_dict())
    states = random_state(np.random.default_rng(c['seed'] + 3000))
    del model
    return initial, states


def stream_replay(c, p, latest, result, warm=None):
    rng = np.random.default_rng(c['seed'] + 3000)
    bh = hashlib.sha256(b'').hexdigest()
    total = c['steps'] if warm is None else p['warm_steps'] + c['steps']
    for step in range(total):
        idx = rng.choice(100, c['batch_size'], replace=False)
        rng.uniform(0, 1, size=(32, 2))
        bh = hashlib.sha256(bytes.fromhex(bh) + idx.astype('<i8').tobytes()).hexdigest()
        if warm is not None and step + 1 == p['warm_steps']:
            if bh != warm['batch_hash']:
                raise ValueError('fork warm batch receipt mismatch')
            same_state(rng.bit_generator.state, warm['rng']['generator'], 'warm_generator')
    if bh != result['batch_hash'] or bh != latest['batch_hash']:
        raise ValueError('batch hash replay failed')
    same_state(rng.bit_generator.state, latest['rng']['generator'], 'final_generator')
    crng = np.random.default_rng(c['seed'] + 4000)
    ch = hashlib.sha256(b'').hexdigest()
    for _ in range(c['steps']):
        qt = crng.uniform(c['time_lower'], c['time_upper'], c['batch_size']).astype('<f4')
        ch = hashlib.sha256(bytes.fromhex(ch) + qt.tobytes()).hexdigest()
    if ch != result['collocation_hash'] or ch != latest['collocation_hash']:
        raise ValueError('query hash replay failed')
    same_state(crng.bit_generator.state, latest['collocation_rng'], 'query_generator')
    if warm is not None:
        for key in ['python', 'numpy', 'torch', 'cuda']:
            same_state(latest['rng'][key], warm['rng'][key], 'unchanged_global_rng/' + key)
    return bh, ch


def gradient_norm(loss, model):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(g.abs().square().sum() for g in gradients if g is not None)).detach())


def replay_calibration(c, a, warm, device):
    model = make_model(c, device, warm['model'])
    raw = np.asarray(a['train_x'], dtype=float)
    truth = np.asarray(a['train_y'], dtype=float)
    norm, scale = warm['normalization'], warm['scale']
    x = torch.tensor((raw - norm['mean']) / norm['std'], dtype=torch.float32, device=device)
    target = torch.tensor((truth - raw.mean(1)[:, None, None]) / scale, dtype=torch.float32, device=device)
    weights, _ = sample_weights(truth.reshape(-1, 256), 'normalized')
    weights = torch.tensor(weights.reshape(100, len(a['times']), 1), dtype=torch.float32, device=device)
    offsets = torch.tensor(raw.mean(1), dtype=torch.float32, device=device)
    probe = np.arange(16)
    times = torch.tensor(a['times'], dtype=torch.float32, device=device)
    query = torch.linspace(c['time_lower'], c['time_upper'], 16, device=device)
    coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
    data_loss = (weights[probe] * (predict(model, x[probe], times, coords) - target[probe]).square()).mean()
    objectives = replication_penalties(model, x[probe], query, coords, scale, offsets[probe], c)
    if set(objectives) != {'full', 'interval_uniform'}:
        raise ValueError('unexpected objective/map introduced by wrapper')
    gd = gradient_norm(data_loss, model)
    gp = {key: gradient_norm(value, model) for key, value in objectives.items()}
    return dict(data_gradient_norm=gd, penalty_gradient_norms=gp, unit_weights={key: gd / value for key, value in gp.items()})


def inference_cost(model, x, times, coords):
    with torch.no_grad():
        for _ in range(3):
            predict(model, x[:16], times, coords)
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(model, x[:16], times, coords)
            torch.cuda.synchronize()
            elapsed.append(1000 * (time.perf_counter() - start))
        peak = int(torch.cuda.max_memory_allocated())
    return dict(batch_size=16, query_times=5, grid=256, warmup_repeats=3, timed_repeats=20,
        batch_time_ms=float(np.median(elapsed)), ms_per_trajectory=float(np.median(elapsed)) / 16,
        timed_batch_ms=elapsed, peak_total_allocated_bytes=peak,
        additional_peak_allocated_bytes_above_baseline=max(peak - baseline, 0),
        policy='CUDA-synchronized median direct normalized model evaluation; excludes transfers and physical postprocessing. Same architecture: measurement fluctuations are not architectural speedups.')


def field_scores(error, columns):
    selected = error[:, columns]
    return dict(primary=float(selected.mean()), per_sample=selected.mean(1).tolist(), by_time=error.mean(0).tolist())


def run():
    plan_path = OUT / 'plan.json'
    p = read_json(plan_path)
    if (p['conditions'] != CONDITIONS or p['kinds'] != KINDS or p['seeds'] != [0, 1, 2] or
            p['warm_steps'] != 1500 or p['continuation_steps'] != 750 or p['strength'] != .3 or
            p['interval_half_widths'] != [.005, .01] or p['interval_nodes'] != 5 or
            p['evaluation_columns'] != {'dense': [1, 2, 3, 4], 'sparse': [1, 2, 3]} or p['common_columns'] != [1, 2, 3]):
        raise ValueError('registered split/observation/arm/budget rules changed')
    for name, expected in {**p['source'], **p['parents'], **p['data_chunks']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent/cache changed: ' + name)
    expected_names = {f'{condition}/{kind}_s{seed}' for condition in CONDITIONS
        for kind in ['warm', *KINDS] for seed in p['seeds']}
    paths = {path.parent.relative_to(OUT).as_posix(): path for path in OUT.glob('*/*/result.json')}
    if set(paths) != expected_names:
        raise ValueError('all and only six warm plus18final records required')
    clean, data = verify_split_and_cache(p)
    prepared = {condition: verify_observations(clean, condition) for condition in CONDITIONS}
    rows, warm_states, fresh, calibration_cache = {}, {}, {}, {}
    fork_pair, cross_condition = {}, {}
    device = torch.device('cuda')
    for condition in CONDITIONS:
        a, obs, normalization, scale = prepared[condition]
        expected_context = dict(plan=digest(p), task='allen_cahn', condition=condition, data=data, observations=obs)
        for kind in ['warm', *KINDS]:
            for seed in p['seeds']:
                name = f'{condition}/{kind}_s{seed}'
                path = paths[name]
                r = read_json(path)
                identity, c = r['identity'], r['identity']['config']
                warm_key = (condition, seed)
                is_warm = kind == 'warm'
                steps = p['warm_steps'] if is_warm else p['continuation_steps']
                if (identity['context'] != expected_context or c['kind'] != kind or c['seed'] != seed or
                        c['steps'] != steps or c['strength'] != (0. if kind in ['warm', 'none'] else .3) or
                        c['position'] != 'periodic' or c['evaluation_columns'] != p['evaluation_columns'][condition]):
                    raise ValueError('run context/arm/primary columns mismatch: ' + name)
                for key in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every', 'checkpoint_every',
                            'resolved_max_mode', 'interval_half_widths', 'interval_nodes']:
                    if c[key] != p[key]:
                        raise ValueError('configuration differs from sealed plan: ' + key)
                if (any(c[k] != v for k, v in p['task_spec'].items()) or
                        c['position_center'] != float(np.mean(a['grid256'])) or
                        c['position_scale'] != float(np.std(a['grid256']) * np.sqrt(2)) or
                        c['lr'] != (.001 if is_warm else .0003) or c['lr_min'] != .00001 or
                        'reliability_map' in c):
                    raise ValueError('PDE/position/schedule/uniform-only configuration changed')
                manifest = read_json(path.parent / 'manifest.json')
                progress = read_json(path.parent / 'progress.json')
                if manifest['identity'] != identity or manifest['status'] != 'completed' or progress['status'] != 'completed' or progress['step'] != steps:
                    raise ValueError('matching completed manifest/progress required')
                final_path, latest_path = path.parent / 'checkpoints/final.pt', path.parent / 'checkpoints/latest.pt'
                if file_hash(final_path) != r['final_sha256']:
                    raise ValueError('final checkpoint SHA changed')
                final = torch.load(final_path, map_location='cpu', weights_only=False)
                latest = torch.load(latest_path, map_location='cpu', weights_only=False)
                if final['identity'] != identity or latest['identity'] != identity or final['step'] != steps or latest['step'] != steps:
                    raise ValueError('checkpoint identity/stage mismatch')
                same_state(final['model'], latest['model'], name + '/final_latest')
                for cp in [final, latest]:
                    if cp['normalization'] != normalization or cp['scale'] != scale:
                        raise ValueError('normalization uses anything except selected training observations')
                if (latest['history'] != r['history'] or latest['calibration'] != r['calibration'] or
                        [v['step'] for v in r['history']] != list(range(0, steps + 1, 250)) or
                        r['metrics'] != r['history'][-1]['metrics']):
                    raise ValueError('fixed final-step history/calibration receipt mismatch')
                warm = None if is_warm else warm_states[warm_key]
                if is_warm:
                    if identity['warm_sha256'] is not None or r['calibration'] is not None or r['penalty_weight'] != 0.:
                        raise ValueError('warm must start from scratch without a fitted objective')
                    if seed not in fresh:
                        fresh[seed] = fresh_initial(c, device)
                    initial, initial_rng = fresh[seed]
                    for key in ['python', 'numpy', 'torch', 'cuda']:
                        same_state(latest['rng'][key], initial_rng[key], name + '/fresh_global_rng/' + key)
                else:
                    warm_path = paths[f'{condition}/warm_s{seed}'].parent / 'checkpoints/latest.pt'
                    if identity['warm_sha256'] != file_hash(warm_path):
                        raise ValueError('fork does not use its own condition full-state warm')
                    initial = model_hash(warm['model'])
                    if warm_key not in calibration_cache:
                        calibration_cache[warm_key] = replay_calibration(c, a, warm, device)
                    strict(calibration_cache[warm_key], r['calibration'], name + '/training_probe_calibration')
                    expected_lambda = 0. if kind == 'none' else .3 * r['calibration']['unit_weights'][kind]
                    if r['penalty_weight'] != expected_lambda:
                        raise ValueError('fixed gradient-ratio objective weight changed')
                if r['initial'] != initial or final['initial'] != initial or latest['initial'] != initial:
                    raise ValueError('fresh/warm initial tensor receipt mismatch')
                bh, ch = stream_replay(c, p, latest, r, warm)
                if final['batch_hash'] != bh or final['collocation_hash'] != ch:
                    raise ValueError('final state stream receipts differ from replay')
                optimizer = latest['optimizer']
                if (len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['lr'] != learning_rate(c, steps - 1) or
                        optimizer['param_groups'][0]['weight_decay'] != c['weight_decay'] or not optimizer['state'] or
                        any(int(v['step']) != (1500 if is_warm else 2250) for v in optimizer['state'].values())):
                    raise ValueError('optimizer step/group lineage failed')
                pair = (initial, bh, ch, r['calibration'], r['parameters_real'])
                if not is_warm:
                    if warm_key in fork_pair and fork_pair[warm_key] != pair:
                        raise ValueError('three forks must share initial/streams/calibration/capacity')
                    fork_pair[warm_key] = pair
                cross_key = (kind, seed)
                cross_value = (bh, ch, r['parameters_real'])
                if cross_key in cross_condition and cross_condition[cross_key] != cross_value:
                    raise ValueError('cross-condition batch/query/capacity pairing failed')
                cross_condition[cross_key] = cross_value
                model = make_model(c, device, final['model'])
                if r['parameters_real'] != 34209:
                    raise ValueError('reported capacity differs from active parameters')
                raw = np.asarray(a['val_x'], dtype=float)
                x = torch.tensor((raw - normalization['mean']) / normalization['std'], dtype=torch.float32, device=device)
                times = torch.tensor(a['val_times'], dtype=torch.float32, device=device)
                coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred = torch.cat([predict(model, b, times, coords) for b in x.split(16)]).cpu().numpy().astype(float)
                pred = pred * scale + raw.mean(1)[:, None, None]
                error = trajectory_errors(pred, a['val_y'])
                metrics = field_scores(error, p['evaluation_columns'][condition])
                strict(metrics, r['metrics']['val'], name + '/all200field_replay')
                common = field_scores(error, p['common_columns'])
                all_future = field_scores(error, [1, 2, 3, 4])
                scores = safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], p['evaluation_columns'][condition])
                common_scores = safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], p['common_columns'])
                by_time_scores = [safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], [i]) for i in range(5)]
                if not np.isfinite(r['seconds']) or r['seconds'] < 0 or r['peak_allocated_bytes_last_segment'] < 0:
                    raise ValueError('invalid persisted cost receipts')
                rows[name] = dict(condition=condition, kind=kind, seed=seed, primary=metrics['primary'],
                    per_case_primary=metrics['per_sample'], common_interior=common['primary'],
                    per_case_common_interior=common['per_sample'], all_future=all_future['primary'], by_time=metrics['by_time'],
                    diagnostics=scores, common_diagnostics=common_scores, by_time_diagnostics=by_time_scores,
                    inference=inference_cost(model, x, times, coords),
                    training_step_seconds=r['seconds'], peak_allocated_bytes_last_segment=r['peak_allocated_bytes_last_segment'],
                    result_sha256=file_hash(path), final_sha256=r['final_sha256'], latest_sha256=file_hash(latest_path),
                    manifest_sha256=file_hash(path.parent / 'manifest.json'), initial=initial, batch_hash=bh, collocation_hash=ch,
                    optimizer_steps=1500 if is_warm else 2250, selected_target_scale=scale,
                    observation_columns=obs['observation_columns'], normalization_sha256=obs['normalization_sha256'],
                    training_probe_calibration_replayed=not is_warm)
                if is_warm:
                    warm_states[warm_key] = latest
                print('ALLEN_REPLICATION_AUDIT', len(rows), '/24', name, flush=True)
                del model, final
    if len(fresh) != 3 or len({value[0] for value in fresh.values()}) != 3:
        raise ValueError('three independently seeded initializations required')
    groups, descriptive = {}, {}
    summary = read_json(OUT / 'summary.json')
    if (summary['plan'] != digest(p) or summary['completed_runs'] != 24 or summary['total_runs'] != 24 or
            summary['warm_count'] != 6 or summary['continuation_runs'] != 18 or summary['confirmation_opened'] is not False):
        raise ValueError('complete matching runner summary required')
    expected_warm_receipts = {f'{condition}/warm_s{seed}': dict(
        primary=rows[f'{condition}/warm_s{seed}']['primary'],
        seconds=rows[f'{condition}/warm_s{seed}']['training_step_seconds'],
        final_sha256=rows[f'{condition}/warm_s{seed}']['final_sha256'],
        initial=rows[f'{condition}/warm_s{seed}']['initial'])
        for condition in CONDITIONS for seed in p['seeds']}
    if summary['warm_runs'] != expected_warm_receipts:
        raise ValueError('fresh warm summary receipts differ from the six actual runs')
    for condition in CONDITIONS:
        for kind in KINDS:
            key = condition + '/' + kind
            items = [rows[f'{condition}/{kind}_s{seed}'] for seed in p['seeds']]
            values = [r['primary'] for r in items]
            groups[key] = dict(count=3, primary_mean=float(np.mean(values)), per_seed=values,
                seed_sd=float(np.std(values, ddof=1)),
                common_interior_mean=float(np.mean([r['common_interior'] for r in items])),
                common_interior_per_seed=[r['common_interior'] for r in items],
                all_future_mean=float(np.mean([r['all_future'] for r in items])),
                by_time=np.mean([r['by_time'] for r in items], axis=0).tolist(),
                gradient_mean=float(np.mean([r['diagnostics']['gradient']['relative_mean'] for r in items])),
                gradient_per_seed=[r['diagnostics']['gradient']['relative_mean'] for r in items],
                curvature_mean=float(np.mean([r['diagnostics']['curvature']['relative_mean'] for r in items])),
                curvature_per_seed=[r['diagnostics']['curvature']['relative_mean'] for r in items],
                common_gradient_mean=float(np.mean([r['common_diagnostics']['gradient']['relative_mean'] for r in items])),
                common_gradient_per_seed=[r['common_diagnostics']['gradient']['relative_mean'] for r in items],
                common_curvature_mean=float(np.mean([r['common_diagnostics']['curvature']['relative_mean'] for r in items])),
                common_curvature_per_seed=[r['common_diagnostics']['curvature']['relative_mean'] for r in items],
                by_time_gradient=np.mean([[v['gradient']['relative_mean'] for v in r['by_time_diagnostics']] for r in items], axis=0).tolist(),
                by_time_curvature=np.mean([[v['curvature']['relative_mean'] for v in r['by_time_diagnostics']] for r in items], axis=0).tolist(),
                training_seconds=[r['training_step_seconds'] for r in items],
                peak_allocated_bytes_last_segment=[r['peak_allocated_bytes_last_segment'] for r in items],
                inference_batch_ms=[r['inference']['batch_time_ms'] for r in items])
            for metric in ['primary_mean', 'per_seed', 'seed_sd', 'common_interior_mean', 'all_future_mean', 'by_time', 'training_seconds']:
                strict(groups[key][metric], summary['groups'][key][metric], 'runner_summary/' + key + '/' + metric)
        base = groups[condition + '/none']
        for kind in ['full', 'interval_uniform']:
            candidate = groups[condition + '/' + kind]
            gain = 1 - candidate['primary_mean'] / base['primary_mean']
            all_three = all(v < b for v, b in zip(candidate['per_seed'], base['per_seed']))
            descriptive[condition + '/' + kind] = dict(primary_relative_gain=gain,
                paired_primary_changes=[v - b for v, b in zip(candidate['per_seed'], base['per_seed'])],
                all_three_seed_primary_improve=all_three,
                fixed_descriptive_support=bool(gain >= .01 and all_three) if kind == 'interval_uniform' else None,
                by_time_field_change_vs_none=[v - b for v, b in zip(candidate['by_time'], base['by_time'])],
                gradient_relative_gain=1 - candidate['gradient_mean'] / base['gradient_mean'],
                curvature_relative_gain=1 - candidate['curvature_mean'] / base['curvature_mean'],
                rule=('Uniform only: fixed arbitrary descriptive screen primarymean gain>=1% AND all3seed gains>0. '
                      'Full is a descriptive strong-control comparison, with no additional registered eligibility screen. '
                      'Not statistical significance, proven practical meaning, domain tolerability or authority to open confirmation.'))
    common_gains = {}
    for condition in CONDITIONS:
        base, candidate = groups[condition + '/none'], groups[condition + '/interval_uniform']
        common_gains[condition] = dict(relative_gain=1 - candidate['common_interior_mean'] / base['common_interior_mean'],
            per_seed_relative_gains=[1 - v / b for v, b in zip(candidate['common_interior_per_seed'], base['common_interior_per_seed'])])
    interaction = dict(dense=common_gains['dense'], sparse=common_gains['sparse'],
        sparse_minus_dense_relative_gain=common_gains['sparse']['relative_gain'] - common_gains['dense']['relative_gain'],
        per_seed_sparse_minus_dense=[s - d for s, d in zip(common_gains['sparse']['per_seed_relative_gains'], common_gains['dense']['per_seed_relative_gains'])],
        evaluation_columns=[1, 2, 3], evaluation_times=[.05, .125, .25],
        policy='Same200development cases andthreeinterior times. Different conditionprimary metrics are NOT compared directly. Descriptive interaction of complete training procedures, not an independent data replication or causal isolation of a temporal mechanism.')
    warm_seconds = {condition: [rows[f'{condition}/warm_s{seed}']['training_step_seconds'] for seed in p['seeds']] for condition in CONDITIONS}
    audit = dict(plan=digest(p), plan_sha256=file_hash(plan_path), audit_source_sha256=file_hash(Path(__file__)),
        summary_sha256=file_hash(OUT / 'summary.json'), checked_runs=24, checked_warm_runs=6,
        checked_continuation_runs=18, complete=True, groups=groups, descriptive_support=descriptive,
        common_interior_interaction=interaction, evidence=rows,
        split=dict(seed=p['split_seed'], train_ids=p['train_ids'], val_ids=p['val_ids'], pool_ids=p['parent_ids_pool'],
            old_train_cases_in_new_development=len(set(range(200000, 200100)) & set(p['val_ids'])),
            independent_confirmation=False, previously_consumed_cases_only=True),
        observation_proofs={condition: prepared[condition][1] for condition in CONDITIONS},
        fresh_initial_hashes={str(seed): value[0] for seed, value in fresh.items()},
        exact_stream_replays=24, within_condition_paired_forks=6, cross_condition_initial_pairs=3,
        frozen_probe_calibrations_replayed=6, field_cases_per_run=200,
        warm_training_step_seconds=warm_seconds,
        total_persisted_training_step_seconds=float(sum(r['training_step_seconds'] for r in rows.values())),
        newly_trained_optimizer_steps=6 * 1500 + 18 * 750,
        costs='Every warm counted once, not once per fork. Persisted seconds time optimizer-step segments only, excluding calibration/evaluation/cache reading/audit. Allocated-byte peaks cover last segment, not whole-run/resume/total-device memory. Equal steps need not mean equal FLOPs.',
        new_solver_performed=False, reliability_map_used=False, new_midpoint_reference_accessed=False,
        confirmation_opened=False, exploratory=True,
        limitations='One deterministic reshuffle of300previously consumed cases andthree model seeds, not independent confirmation or multiple new datasets. Omitted sparse TRAIN labels are present in raw integrity-checked caches but not fitting statistics/targets/calibration. Shared objective/Fourier metric replays do not constitute a second PDE solver. No checkpoint/strength/time/seed selection or architecture novelty established; no selected predictor exported.')
    atomic_json(OUT / 'audit.json', audit)
    print('ALLEN REPLICATION AUDIT COMPLETE', json.dumps(groups), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true', help='Compatibility flag: all24 runs are always required.')
    parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
