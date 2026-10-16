"""Fail-closed audit of the 15 Allen-only exploratory interval continuations.

Reads the already sealed plan; never invokes a planner, solver, trainer, or
reference builder. Model construction, streams, frozen probe calibration and
prediction scores are replayed. Shared loss/Fourier helpers are identified as
replays, not independent PDE derivations or independent data confirmation.
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
from scripts.allen_interval_pilot_core import interval_penalties
from scripts.allen_interval_reference import compute_map
from scripts.audit_interval_diagnosis import load_reference
from scripts.audit_temporal_after_periodic_v3 import close_structure
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from scripts.fno_regularization_core import gradient_norm
from scripts.interval_diagnostics_core import indices, interval_metrics
from scripts.objective_modal_core import sample_weights
from scripts.physics_token_core import trajectory_errors
from scripts.run_physics_stress import observations
from scripts.run_public_pod_readout import learning_rate
from scripts.run_public_ridge_residual import digest
from scripts.time_fno_baseline import predict
from src.periodic_position_fno import PositionTimeFNO1d
from src.public_benchmark import atomic_json, file_hash

OUT = ROOT / 'results/allen_interval_pilot_v1'
KINDS = ['none', 'low', 'full', 'interval_uniform', 'interval_reliable']
RTOL, ATOL = 1e-10, 1e-11


def read_json(path):
    return json.loads(Path(path).read_text())


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def strict(actual, expected, label):
    close_structure(actual, expected, label, rtol=RTOL, atol=ATOL)


def same_state(actual, expected, label):
    """Exact nested tensor/NumPy/Python equality, including RNG states."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError('tensor state changed: ' + label)
    elif isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            raise ValueError('NumPy state changed: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('state keys changed: ' + label)
        for key in expected:
            same_state(actual[key], expected[key], label + '/' + str(key))
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('state sequence changed: ' + label)
        for i, value in enumerate(expected):
            same_state(actual[i], value, label + '/' + str(i))
    elif actual != expected:
        raise ValueError('state leaf changed: ' + label)


def model_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def make_model(c, state, device):
    model = PositionTimeFNO1d(c['width'], c['modes'], c['layers'],
        position=c['position'], length=c['length'], position_center=c['position_center'],
        position_scale=c['position_scale']).to(device)
    model.load_state_dict(state)
    model.eval()
    count = sum(v.numel() * (2 if v.is_complex() else 1) for v in model.parameters())
    if count != 34209:
        raise ValueError('FNO capacity changed')
    return model


def load_development(p):
    """Only read six existing train/dev cache files; never call clean_data."""
    old = read_json(ROOT / 'results/physics_stress_v1/plan.json')
    params = old['tasks']['allen_cahn']
    if p['train_ids'] != params['train_ids'] or p['val_ids'] != params['val_ids']:
        raise ValueError('old train/development IDs changed')
    qa_path = ROOT / 'results/physics_stress_v1/allen_cahn_solver_qa.json'
    if not read_json(qa_path)['passed']:
        raise ValueError('historical solver QA failed')
    arrays, hashes = {}, {}
    for split, key in [('train', 'train_ids'), ('val', 'val_ids')]:
        parts = []
        ids = params[key]
        for start in range(0, len(ids), 50):
            path = ROOT / f'external_data/physics_stress_v1/allen_{split}_{start:04d}.pt'
            obj = torch.load(path, map_location='cpu', weights_only=False)
            if obj['plan'] != digest(old) or obj['ids'] != ids[start:start + 50]:
                raise ValueError('old clean cache identity changed')
            parts.append(obj)
            hashes[path.name] = file_hash(path)
        arrays[split + '_x'] = np.concatenate([v['x'] for v in parts])
        arrays[split + '_y'] = np.concatenate([v['y'] for v in parts])
    arrays.update(times=params['times'], grid256=np.arange(256) * 2 / 256)
    a, obs = observations(arrays, 'dense_clean')
    data = dict(plan=digest(old), chunks=hashes, qa_sha256=file_hash(qa_path))
    if len(a['train_x']) != 100 or len(a['val_x']) != 200 or list(a['times']) != [0., .05, .125, .25, .5]:
        raise ValueError('fixed100/200/five-label budget changed')
    return a, obs, data


def validate_map(p, scale):
    manifest_path = OUT / 'reference_manifest.json'
    manifest = read_json(manifest_path)
    folder = ROOT / manifest['reference_path']
    expected_folder = ROOT / 'external_data/allen_interval_pilot_v1/train_reference'
    if folder.resolve() != expected_folder.resolve():
        raise ValueError('unexpected training-only map folder')
    map_path, external_json = folder / 'weight_map.pt', folder / 'weight_map.json'
    local_json = OUT / 'weight_map.json'
    artifacts = dict(map_sha256=file_hash(map_path), map_json_sha256=file_hash(local_json))
    if manifest != dict(artifacts=artifacts, reference_path=folder.relative_to(ROOT).as_posix(),
            train_ids=p['weight_train_ids'], context=dict(plan=digest(p))):
        raise ValueError('reference manifest identity changed')
    obj = torch.load(map_path, map_location='cpu', weights_only=False)
    if read_json(local_json) != native(obj) or read_json(external_json) != native(obj):
        raise ValueError('map JSON/binary companions differ')
    identity = obj['identity']
    if (identity != read_json(folder / 'identity.json') or
            identity['train_ids'] != p['weight_train_ids'] or
            p['weight_train_ids'] != p['train_ids'][:16] or
            p['weight_train_ids'] != list(range(200000, 200016)) or
            identity['target_scale'] != scale or identity['context'] != dict(plan=digest(p))):
        raise ValueError('training-only map identity/scale changed')
    if file_hash(ROOT / 'scripts/allen_interval_reference.py') != identity['source_sha256']:
        raise ValueError('training reference source changed')
    for name, expected in identity['frozen_sources'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('training reference dependency changed: ' + name)
    if (identity['centers'] != p['centers'] or identity['half_widths'] != p['interval_half_widths'] or
            identity['params'] != {k: p['task_spec'][k] for k in ['pde', 'length', 'diffusion', 'reaction']} |
                {'resolved_max_mode': p['resolved_max_mode']} or identity['final_time'] != p['task_spec']['time_upper'] or
            identity['actual_grid'] != 256 or identity['simpson_points'] != 9 or
            identity['comparison_simpson_points'] != 5 or identity['chunk_size'] != 8):
        raise ValueError('map numerical protocol changed')
    if obj['centers'] != p['centers'] or obj['half_widths'] != p['interval_half_widths'] or obj['band'] != 16:
        raise ValueError('map window/band scope changed')
    shape = (5, 2, 17)
    w64, w32, epsilon = [np.asarray(obj[k]) for k in ['weights64', 'weights32', 'epsilon']]
    if w64.shape != shape or w64.dtype != np.float64 or w32.shape != shape or w32.dtype != np.float32:
        raise ValueError('map weight shape/dtype changed')
    if epsilon.shape != shape or not np.isfinite(epsilon).all() or np.any(epsilon < 0):
        raise ValueError('invalid map numerical discrepancy')
    if not np.array_equal(w32, w64.astype(np.float32)) or not np.isfinite(w32).all() or np.any((w32 <= 0) | (w32 > 1)):
        raise ValueError('effective float32 weights not exact positive cast')
    components = obj['diagnostics']['rms_rate_components']
    names = {'timestep_endpoint_rate', 'timestep_rhs_average', 'space_endpoint_rate',
        'space_rhs_average', 'simpson5_vs9_rhs_average', 'native_closure', 'actual_grid_rhs_average'}
    if set(components) != names or not np.array_equal(epsilon, np.maximum.reduce(list(components.values()))):
        raise ValueError('map epsilon does not equal declared component maximum')
    tau = .001 * scale / p['task_spec']['time_upper']
    if obj['tau'] != tau or not np.array_equal(w64, tau * tau / (tau * tau + epsilon * epsilon)):
        raise ValueError('absolute tolerance/continuous weighting changed')
    keys = ('identity', 'centers', 'half_widths', 'band', 'weights64', 'weights32',
            'epsilon', 'tau', 'diagnostics', 'provenance', 'hashes', 'seconds')
    if obj['payload_sha256'] != digest(native({key: obj[key] for key in keys})):
        raise ValueError('map payload SHA changed')
    if (obj['provenance']['training_only_extra_solver_information'] is not True or
            obj['provenance']['no_development_or_reserved_access'] is not True or
            obj['provenance']['train_ids'] != p['weight_train_ids']):
        raise ValueError('map training-only provenance changed')
    stages = [('native_fine', 1024, .000125), ('native_coarse', 1024, .00025), ('spatial_fine', 512, .000125)]
    if identity['stages'] != native(stages):
        raise ValueError('map reference solver stages changed')
    times = np.asarray(obj['provenance']['times'], dtype=np.float64)
    expected_times = np.asarray(sorted({0.} | {round(float(t), 12) for center in p['centers']
        for h in p['interval_half_widths'] for t in np.linspace(center - h, center + h, 9)}))
    np.testing.assert_array_equal(times, expected_times)
    expected_stage_names = {f'{stage}_chunk_{chunk:04d}.pt' for stage, _, _ in stages for chunk in range(2)}
    if set(obj['hashes']['stages']) != expected_stage_names:
        raise ValueError('all six training reference stages required')
    pieces = {stage: [] for stage, _, _ in stages}
    stage_evidence = []
    for stage, n, dt in stages:
        for chunk in range(2):
            name = f'{stage}_chunk_{chunk:04d}.pt'
            path = folder / name
            if file_hash(path) != obj['hashes']['stages'][name]:
                raise ValueError('training reference stage file changed: ' + name)
            record = torch.load(path, map_location='cpu', weights_only=False)
            if record['identity'] != dict(map_identity=digest(identity), stage=stage,
                    ids=p['weight_train_ids'][chunk * 8:(chunk + 1) * 8], times=times.tolist()):
                raise ValueError('training reference stage identity changed')
            value = record['values']
            if value.shape != (8, len(times), n) or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('training reference stage array changed')
            if array_hash(value) != record['values_sha256']:
                raise ValueError('training reference stage array SHA changed')
            pieces[stage].append(value)
            stage_evidence.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=file_hash(path), cases=8, grid=n, dt=dt))
    # Replay the map calculation from saved TRAINING stages only. This is the
    # same equation/map helper, not a new numerical PDE integration.
    replay = compute_map(*(np.concatenate(pieces[name]) for name in
        ['native_fine', 'native_coarse', 'spatial_fine']), times, scale)
    strict(native(replay), native({key: obj[key] for key in replay}), 'training_map_from_cached_stages')
    report = dict(map_sha256=artifacts['map_sha256'], map_json_sha256=artifacts['map_json_sha256'],
        payload_sha256=obj['payload_sha256'], train_ids=p['weight_train_ids'], stage_evidence=stage_evidence,
        training_reference_seconds=float(obj['seconds']), tau=tau, epsilon_min=float(epsilon.min()),
        epsilon_max=float(epsilon.max()), weights64_min=float(w64.min()), weights64_max=float(w64.max()),
        weights64_mean=float(w64.mean()), weights64_max_distance_from_one=float(abs(w64 - 1).max()),
        weights32_min=float(w32.min()), weights32_max=float(w32.max()), weights32_mean=float(w32.mean()),
        weights32_max_distance_from_one=float(abs(w32.astype(float) - 1).max()),
        weights32_exactly_all_one=bool(np.array_equal(w32, np.ones(shape, dtype=np.float32))),
        weighted_mode_mass_fraction=float((w32 * np.r_[1., np.full(16, 2.)]).sum() / (5 * 2 * 33)),
        map_calculation_replayed=True, new_solver_performed=False,
        interpretation='Fixed absolute numerical tolerance, not calibrated probability. Constant weighting cancels under per-objective initial gradient calibration; tiny/all-one contrast is retained, not amplified.')
    return obj, artifacts, report


def replay_streams(seed, p, warm, latest, result, n=100):
    rng = np.random.default_rng(seed + 3000)
    bh = hashlib.sha256(b'').hexdigest()
    for stage, steps in [('warm', p['warm_steps']), ('continuation', p['continuation_steps'])]:
        for _ in range(steps):
            idx = rng.choice(n, min(p['batch_size'], n), replace=False)
            rng.uniform(0, 1, size=(32, 2))
            bh = hashlib.sha256(bytes.fromhex(bh) + idx.astype('<i8').tobytes()).hexdigest()
        if stage == 'warm':
            if bh != warm['batch_hash']:
                raise ValueError('warm batch stream mismatch')
            same_state(rng.bit_generator.state, warm['rng']['generator'], 'warm_generator')
    if bh != result['batch_hash'] or bh != latest['batch_hash']:
        raise ValueError('continuation batch stream mismatch')
    same_state(rng.bit_generator.state, latest['rng']['generator'], 'final_generator')
    c = result['identity']['config']
    query_rng = np.random.default_rng(seed + 4000)
    ch = hashlib.sha256(b'').hexdigest()
    for _ in range(c['steps']):
        qt = query_rng.uniform(c['time_lower'], c['time_upper'], min(c['batch_size'], n)).astype('<f4')
        ch = hashlib.sha256(bytes.fromhex(ch) + qt.tobytes()).hexdigest()
    if ch != result['collocation_hash'] or ch != latest['collocation_hash']:
        raise ValueError('query RNG/hash mismatch')
    same_state(query_rng.bit_generator.state, latest['collocation_rng'], 'final_query_generator')
    for name in ['python', 'numpy', 'torch', 'cuda']:
        same_state(latest['rng'][name], warm['rng'][name], 'unchanged_global_rng/' + name)
    return bh, ch


def calibration(c, a, warm, device):
    model = make_model(c, warm['model'], device)
    raw = np.asarray(a['train_x'], dtype=float)
    truth = np.asarray(a['train_y'], dtype=float)
    norm, scale = warm['normalization'], warm['scale']
    x = torch.tensor((raw - norm['mean']) / norm['std'], dtype=torch.float32, device=device)
    target = torch.tensor((truth - raw.mean(1)[:, None, None]) / scale, dtype=torch.float32, device=device)
    weights, _ = sample_weights(truth.reshape(-1, raw.shape[-1]), 'normalized')
    weights = torch.tensor(weights.reshape(len(raw), len(a['times']), 1), dtype=torch.float32, device=device)
    offsets = torch.tensor(raw.mean(1), dtype=torch.float32, device=device)
    probe = np.arange(16)
    times = torch.tensor(a['times'], dtype=torch.float32, device=device)
    query = torch.linspace(c['time_lower'], c['time_upper'], 16, device=device)
    coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
    loss = (weights[probe] * (predict(model, x[probe], times, coords) - target[probe]).square()).mean()
    penalties = interval_penalties(model, x[probe], query, coords, scale, offsets[probe], c)
    gd = gradient_norm(loss, model)
    gp = {key: gradient_norm(value, model) for key, value in penalties.items()}
    return dict(data_gradient_norm=gd, penalty_gradient_norms=gp, unit_weights={key: gd / value for key, value in gp.items()})


def inference_cost(model, x, times, coords):
    """Identical batch16 x five-time warmup3/repeat20 CUDA timing per arm."""
    sample = x[:16]
    with torch.no_grad():
        for _ in range(3):
            predict(model, sample, times, coords)
        torch.cuda.synchronize()
        elapsed = []
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(model, sample, times, coords)
            torch.cuda.synchronize()
            elapsed.append(1000 * (time.perf_counter() - start))
        peak = int(torch.cuda.max_memory_allocated())
    batch_ms = float(np.median(elapsed))
    return dict(batch_size=16, query_times=5, spatial_grid=256, warmup_repeats=3, timed_repeats=20,
        batch_time_ms=batch_ms, ms_per_trajectory=batch_ms / 16, timed_batch_ms=elapsed,
        peak_total_allocated_bytes=peak, additional_peak_allocated_bytes_above_baseline=max(peak - baseline, 0),
        policy='Median wall time, CUDA synchronized around direct normalized model evaluation; input/output transfer and postprocessing excluded.')


def old_interval_reference(p, a):
    path = ROOT / 'results/interval_diagnosis_v1'
    previous = read_json(path / 'plan.json')
    audit = read_json(path / 'audit.json')
    if not audit['complete'] or audit['checked_models'] != 36:
        raise ValueError('complete prior frozen interval audit required')
    if file_hash(ROOT / 'scripts/audit_interval_diagnosis.py') != audit['audit_source_sha256']:
        raise ValueError('sealed prior reference checker changed')
    ref, rhash, evidence = load_reference(previous, 'allen_cahn', None)
    old_ref = next(r for r in audit['reference_evidence'] if r['task'] == 'allen_cahn')
    if rhash != old_ref['reference_sha256'] or ref['ids'] != p['val_ids'][:32]:
        raise ValueError('consumed development reference changed')
    np.testing.assert_allclose(ref['fields'][256][:, indices(ref['times'], a['val_times'])],
        a['val_y'][:32], rtol=1e-5, atol=1e-6)
    return previous['tasks']['allen_cahn'], ref, evidence


def interval_diagnostics(model, x, coords, scale, raw, a, c, spec, ref):
    times = sorted({t for window in spec['windows'] for t in window['model_times']})
    ts = torch.tensor(times, dtype=torch.float32, device=x.device)
    with torch.no_grad():
        normalized = torch.cat([model(q.repeat_interleave(ts.numel(), 0), ts.repeat(q.shape[0]), coords)
            .reshape(q.shape[0], ts.numel(), q.shape[-1]) for q in x[:32].split(1)]).cpu().numpy().astype(float)
    pred = normalized * scale + raw[:32].mean(1)[:, None, None]
    centers = spec['centers']
    truth = ref['fields'][1024][:, indices(ref['times'], centers)]
    pc = pred[:, indices(times, centers)]
    windows = []
    for window in spec['windows']:
        value = interval_metrics(pred, times, ref, c, window)
        value['secondary_late'] = window['secondary_late']
        windows.append(value)
    primary = [w for w in windows if not w['secondary_late']]
    late = [w for w in windows if w['secondary_late']]
    return dict(midpoint=safe_band_errors(pc, truth[..., ::4], a['train_x'], c['length'], [0, 1, 2, 3]),
        late=safe_band_errors(pc, truth[..., ::4], a['train_x'], c['length'], [4]), windows=windows,
        primary_increment_mean=float(np.mean([w['increment']['mean'] for w in primary])),
        late_increment_mean=float(np.mean([w['increment']['mean'] for w in late])),
        primary_defect_rms_mean=float(np.mean([w['predicted_defect_rms'] for w in primary])),
        late_defect_rms_mean=float(np.mean([w['predicted_defect_rms'] for w in late])),
        reference_policy='Existing consumed32development IDs/times, old hashed reference; Simpson17 evaluation, not the Simpson5 training quadrature. No new labels, IDs, independent verification or correction export.')


def run():
    ppath = OUT / 'plan.json'
    p = read_json(ppath)
    if (p['seeds'] != [0, 1, 2] or p['kinds'] != KINDS or p['warm_steps'] != 1500 or
            p['continuation_steps'] != 750 or p['interval_nodes'] != 5 or
            p['interval_half_widths'] != [.005, .01] or p['strength'] != .3):
        raise ValueError('registered small-pilot budget changed')
    for name, expected in {**p['source'], **p['parents']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent changed: ' + name)
    expected_names = {f'{kind}_s{seed}' for kind in KINDS for seed in p['seeds']}
    paths = {path.parent.name: path for path in OUT.glob('*/result.json')}
    if set(paths) != expected_names:
        raise ValueError('all and only 15 completed continuation records required')
    a, obs, data = load_development(p)
    raw_train = np.asarray(a['train_x'], dtype=float)
    scale = float(np.asarray(a['train_y'], dtype=float).std())
    normalization = dict(mean=float(raw_train.mean()), std=float(raw_train.std()))
    weight_map, reference_identity, map_report = validate_map(p, scale)
    context = dict(plan=digest(p), task='allen_cahn', data=data, observations=obs, training_reference=reference_identity)
    embedded_map = dict(centers=weight_map['centers'], half_widths=weight_map['half_widths'], weights32=weight_map['weights32'].tolist())
    spec, ref, ref_evidence = old_interval_reference(p, a)
    warm_states, warm_evidence = {}, []
    for seed in p['seeds']:
        lineage = p['warm_lineage'][str(seed)]
        latest_path, final_path = ROOT / lineage['latest_path'], ROOT / lineage['final_path']
        if file_hash(latest_path) != lineage['latest_sha256'] or file_hash(final_path) != lineage['final_sha256']:
            raise ValueError('sealed warm checkpoint changed')
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        if latest['step'] != 1500 or final['step'] != 1500:
            raise ValueError('full-state warm step changed')
        same_state(latest['model'], final['model'], 'warm latest/final')
        if latest['normalization'] != normalization or latest['scale'] != scale:
            raise ValueError('warm normalization/observation scale changed')
        for key in ['optimizer', 'rng', 'collocation_rng', 'batch_hash']:
            if key not in latest:
                raise ValueError('warm checkpoint lacks continuation state')
        warm_states[seed] = latest
        warm_evidence.append(dict(seed=seed, **lineage, step=1500, full_state_restored=True,
            warm_model_hash=model_hash(latest['model'])))
    rows, pairing, calibrations, final_states = {}, {}, {}, {}
    historical_matches = 0
    device = torch.device('cuda')
    for kind in KINDS:
        for seed in p['seeds']:
            name = f'{kind}_s{seed}'
            path = paths[name]
            result = read_json(path)
            identity, c = result['identity'], result['identity']['config']
            if (identity['context'] != context or identity['warm_sha256'] != p['warm_lineage'][str(seed)]['latest_sha256'] or
                    c['kind'] != kind or c['seed'] != seed or c['position'] != 'periodic' or
                    c['steps'] != 750 or c['strength'] != (0. if kind == 'none' else .3) or
                    c['reliability_map'] != embedded_map):
                raise ValueError('arm/lineage/map identity changed: ' + name)
            for key in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every',
                        'checkpoint_every', 'resolved_max_mode', 'interval_half_widths', 'interval_nodes']:
                if c[key] != p[key]:
                    raise ValueError('configuration differs from sealed budget: ' + key)
            if any(c[k] != v for k, v in p['task_spec'].items()) or c['lr'] != .0003 or c['lr_min'] != .00001:
                raise ValueError('PDE or optimization schedule changed')
            if c['evaluation_columns'] != [1, 2, 3, 4]:
                raise ValueError('primary evaluation columns changed')
            if (c['position_center'] != float(np.mean(a['grid256'])) or
                    c['position_scale'] != float(np.std(a['grid256']) * np.sqrt(2))):
                raise ValueError('fixed periodic position encoding changed')
            manifest = read_json(path.parent / 'manifest.json')
            if manifest['identity'] != identity or manifest['status'] != 'completed':
                raise ValueError('manifest is not a completed matching run')
            final_path = path.parent / 'checkpoints/final.pt'
            latest_path = path.parent / 'checkpoints/latest.pt'
            if file_hash(final_path) != result['final_sha256']:
                raise ValueError('final checkpoint SHA changed')
            final = torch.load(final_path, map_location='cpu', weights_only=False)
            latest = torch.load(latest_path, map_location='cpu', weights_only=False)
            if final['identity'] != identity or latest['identity'] != identity or final['step'] != 750 or latest['step'] != 750:
                raise ValueError('final/latest identity/stage mismatch')
            same_state(final['model'], latest['model'], name + '/final/latest_model')
            if final['normalization'] != normalization or final['scale'] != scale or latest['normalization'] != normalization or latest['scale'] != scale:
                raise ValueError('training-only normalization changed')
            warm = warm_states[seed]
            initial = model_hash(warm['model'])
            if final['initial'] != initial or latest['initial'] != initial or result['initial'] != initial:
                raise ValueError('fork does not start from registered warm tensors')
            bh, ch = replay_streams(seed, p, warm, latest, result)
            if final['batch_hash'] != bh or final['collocation_hash'] != ch:
                raise ValueError('final model stream receipts changed')
            if latest['calibration'] != result['calibration'] or latest['history'] != result['history']:
                raise ValueError('full-state calibration/history differs from result')
            if ([entry['step'] for entry in result['history']] != [0, 250, 500, 750] or
                    result['metrics'] != result['history'][-1]['metrics']):
                raise ValueError('fixed final-step evaluation history changed')
            optimizer = latest['optimizer']
            expected_lr = learning_rate(c, c['steps'] - 1)
            if len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['lr'] != expected_lr or optimizer['param_groups'][0]['weight_decay'] != c['weight_decay']:
                raise ValueError('final optimizer group/schedule mismatch')
            if not optimizer['state'] or any(int(v['step']) != 2250 for v in optimizer['state'].values()):
                raise ValueError('Adam optimizer step lineage does not equal1500+750')
            if seed not in calibrations:
                calibrations[seed] = calibration(c, a, warm, device)
            strict(calibrations[seed], result['calibration'], name + '/warm_training_probe_calibration')
            expected_lambda = 0. if kind == 'none' else .3 * result['calibration']['unit_weights'][kind]
            if result['penalty_weight'] != expected_lambda:
                raise ValueError('frozen gradient-calibrated objective weight changed')
            pair = (initial, bh, ch, result['calibration'], result['parameters_real'])
            if seed in pairing and pairing[seed] != pair:
                raise ValueError('five-arm paired fork/calibration/streams mismatch')
            pairing[seed] = pair
            if kind in ['none', 'low', 'full']:
                old_path = ROOT / f'results/periodic_curvature_v1/allen_cahn/periodic/{kind}_s{seed}/checkpoints/final.pt'
                old_result = read_json(old_path.parent.parent / 'result.json')
                if file_hash(old_path) != old_result['final_sha256']:
                    raise ValueError('historical control checkpoint changed')
                old_cp = torch.load(old_path, map_location='cpu', weights_only=False)
                same_state(final['model'], old_cp['model'], name + '/historical_control')
                historical_matches += 1
            model = make_model(c, final['model'], device)
            if result['parameters_real'] != 34209:
                raise ValueError('reported model capacity changed')
            raw = np.asarray(a['val_x'], dtype=float)
            x = torch.tensor((raw - normalization['mean']) / normalization['std'], dtype=torch.float32, device=device)
            coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
            times = torch.tensor(a['val_times'], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = torch.cat([predict(model, q, times, coords) for q in x.split(16)]).cpu().numpy().astype(float)
            pred = pred * scale + raw.mean(1)[:, None, None]
            error = trajectory_errors(pred, a['val_y'])
            expected_metrics = dict(primary=float(error[:, 1:].mean()), per_sample=error[:, 1:].mean(1).tolist(), by_time=error.mean(0).tolist())
            strict(expected_metrics, result['metrics']['val'], name + '/200case_field_replay')
            diagnostics = safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], [1, 2, 3, 4])
            intervals = interval_diagnostics(model, x, coords, scale, raw, a, c, spec, ref)
            timing = inference_cost(model, x, times, coords)
            if not np.isfinite(result['seconds']) or result['seconds'] < 0:
                raise ValueError('invalid persisted training time')
            peak = result['peak_allocated_bytes_last_segment']
            if not isinstance(peak, int) or peak < 0:
                raise ValueError('invalid persisted memory receipt')
            rows[name] = dict(kind=kind, seed=seed, field=result['metrics']['val']['primary'],
                diagnostics=diagnostics, interval_diagnostics=intervals, inference=timing,
                training_step_seconds=result['seconds'], peak_allocated_bytes_last_segment=peak,
                cost_policy='Persisted seconds are timed optimizer-step segments only, excluding calibration/evaluation/reference/warm reuse. Peak is last segment, not whole-run or resumed-run peak.',
                result_sha256=file_hash(path), final_sha256=result['final_sha256'], latest_sha256=file_hash(latest_path),
                manifest_sha256=file_hash(path.parent / 'manifest.json'), initial=initial, batch_hash=bh,
                collocation_hash=ch, calibration_replayed=True, optimizer_rng_checkpoint_checked=True,
                historical_tensor_match=kind in ['none', 'low', 'full'])
            if kind in ['interval_uniform', 'interval_reliable']:
                final_states[(kind, seed)] = {k: v.clone() for k, v in final['model'].items()}
            print('ALLEN_INTERVAL_AUDIT', len(rows), '/15', name, flush=True)
            del model, final, latest
    if historical_matches != 9:
        raise ValueError('all nine historical tensor controls must reproduce')
    tensor_contrasts = []
    for seed in p['seeds']:
        left, right = final_states[('interval_uniform', seed)], final_states[('interval_reliable', seed)]
        same = left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
        maximum = max(float((left[k] - right[k]).abs().max()) for k in left)
        if map_report['weights32_exactly_all_one'] and not same:
            raise ValueError('all-one map must reproduce uniform model exactly')
        tensor_contrasts.append(dict(seed=seed, exact_tensor_equality=same,
            maximum_absolute_tensor_difference=maximum, required_equal_because_map_all_one=map_report['weights32_exactly_all_one']))
    groups = {}
    summary = read_json(OUT / 'summary.json')
    if summary['plan'] != digest(p) or summary['completed_runs'] != 15 or summary['total_runs'] != 15 or summary['confirmation_opened'] is not False:
        raise ValueError('incomplete/mismatched runner summary')
    for kind in KINDS:
        items = [rows[f'{kind}_s{seed}'] for seed in p['seeds']]
        fields = [r['field'] for r in items]
        report = dict(count=3, field_mean=float(np.mean(fields)), seed_sd=float(np.std(fields, ddof=1)), per_seed=fields,
            gradient_mean=float(np.mean([r['diagnostics']['gradient']['relative_mean'] for r in items])),
            gradient_per_seed=[r['diagnostics']['gradient']['relative_mean'] for r in items],
            curvature_mean=float(np.mean([r['diagnostics']['curvature']['relative_mean'] for r in items])),
            curvature_per_seed=[r['diagnostics']['curvature']['relative_mean'] for r in items],
            midpoint_field_mean=float(np.mean([r['interval_diagnostics']['midpoint']['field']['relative_mean'] for r in items])),
            late_field_mean=float(np.mean([r['interval_diagnostics']['late']['field']['relative_mean'] for r in items])),
            interval_increment_mean=float(np.mean([r['interval_diagnostics']['primary_increment_mean'] for r in items])),
            interval_increment_per_seed=[r['interval_diagnostics']['primary_increment_mean'] for r in items],
            late_interval_increment_mean=float(np.mean([r['interval_diagnostics']['late_increment_mean'] for r in items])),
            interval_defect_rms_mean=float(np.mean([r['interval_diagnostics']['primary_defect_rms_mean'] for r in items])),
            training_step_seconds=[r['training_step_seconds'] for r in items],
            peak_allocated_bytes_last_segment=[r['peak_allocated_bytes_last_segment'] for r in items],
            inference_batch_time_ms=[r['inference']['batch_time_ms'] for r in items])
        for key in ['field_mean', 'seed_sd', 'per_seed']:
            strict(report[key], summary['groups'][kind][key], 'runner_summary/' + kind + '/' + key)
        if summary['groups'][kind]['training_seconds'] != report['training_step_seconds']:
            raise ValueError('runner time receipts differ from run records')
        groups[kind] = report
    for kind in KINDS:
        groups[kind]['paired_field_change_vs_none'] = [v - b for v, b in zip(groups[kind]['per_seed'], groups['none']['per_seed'])]
    result = dict(plan=digest(p), plan_sha256=file_hash(ppath), audit_source_sha256=file_hash(Path(__file__)),
        summary_sha256=file_hash(OUT / 'summary.json'), checked_runs=15, complete=True,
        historical_control_tensor_matches=historical_matches, warm_lineage=warm_evidence,
        training_reliability=map_report, uniform_reliable_tensor_contrasts=tensor_contrasts,
        consumed_development_reference=ref_evidence, groups=groups, evidence=rows,
        source_verified=True, exact_stream_and_pairing_replays=15, warm_probe_calibrations_replayed=3,
        field_cases_per_run=200, diagnostic_interval_cases_per_run=32, diagnostic_intervals=150,
        reference_generation_seconds=map_report['training_reference_seconds'],
        newly_trained_optimizer_steps=15 * 750, reused_warm_optimizer_steps=3 * 1500,
        exploratory=True, confirmation_opened=False, new_solver_performed=False,
        limitations='Shared loss/calibration/map/Fourier helpers are numerical replays, not a separately derived PDE solver. TRAIN16 reference stages are additional training information, not fitted intermediate labels. Existing200dev fields and32dev intervals remain development. Three seeds and one transferred strength do not establish significance, optimality, architecture novelty, or independent benefit; no selected predictor exported.')
    atomic_json(OUT / 'audit.json', result)
    print('ALLEN INTERVAL AUDIT COMPLETE', json.dumps(groups), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true', help='Compatibility flag: all15 runs always required.')
    parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
