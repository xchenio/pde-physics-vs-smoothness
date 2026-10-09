"""Replay all frozen midpoint/interval diagnostics before declaring completion.

The model is constructed directly, forward predictions preserve the runner's
one-trajectory batching, and forward-mode AD is compared with the original
reverse-over-reverse path under the previously registered scale-aware policy.
SciPy Simpson integration supplies an independent quadrature check. Shared
Fourier/PDE/metric helpers are replayed; this is not a second PDE solver or
independent data confirmation. No model, reference or old artifact is written.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.integrate import simpson as scipy_simpson
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_temporal_after_periodic_v3 import close_structure, derivative_precision
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from scripts.field_derivative_diagnostics import coefficients
from scripts.interval_diagnostics_core import indices, interval_metrics, simpson
from scripts.run_public_ridge_residual import digest
from scripts.temporal_after_periodic_core import rhs_coefficients, score_coefficients
from src.periodic_position_fno import PositionTimeFNO1d
from src.public_benchmark import atomic_json, file_hash
from src.time_fno import time_fields

OUT = ROOT / 'results/interval_diagnosis_v1'
DATA = ROOT / 'external_data/interval_diagnosis_v1'
FIELD_RTOL = 1e-10
FIELD_ATOL = 1e-11
TIME_RTOL = 2e-5
TIME_ATOL = 1e-7


def read_json(path):
    return json.loads(Path(path).read_text())


def strict(actual, expected, label):
    close_structure(actual, expected, label, rtol=FIELD_RTOL, atol=FIELD_ATOL)


def expected_times(p, task):
    """Independently reconstruct the registered reference time union."""
    spec = p['tasks'][task]
    values = [0., *spec['label_times']]
    for window in spec['windows']:
        values.extend(window['reference_times'])
    return np.asarray(sorted({round(float(t), 12) for t in values}), dtype=np.float64)


def validate_array(values, count, times, n, label):
    values = np.asarray(values)
    if values.dtype != np.float64 or values.shape != (count, len(times), n):
        raise ValueError('reference array shape/dtype changed: ' + label)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite reference array: ' + label)


def load_cached_development(stress, task):
    """Read existing caches only: never regenerate a PDE reference or labels."""
    if task == 'burgers':
        p = read_json(ROOT / 'results/physics_token_v1/plan.json')
        path = ROOT / 'external_data/physics_token_v1/development.pt'
        obj = torch.load(path, map_location='cpu', weights_only=False)
        if obj['plan'] != digest(p):
            raise ValueError('Burgers development cache identity changed')
        return obj['arrays'], {path.relative_to(ROOT).as_posix(): file_hash(path)}
    qa = read_json(ROOT / 'results/physics_stress_v1/allen_cahn_solver_qa.json')
    if not qa['passed']:
        raise ValueError('historical Allen solver QA was not successful')
    arrays, hashes = {}, {}
    for split, key in [('train', 'train_ids'), ('val', 'val_ids')]:
        parts = []
        ids = stress['tasks'][task][key]
        for start in range(0, len(ids), 50):
            path = ROOT / f'external_data/physics_stress_v1/allen_{split}_{start:04d}.pt'
            obj = torch.load(path, map_location='cpu', weights_only=False)
            if obj['plan'] != digest(stress) or obj['ids'] != ids[start:start + 50]:
                raise ValueError('Allen development cache identity changed')
            parts.append(obj)
            hashes[path.relative_to(ROOT).as_posix()] = file_hash(path)
        arrays[split + '_x'] = np.concatenate([v['x'] for v in parts])
        arrays[split + '_y'] = np.concatenate([v['y'] for v in parts])
    arrays.update(times=stress['tasks'][task]['times'], grid256=np.arange(256) * 2 / 256)
    return arrays, hashes


def load_reference(p, task, dataset_path):
    """Check complete chunk lineage and content, without running a solver."""
    ids = p['dev_ids'][task]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('exactly 32 existing development IDs required')
    times = expected_times(p, task)
    path = DATA / f'{task}_reference.pt'
    obj = torch.load(path, map_location='cpu', weights_only=False)
    identity = dict(plan=digest(p), task=task, ids=ids, times=times.tolist())
    if obj['identity'] != identity or obj['plan'] != digest(p) or obj['ids'] != ids:
        raise ValueError('reference identity changed: ' + task)
    if not np.array_equal(obj['times'], times):
        raise ValueError('reference times changed: ' + task)
    grids = {256, 1024, 512} if task == 'allen_cahn' else {256, 1024}
    if set(obj['fields']) != grids:
        raise ValueError('reference grid set changed: ' + task)
    for n, values in obj['fields'].items():
        validate_array(values, 32, times, n, task + '/' + str(n))
    if not np.array_equal(obj['fields'][256], obj['fields'][1024][..., ::4]):
        raise ValueError('reference256 is not declared native subsampling')
    stages = {'native_fine': 1024}
    if task == 'allen_cahn':
        stages.update(native_coarse=1024, independent512_fine=512)
        if set(obj['coarse_fields']) != {1024}:
            raise ValueError('Allen coarse reference grid changed')
        validate_array(obj['coarse_fields'][1024], 32, times, 1024, 'Allen coarse1024')
    elif obj['coarse_fields']:
        raise ValueError('unexpected Burgers solver output')
    names = {f'{task}_chunk_{i:04d}.pt' for i in range(4)}
    if set(obj['chunk_sha256']) != names:
        raise ValueError('reference chunk set changed')
    pieces = {stage: [] for stage in stages}
    chunk_evidence = []
    raw_times = None
    for i in range(4):
        name = f'{task}_chunk_{i:04d}.pt'
        cpath = DATA / name
        expected_hash = obj['chunk_sha256'][name]
        if file_hash(cpath) != expected_hash:
            raise ValueError('reference chunk hash changed: ' + name)
        chunk = torch.load(cpath, map_location='cpu', weights_only=False)
        wanted = dict(plan=digest(p), task=task, ids=ids[i * 8:(i + 1) * 8],
                      times=times.tolist())
        if chunk['identity'] != wanted or set(chunk['stages']) != set(stages):
            raise ValueError('reference chunk identity/stages changed: ' + name)
        for stage, n in stages.items():
            values = chunk['stages'][stage]
            validate_array(values, 8, times, n, name + '/' + stage)
            pieces[stage].append(values)
        if task == 'burgers':
            current = chunk['stored_time_coordinates']
            if raw_times is not None and current != raw_times:
                raise ValueError('Burgers raw coordinates differ between chunks')
            if np.max(abs(np.asarray(current) - times)) > 2e-7:
                raise ValueError('Burgers raw stored coordinates changed')
            raw_times = current
        chunk_evidence.append(dict(path=cpath.relative_to(ROOT).as_posix(),
                                   sha256=expected_hash, cases=8, stages=len(stages)))
    np.testing.assert_array_equal(np.concatenate(pieces['native_fine']), obj['fields'][1024])
    if task == 'allen_cahn':
        np.testing.assert_array_equal(np.concatenate(pieces['native_coarse']), obj['coarse_fields'][1024])
        np.testing.assert_array_equal(np.concatenate(pieces['independent512_fine']), obj['fields'][512])
    provenance = obj['provenance']
    if (provenance['source'] != p['source'] or
            provenance['task_parameters'] != p['task_spec'][task] or
            provenance['newly_opened_ids'] != [] or
            provenance['training_performed'] is not False or
            provenance['confirmation_opened'] is not False):
        raise ValueError('reference provenance changed')
    labels = {round(float(t), 12) for t in p['tasks'][task]['label_times']}
    extra = [float(t) for t in times if round(float(t), 12) not in labels]
    if provenance['diagnostic_times_not_in_supervised_labels'] != extra:
        raise ValueError('extra diagnostic time disclosure changed')
    if obj['extra_development_time_snapshots'] != times.tolist():
        raise ValueError('reference snapshot time declaration changed')
    if task == 'allen_cahn':
        if (provenance['fine_dt'] != .0000625 or provenance['coarse_dt'] != .000125 or
                provenance['independent512_dt'] != .0000625):
            raise ValueError('Allen reference timesteps changed')
    else:
        if (provenance['dataset'] != p['dataset'] or provenance['nominal_stored_dt'] != .01 or
                provenance['raw_stored_time_coordinates'] != raw_times):
            raise ValueError('Burgers source provenance changed')
        # Independently verify publisher values at these EXISTING IDs/times.
        ti = np.rint(times / .01).astype(int)
        with h5py.File(dataset_path, 'r') as source:
            native = np.stack([source['tensor'][int(case), ti, :] for case in ids]).astype(float)
            np.testing.assert_array_equal(native, obj['fields'][1024])
            np.testing.assert_array_equal(source['t-coordinate'][ti], np.asarray(raw_times))
    return obj, file_hash(path), dict(task=task, reference_sha256=file_hash(path),
        cases=32, time_count=len(times), grids=sorted(grids), chunk_evidence=chunk_evidence,
        chunk_content_reassembled=True, primary256_native_subsampling_exact=True,
        publisher_native_values_replayed=task == 'burgers',
        new_solver_performed=False)


def check_simpson_and_identity(pred, model_times, ref, c, window, row, label):
    """Independent SciPy quadrature and signed coefficient/energy identities."""
    mt = np.asarray(window['model_times'])
    rt = np.asarray(window['reference_times'])
    v = pred[:, indices(model_times, mt)]
    truth = ref['fields'][1024][:, indices(ref['times'], rt)]
    streams = [('model', v, mt), ('native', truth, rt)]
    if 'coarse_fields' in ref and 1024 in ref['coarse_fields']:
        streams.append(('solver_coarse', ref['coarse_fields'][1024][:, indices(ref['times'], rt)], rt))
    if c['pde'] == 'allen_cahn' and 512 in ref['fields']:
        streams.append(('space512', ref['fields'][512][:, indices(ref['times'], rt)], rt))
    integrals = {}
    checks = 0
    for name, fields, nodes in streams:
        rhs = rhs_coefficients(fields, c, 16)
        for stride in (1, 2):
            actual = scipy_simpson(rhs[:, ::stride], x=nodes[::stride], axis=-2)
            declared = simpson(rhs[:, ::stride], nodes[::stride])
            np.testing.assert_allclose(actual, declared, rtol=FIELD_RTOL, atol=FIELD_ATOL,
                equal_nan=False, err_msg=label + '/SciPy/' + name + '/stride' + str(stride))
            integrals[(name, stride)] = actual
            checks += 1
    dp = coefficients(v[:, -1] - v[:, 0])[..., :17]
    dt = coefficients(truth[:, -1] - truth[:, 0])[..., :17]
    ip, it = integrals[('model', 1)], integrals[('native', 1)]
    et, er = dp - dt, ip - it
    native, residual = dt - it, dp - ip
    np.testing.assert_allclose(residual, native + et - er, rtol=FIELD_RTOL, atol=FIELD_ATOL)
    weight = np.r_[1., np.full(16, 2.)]
    energy = lambda value: np.sum(abs(value) ** 2 * weight, axis=-1)
    denominator = np.maximum(energy(dt), max(1e-24, 1e-6 * float(energy(dt).mean())))
    te, se = energy(et), energy(er)
    cross = -2 * np.sum((et * er.conj()).real * weight, axis=-1)
    np.testing.assert_allclose(energy(et - er), te + se + cross,
                               rtol=FIELD_RTOL, atol=FIELD_ATOL)
    replay = dict(time_increment_common_energy=float((te / denominator).mean()),
        state_rhs_integral_common_energy=float((se / denominator).mean()),
        cross_common_energy=float((cross / denominator).mean()),
        error_discrepancy_common_energy=float((energy(et - er) / denominator).mean()),
        predicted_defect_rms=float(np.sqrt(energy(residual).mean())),
        native_defect_rms=float(np.sqrt(energy(native).mean())),
        oracle_increment_defect_rms=float(np.sqrt(energy(dt - ip).mean())),
        oracle_state_rhs_defect_rms=float(np.sqrt(energy(dp - it).mean())))
    strict(replay, {key: row[key] for key in replay}, label + '/independent_energy_oracles')
    np.testing.assert_allclose(row['time_increment_common_energy'] +
        row['state_rhs_integral_common_energy'] + row['cross_common_energy'],
        row['error_discrepancy_common_energy'], rtol=FIELD_RTOL, atol=FIELD_ATOL)
    if row['coefficient_identity_closed'] is not True:
        raise ValueError('stored coefficient identity flag is not true')
    gate = bool(row['increment']['mean'] >= .1 and
        row['increment']['mean'] >= 10 * max(v['mean'] for v in row['reference_uncertainty'].values()) and
        row['model_quadrature']['mean'] <= .1 * row['increment']['mean'] and
        row['increment']['floor_active_fraction'] == 0)
    if row['descriptive_gate'] is not gate:
        raise ValueError('per-window gate does not reproduce: ' + label)
    return checks


def independent_aggregate(rows):
    """Explicit saved-row aggregation/gates; do not call the runner aggregate."""
    groups = {}
    for name, row in rows.items():
        groups.setdefault(name.rsplit('_s', 1)[0], []).append(row)
    report = {}
    for name, items in sorted(groups.items()):
        if len(items) != 3:
            raise ValueError('all three seeds required: ' + name)
        windows = [w for r in items for w in r['intervals']]
        primary = [w for w in windows if not w['secondary_late']]
        report[name] = dict(count=len(items),
            midpoint_field=float(np.mean([r['midpoint_field']['field']['relative_mean'] for r in items])),
            midpoint_time_resolved=float(np.mean([r['midpoint_time_resolved']['mean'] for r in items])),
            late_field=float(np.mean([r['late_field']['field']['relative_mean'] for r in items])),
            late_time_resolved=float(np.mean([r['late_time_resolved']['mean'] for r in items])),
            interval_increment=float(np.mean([w['increment']['mean'] for w in primary])),
            interval_increment_per_seed=[float(np.mean([
                w['increment']['mean'] for w in r['intervals'] if not w['secondary_late']])) for r in items],
            interval_native_uncertainty=float(np.mean([w['reference_uncertainty_max'] for w in primary])),
            interval_model_quadrature=float(np.mean([w['model_quadrature']['mean'] for w in primary])),
            increment_common_energy=float(np.mean([w['time_increment_common_energy'] for w in primary])),
            rhs_integral_common_energy=float(np.mean([w['state_rhs_integral_common_energy'] for w in primary])),
            cross_common_energy=float(np.mean([w['cross_common_energy'] for w in primary])),
            endpoint_field=float(np.mean([w['endpoint_field']['mean'] for w in primary])),
            repair_field=float(np.mean([w['fixed_integral_repair_field']['mean'] for w in primary])),
            center_h_gates=[dict(run_seed=i, center=w['center'], half_width=w['half_width'],
                passed=w['descriptive_gate']) for i, r in enumerate(items) for w in r['intervals']])
    decision = {}
    for name in ['allen_cahn/periodic/none', 'burgers/linear/low']:
        if name not in groups:
            raise ValueError('strong baseline missing from independent aggregation')
        items = groups[name]
        primary = [w for r in items for w in r['intervals'] if not w['secondary_late']]
        decision[name] = dict(
            all_primary_centers_windows_seeds=len(items) == 3 and all(w['descriptive_gate'] for w in primary),
            pure_increment_dominance_all=all(w['time_increment_common_energy'] >
                w['state_rhs_integral_common_energy'] for w in primary),
            limits='Descriptive eligibility only. A failed strict all-center gate does not prove accurate dynamics; never choose favorable centers. No training authorized by screen alone.')
    return report, decision


def run():
    ppath = OUT / 'plan.json'
    p = read_json(ppath)
    plan_id = digest(p)
    summary_path = OUT / 'summary.json'
    summary = read_json(summary_path)
    parent_path = ROOT / 'results/temporal_after_periodic_v2'
    if (file_hash(parent_path / 'plan.json') != p['parent_plan_sha256'] or
            file_hash(parent_path / 'audit_v3.json') != p['parent_audit_sha256']):
        raise ValueError('sealed parent artifacts changed')
    parent = read_json(parent_path / 'plan.json')
    paudit = read_json(parent_path / 'audit_v3.json')
    if not paudit['complete'] or paudit['checked_models'] != 36:
        raise ValueError('complete temporal parent audit required')
    if file_hash(ROOT / 'scripts/audit_temporal_after_periodic_v3.py') != paudit['audit_source_sha256']:
        raise ValueError('previously sealed derivative QA helper changed')
    if p['models'] != parent['models'] or p['dev_ids'] != parent['dev_ids']:
        raise ValueError('frozen model/old development scope changed')
    expected = {f'{task}/{position}/{kind}_s{seed}'
        for task in ('allen_cahn', 'burgers') for position in ('linear', 'periodic')
        for kind in ('none', 'low', 'full') for seed in range(3)}
    if set(p['models']) != expected or p['resolved_max_mode'] != 16:
        raise ValueError('all 36 frozen models and native band16 required')
    for name, expected_hash in p['source'].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError('sealed source changed: ' + name)
    expected_rows = {name.replace('/', '__') + '.json' for name in expected}
    if {f.name for f in (OUT / 'runs').glob('*.json')} != expected_rows:
        raise ValueError('exactly 36 completed diagnostic records required')
    stress = read_json(ROOT / 'results/physics_stress_v1/plan.json')
    if p['task_spec'] != stress['tasks']:
        raise ValueError('PDE/data task specification changed')
    dataset_path = Path(p['dataset']['path'])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if file_hash(dataset_path) != p['dataset']['sha256']:
        raise ValueError('publisher dataset hash changed')
    cache = {}
    reference_evidence = []
    development_evidence = {}
    for task in ('allen_cahn', 'burgers'):
        arrays, chashes = load_cached_development(stress, task)
        ref, rhash, evidence = load_reference(p, task, dataset_path)
        if stress['tasks'][task]['val_ids'][:32] != ref['ids']:
            raise ValueError('existing development order changed')
        np.testing.assert_allclose(ref['fields'][256][:, indices(ref['times'], arrays['times'])],
            arrays['val_y'][:32], rtol=1e-5, atol=1e-6, equal_nan=False,
            err_msg='historical labeled-time reference consistency: ' + task)
        cache[task] = (arrays, ref, rhash)
        reference_evidence.append(evidence)
        development_evidence[task] = chashes
    rows, evidence = {}, []
    device = torch.device('cuda')
    total_quadrature_checks = 0
    for name, entry in p['models'].items():
        task, position, arm_seed = name.split('/')
        kind, seed = arm_seed.rsplit('_s', 1)
        arrays, ref, rhash = cache[task]
        model_path = ROOT / entry['path']
        result_path = model_path.parent.parent / 'result.json'
        if (file_hash(model_path) != entry['sha256'] or
                file_hash(result_path) != entry['result_sha256']):
            raise ValueError('frozen model/parent result hash changed: ' + name)
        row_path = OUT / 'runs' / (name.replace('/', '__') + '.json')
        row = read_json(row_path)
        if row['identity'] != dict(plan=plan_id, model=entry['sha256'], reference=rhash):
            raise ValueError('diagnostic row identity changed: ' + name)
        cp = torch.load(model_path, map_location='cpu', weights_only=False)
        c = cp['identity']['config']
        saved_parent = read_json(result_path)
        if cp['identity'] != saved_parent['identity'] or cp['step'] != c['steps']:
            raise ValueError('checkpoint identity/final step changed: ' + name)
        if (c['pde'] != task or c['position'] != position or c['kind'] != kind or
                c['seed'] != int(seed)):
            raise ValueError('checkpoint arm does not match registered name')
        model = PositionTimeFNO1d(c['width'], c['modes'], c['layers'],
            position=c['position'], length=c['length'], position_center=c['position_center'],
            position_scale=c['position_scale']).to(device)
        model.load_state_dict(cp['model'])
        model.eval()
        real_parameters = sum(v.numel() * (2 if v.is_complex() else 1) for v in model.parameters())
        if real_parameters != 34209:
            raise ValueError('architecture capacity changed')
        raw = np.asarray(arrays['val_x'][:32], dtype=float)
        norm = cp['normalization']
        x = torch.tensor((raw - norm['mean']) / norm['std'], dtype=torch.float32, device=device)
        coords = torch.tensor(arrays['grid256'], dtype=torch.float32, device=device)
        spec = p['tasks'][task]
        times = sorted({t for window in spec['windows'] for t in window['model_times']})
        ts = torch.tensor(times, dtype=torch.float32, device=device)
        with torch.no_grad():
            normalized = torch.cat([model(q.repeat_interleave(ts.numel(), 0),
                ts.repeat(q.shape[0]), coords).reshape(q.shape[0], ts.numel(), q.shape[-1])
                for q in x.split(1)]).cpu().numpy().astype(float)
        pred = normalized * cp['scale'] + raw.mean(1)[:, None, None]
        centers = spec['centers']
        pc = pred[:, indices(times, centers)]
        truth = ref['fields'][1024][:, indices(ref['times'], centers)]
        midpoint_field = safe_band_errors(pc, truth[..., ::4], arrays['train_x'], c['length'], [0, 1, 2, 3])
        late_field = safe_band_errors(pc, truth[..., ::4], arrays['train_x'], c['length'], [4])
        strict(midpoint_field, row['midpoint_field'], name + '/midpoint_field')
        strict(late_field, row['late_field'], name + '/late_field')
        if len(row['intervals']) != len(spec['windows']) or len(spec['windows']) != 10:
            raise ValueError('all 10 primary/secondary windows required')
        qchecks = 0
        for i, window in enumerate(spec['windows']):
            replay = interval_metrics(pred, times, ref, c, window)
            replay['secondary_late'] = window['secondary_late']
            strict(replay, row['intervals'][i], name + '/interval/' + str(i))
            qchecks += check_simpson_and_identity(pred, times, ref, c, window,
                row['intervals'][i], name + '/interval/' + str(i))
        tt = torch.tensor(centers, dtype=torch.float32, device=device).repeat(32)
        branch = x.repeat_interleave(len(centers), 0)
        offset = torch.tensor(raw.mean(1), dtype=torch.float32, device=device).repeat_interleave(len(centers))
        def physical_field(query):
            return model(branch, query, coords) * cp['scale'] + offset[:, None]
        fvalue, fut = torch.func.jvp(physical_field, (tt,), (torch.ones_like(tt),), strict=True)
        rvalue, rut = time_fields(model, branch, tt, coords, cp['scale'], offset, create_graph=False)
        forward = fut.detach().cpu().numpy().astype(float).reshape(32, len(centers), 256)
        reverse = rut.detach().cpu().numpy().astype(float).reshape(32, len(centers), 256)
        # Signal-aware raw derivative QA preserves the preceding failure-aware
        # policy; no elementwise near-zero derivative criterion is asserted.
        precision = derivative_precision(forward, reverse)
        np.testing.assert_allclose(fvalue.detach().cpu().numpy(), rvalue.detach().cpu().numpy(),
            rtol=TIME_RTOL, atol=TIME_ATOL, equal_nan=False,
            err_msg='forward/reverse primal: ' + name)
        if task == 'allen_cahn':
            truecf = rhs_coefficients(truth, c, 16)
        else:
            plus = ref['fields'][1024][:, indices(ref['times'], np.asarray(centers) + .01)]
            minus = ref['fields'][1024][:, indices(ref['times'], np.asarray(centers) - .01)]
            truecf = coefficients((plus - minus) / .02)[..., :17]
        weights = np.r_[1., np.full(16, 2.)]
        fcf = coefficients(forward)[..., :17]
        rcf = coefficients(reverse)[..., :17]
        for label, sl in [('midpoint_time_resolved', slice(None, 4)), ('late_time_resolved', slice(4, None))]:
            primary = score_coefficients(fcf[:, sl], truecf[:, sl], weights)
            original = score_coefficients(rcf[:, sl], truecf[:, sl], weights)
            strict(original, row[label], name + '/' + label + '/original_AD')
            close_structure(primary, row[label], name + '/' + label + '/alternate_AD',
                rtol=TIME_RTOL, atol=TIME_ATOL)
        rows[name] = row
        total_quadrature_checks += qchecks
        evidence.append(dict(run=name, row_sha256=file_hash(row_path),
            model_sha256=entry['sha256'], parent_result_sha256=entry['result_sha256'],
            reference_sha256=rhash, real_parameters=real_parameters,
            frozen_forward_field_replayed=True, intervals_replayed=10,
            independent_scipy_quadrature_checks=qchecks, signed_energy_oracle_checks=10,
            reverse_AD_metrics_replayed=True, alternate_forward_AD_metrics_replayed=True,
            derivative_precision=precision))
        print('INTERVAL_AUDIT', len(rows), '/36', name, flush=True)
        del model, cp, fvalue, fut, rvalue, rut
    groups, decision = independent_aggregate(rows)
    wanted_summary = dict(plan=plan_id, checked_models=36, groups=groups, decision=decision,
        confirmation_opened=False, training_performed=False)
    if summary != wanted_summary:
        raise ValueError('summary/decisions do not reproduce exactly from saved rows')
    audit = dict(plan=plan_id, plan_sha256=file_hash(ppath), summary_sha256=file_hash(summary_path),
        audit_source_sha256=file_hash(Path(__file__)), checked_models=len(rows), complete=True,
        parent_plan_sha256=p['parent_plan_sha256'], parent_audit_sha256=p['parent_audit_sha256'],
        previous_derivative_precision_source_sha256=file_hash(ROOT / 'scripts/audit_temporal_after_periodic_v3.py'),
        reference_evidence=reference_evidence, development_cache_evidence=development_evidence,
        checked_reference_chunks=sum(len(r['chunk_evidence']) for r in reference_evidence),
        checked_reference_solver_stages=sum(c['stages'] for r in reference_evidence for c in r['chunk_evidence']),
        checked_intervals=len(rows) * 10, independent_scipy_quadrature_checks=total_quadrature_checks,
        strict_metric_replay_tolerance=dict(rtol=FIELD_RTOL, atol=FIELD_ATOL),
        alternate_AD_metric_tolerance=dict(rtol=TIME_RTOL, atol=TIME_ATOL),
        saved_aggregate_replay='exact, explicit independent aggregation and gate recomputation',
        evidence=evidence, groups=groups, decision=decision, training_performed=False, confirmation_opened=False,
        limitations='Direct frozen-model forward/alternate-AD and shared Fourier/PDE/metric replay, plus independent SciPy quadrature and energy identities. Refined-step and512 consistency metrics replay cached references; no new or independent PDE solve. No causal field attribution, independent confirmation, time-loss fit or predictor export.')
    # This success artifact is written only after every one of the 36 records,
    # 360 intervals and independently reconstructed aggregate decisions passes.
    atomic_json(OUT / 'audit.json', audit)
    print('INTERVAL AUDIT COMPLETE', len(rows), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true',
        help='Compatibility flag: this auditor always requires all 36 records.')
    parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
