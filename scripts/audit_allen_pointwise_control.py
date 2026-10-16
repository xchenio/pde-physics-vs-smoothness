"""Audit a same-node pointwise-PDE control against frozen interval controls.

Only the new24 continuation records, previously audited six warms, old18
controls and six old data caches are read. No solver, reliability map, planner,
trainer or new reference data is invoked. Shared helper replays and fixed
TRAIN16 algebra QA are not independent predictive validation.
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
from scripts.allen_pointwise_control_core import corecontrol_penalties
from scripts.allen_interval_pilot_core import interval_windows
from scripts.audit_allen_interval_replication import (
    verify_split_and_cache, verify_observations, strict, same_state, make_model,
    stream_replay, inference_cost, field_scores, model_hash,
)
from scripts.diagnose_physics_stress_v2 import safe_band_errors
from scripts.field_derivative_diagnostics import coefficients
from scripts.objective_modal_core import sample_weights
from scripts.physics_token_core import trajectory_errors
from scripts.run_public_pod_readout import learning_rate
from scripts.run_public_ridge_residual import digest
from scripts.time_fno_baseline import predict
from src.physics_equations_v1 import equation_residual
from src.public_benchmark import atomic_json, file_hash
from src.time_fno import time_fields

OUT = ROOT / 'results/allen_pointwise_control_v1'
PARENT = ROOT / 'results/allen_interval_replication_v1'
CONDITIONS = ['dense', 'sparse']
KINDS = ['none', 'full', 'interval_uniform', 'pointwise_nodes']


def read_json(path):
    return json.loads(Path(path).read_text())


def gradient_norm(loss, model):
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(g.abs().square().sum() for g in gradients if g is not None)).detach())


def training_algebra(model, x, query, coords, scale, offsets, c, original_penalties):
    """TRAIN16 only: weighted means/variance identity and AD quadrature gap.

    The scalar tolerances were registered before pointwise outcomes. The gap
    has no pass/fail or predictive-benefit threshold and is not a truth error.
    """
    center, h, _ = interval_windows(query, c)
    nodes = torch.linspace(-1, 1, 5, dtype=query.dtype, device=query.device)
    times = center[:, None] + h[:, None] * nodes
    u, ut = time_fields(model, x.repeat_interleave(5, 0), times.reshape(-1), coords,
        scale, offsets.repeat_interleave(5), create_graph=False)
    with torch.no_grad():
        residual = equation_residual(u, ut, c) * (c['time_upper'] / scale)
    r = residual.detach().cpu().numpy().astype(np.complex128).reshape(16, 5, 17)
    q = np.asarray([1., 4., 2., 4., 1.]) / 12
    multiplicity = np.r_[1., np.full(16, 2.)]
    energy = lambda value: np.sum(abs(value) ** 2 * multiplicity, axis=-1)
    average = np.sum(r * q[None, :, None], axis=1)
    pointwise = float(np.mean(np.sum(energy(r) * q[None, :], axis=1)))
    mean_energy = float(np.mean(energy(average)))
    variance = float(np.mean(np.sum(energy(r - average[:, None]) * q[None, :], axis=1)))
    np.testing.assert_allclose(pointwise, mean_energy + variance, rtol=1e-5, atol=1e-7,
        equal_nan=False, err_msg='registered TRAIN Simpson mean/variance identity')
    stored_pointwise = float(original_penalties['pointwise_nodes'].detach())
    np.testing.assert_allclose(pointwise, stored_pointwise, rtol=1e-5, atol=1e-7,
        equal_nan=False, err_msg='TRAIN independent weighted pointwise scalar replay')
    uf = u.detach().cpu().numpy().astype(float).reshape(16, 5, 256)
    utf = ut.detach().cpu().numpy().astype(float).reshape(16, 5, 256)
    width = h.detach().cpu().numpy().astype(float)
    endpoint_rate = coefficients((uf[:, -1] - uf[:, 0]) / (2 * width[:, None]))[..., :17]
    average_ut = np.sum(coefficients(utf)[..., :17] * q[None, :, None], axis=1)
    factor = c['time_upper'] / scale
    gap_rms = float(np.sqrt(np.mean(energy((endpoint_rate - average_ut) * factor))))
    ut_rms = float(np.sqrt(np.mean(energy(average_ut * factor))))
    return dict(training_cases=16, query_source='same fixed observed-TRAIN calibration probe',
        squared_pointwise_mean=pointwise, squared_mean_residual=mean_energy,
        within_window_residual_variance=variance,
        algebra_absolute_difference=abs(pointwise - mean_energy - variance),
        implementation_pointwise_loss=stored_pointwise,
        implementation_interval_loss=float(original_penalties['interval_uniform'].detach()),
        endpoint_minus_average_AD_ut_dimensionless_rms=gap_rms,
        average_AD_ut_dimensionless_rms=ut_rms,
        gap_relative_to_model_AD_signal=gap_rms / max(ut_rms, 1e-30),
        numerical_QA_tolerance=dict(rtol=1e-5, atol=1e-7),
        policy='Registered technical mean/variance and implementation checks, not a predictive threshold. Endpoint/AD gap reported only; Simpson approximation prevents assuming interval loss is always <= pointwise loss. No true future field, development value or solver used.')


def calibration_and_algebra(c, a, warm, device):
    model = make_model(c, device, warm['model'])
    raw, truth = np.asarray(a['train_x'], dtype=float), np.asarray(a['train_y'], dtype=float)
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
    loss = (weights[probe] * (predict(model, x[probe], times, coords) - target[probe]).square()).mean()
    penalties = corecontrol_penalties(model, x[probe], query, coords, scale, offsets[probe], c)
    if set(penalties) != {'full', 'interval_uniform', 'pointwise_nodes'}:
        raise ValueError('same-node objective dictionary changed')
    gd = gradient_norm(loss, model)
    gp = {key: gradient_norm(value, model) for key, value in penalties.items()}
    calibrated = dict(data_gradient_norm=gd, penalty_gradient_norms=gp,
        unit_weights={key: gd / value for key, value in gp.items()})
    algebra = training_algebra(model, x[probe], query, coords, scale, offsets[probe], c, penalties)
    return calibrated, algebra


def verify_warm_lineage(p, parent_plan, parent_audit, prepared):
    expected = {f'{condition}/s{seed}' for condition in CONDITIONS for seed in p['seeds']}
    if set(p['warm_lineage']) != expected:
        raise ValueError('exactly six previously audited condition-owned warms required')
    states, receipts = {}, {}
    for condition in CONDITIONS:
        a, obs, norm, scale = prepared[condition]
        for seed in p['seeds']:
            key = f'{condition}/s{seed}'
            old_key = f'{condition}/warm_s{seed}'
            line = p['warm_lineage'][key]
            for path_key, hash_key in [('latest_path', 'latest_sha256'), ('final_path', 'final_sha256'), ('result_path', 'result_sha256')]:
                if file_hash(ROOT / line[path_key]) != line[hash_key]:
                    raise ValueError('reused warm artifact changed: ' + key + '/' + path_key)
            expected_folder = PARENT / condition / f'warm_s{seed}'
            if (ROOT / line['latest_path']).resolve() != (expected_folder / 'checkpoints/latest.pt').resolve():
                raise ValueError('warm must belong to its audited parent condition/seed')
            if (ROOT / line['final_path']).resolve() != (expected_folder / 'checkpoints/final.pt').resolve() or (ROOT / line['result_path']).resolve() != (expected_folder / 'result.json').resolve():
                raise ValueError('warm final/result lineage path mismatch')
            old = parent_audit['evidence'][old_key]
            if (line['latest_sha256'] != old['latest_sha256'] or line['final_sha256'] != old['final_sha256'] or
                    line['result_sha256'] != old['result_sha256']):
                raise ValueError('warm SHA is not the one previously audited')
            latest = torch.load(ROOT / line['latest_path'], map_location='cpu', weights_only=False)
            final = torch.load(ROOT / line['final_path'], map_location='cpu', weights_only=False)
            result = read_json(ROOT / line['result_path'])
            if latest['step'] != 1500 or final['step'] != 1500 or result['identity']['warm_sha256'] is not None:
                raise ValueError('reused warm is not the audited from-scratch1500 stage')
            same_state(latest['model'], final['model'], key + '/latest_final')
            if latest['identity'] != result['identity'] or final['identity'] != result['identity']:
                raise ValueError('warm checkpoint/result identities differ')
            if latest['identity']['context']['plan'] != digest(parent_plan) or latest['identity']['context']['observations'] != obs:
                raise ValueError('warm observation/parent identity changed')
            if latest['normalization'] != norm or latest['scale'] != scale or final['normalization'] != norm or final['scale'] != scale:
                raise ValueError('reused warm has an incompatible observation-only scale')
            if not latest['optimizer']['state'] or any(int(v['step']) != 1500 for v in latest['optimizer']['state'].values()):
                raise ValueError('reused full-state warm optimizer step changed')
            states[(condition, seed)] = latest
            receipts[key] = dict(**line, step=1500, prior_training_step_seconds=result['seconds'],
                warm_model_hash=model_hash(latest['model']), parent_audit_bound=True)
    return states, receipts


def verify_parent_controls(p, parent_audit):
    expected = {f'{condition}/{kind}_s{seed}' for condition in CONDITIONS
        for kind in ['none', 'full', 'interval_uniform'] for seed in p['seeds']}
    if set(p['parent_controls']) != expected:
        raise ValueError('all18 frozen parent control records required')
    for key, entry in p['parent_controls'].items():
        old = parent_audit['evidence'][key]
        if entry['result_sha256'] != old['result_sha256'] or entry['final_sha256'] != old['final_sha256']:
            raise ValueError('parent control receipt not bound to completed prior audit')
        folder = PARENT / key
        if (ROOT / entry['result_path']).resolve() != (folder / 'result.json').resolve() or (ROOT / entry['final_path']).resolve() != (folder / 'checkpoints/final.pt').resolve():
            raise ValueError('parent control paths changed')
        if file_hash(ROOT / entry['result_path']) != entry['result_sha256'] or file_hash(ROOT / entry['final_path']) != entry['final_sha256']:
            raise ValueError('parent control artifact changed')


def calibration_subset(calibration, old):
    return dict(data_gradient_norm=calibration['data_gradient_norm'],
        penalty_gradient_norms={key: calibration['penalty_gradient_norms'][key] for key in old['penalty_gradient_norms']},
        unit_weights={key: calibration['unit_weights'][key] for key in old['unit_weights']})


def run():
    plan_path = OUT / 'plan.json'
    p = read_json(plan_path)
    parent_plan_path, parent_audit_path = PARENT / 'plan.json', PARENT / 'audit.json'
    if file_hash(parent_plan_path) != p['replication_plan_sha256']:
        raise ValueError('audited replication parent plan changed')
    for name, expected in {**p['source'], **p['parents'], **p['data_chunks']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed source/parent/cache changed: ' + name)
    if p['parents'][parent_audit_path.relative_to(ROOT).as_posix()] != file_hash(parent_audit_path):
        raise ValueError('parent audit not sealed by new plan')
    parent_plan, parent_audit = read_json(parent_plan_path), read_json(parent_audit_path)
    if not parent_audit['complete'] or parent_audit['checked_runs'] != 24:
        raise ValueError('completed parent replication audit required')
    if (p['conditions'] != CONDITIONS or p['kinds'] != KINDS or p['seeds'] != [0, 1, 2] or
            p['warm_steps'] != 1500 or p['continuation_steps'] != 750 or p['strength'] != .3 or
            p['interval_half_widths'] != [.005, .01] or p['interval_nodes'] != 5):
        raise ValueError('fixed same-node control budget changed')
    for key in ['train_ids', 'val_ids', 'parent_ids_pool', 'split_seed', 'data_chunks', 'task_spec',
                'evaluation_columns', 'common_columns', 'batch_size', 'width', 'modes', 'layers']:
        if p[key] != parent_plan[key]:
            raise ValueError('split/model/observation rule differs from audited parent: ' + key)
    names = {f'{condition}/{kind}_s{seed}' for condition in CONDITIONS for kind in KINDS for seed in p['seeds']}
    paths = {f.parent.relative_to(OUT).as_posix(): f for f in OUT.glob('*/*/result.json')}
    if set(paths) != names:
        raise ValueError('all and only24 continuation records required, no new warm fits')
    clean, data = verify_split_and_cache(p)
    prepared = {condition: verify_observations(clean, condition) for condition in CONDITIONS}
    warms, warm_receipts = verify_warm_lineage(p, parent_plan, parent_audit, prepared)
    verify_parent_controls(p, parent_audit)
    rows, pairings, streams, calibrations, algebra = {}, {}, {}, {}, {}
    matches = 0
    device = torch.device('cuda')
    for condition in CONDITIONS:
        a, obs, norm, scale = prepared[condition]
        context = dict(plan=digest(p), task='allen_cahn', condition=condition, data=data, observations=obs)
        for kind in KINDS:
            for seed in p['seeds']:
                key, warm_key = f'{condition}/{kind}_s{seed}', (condition, seed)
                path = paths[key]
                r = read_json(path)
                identity, c = r['identity'], r['identity']['config']
                warm = warms[warm_key]
                line = p['warm_lineage'][f'{condition}/s{seed}']
                if (identity['context'] != context or identity['warm_sha256'] != line['latest_sha256'] or
                        c['kind'] != kind or c['seed'] != seed or c['position'] != 'periodic' or c['steps'] != 750 or
                        c['strength'] != (0. if kind == 'none' else .3) or 'reliability_map' in c):
                    raise ValueError('run identity/arm/lineage changed: ' + key)
                for field in ['width', 'modes', 'layers', 'weight_decay', 'batch_size', 'eval_every',
                              'checkpoint_every', 'resolved_max_mode', 'interval_half_widths', 'interval_nodes']:
                    if c[field] != p[field]:
                        raise ValueError('configuration differs from fixed budget: ' + field)
                if (any(c[k] != v for k, v in p['task_spec'].items()) or c['lr'] != .0003 or c['lr_min'] != .00001 or
                        c['position_center'] != float(np.mean(a['grid256'])) or c['position_scale'] != float(np.std(a['grid256']) * np.sqrt(2)) or
                        c['evaluation_columns'] != p['evaluation_columns'][condition]):
                    raise ValueError('equation/coordinate/schedule/evaluation changed')
                manifest, progress = read_json(path.parent / 'manifest.json'), read_json(path.parent / 'progress.json')
                if manifest['identity'] != identity or manifest['status'] != 'completed' or progress['status'] != 'completed' or progress['step'] != 750:
                    raise ValueError('complete matching manifest/progress required')
                final_path, latest_path = path.parent / 'checkpoints/final.pt', path.parent / 'checkpoints/latest.pt'
                if file_hash(final_path) != r['final_sha256']:
                    raise ValueError('final SHA changed')
                final = torch.load(final_path, map_location='cpu', weights_only=False)
                latest = torch.load(latest_path, map_location='cpu', weights_only=False)
                if final['identity'] != identity or latest['identity'] != identity or final['step'] != 750 or latest['step'] != 750:
                    raise ValueError('checkpoint identity/stage mismatch')
                same_state(final['model'], latest['model'], key + '/final_latest')
                initial = model_hash(warm['model'])
                if any(cp['initial'] != initial for cp in [r, final, latest]):
                    raise ValueError('continuation does not start from its audited warm tensors')
                if any(cp['normalization'] != norm or cp['scale'] != scale for cp in [final, latest]):
                    raise ValueError('selected-observation normalization changed')
                if (latest['history'] != r['history'] or latest['calibration'] != r['calibration'] or
                        [v['step'] for v in r['history']] != [0, 250, 500, 750] or r['metrics'] != r['history'][-1]['metrics']):
                    raise ValueError('fixed-step history/calibration/result receipt mismatch')
                bh, ch = stream_replay(c, p, latest, r, warm)
                if final['batch_hash'] != bh or final['collocation_hash'] != ch:
                    raise ValueError('final random stream receipt mismatch')
                optimizer = latest['optimizer']
                if (len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['lr'] != learning_rate(c, 749) or
                        optimizer['param_groups'][0]['weight_decay'] != c['weight_decay'] or not optimizer['state'] or
                        any(int(value['step']) != 2250 for value in optimizer['state'].values())):
                    raise ValueError('optimizer continuation lineage/schedule mismatch')
                if warm_key not in calibrations:
                    calibrations[warm_key], algebra[f'{condition}/s{seed}'] = calibration_and_algebra(c, a, warm, device)
                strict(calibrations[warm_key], r['calibration'], key + '/observed_TRAIN_calibration')
                lam = 0. if kind == 'none' else .3 * r['calibration']['unit_weights'][kind]
                if r['penalty_weight'] != lam:
                    raise ValueError('initial-gradient-rule constraint weight changed')
                value = (initial, bh, ch, r['calibration'], r['parameters_real'])
                if warm_key in pairings and pairings[warm_key] != value:
                    raise ValueError('four arms do not share warm/calibration/streams/capacity')
                pairings[warm_key] = value
                cross_key = (kind, seed)
                stream_value = (bh, ch, r['parameters_real'])
                if cross_key in streams and streams[cross_key] != stream_value:
                    raise ValueError('dense/sparse random-stream pairing failed')
                streams[cross_key] = stream_value
                if kind != 'pointwise_nodes':
                    old_entry = p['parent_controls'][key]
                    old_result = read_json(ROOT / old_entry['result_path'])
                    old_cp = torch.load(ROOT / old_entry['final_path'], map_location='cpu', weights_only=False)
                    same_state(final['model'], old_cp['model'], key + '/exact_parent_control')
                    strict(r['metrics'], old_result['metrics'], key + '/parent_control_metrics')
                    strict(calibration_subset(r['calibration'], old_result['calibration']),
                           old_result['calibration'], key + '/old_calibration_subset')
                    matches += 1
                model = make_model(c, device, final['model'])
                if r['parameters_real'] != 34209:
                    raise ValueError('reported architecture capacity changed')
                raw = np.asarray(a['val_x'], dtype=float)
                x = torch.tensor((raw - norm['mean']) / norm['std'], dtype=torch.float32, device=device)
                times = torch.tensor(a['val_times'], dtype=torch.float32, device=device)
                coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred = torch.cat([predict(model, b, times, coords) for b in x.split(16)]).cpu().numpy().astype(float)
                pred = pred * scale + raw.mean(1)[:, None, None]
                error = trajectory_errors(pred, a['val_y'])
                metrics = field_scores(error, p['evaluation_columns'][condition])
                strict(metrics, r['metrics']['val'], key + '/all200field_replay')
                common, future = field_scores(error, p['common_columns']), field_scores(error, [1, 2, 3, 4])
                scores = safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], p['evaluation_columns'][condition])
                common_scores = safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], p['common_columns'])
                by_time = [safe_band_errors(pred, a['val_y'], a['train_x'], c['length'], [i]) for i in range(5)]
                if not np.isfinite(r['seconds']) or r['seconds'] < 0 or r['peak_allocated_bytes_last_segment'] < 0:
                    raise ValueError('invalid persisted time/memory receipt')
                rows[key] = dict(condition=condition, kind=kind, seed=seed, primary=metrics['primary'],
                    per_case_primary=metrics['per_sample'], common_interior=common['primary'], per_case_common_interior=common['per_sample'],
                    all_future=future['primary'], by_time=metrics['by_time'], diagnostics=scores,
                    common_diagnostics=common_scores, by_time_diagnostics=by_time,
                    inference=inference_cost(model, x, times, coords), training_step_seconds=r['seconds'],
                    peak_allocated_bytes_last_segment=r['peak_allocated_bytes_last_segment'],
                    result_sha256=file_hash(path), final_sha256=r['final_sha256'], latest_sha256=file_hash(latest_path),
                    manifest_sha256=file_hash(path.parent / 'manifest.json'), warm_sha256=line['latest_sha256'],
                    initial=initial, batch_hash=bh, collocation_hash=ch, calibration_replayed=True,
                    historical_control_tensor_match=kind != 'pointwise_nodes')
                print('ALLEN_POINTWISE_AUDIT', len(rows), '/24', key, flush=True)
                del model, final, latest
    if matches != 18:
        raise ValueError('all18 original interval/none/full controls must replay exactly')
    summary = read_json(OUT / 'summary.json')
    if (summary['plan'] != digest(p) or summary['completed_runs'] != 24 or summary['total_runs'] != 24 or
            summary['reused_warm_count'] != 6 or summary['confirmation_opened'] is not False):
        raise ValueError('complete matching24-continuation summary required')
    groups = {}
    for condition in CONDITIONS:
        for kind in KINDS:
            key = condition + '/' + kind
            items = [rows[f'{condition}/{kind}_s{seed}'] for seed in p['seeds']]
            values = [r['primary'] for r in items]
            groups[key] = dict(count=3, primary_mean=float(np.mean(values)), per_seed=values,
                seed_sd=float(np.std(values, ddof=1)),
                primary_columns=p['evaluation_columns'][condition],
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
            for metric in ['primary_mean', 'per_seed', 'seed_sd', 'primary_columns', 'common_interior_mean', 'all_future_mean', 'by_time', 'training_seconds']:
                strict(groups[key][metric], summary['groups'][key][metric], 'runner_summary/' + key + '/' + metric)
    contrasts = {}
    for condition in CONDITIONS:
        interval = groups[condition + '/interval_uniform']
        pointwise = groups[condition + '/pointwise_nodes']
        contrasts[condition] = dict(
            primary_relative_change_interval_vs_pointwise=interval['primary_mean'] / pointwise['primary_mean'] - 1,
            primary_paired_changes=[i - q for i, q in zip(interval['per_seed'], pointwise['per_seed'])],
            primary_per_seed_relative_changes=[i / q - 1 for i, q in zip(interval['per_seed'], pointwise['per_seed'])],
            common_interior_relative_change=interval['common_interior_mean'] / pointwise['common_interior_mean'] - 1,
            common_interior_paired_changes=[i - q for i, q in zip(interval['common_interior_per_seed'], pointwise['common_interior_per_seed'])],
            common_interior_per_seed_relative_changes=[i / q - 1 for i, q in zip(interval['common_interior_per_seed'], pointwise['common_interior_per_seed'])],
            gradient_relative_change=interval['gradient_mean'] / pointwise['gradient_mean'] - 1,
            gradient_paired_changes=[i - q for i, q in zip(interval['gradient_per_seed'], pointwise['gradient_per_seed'])],
            curvature_relative_change=interval['curvature_mean'] / pointwise['curvature_mean'] - 1,
            curvature_paired_changes=[i - q for i, q in zip(interval['curvature_per_seed'], pointwise['curvature_per_seed'])],
            common_gradient_relative_change=interval['common_gradient_mean'] / pointwise['common_gradient_mean'] - 1,
            common_gradient_paired_changes=[i - q for i, q in zip(interval['common_gradient_per_seed'], pointwise['common_gradient_per_seed'])],
            common_curvature_relative_change=interval['common_curvature_mean'] / pointwise['common_curvature_mean'] - 1,
            common_curvature_paired_changes=[i - q for i, q in zip(interval['common_curvature_per_seed'], pointwise['common_curvature_per_seed'])],
            by_time_field_changes=[i - q for i, q in zip(interval['by_time'], pointwise['by_time'])],
            by_time_per_seed_field_changes=[[i - q for i, q in zip(rows[f'{condition}/interval_uniform_s{seed}']['by_time'],
                rows[f'{condition}/pointwise_nodes_s{seed}']['by_time'])] for seed in p['seeds']],
            by_time_gradient_changes=[i - q for i, q in zip(interval['by_time_gradient'], pointwise['by_time_gradient'])],
            by_time_curvature_changes=[i - q for i, q in zip(interval['by_time_curvature'], pointwise['by_time_curvature'])],
            sign_policy='Negative means lower interval error than pointwise. Signed descriptive paired differences only: no new success threshold, significance test or winner export.')
    audit = dict(plan=digest(p), plan_sha256=file_hash(plan_path), audit_source_sha256=file_hash(Path(__file__)),
        replication_plan_sha256=p['replication_plan_sha256'], parent_audit_sha256=file_hash(parent_audit_path),
        summary_sha256=file_hash(OUT / 'summary.json'), checked_runs=24, complete=True,
        historical_control_tensor_matches=matches, exact_stream_replays=24,
        within_condition_paired_forks=6, reused_warm_count=6, frozen_probe_calibrations_replayed=6,
        warm_lineage=warm_receipts, training_probe_algebra=algebra,
        observation_proofs={condition: prepared[condition][1] for condition in CONDITIONS},
        groups=groups, interval_vs_pointwise=contrasts, evidence=rows,
        field_cases_per_run=200, new_continuation_optimizer_steps=24 * 750,
        reused_warm_optimizer_steps=6 * 1500,
        total_new_training_step_seconds=float(sum(r['training_step_seconds'] for r in rows.values())),
        prior_reused_warm_step_seconds=float(sum(v['prior_training_step_seconds'] for v in warm_receipts.values())),
        cost_policy='750 optimization steps andexact5nodes are matched, NOT FLOPs. All arms compute allthree penalty forwards; selected backward/JVP costs differ. Persisted seconds exclude calibration/evaluation/I/O/audit. Warm costs are previous reused work, not new fits. Allocated memory peaks are lastsegment only. Samearchitecture timing differences are not inference speedups.',
        new_warm_trained=False, new_solver_performed=False, reliability_map_used=False,
        new_reference_data_accessed=False, confirmation_opened=False, exploratory=True,
        limitations='One previously consumed development split, two supervision conditions, three model seeds and one fixed initial-gradientratio rule. Same-node coverage does not equal full objective/optimization tuning or identical effective regularization through training. Shared operator/metric replay and TRAINalgebra are not independent PDE accuracy or new confirmation. No module, architecture novelty, general integral superiority or selected deployable predictor established.')
    atomic_json(OUT / 'audit.json', audit)
    print('ALLEN POINTWISE AUDIT COMPLETE', json.dumps(groups), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true', help='Compatibility flag: all24 continuations required.')
    parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
