"""Read-only empirical precision probe; does not waive the sealed v2 audit.

Records alternate automatic-differentiation agreement, all stored metric-leaf
checks, and temporal signal scales for 36 already-diagnosed frozen models.
Only its explicitly named, source-hashed JSON artifact is written.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_temporal_after_periodic_v2 import OUT, load_task
from scripts.audit_temporal_after_periodic_v2 import close_structure, RTOL, ATOL
from scripts.temporal_after_periodic_core import analyze, temporal_score, native_reference
from scripts.run_public_ridge_residual import digest
from src.periodic_position_fno import PositionTimeFNO1d
from src.time_fno import time_fields
from src.public_benchmark import atomic_json, file_hash


def metric_checks(actual, expected, prefix, rtol=RTOL, atol=ATOL):
    """Use the sealed checker for every scalar or numeric-array metric field."""
    checked = 0
    failures = []
    def visit(got, wanted, name):
        nonlocal checked
        if isinstance(wanted, dict):
            if not isinstance(got, dict) or got.keys() != wanted.keys():
                failures.append(dict(field=name, reason='dictionary structure mismatch'))
                return
            for key in wanted:
                visit(got[key], wanted[key], name + '/' + key)
            return
        checked += 1
        try:
            close_structure(got, wanted, name, rtol=rtol, atol=atol)
        except (AssertionError, ValueError, TypeError) as exc:
            record = dict(field=name, reason=str(exc))
            if isinstance(wanted, (float, int, list)) and not isinstance(wanted, bool):
                ga = np.asarray(got, dtype=float)
                wa = np.asarray(wanted, dtype=float)
                if ga.shape == wa.shape:
                    diff = abs(ga - wa)
                    mask = diff > atol + rtol * abs(wa)
                    record.update(mismatch_elements=int(mask.sum()),
                                  max_absolute_difference=float(diff.max()),
                                  stored_shape=list(wa.shape))
            failures.append(record)
    visit(actual, expected, prefix)
    return dict(checked_fields=checked, failed_fields=len(failures),
                passed=not failures, failures=failures)


def run():
    p = json.loads((OUT / 'plan.json').read_text())
    for name, expected_hash in p['source'].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError('sealed source changed: ' + name)
    expected_names = {f'{task}/{position}/{kind}_s{seed}'
                      for task in ('allen_cahn', 'burgers')
                      for position in ('linear', 'periodic')
                      for kind in ('none', 'low', 'full') for seed in range(3)}
    if set(p['models']) != expected_names:
        raise ValueError('36 frozen final models required')
    cache = {}
    rows = []
    device = torch.device('cuda')
    for name, entry in p['models'].items():
        task = name.split('/')[0]
        if task not in cache:
            cache[task] = load_task(p, task)
        a, ref = cache[task]
        model_path = ROOT / entry['path']
        if file_hash(model_path) != entry['sha256']:
            raise ValueError('frozen model changed: ' + name)
        if file_hash(model_path.parent.parent / 'result.json') != entry['result_sha256']:
            raise ValueError('parent result changed: ' + name)
        stored_path = OUT / 'runs' / (name.replace('/', '__') + '.json')
        stored = json.loads(stored_path.read_text())
        if stored['identity'] != dict(plan=digest(p), model=entry['sha256'],
                                      reference=p['references'][task]['sha256']):
            raise ValueError('stored row identity changed: ' + name)
        cp = torch.load(model_path, map_location='cpu', weights_only=False)
        c = cp['identity']['config']
        model = PositionTimeFNO1d(c['width'], c['modes'], c['layers'],
                                 position=c['position'], length=c['length'],
                                 position_center=c['position_center'],
                                 position_scale=c['position_scale']).to(device)
        model.load_state_dict(cp['model'])
        model.eval()
        raw = np.asarray(a['val_x'][:32], dtype=float)
        norm = cp['normalization']
        x = torch.tensor((raw - norm['mean']) / norm['std'], dtype=torch.float32, device=device)
        coords = torch.tensor(a['grid256'], dtype=torch.float32, device=device)
        times = torch.tensor(a['times'][1:], dtype=torch.float32, device=device)
        b = x.repeat_interleave(4, 0)
        tt = times.repeat(32)
        off = torch.tensor(raw.mean(1), dtype=torch.float32, device=device).repeat_interleave(4)
        def physical_field(t):
            return model(b, t, coords) * cp['scale'] + off[:, None]
        fvalue, fderivative = torch.func.jvp(physical_field, (tt,), (torch.ones_like(tt),), strict=True)
        rvalue, rderivative = time_fields(model, b, tt, coords, cp['scale'], off, create_graph=False)
        forward = fderivative.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        reverse = rderivative.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        primal_forward = fvalue.detach().cpu().numpy().astype(float)
        primal_reverse = rvalue.detach().cpu().numpy().astype(float)
        difference = forward - reverse
        signal_rms = float(np.sqrt(np.mean(reverse ** 2)))
        case_signal = np.sqrt(np.mean(reverse ** 2, axis=-1))
        case_difference = np.sqrt(np.mean(difference ** 2, axis=-1))
        case_max = abs(difference).max(-1)
        mask = abs(difference) > ATOL + RTOL * abs(reverse)
        worst = np.unravel_index(np.argmax(abs(difference)), difference.shape)
        mismatch_examples = [dict(index=index.tolist(), forward=float(forward[tuple(index)]),
                                  reverse=float(reverse[tuple(index)]),
                                  absolute_difference=float(abs(difference[tuple(index)])))
                             for index in np.argwhere(mask)[:5]]
        all_times = torch.tensor(a['times'], dtype=torch.float32, device=device)
        with torch.no_grad():
            normalized = torch.cat([
                model(q.repeat_interleave(all_times.numel(), 0), all_times.repeat(q.shape[0]), coords)
                .reshape(q.shape[0], all_times.numel(), q.shape[-1])
                for q in x.split(16)]).cpu().numpy().astype(float)
        pred = (normalized * cp['scale'] + raw.mean(1)[:, None, None])[:, 1:]
        field = temporal_score(pred, np.asarray(a['val_y'][:32, 1:], dtype=float))
        temporal = analyze(pred, forward, ref, c)
        own = {}
        with torch.no_grad():
            for h in p['model_fd_half_widths']:
                fd = ((model(b, tt + h, coords) - model(b, tt - h, coords)) * cp['scale'] / (2 * h))
                own[f'{h:g}'] = temporal_score(fd.cpu().numpy().astype(float).reshape(32, 4, 256), forward)
        checks = dict(field32=metric_checks(field, stored['field32'], name + '/field32',
                                           rtol=1e-10, atol=1e-11),
                      temporal=metric_checks(temporal, stored['temporal'], name + '/temporal'),
                      own_fd=metric_checks(own, stored['own_fd'], name + '/own_fd'))
        truth_native, _ = native_reference(ref, c)
        truth = truth_native[..., ::4]
        # Signal/error scales use the primary reverse-mode derivative; the
        # forward-mode precision probe must not replace the original scores.
        case_true_rms = np.sqrt(np.mean(truth ** 2, axis=-1))
        record = dict(
            run=name, model_sha256=entry['sha256'], row_sha256=file_hash(stored_path),
            raw_elements=int(reverse.size), strict_raw_mismatch_elements=int(mask.sum()),
            strict_raw_mismatch_examples=mismatch_examples,
            signal_rms=signal_rms, difference_rms=float(np.sqrt(np.mean(difference ** 2))),
            norm_relative=float(np.linalg.norm(difference) / max(np.linalg.norm(reverse), 1e-30)),
            max_absolute_difference=float(abs(difference).max()),
            max_absolute_over_signal_rms=float(abs(difference).max() / max(signal_rms, 1e-30)),
            max_case_time_rms_relative=float((case_difference / np.maximum(case_signal, 1e-30)).max()),
            max_case_time_absolute_over_signal_rms=float((case_max / np.maximum(case_signal, 1e-30)).max()),
            worst_absolute_index=[int(i) for i in worst],
            primal_max_absolute_difference=float(abs(primal_forward - primal_reverse).max()),
            metric_checks=checks,
            primary_signal_scale_source='unchanged stored-runner reverse-over-reverse JVP',
            times=np.asarray(a['times'][1:], dtype=float).tolist(),
            predicted_ut_rms_by_time=np.sqrt(np.mean(reverse ** 2, axis=(0, 2))).tolist(),
            forward_ut_rms_by_time=np.sqrt(np.mean(forward ** 2, axis=(0, 2))).tolist(),
            true_native_sampled_ut_rms_by_time=np.sqrt(np.mean(truth ** 2, axis=(0, 2))).tolist(),
            error_rms_by_time=np.sqrt(np.mean((reverse - truth) ** 2, axis=(0, 2))).tolist(),
            true_native_sampled_case_rms_quantiles_by_time={
                str(q):np.quantile(case_true_rms, q, axis=0).tolist() for q in (0., .05, .5, .95, 1.)},
            stored_relative_time_error_by_time=stored['temporal']['truth_full_sampled']['by_time'],
            stored_denominator_floor_active=stored['temporal']['truth_full_sampled']['floor_active_fraction'])
        rows.append(record)
        print('PRECISION_PROBE', len(rows), '/36', name, 'raw_mismatches', int(mask.sum()),
              'relative', record['norm_relative'], 'metric_failed_fields',
              sum(check['failed_fields'] for check in checks.values()), flush=True)
        del model, cp, fvalue, fderivative, rvalue, rderivative
    source_paths = [Path(__file__), ROOT / 'scripts/audit_temporal_after_periodic_v2.py',
                    ROOT / 'scripts/diagnose_temporal_after_periodic_v2.py',
                    ROOT / 'scripts/temporal_after_periodic_core.py',
                    ROOT / 'src/periodic_position_fno.py', ROOT / 'src/time_fno.py', ROOT / 'src/fno.py']
    result = dict(
        plan=digest(p), plan_sha256=file_hash(OUT / 'plan.json'), checked_models=len(rows),
        probe_source_sha256=file_hash(Path(__file__)),
        dependency_sources={path.relative_to(ROOT).as_posix():file_hash(path) for path in source_paths},
        strict_existing_tolerance=dict(rtol=RTOL, atol=ATOL),
        summary=dict(
            models_with_raw_mismatches=sum(r['strict_raw_mismatch_elements'] > 0 for r in rows),
            total_raw_mismatches=sum(r['strict_raw_mismatch_elements'] for r in rows),
            total_raw_elements=sum(r['raw_elements'] for r in rows),
            maximum_norm_relative=max(r['norm_relative'] for r in rows),
            maximum_absolute_difference=max(r['max_absolute_difference'] for r in rows),
            maximum_absolute_over_signal_rms=max(r['max_absolute_over_signal_rms'] for r in rows),
            maximum_case_time_rms_relative=max(r['max_case_time_rms_relative'] for r in rows),
            maximum_case_time_absolute_over_signal_rms=max(r['max_case_time_absolute_over_signal_rms'] for r in rows),
            models_with_metric_failures=sum(any(not c['passed'] for c in r['metric_checks'].values()) for r in rows),
            total_failed_metric_fields=sum(c['failed_fields'] for r in rows for c in r['metric_checks'].values())),
        models=rows, training_performed=False, confirmation_opened=False,
        limitation='Empirical floating-point probe of fixed models, not an audit pass, new predictive evidence or authorization to relax sealed criteria.')
    atomic_json(OUT / 'ad_precision_probe.json', result)
    print('PRECISION_PROBE COMPLETE', json.dumps(result['summary']), flush=True)
    return result


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
