"""Independent forward-mode replay of all frozen temporal-diagnosis rows.

Stored identities and aggregates must match exactly. Numerical replay permits
the registered float32 tolerance because torch.func.jvp uses forward-mode AD,
unlike the diagnostic runner's reverse-over-reverse functional.jvp. Shared
NumPy score routines are replayed, not represented as an independent derivation
of the PDE or a causal field-error decomposition.
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
from scripts.diagnose_temporal_after_periodic_v2 import load_task
from scripts.temporal_after_periodic_core import analyze, temporal_score
from scripts.physics_token_core import trajectory_errors
from scripts.run_public_ridge_residual import digest
from src.periodic_position_fno import PositionTimeFNO1d
from src.time_fno import time_fields
from src.public_benchmark import atomic_json, file_hash

OUT = ROOT / 'results/temporal_after_periodic_v2'
PILOT = ROOT / 'results/temporal_after_periodic_v1'
PARENT = ROOT / 'results/periodic_curvature_v1'
RTOL = 2e-5
ATOL = 1e-7


def close_structure(actual, expected, label, rtol=RTOL, atol=ATOL):
    """Check every replayed numeric leaf and all nonnumeric structure."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('dictionary structure mismatch: ' + label)
        for key in expected:
            close_structure(actual[key], expected[key], label + '/' + key, rtol, atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('list structure mismatch: ' + label)
        for index, item in enumerate(expected):
            close_structure(actual[index], item, label + '/' + str(index), rtol, atol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('exact leaf mismatch: ' + label)
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError('numeric leaf type mismatch: ' + label)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   equal_nan=False, err_msg=label)
    else:
        raise TypeError('unsupported JSON leaf: ' + label)


def check_energy_closure(row, name):
    d = row['temporal']['decomposition_resolved16']
    if d['identity_closed'] is not True:
        raise ValueError('unclosed coefficient identity: ' + name)
    np.testing.assert_allclose(
        d['time_error_common_normalized_squared'] +
        d['rhs_error_common_normalized_squared'] +
        d['cross_common_normalized'],
        d['discrepancy_common_normalized_squared'], rtol=1e-10, atol=1e-10,
        err_msg='decomposition normalized-energy closure: ' + name)


def independent_aggregate(rows):
    """Reproduce saved-row aggregates and the descriptive gate independently."""
    grouped = {}
    for name, row in rows.items():
        grouped.setdefault(name.rsplit('_s', 1)[0], []).append(row)
    report = {}
    for name, items in sorted(grouped.items()):
        if len(items) != 3:
            raise ValueError('expected all three seeds: ' + name)
        temporal = [r['temporal'] for r in items]
        decomposed = [t['decomposition_resolved16'] for t in temporal]
        report[name] = dict(
            count=len(items),
            field32=float(np.mean([r['field32']['mean'] for r in items])),
            time_full=float(np.mean([t['truth_full_sampled']['mean'] for t in temporal])),
            time_resolved=float(np.mean([t['truth_resolved_native']['mean'] for t in temporal])),
            time_resolved_per_seed=[t['truth_resolved_native']['mean'] for t in temporal],
            time_resolved_by_time=np.mean(
                [t['truth_resolved_native']['by_time'] for t in temporal], axis=0).tolist(),
            own_fd_max=max(v['mean'] for r in items for v in r['own_fd'].values()),
            time_error_energy=float(np.mean(
                [d['time_error_common_normalized_squared'] for d in decomposed])),
            rhs_error_energy=float(np.mean(
                [d['rhs_error_common_normalized_squared'] for d in decomposed])),
            residual_rms=float(np.mean([d['predicted_residual_rms'] for d in decomposed])),
            oracle_time_residual_rms=float(np.mean(
                [d['oracle_time_residual_rms'] for d in decomposed])),
            oracle_field_rhs_residual_rms=float(np.mean(
                [d['oracle_field_rhs_residual_rms'] for d in decomposed])))
    checks = []
    for name, row in sorted(rows.items()):
        if not name.startswith('allen_cahn/periodic/none_s'):
            continue
        t = row['temporal']
        d = t['decomposition_resolved16']
        uncertainty = max(t['reference']['width_sensitivity']['resolved16']['mean'],
                          t['reference']['fd_vs_rhs']['resolved16']['mean'])
        checks.append(dict(
            run=name,
            nontrivial=t['truth_resolved_native']['mean'] >= .1,
            exceeds_reference_uncertainty=t['truth_resolved_native']['mean'] >= 10 * uncertainty,
            exceeds_rhs_field_error=(d['time_error_common_normalized_squared'] >
                                     d['rhs_error_common_normalized_squared']),
            oracle_time_reduces_residual=(d['oracle_time_residual_rms'] <
                                         d['predicted_residual_rms'])))
    decision = dict(
        checks=checks,
        all_three_support=(len(checks) == 3 and all(
            c['nontrivial'] and c['exceeds_reference_uncertainty'] and
            c['exceeds_rhs_field_error'] and c['oracle_time_reduces_residual']
            for c in checks)),
        meaning='Descriptive eligibility for a future time-constraint comparison; not causal field-error attribution, significance or verification.')
    return report, decision


def run():
    plan_path = OUT / 'plan.json'
    p = json.loads(plan_path.read_text())
    identity = digest(p)
    pilot_plan_path = PILOT / 'plan.json'
    if file_hash(pilot_plan_path) != p['supersedes_plan_sha256']:
        raise ValueError('sealed pilot plan changed')
    pilot_plan = json.loads(pilot_plan_path.read_text())
    if (pilot_plan['models'] != p['models'] or
            pilot_plan['references'] != p['references'] or
            pilot_plan['dev_ids'] != p['dev_ids'] or
            pilot_plan['model_fd_half_widths'] != p['model_fd_half_widths']):
        raise ValueError('precision repair changed models, references, IDs or FD widths')
    for name, expected_hash in pilot_plan['source'].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError('sealed pilot source changed: ' + name)
    expected_names = {f'{task}/{position}/{kind}_s{seed}'
                      for task in ('allen_cahn', 'burgers')
                      for position in ('linear', 'periodic')
                      for kind in ('none', 'low', 'full') for seed in range(3)}
    if set(p['models']) != expected_names:
        raise ValueError('plan must contain all 36 final paired models')
    expected_paths = {OUT / 'runs' / (name.replace('/', '__') + '.json')
                      for name in p['models']}
    if set((OUT / 'runs').glob('*.json')) != expected_paths:
        raise ValueError('incomplete or unexpected temporal rows')
    expected_pilot_names = {name for name in expected_names if name.startswith('allen_cahn/')}
    expected_pilot_paths = {PILOT / 'runs' / (name.replace('/', '__') + '.json')
                            for name in expected_pilot_names}
    if set((PILOT / 'runs').glob('*.json')) != expected_pilot_paths:
        raise ValueError('expected exactly 18 preserved Allen pilot rows')
    pilot_rows = {}
    for name in expected_pilot_names:
        path = PILOT / 'runs' / (name.replace('/', '__') + '.json')
        pilot_row = json.loads(path.read_text())
        expected_identity = dict(plan=digest(pilot_plan),
                                 model=pilot_plan['models'][name]['sha256'],
                                 reference=pilot_plan['references']['allen_cahn']['sha256'])
        if pilot_row['identity'] != expected_identity:
            raise ValueError('preserved pilot row identity changed: ' + name)
        pilot_rows[name] = pilot_row
    for name, expected_hash in p['source'].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError('sealed source changed: ' + name)
    for key, path in (
            ('parent_plan_sha256', PARENT / 'plan.json'),
            ('parent_audit_sha256', PARENT / 'audit.json'),
            ('reference_plan_sha256', ROOT / 'results/physics_diagnosis_v2/plan.json')):
        if file_hash(path) != p[key]:
            raise ValueError('sealed parent/reference record changed: ' + key)
    parent_plan = json.loads((PARENT / 'plan.json').read_text())
    parent_audit = json.loads((PARENT / 'audit.json').read_text())
    if parent_audit['complete'] is not True or parent_audit['historical_linear_tensor_matches'] != 18:
        raise ValueError('parent paired training audit is incomplete')
    reference_plan = json.loads((ROOT / 'results/physics_diagnosis_v2/plan.json').read_text())
    if digest(reference_plan) != p['reference_plan_digest'] or reference_plan['dev_ids'] != p['dev_ids']:
        raise ValueError('reference-plan/ID identity mismatch')
    if set(p['references']) != {'allen_cahn', 'burgers'}:
        raise ValueError('reference tasks changed')
    if p['model_fd_half_widths'] != [.002, .001, .0005]:
        raise ValueError('registered model FD widths changed')
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    rows = {}
    evidence = []
    repair_evidence = []
    cache = {}
    device = torch.device('cuda')
    for name, entry in p['models'].items():
        task, position, seed_arm = name.split('/')
        kind, seed_text = seed_arm.rsplit('_s', 1)
        seed = int(seed_text)
        if task not in cache:
            # Loading only: reference hash/plan/IDs and the cached data are checked
            # by this read-only helper. No reference generation is authorized.
            cache[task] = load_task(p, task)
        a, ref = cache[task]
        model_path = ROOT / entry['path']
        if file_hash(model_path) != entry['sha256']:
            raise ValueError('frozen model changed: ' + name)
        parent_path = model_path.parent.parent / 'result.json'
        if file_hash(parent_path) != entry['result_sha256']:
            raise ValueError('frozen parent model record changed: ' + name)
        parent = json.loads(parent_path.read_text())
        cp = torch.load(model_path, map_location='cpu', weights_only=False)
        if cp['identity'] != parent['identity'] or cp['step'] != cp['identity']['config']['steps']:
            raise ValueError('checkpoint/result identity or final step mismatch: ' + name)
        c = cp['identity']['config']
        context = cp['identity']['context']
        if ((context['task'], c['position'], c['kind'], c['seed']) !=
                (task, position, kind, seed) or context['plan'] != digest(parent_plan)):
            raise ValueError('checkpoint model-arm/parent-plan mismatch: ' + name)
        if parent['final_sha256'] != entry['sha256']:
            raise ValueError('parent final-checkpoint digest mismatch: ' + name)
        row_path = OUT / 'runs' / (name.replace('/', '__') + '.json')
        row = json.loads(row_path.read_text())
        row_identity = dict(plan=identity, model=entry['sha256'],
                            reference=p['references'][task]['sha256'])
        if row['identity'] != row_identity or set(row) != {'identity', 'field32', 'temporal', 'own_fd', 'git_commit'}:
            raise ValueError('temporal row identity/structure mismatch: ' + name)
        if task == 'allen_cahn':
            pilot_row = pilot_rows[name]
            for metric in ('field32', 'temporal', 'own_fd'):
                if row[metric] != pilot_row[metric]:
                    raise ValueError('precision repair changed preserved numerical dictionary: ' + name + '/' + metric)
            repair_evidence.append(dict(
                run=name, exact_numeric_dictionaries=['field32', 'temporal', 'own_fd'],
                pilot_row_sha256=file_hash(PILOT / 'runs' / (name.replace('/', '__') + '.json')),
                repaired_row_sha256=file_hash(row_path)))
        model = PositionTimeFNO1d(
            c['width'], c['modes'], c['layers'], position=c['position'],
            length=c['length'], position_center=c['position_center'],
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
        value, ut = torch.func.jvp(physical_field, (tt,), (torch.ones_like(tt),), strict=True)
        jvp_primal = value.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        tangent = ut.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        # Direct implementation-path comparison complements scalar-score replay.
        reverse_value, reverse_ut = time_fields(model, b, tt, coords, cp['scale'], off,
                                                create_graph=False)
        reverse_pred = reverse_value.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        reverse_tangent = reverse_ut.detach().cpu().numpy().astype(float).reshape(32, 4, 256)
        np.testing.assert_allclose(jvp_primal, reverse_pred, rtol=RTOL, atol=ATOL,
                                   err_msg='forward/reverse primal replay: ' + name)
        np.testing.assert_allclose(tangent, reverse_tangent, rtol=RTOL, atol=ATOL,
                                   err_msg='forward/reverse JVP replay: ' + name)
        # Preserve the parent's evaluation batch layout and float64 physical
        # rescaling. The JVP primal adds a rounded float32 offset, so it is not
        # substituted for that historical field prediction.
        all_times = torch.tensor(a['times'], dtype=torch.float32, device=device)
        with torch.no_grad():
            normalized = torch.cat([
                model(q.repeat_interleave(all_times.numel(), 0),
                      all_times.repeat(q.shape[0]), coords).reshape(
                          q.shape[0], all_times.numel(), q.shape[-1])
                for q in x.split(16)]).cpu().numpy().astype(float)
        pred = (normalized * cp['scale'] + raw.mean(1)[:, None, None])[:, 1:]
        field = temporal_score(pred, np.asarray(a['val_y'][:32, 1:], dtype=float))
        temporal = analyze(pred, tangent, ref, c)
        own = {}
        with torch.no_grad():
            for h in p['model_fd_half_widths']:
                fd = ((model(b, tt + h, coords) - model(b, tt - h, coords)) *
                      cp['scale'] / (2 * h)).cpu().numpy().astype(float).reshape(32, 4, 256)
                own[f'{h:g}'] = temporal_score(fd, tangent)
        close_structure(field, row['field32'], name + '/field32', rtol=1e-10, atol=1e-11)
        close_structure(temporal, row['temporal'], name + '/temporal')
        close_structure(own, row['own_fd'], name + '/own_fd')
        np.testing.assert_allclose(trajectory_errors(pred, a['val_y'][:32, 1:]).mean(1),
                                   parent['metrics']['val']['per_sample'][:32],
                                   rtol=1e-10, atol=1e-11, err_msg='parent first32 field replay: ' + name)
        check_energy_closure(row, name)
        check_energy_closure(dict(temporal=temporal), name + '/forward-mode')
        rows[name] = row
        evidence.append(dict(
            run=name, model_sha256=entry['sha256'], reference_sha256=row_identity['reference'],
            row_sha256=file_hash(row_path), field32_replayed=True,
            temporal_metrics_replayed=True, all_fd_widths_replayed=True,
            derivative_path='torch.func.jvp forward mode versus torch.autograd.functional.jvp reverse-over-reverse',
            forward_reverse_jvp_absolute_rms=float(np.sqrt(np.mean((tangent - reverse_tangent) ** 2))),
            forward_reverse_jvp_relative_rms=float(
                np.linalg.norm(tangent - reverse_tangent) / max(np.linalg.norm(reverse_tangent), 1e-30)),
            decomposition_energy_closed=True))
        print('TEMPORAL_AUDIT', len(rows), '/36', name, flush=True)
        del model, cp, value, ut, reverse_value, reverse_ut
    groups, decision = independent_aggregate(rows)
    if len(repair_evidence) != 18:
        raise ValueError('incomplete precision-repair reproduction evidence')
    expected_summary = dict(plan=identity, checked_models=36, groups=groups, decision=decision,
                            training_performed=False, confirmation_opened=False)
    if summary != expected_summary:
        raise ValueError('summary/decision do not exactly reproduce from saved rows')
    result = dict(
        plan=identity, plan_sha256=file_hash(plan_path), summary_sha256=file_hash(summary_path),
        audit_source_sha256=file_hash(Path(__file__)), checked_models=len(rows), complete=True,
        supersedes_plan_sha256=p['supersedes_plan_sha256'],
        sealed_pilot_auditor_source_sha256=file_hash(ROOT / 'scripts/audit_temporal_after_periodic.py'),
        repair_reproductions=len(repair_evidence), repair_evidence=repair_evidence,
        numerical_replay_tolerance=dict(rtol=RTOL, atol=ATOL),
        stored_aggregate_replay='exact, independent aggregation of stored rows',
        evidence=evidence, groups=groups, decision=decision,
        limitations='Alternate AD path and shared NumPy metric replay, not a separately derived PDE solver, causal decomposition or independent data confirmation.',
        training_performed=False, confirmation_opened=False)
    atomic_json(OUT / 'audit.json', result)
    print('TEMPORAL AUDIT COMPLETE', len(rows), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true',
                        help='Compatibility flag: this auditor always requires all 36 rows.')
    parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run()
