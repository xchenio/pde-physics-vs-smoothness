"""Training-only numerical reliability map for an Allen--Cahn interval pilot.

The extra solver trajectories are additional TRAINING information, not ordinary
supervised labels. No development or reserved case is opened by this module.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.allen_cahn_data import initial
from scripts.allen_cahn_trajectory_v1 import trajectories
from src.physics_equations_v1 import equation_residual
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save

TRAIN_IDS = list(range(200000, 200016))
CENTERS = [.025, .0875, .1875, .375, .475]
HALF_WIDTHS = [.005, .01]
PARAMS = dict(pde='allen_cahn', length=2., diffusion=.01, reaction=5.,
              resolved_max_mode=16)
FINAL_TIME = .5
CHUNK_SIZE = 8
STAGES = [('native_fine', 1024, .000125),
          ('native_coarse', 1024, .00025),
          ('spatial_fine', 512, .000125)]
FROZEN_SOURCES = ['scripts/allen_cahn_data.py', 'scripts/allen_cahn_trajectory_v1.py',
                  'src/physics_equations_v1.py', 'src/time_fno.py', 'src/fno.py',
                  'src/public_benchmark.py', 'src/resume.py']


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def interval_times(centers=CENTERS, half_widths=HALF_WIDTHS):
    return np.array(sorted({0.} | {round(float(t), 12) for center in centers
                                  for h in half_widths
                                  for t in np.linspace(center-h, center+h, 9)}), dtype=np.float64)


def _indices(times, requested):
    ix = np.abs(np.asarray(times)[:, None] - np.asarray(requested)[None, :]).argmin(0)
    if np.max(np.abs(np.asarray(times)[ix] - requested)) > 1e-10:
        raise ValueError('required interval reference time absent')
    return ix


def _average(values):
    """Composite Simpson average, not integral: the output has rate units."""
    count = values.shape[-2]
    if count < 3 or count % 2 != 1:
        raise ValueError('odd Simpson point count required')
    w = np.ones(count)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return np.sum(values * w[None, :, None], axis=-2) / (3 * (count-1))


def _rhs(values, params, band):
    c = dict(params, resolved_max_mode=band)
    field = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float64)
    # Exactly the same 4x-dealiased cubic equation as the pilot constraint.
    with torch.no_grad():
        return (-equation_residual(field, torch.zeros_like(field), c)).cpu().numpy()


def _weight_statistics(values):
    """Expose whether the implemented map differs materially from all ones."""
    a = np.asarray(values, dtype=np.float64)
    multiplicity = np.full(a.shape[-1], 2.)
    multiplicity[0] = 1.  # The resolved band excludes the grid Nyquist mode.
    mass = np.broadcast_to(multiplicity, a.shape)
    return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                std=float(a.std()), max_abs_one_deviation=float(np.max(np.abs(a-1))),
                unique_count=int(np.unique(values).size),
                parseval_weighted_mass_fraction=float(np.sum(a*mass)/np.sum(mass)),
                all_ones_exact=bool(np.all(values == 1)))


def compute_map(fine, coarse, spatial, times, target_scale, *, centers=CENTERS,
                half_widths=HALF_WIDTHS, params=PARAMS, final_time=FINAL_TIME,
                actual_n=256, band=16):
    """Pure numerical map calculation; inputs are case x time x physical grid.

    Every epsilon component is an RMS over cases of a complex Fourier-mode RATE
    discrepancy. No per-mode percentage normalization, median rescaling,
    threshold deletion, development evaluation or model outputs enter the map.
    """
    fine, coarse, spatial = [np.asarray(a, dtype=np.float64) for a in (fine, coarse, spatial)]
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(target_scale) or target_scale <= 0 or final_time <= 0:
        raise ValueError('positive finite target scale and final time required')
    if times.ndim != 1 or len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('increasing reference time axis required')
    if fine.ndim != 3 or coarse.shape != fine.shape or spatial.shape[:2] != fine.shape[:2]:
        raise ValueError('reference case/time shapes differ')
    if fine.shape[1] != len(times) or fine.shape[0] < 1:
        raise ValueError('reference case/time count invalid')
    if actual_n <= 0 or fine.shape[-1] % actual_n or not 0 <= band < min(actual_n, spatial.shape[-1])//2:
        raise ValueError('invalid actual grid or resolved band')
    if not all(np.isfinite(a).all() for a in (fine, coarse, spatial)):
        raise ValueError('nonfinite reference fields')
    names = ['timestep_endpoint_rate', 'timestep_rhs_average', 'space_endpoint_rate',
             'space_rhs_average', 'simpson5_vs9_rhs_average', 'native_closure',
             'actual_grid_rhs_average']
    terms = {name: np.empty((len(centers), len(half_widths), band+1)) for name in names}
    cf = lambda a: np.fft.rfft(a, axis=-1, norm='forward')[..., :band+1]
    rms = lambda a: np.sqrt(np.mean(np.abs(a)**2, axis=0))
    for i, center in enumerate(centers):
        for j, h in enumerate(half_widths):
            if not np.isfinite(h) or h <= 0 or center-h < 0 or center+h > final_time:
                raise ValueError('invalid reference interval')
            ix = _indices(times, np.round(np.linspace(center-h, center+h, 9), 12))
            f, c, s = fine[:, ix], coarse[:, ix], spatial[:, ix]
            f_rate = (cf(f[:, -1])-cf(f[:, 0]))/(2*h)
            c_rate = (cf(c[:, -1])-cf(c[:, 0]))/(2*h)
            s_rate = (cf(s[:, -1])-cf(s[:, 0]))/(2*h)
            fr = _rhs(f, params, band)
            f_rhs = _average(fr)
            c_rhs = _average(_rhs(c, params, band))
            s_rhs = _average(_rhs(s, params, band))
            actual_rhs = _average(_rhs(f[..., ::fine.shape[-1]//actual_n], params, band))
            errors = [f_rate-c_rate, f_rhs-c_rhs, f_rate-s_rate, f_rhs-s_rhs,
                      f_rhs-_average(fr[:, ::2]), f_rate-f_rhs, actual_rhs-f_rhs]
            for name, error in zip(names, errors):
                terms[name][i, j] = rms(error)
    epsilon = np.maximum.reduce(list(terms.values()))
    tau = .001*float(target_scale)/float(final_time)
    weights64 = tau*tau/(tau*tau+epsilon*epsilon)
    if not np.isfinite(weights64).all() or np.any(weights64 <= 0) or np.any(weights64 > 1):
        raise ValueError('continuous reliability weights outside (0,1]')
    weights32 = weights64.astype(np.float32)
    if not np.isfinite(weights32).all() or np.any(weights32 <= 0) or np.any(weights32 > 1):
        raise ValueError('float32 reliability weights outside (0,1]; possible cast underflow')
    return dict(centers=list(centers), half_widths=list(half_widths), band=band,
                weights64=weights64, weights32=weights32,
                epsilon=epsilon, tau=tau, diagnostics=dict(rms_rate_components=terms,
                    epsilon_rule='per center/half_width/mode maximum across seven case-RMS rate errors',
                    units='physical field / time; no Parseval multiplicity applied to per-mode epsilon',
                    calibration_case_count=int(fine.shape[0]), actual_grid_size=actual_n,
                    weight64_statistics=_weight_statistics(weights64),
                    weight32_statistics=_weight_statistics(weights32),
                    global_scale_note='A globally constant positive weight is cancelled by training-gradient calibration; only relative variation can distinguish this arm from uniform weights. No variation is amplified.',
                    median_rescaling=False, hard_thresholding=False))


def _identity(train_ids, target_scale, context):
    return dict(version=1, source_sha256=file_hash(Path(__file__)),
                frozen_sources={name:file_hash(ROOT/name) for name in FROZEN_SOURCES},
                train_ids=train_ids, target_scale=float(target_scale), context=context,
                centers=CENTERS, half_widths=HALF_WIDTHS, params=PARAMS,
                final_time=FINAL_TIME, actual_grid=256, simpson_points=9,
                comparison_simpson_points=5, chunk_size=CHUNK_SIZE, stages=STAGES,
                tau_rule='.001 * training_target_std / T',
                access_policy='TRAINING-ONLY additional solver information. No dev/reserved IDs, values, statistics, model predictions or decisions.')


def _validate_stage(obj, identity, stage, ids, times, n):
    expected = dict(map_identity=_digest(identity), stage=stage, ids=ids, times=times.tolist())
    if obj['identity'] != expected:
        raise ValueError('reference stage identity changed')
    values = obj['values']
    if values.dtype != np.float64 or values.shape != (len(ids), len(times), n) or not np.isfinite(values).all():
        raise ValueError('invalid reference stage values')
    if _array_digest(values) != obj['values_sha256']:
        raise ValueError('reference stage array SHA-256 changed')


def _map_payload(obj):
    return {name:obj[name] for name in ('identity', 'centers', 'half_widths', 'band',
                                      'weights64', 'weights32', 'epsilon', 'tau',
                                      'diagnostics', 'provenance', 'hashes', 'seconds')}


def _validate_map(obj, identity, folder):
    if obj['identity'] != identity:
        raise ValueError('weight map identity changed')
    shape = (len(CENTERS), len(HALF_WIDTHS), PARAMS['resolved_max_mode']+1)
    if obj['weights64'].shape != shape or obj['weights64'].dtype != np.float64:
        raise ValueError('invalid weight64 shape or dtype')
    if obj['weights32'].dtype != np.float32 or not np.array_equal(obj['weights32'], obj['weights64'].astype(np.float32)):
        raise ValueError('weight32 is not the exact weight64 cast')
    if not np.isfinite(obj['weights32']).all() or np.any(obj['weights32'] <= 0) or np.any(obj['weights32'] > 1):
        raise ValueError('float32 reliability weights outside (0,1]')
    expected_tau = .001*identity['target_scale']/FINAL_TIME
    if obj['tau'] != expected_tau:
        raise ValueError('weight-map tau changed')
    expected = expected_tau**2/(expected_tau**2+obj['epsilon']**2)
    if not np.array_equal(obj['weights64'], expected):
        raise ValueError('weight-map continuous shrinkage changed')
    for name, expected_sha in obj['hashes']['stages'].items():
        if file_hash(folder/name) != expected_sha:
            raise ValueError('reference stage file SHA-256 changed')
    if obj['payload_sha256'] != _digest(_json_value(_map_payload(obj))):
        raise ValueError('weight map payload SHA-256 changed')


def build_reference(folder, train_ids, target_scale, context):
    """Return the immutable, resumable, TRAINING-only continuous weight map."""
    folder = Path(folder)
    ids = [int(i) for i in train_ids]
    if ids != TRAIN_IDS:
        raise ValueError('exact first16 TRAIN IDs200000..200015 required; dev/reserves forbidden')
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError('positive finite training target std required')
    # JSON roundtrip freezes the passed context/parameters against later mutation.
    identity = json.loads(json.dumps(_identity(ids, target_scale, context), allow_nan=False))
    folder.mkdir(parents=True, exist_ok=True)
    plan_path = folder/'identity.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != identity:
            raise ValueError('reference folder identity changed')
    else:
        atomic_json(plan_path, identity)
    path = folder/'weight_map.pt'
    json_path = folder/'weight_map.json'
    if path.exists():
        obj = torch.load(path, map_location='cpu', weights_only=False)
        _validate_map(obj, identity, folder)
        # Repair only a missing JSON companion after an interrupted atomic save.
        expected_json = _json_value(obj)
        if json_path.exists():
            if json.loads(json_path.read_text()) != expected_json:
                raise ValueError('weight map JSON companion changed')
        else:
            atomic_json(json_path, expected_json)
        return obj
    times = interval_times()
    pieces = {stage:[] for stage, _, _ in STAGES}
    stage_hashes = {}
    seconds = 0.
    count = 0
    total = len(STAGES)*((len(ids)+CHUNK_SIZE-1)//CHUNK_SIZE)
    for start in range(0, len(ids), CHUNK_SIZE):
        part = ids[start:start+CHUNK_SIZE]
        for stage, n, dt in STAGES:
            stage_path = folder/f'{stage}_chunk_{start//CHUNK_SIZE:04d}.pt'
            if stage_path.exists():
                record = torch.load(stage_path, map_location='cpu', weights_only=False)
                _validate_stage(record, identity, stage, part, times, n)
            else:
                started = time.perf_counter()
                values = trajectories(initial(part, n), times, dt=dt,
                                      length=PARAMS['length'], nu=PARAMS['diffusion'],
                                      rho=PARAMS['reaction'])
                record = dict(identity=dict(map_identity=_digest(identity), stage=stage,
                              ids=part, times=times.tolist()), values=values,
                              values_sha256=_array_digest(values), seconds=time.perf_counter()-started)
                _validate_stage(record, identity, stage, part, times, n)
                atomic_save(record, stage_path)
            pieces[stage].append(record['values'])
            stage_hashes[stage_path.name] = file_hash(stage_path)
            seconds += float(record['seconds'])
            count += 1
            atomic_json(folder/'progress.json', dict(identity=_digest(identity), completed_stages=count,
                        total_stages=total, last_stage=stage, last_chunk=start//CHUNK_SIZE,
                        completed_stage_seconds=seconds, complete=False, stage_sha256=stage_hashes,
                        training_only_extra_solver_information=True, confirmation_opened=False))
    started = time.perf_counter()
    obj = compute_map(*(np.concatenate(pieces[name], axis=0)
                        for name in ('native_fine', 'native_coarse', 'spatial_fine')),
                      times, target_scale)
    obj.update(identity=identity, hashes=dict(stages=stage_hashes),
               seconds=seconds+time.perf_counter()-started,
               provenance=dict(training_only_extra_solver_information=True,
                   train_ids=ids, no_development_or_reserved_access=True,
                   times=times.tolist(), sources=identity['frozen_sources'],
                   source_sha256=identity['source_sha256'],
                   spatial_policy='actual256 is native1024[...,::4]; spatial512 is an independent solve from same training IDs',
                   rhs_policy='src.physics_equations_v1.equation_residual, torch float64, 4x dealiased cubic',
                   calibration='No model outputs, development errors, thresholds, median normalization or learned map parameters'))
    obj['payload_sha256'] = _digest(_json_value(_map_payload(obj)))
    _validate_map(obj, identity, folder)
    atomic_save(obj, path)
    atomic_json(json_path, _json_value(obj))
    atomic_json(folder/'progress.json', dict(identity=_digest(identity), completed_stages=count,
                total_stages=total, complete=True, seconds=obj['seconds'],
                weight_map_sha256=file_hash(path), weight_map_json_sha256=file_hash(json_path),
                training_only_extra_solver_information=True, confirmation_opened=False))
    return obj
