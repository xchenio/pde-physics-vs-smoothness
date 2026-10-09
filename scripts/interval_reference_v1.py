"""Chunk-resumable, development-only references for fixed interval diagnostics.

This module never trains a model, changes historical solvers, or opens a
confirmation split. The caller must seal its plan before calling build_reference.
"""
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.allen_cahn_data import initial
from scripts.allen_cahn_trajectory_v1 import trajectories
from scripts.run_public_ridge_residual import digest
from src.public_benchmark import atomic_json, file_hash
from src.resume import atomic_save

OUTDATA_ROOT = ROOT / 'external_data/interval_diagnosis_v1'
CHUNK_SIZE = 8
ALLEN_FINE_DT = .0000625
ALLEN_COARSE_DT = .000125


def reference_times(p, task):
    """Return nominal, rounded times, not float32 HDF coordinate roundoff."""
    spec = p['tasks'][task]
    values = [0., *spec['label_times']]
    for window in spec['windows']:
        values.extend(window['reference_times'])
    result = np.array(sorted({round(float(t), 12) for t in values}), dtype=np.float64)
    if not np.isfinite(result).all() or result[0] != 0. or np.any(np.diff(result) <= 0):
        raise ValueError('finite, increasing reference times starting at zero required')
    dt = .01 if task == 'burgers' else ALLEN_COARSE_DT
    if np.max(np.abs(np.rint(result / dt) * dt - result)) > 1e-10:
        raise ValueError('reference times do not align with the pinned reference grid')
    return result


def time_indices(times, requested, atol=1e-10):
    """Strict nearest lookup; reject accidental time substitution."""
    times = np.asarray(times, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('invalid source time axis')
    indices = np.abs(times[:, None] - requested.reshape(1, -1)).argmin(0)
    if np.any(np.abs(times[indices] - requested.reshape(-1)) > atol):
        raise ValueError('requested time not present in reference')
    return indices


def _source_check(p):
    for name, expected in p['source'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('sealed reference source changed: ' + name)


def _validate_values(values, ids, times, n):
    values = np.asarray(values)
    if values.dtype != np.float64 or values.shape != (len(ids), len(times), n):
        raise ValueError('invalid reference shape or dtype')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite reference')


def _identity(p, task, ids, times):
    return dict(plan=digest(p), task=task, ids=ids, times=times.tolist())


def _stages(task):
    if task == 'burgers':
        return [('native_fine', 1024, None)]
    return [('native_fine', 1024, ALLEN_FINE_DT),
            ('native_coarse', 1024, ALLEN_COARSE_DT),
            ('independent512_fine', 512, ALLEN_FINE_DT)]


def _dataset_path(p):
    path = Path(p['dataset']['path'])
    return path if path.is_absolute() else ROOT / path


def _burgers_read(p, ids, times):
    # build_reference checks the pinned whole-file SHA before the first open.
    with h5py.File(_dataset_path(p), 'r') as source:
        data = source['tensor']
        if data.ndim != 3 or data.shape[-1] != 1024:
            raise ValueError('unexpected native Burgers layout')
        if any(i < 0 or i >= data.shape[0] for i in ids):
            raise ValueError('Burgers development ID out of range')
        indices = np.rint(times / .01).astype(int)
        if indices[-1] >= data.shape[1]:
            raise ValueError('Burgers reference time beyond stored solution')
        stored = np.asarray(source['t-coordinate'][indices], dtype=np.float64)
        # The publisher stores float32 coordinates; preserve the nominal .01
        # grid for interval lookup, and disclose the raw coordinates separately.
        if np.max(np.abs(stored - times)) > 2e-7:
            raise ValueError('Burgers stored time coordinates changed')
        values = np.stack([data[int(i), indices, :] for i in ids]).astype(np.float64)
    return values, stored.tolist()


def _validate_final(obj, p, task, ids, times):
    if obj['identity'] != _identity(p, task, ids, times) or obj['plan'] != digest(p):
        raise ValueError('reference identity changed')
    if obj['ids'] != ids or not np.array_equal(obj['times'], times):
        raise ValueError('reference IDs or times changed')
    for n, values in obj['fields'].items():
        _validate_values(values, ids, times, n)
    if not np.array_equal(obj['fields'][256], obj['fields'][1024][..., ::4]):
        raise ValueError('primary256 is not the declared native subsampling')
    if task == 'allen_cahn':
        _validate_values(obj['coarse_fields'][1024], ids, times, 1024)
    for name, expected in obj['chunk_sha256'].items():
        if file_hash(OUTDATA_ROOT / name) != expected:
            raise ValueError('cached reference chunk changed')


def build_reference(p, task):
    """Build/reload a sealed task reference; return (object, file SHA-256).

    Each eight-case chunk is atomically saved after each independently solved
    grid/time-step stage. An interruption loses at most the current stage of
    one chunk. Completed stages and completed references are never overwritten.
    """
    if task not in ('allen_cahn', 'burgers'):
        raise ValueError('unsupported reference PDE')
    _source_check(p)
    ids = [int(i) for i in p['dev_ids'][task]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('nonempty unique development IDs required')
    times = reference_times(p, task)
    path = OUTDATA_ROOT / f'{task}_reference.pt'
    if path.exists():
        obj = torch.load(path, map_location='cpu', weights_only=False)
        _validate_final(obj, p, task, ids, times)
        return obj, file_hash(path)

    stages = _stages(task)
    pieces = {name: [] for name, _, _ in stages}
    chunk_hashes = {}
    dataset_verified = False
    stored_times = None
    completed_stages = 0
    total_chunks = (len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_index, start in enumerate(range(0, len(ids), CHUNK_SIZE)):
        chunk_ids = ids[start:start + CHUNK_SIZE]
        chunk_path = OUTDATA_ROOT / f'{task}_chunk_{chunk_index:04d}.pt'
        identity = _identity(p, task, chunk_ids, times)
        if chunk_path.exists():
            chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
            if chunk['identity'] != identity:
                raise ValueError('partial reference identity changed')
        else:
            chunk = dict(identity=identity, stages={})
        if set(chunk['stages']) - set(pieces):
            raise ValueError('unknown cached reference stage')
        for name, n, dt in stages:
            if name not in chunk['stages']:
                if task == 'burgers':
                    if not dataset_verified:
                        if file_hash(_dataset_path(p)) != p['dataset']['sha256']:
                            raise ValueError('Burgers pinned dataset SHA-256 mismatch')
                        dataset_verified = True
                    values, stored = _burgers_read(p, chunk_ids, times)
                    chunk['stored_time_coordinates'] = stored
                else:
                    spec = p['task_spec'][task]
                    values = trajectories(initial(chunk_ids, n), times, dt=dt,
                                          length=spec['length'], nu=spec['diffusion'],
                                          rho=spec['reaction'])
                _validate_values(values, chunk_ids, times, n)
                chunk['stages'][name] = values
                atomic_save(chunk, chunk_path)
            values = chunk['stages'][name]
            _validate_values(values, chunk_ids, times, n)
            pieces[name].append(values)
            completed_stages += 1
            atomic_json(OUTDATA_ROOT / f'{task}_reference_progress.json',
                        dict(plan=digest(p), task=task, completed_stages=completed_stages,
                             total_stages=total_chunks * len(stages),
                             chunk_index=chunk_index, stage=name, complete=False,
                             training_performed=False, confirmation_opened=False))
        if task == 'burgers':
            if stored_times is not None and stored_times != chunk['stored_time_coordinates']:
                raise ValueError('stored Burgers time coordinates differ across chunks')
            stored_times = chunk['stored_time_coordinates']
        chunk_hashes[chunk_path.name] = file_hash(chunk_path)

    native = np.concatenate(pieces['native_fine'], axis=0)
    fields = {1024: native, 256: native[..., ::4].copy()}
    coarse = {}
    if task == 'allen_cahn':
        fields[512] = np.concatenate(pieces['independent512_fine'], axis=0)
        coarse[1024] = np.concatenate(pieces['native_coarse'], axis=0)
    labels = {round(float(t), 12) for t in p['tasks'][task]['label_times']}
    extra = [float(t) for t in times if round(float(t), 12) not in labels]
    provenance = dict(source=p['source'], task_parameters=p['task_spec'][task],
                      primary_grid_policy='256 is point subsampling native1024[...,::4], not an independent PDE solve',
                      development_access='Existing sealed development IDs only; additional times are diagnostic access, never labels or training data',
                      diagnostic_times_not_in_supervised_labels=extra,
                      newly_opened_ids=[], training_performed=False, confirmation_opened=False)
    if task == 'allen_cahn':
        provenance.update(fine_dt=ALLEN_FINE_DT, coarse_dt=ALLEN_COARSE_DT,
                          independent512_dt=ALLEN_FINE_DT,
                          solver='unchanged deterministic Fourier Strang solver; pinned initial IDs')
    else:
        provenance.update(dataset=p['dataset'], nominal_stored_dt=.01,
                          raw_stored_time_coordinates=stored_times,
                          solver='publisher HDF snapshots only; no new Burgers solver')
    obj = dict(identity=_identity(p, task, ids, times), plan=digest(p), ids=ids,
               times=times, fields=fields, coarse_fields=coarse,
               provenance=provenance, chunk_sha256=chunk_hashes,
               extra_development_time_snapshots=times.tolist())
    _validate_final(obj, p, task, ids, times)
    atomic_save(obj, path)
    atomic_json(OUTDATA_ROOT / f'{task}_reference_progress.json',
                dict(plan=digest(p), task=task, completed_stages=completed_stages,
                     total_stages=total_chunks * len(stages), complete=True,
                     reference_sha256=file_hash(path), training_performed=False,
                     confirmation_opened=False))
    return obj, file_hash(path)
