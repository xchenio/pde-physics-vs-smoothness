"""Read-only reused Allen data for a new DEVELOPMENT split and sparse labels.

The six historical caches contain all five labels and are loaded for integrity
and reindexing. Only observations()'s selected training labels enter fitting
statistics or losses. No generator, solver, confirmation cache or writer is used.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.public_benchmark import file_hash

POOL_IDS = list(range(200000, 200300))
SPLIT_SEED = 2026101601
TIMES = [0., .05, .125, .25, .5]
PARENT_PLAN = Path('results/physics_stress_v1/plan.json')
CACHE_ROOT = Path('external_data/physics_stress_v1')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def split_ids(seed=SPLIT_SEED):
    """New split of exactly the previously consumed300 IDs, no fresh cases."""
    order = np.random.default_rng(seed).permutation(np.arange(200000, 200300))
    return dict(train_ids=np.sort(order[:100]).tolist(), val_ids=np.sort(order[100:]).tolist())


def _validate_split(p):
    train = [int(i) for i in p['train_ids']]
    val = [int(i) for i in p['val_ids']]
    if len(train) != 100 or len(val) != 200 or len(set(train)) != 100 or len(set(val)) != 200:
        raise ValueError('replication requires100 unique train and200 unique development IDs')
    if set(train) & set(val) or set(train) | set(val) != set(POOL_IDS):
        raise ValueError('split must partition exactly the previously used300 IDs; reserves forbidden')
    return train, val


def load_cached_split(p):
    """Return (clean, context) from six verified historical caches, read-only.

    This deliberately does not call historical clean_data: a missing file is a
    hard failure, never permission to generate trajectories or mutate progress.
    """
    train_ids, val_ids = _validate_split(p)
    parent_path = ROOT/PARENT_PLAN
    parent_sha = file_hash(parent_path)
    if 'parent_plan_sha256' in p and p['parent_plan_sha256'] != parent_sha:
        raise ValueError('pinned parent plan SHA-256 changed')
    pinned_parents = p.get('parents', {})
    if PARENT_PLAN.as_posix() in pinned_parents and pinned_parents[PARENT_PLAN.as_posix()] != parent_sha:
        raise ValueError('pinned parent plan SHA-256 changed')
    old = json.loads(parent_path.read_text())
    spec = old['tasks']['allen_cahn']
    if (spec['train_ids'] != list(range(200000, 200100)) or
            spec['val_ids'] != list(range(200100, 200300)) or spec['times'] != TIMES):
        raise ValueError('historical Allen pool or time grid changed')
    old_digest = _digest(old)
    pool_ids, inputs, targets, hashes = [], [], [], {}
    for split, key in [('train', 'train_ids'), ('val', 'val_ids')]:
        ids = spec[key]
        for start in range(0, len(ids), 50):
            part = ids[start:start+50]
            relative = CACHE_ROOT/f'allen_{split}_{start:04d}.pt'
            path = ROOT/relative
            # file_hash naturally fails if a cache is absent; there is no
            # fallback solver, mkdir, save, progress or QA call in this loader.
            sha = file_hash(path)
            obj = torch.load(path, map_location='cpu', weights_only=False)
            if obj['plan'] != old_digest or obj['ids'] != part:
                raise ValueError('historical cache plan digest or chunk IDs changed')
            x, y = np.asarray(obj['x']), np.asarray(obj['y'])
            if x.shape != (len(part), 256) or y.shape != (len(part), 5, 256):
                raise ValueError('historical Allen cache shape changed')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('nonfinite historical cache')
            if not np.array_equal(x, y[:, 0]):
                raise ValueError('historical initial field differs from t0 label')
            pool_ids.extend(part)
            inputs.append(x)
            targets.append(y)
            hashes[relative.as_posix()] = sha
    if pool_ids != POOL_IDS:
        raise ValueError('historical pool ordering or completeness changed')
    raw_x = np.concatenate(inputs, axis=0)
    raw_y = np.concatenate(targets, axis=0)
    lookup = {i:j for j, i in enumerate(pool_ids)}
    clean = dict(times=TIMES.copy(), grid256=np.arange(256, dtype=float)*2/256,
                 train_ids=train_ids, val_ids=val_ids)
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        ix = [lookup[i] for i in ids]
        clean[split+'_x'] = np.array(raw_x[ix], dtype=float, copy=True)
        clean[split+'_y'] = np.array(raw_y[ix], dtype=float, copy=True)
    context = dict(parent_plan=old_digest, parent_plan_sha256=parent_sha,
                   parent_plan_path=PARENT_PLAN.as_posix(), chunks=hashes,
                   train_ids=train_ids, val_ids=val_ids, pool_ids=POOL_IDS,
                   reused_previously_consumed_cases_only=True, new_cases_accessed=False,
                   new_solver_performed=False, cache_generated=False, cache_mutated=False,
                   new_split_is_development_not_independent_confirmation=True,
                   label_access_policy='All five cached labels loaded for integrity/reindexing, not all used for fitting; observations selects before computing any training statistic.')
    return clean, context


def observations(clean, condition):
    """Select training supervision before calculating any training statistic."""
    if condition not in ('dense', 'sparse'):
        raise ValueError('replication condition must be dense or sparse')
    if list(clean['times']) != TIMES:
        raise ValueError('original five-time observation grid required')
    columns = list(range(5)) if condition == 'dense' else [0, 4]
    # No omitted training-time values or development values enter reductions,
    # normalization, scale, hash proofs or any quantity used to fit the model.
    raw = np.array(clean['train_x'], dtype=float, copy=True)
    selected = np.array(clean['train_y'][:, columns], dtype=float, copy=True)
    if raw.ndim != 2 or selected.shape != (len(raw), len(columns), raw.shape[-1]):
        raise ValueError('selected training observation shape mismatch')
    if not np.isfinite(raw).all() or not np.isfinite(selected).all():
        raise ValueError('nonfinite selected training observations')
    norm = dict(mean=float(raw.mean()), std=float(raw.std()))
    scale = float(selected.std())
    if min(norm['std'], scale) <= 0:
        raise ValueError('degenerate selected training observations')
    normalized_x = (raw-norm['mean'])/norm['std']
    normalized_y = (selected-raw.mean(1)[:, None, None])/scale
    a = dict(train_x=raw, train_y=selected,
             val_x=np.array(clean['val_x'], dtype=float, copy=True),
             val_y=np.array(clean['val_y'], dtype=float, copy=True),
             times=[TIMES[i] for i in columns], val_times=TIMES.copy(),
             grid256=np.array(clean['grid256'], dtype=float, copy=True),
             train_ids=list(clean['train_ids']), val_ids=list(clean['val_ids']))
    proof = dict(input_normalization=norm, selected_target_scale=scale,
                 training_inputs_sha256=_array_digest(raw),
                 selected_targets_sha256=_array_digest(selected),
                 normalized_inputs_sha256=_array_digest(normalized_x),
                 normalized_targets_sha256=_array_digest(normalized_y))
    obs = dict(condition=condition, observation_columns=columns,
               observed_train_ids=list(clean['train_ids']),
               observation_times=a['times'], observation_sha256=_digest(selected.tolist()),
               input_normalization=norm, selected_target_scale=scale,
               normalization_proof=proof, normalization_sha256=_digest(proof),
               heldout_training_labels_accessed_for_stats=False,
               development_labels_accessed_for_training_stats=False,
               physical_cache_contains_all_five_labels=True,
               input_noise=False, initial_label_noise=False, noise_sigma=0.,
               policy='Only selected TRAIN labels determine target std, normalized loss targets and their proof hashes. Cached omitted labels are loaded/integrity-checked but are not fitted or summarized. Development labels remain evaluation-only.')
    return a, obs
