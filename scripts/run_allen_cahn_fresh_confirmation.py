"""Generate and evaluate a fresh, resumable Allen--Cahn confirmation set.

The 24 fitted checkpoints are frozen. New trajectories are never used for
training, calibration, model selection, or early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.allen_cahn_data import PARAMS, initial
from scripts.allen_cahn_trajectory_v1 import qa as solver_qa, trajectories
from scripts.field_derivative_diagnostics import band_errors
from scripts.physics_token_core import trajectory_errors
from scripts.run_public_ridge_residual import digest, locked_json
from src.periodic_position_fno import PositionTimeFNO1d
from src.public_benchmark import atomic_json, file_hash


OUT = ROOT / "results/allen_cahn_fresh_confirmation_v1"
MODELS_ROOT = ROOT / "results/allen_pointwise_control_v1"
CONDITIONS = ["dense", "sparse"]
ARMS = ["none", "full", "interval_uniform", "pointwise_nodes"]
SEEDS = [0, 1, 2]
TIMES = np.asarray([0.0, 0.05, 0.125, 0.25, 0.5], dtype=np.float64)
GRID = np.arange(256, dtype=np.float64) * (2.0 / 256)
CONFIRM_IDS = list(range(310000, 310200))
QA_IDS = list(range(300000, 300016))
OLD_POOL = list(range(200000, 200300))
CHUNK_SIZE = 20
BATCH_SIZE = 16
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026102911


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def model_hash(state):
    return hashlib.sha256(b"".join(v.detach().cpu().contiguous().numpy().tobytes()
                                   for v in state.values())).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def log_event(event, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"time_local": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **fields}
    path = OUT / "events.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _model_entries(parent_plan, parent_audit):
    entries = {}
    for condition in CONDITIONS:
        for arm in ARMS:
            for seed in SEEDS:
                key = f"{condition}/{arm}_s{seed}"
                evidence = parent_audit["evidence"].get(key)
                if evidence is None:
                    raise ValueError("missing audited model: " + key)
                folder = MODELS_ROOT / condition / f"{arm}_s{seed}"
                result_path = folder / "result.json"
                checkpoint_path = folder / "checkpoints/final.pt"
                manifest_path = folder / "manifest.json"
                for path_key, path in [("result_sha256", result_path),
                                       ("final_sha256", checkpoint_path),
                                       ("manifest_sha256", manifest_path)]:
                    if not path.is_file() or file_hash(path) != evidence[path_key]:
                        raise ValueError(f"audited model artifact changed: {key}/{path_key}")
                result = read_json(result_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                config = result["identity"]["config"]
                if checkpoint["identity"] != result["identity"]:
                    raise ValueError("checkpoint/result identity mismatch: " + key)
                if config["kind"] != arm or config["seed"] != seed:
                    raise ValueError("unexpected arm or seed in frozen model: " + key)
                entries[key] = dict(
                    condition=condition,
                    arm=arm,
                    seed=seed,
                    result_path=result_path.relative_to(ROOT).as_posix(),
                    result_sha256=evidence["result_sha256"],
                    checkpoint_path=checkpoint_path.relative_to(ROOT).as_posix(),
                    checkpoint_sha256=evidence["final_sha256"],
                    manifest_path=manifest_path.relative_to(ROOT).as_posix(),
                    manifest_sha256=evidence["manifest_sha256"],
                    weights_sha256=model_hash(checkpoint["model"]),
                    config=config,
                    normalization=checkpoint["normalization"],
                    target_scale=float(checkpoint["scale"]),
                    initial_model_hash=result["initial"],
                )
    if len(entries) != 24:
        raise ValueError("expected all 24 frozen checkpoints")
    return entries


def plan(write=True):
    parent_plan_path = MODELS_ROOT / "plan.json"
    parent_audit_path = MODELS_ROOT / "audit.json"
    parent_summary_path = MODELS_ROOT / "summary.json"
    parent_plan = read_json(parent_plan_path)
    parent_audit = read_json(parent_audit_path)
    parent_summary = read_json(parent_summary_path)
    if (parent_audit.get("checked_runs") != 24 or
            parent_audit.get("exact_stream_replays") != 24 or
            len(parent_audit.get("evidence", {})) != 24 or
            parent_summary.get("completed_runs") != 24):
        raise ValueError("the full audited 24-model parent experiment is required")

    if set(CONFIRM_IDS) & (set(OLD_POOL) | set(QA_IDS)):
        raise ValueError("confirmation IDs overlap a prior or solver-QA set")
    if CONFIRM_IDS != list(range(310000, 310200)):
        raise ValueError("confirmation ID range changed")

    source_names = [
        "scripts/run_allen_cahn_fresh_confirmation.py",
        "docs/ALLEN_CAHN_FRESH_CONFIRMATION_V1.md",
        "scripts/allen_cahn_data.py",
        "scripts/allen_cahn_trajectory_v1.py",
        "scripts/field_derivative_diagnostics.py",
        "scripts/physics_token_core.py",
        "src/periodic_position_fno.py",
        "src/time_fno.py",
        "src/public_benchmark.py",
    ]
    sources = {name: file_hash(ROOT / name) for name in source_names}
    parents = {
        parent_plan_path.relative_to(ROOT).as_posix(): file_hash(parent_plan_path),
        parent_audit_path.relative_to(ROOT).as_posix(): file_hash(parent_audit_path),
        parent_summary_path.relative_to(ROOT).as_posix(): file_hash(parent_summary_path),
    }
    model_entries = _model_entries(parent_plan, parent_audit)
    training_cache_hashes = {
        name: sha for name, sha in parent_plan["data_chunks"].items()
        if "allen_train_" in name
    }
    if len(training_cache_hashes) != 2:
        raise ValueError("expected the two old Allen--Cahn training cache chunks")

    payload = dict(
        version="allen_cahn_fresh_confirmation_v1",
        protocol="frozen-model independent-trajectory confirmation; evaluation only",
        prepared_date="2026-10-29",
        parent_plan_sha256=digest(parent_plan),
        parent_audit_sha256=file_hash(parent_audit_path),
        parent_summary_sha256=file_hash(parent_summary_path),
        sources=sources,
        parents=parents,
        training_cache_hashes=training_cache_hashes,
        models=model_entries,
        conditions=CONDITIONS,
        arms=ARMS,
        seeds=SEEDS,
        old_trajectory_ids=OLD_POOL,
        solver_qa_ids=QA_IDS,
        confirmation_ids=CONFIRM_IDS,
        generation=dict(
            initial_condition_source="scripts/allen_cahn_data.py::initial",
            base_seed=int(PARAMS["seed"]),
            modes=int(PARAMS["modes"]),
            grid_size=256,
            length=2.0,
            times=TIMES.tolist(),
            dt=0.0005,
            diffusion=0.01,
            reaction=5.0,
            solver="Strang-splitting Fourier solver in scripts/allen_cahn_trajectory_v1.py",
            chunk_size=CHUNK_SIZE,
        ),
        evaluation=dict(
            batch_size=BATCH_SIZE,
            dense_primary_columns=[1, 2, 3, 4],
            sparse_primary_columns=[1, 2, 3],
            common_interior_columns=[1, 2, 3],
            all_time_columns=[0, 1, 2, 3, 4],
            derivative_diagnostic="band_errors from the existing shared numerical helper; denominator floors use the original 100 training initial fields only",
            no_training=True,
            no_calibration=True,
            no_model_selection=True,
        ),
        primary_comparison=dict(
            arm_a="interval_uniform",
            arm_b="pointwise_nodes",
            estimand="paired per-trajectory relative-L2 difference, interval minus pointwise, averaged over the three frozen model seeds",
            conditions=["dense", "sparse"],
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
            two_sided_per_condition_confidence=0.975,
            familywise_alpha=0.05,
            adjustment="Bonferroni across the two condition-specific primary contrasts",
            interpretation="no equivalence margin or success threshold; no formal equivalence claim",
        ),
        secondary_comparisons="Report all four arms and all seeds; contrasts against none and full curvature are descriptive.",
        evidence_boundary="The confirmation trajectories are from the declared existing initial-condition distribution but have new independent IDs and are never used for model development. QA cases are disjoint. The confirmation set is consumed once evaluation begins.",
    )
    if write:
        locked_json(OUT / "plan.json", payload)
    return payload


def verify_plan(p):
    for name, expected in {**p["sources"], **p["parents"], **p["training_cache_hashes"]}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("sealed dependency changed: " + name)
    if digest(p) != digest(read_json(OUT / "plan.json")):
        raise ValueError("locked plan changed")
    if set(p["confirmation_ids"]) & (set(p["old_trajectory_ids"]) | set(p["solver_qa_ids"])):
        raise ValueError("confirmed set overlaps a prior or QA ID")
    for key, entry in p["models"].items():
        for path_key, hash_key in [("result_path", "result_sha256"),
                                   ("checkpoint_path", "checkpoint_sha256"),
                                   ("manifest_path", "manifest_sha256")]:
            if file_hash(ROOT / entry[path_key]) != entry[hash_key]:
                raise ValueError("frozen model receipt changed: " + key + "/" + path_key)


def _save_chunk(path, ids, x, y):
    atomic_npz(path, ids=np.asarray(ids, dtype=np.int64), x=np.asarray(x, dtype=np.float64),
               y=np.asarray(y, dtype=np.float64), times=TIMES, grid=GRID)


def prepare():
    p = read_json(OUT / "plan.json")
    verify_plan(p)
    OUT.mkdir(parents=True, exist_ok=True)
    qa_path = OUT / "solver_qa.json"
    if qa_path.is_file():
        q = read_json(qa_path)
        if q.get("plan_sha256") != digest(p) or q.get("solver_sha256") != p["sources"]["scripts/allen_cahn_trajectory_v1.py"]:
            raise ValueError("stored solver QA receipt does not match the locked protocol")
        qa_result = q["qa"]
    else:
        qa_result = solver_qa()
        if qa_result.get("qa_ids") != p["solver_qa_ids"] or not qa_result.get("passed"):
            raise ValueError("independent solver QA failed or used unexpected IDs")
        atomic_json(qa_path, dict(plan_sha256=digest(p),
            solver_sha256=p["sources"]["scripts/allen_cahn_trajectory_v1.py"], qa=qa_result))
        log_event("solver_qa_passed", ids=p["solver_qa_ids"], metrics=qa_result)
    if not qa_result.get("passed"):
        raise ValueError("solver QA did not pass")

    chunk_dir = OUT / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "prepare_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else dict(plan_sha256=digest(p), chunks={})
    if progress.get("plan_sha256") != digest(p):
        raise ValueError("generation progress belongs to a different protocol")

    chunks = []
    for start in range(0, len(p["confirmation_ids"]), CHUNK_SIZE):
        ids = p["confirmation_ids"][start:start + CHUNK_SIZE]
        number = start // CHUNK_SIZE
        path = chunk_dir / f"confirmation_{number:03d}.npz"
        name = path.relative_to(ROOT).as_posix()
        receipt = progress["chunks"].get(name)
        if path.exists() and receipt:
            if file_hash(path) != receipt["file_sha256"]:
                raise ValueError("saved generation chunk hash changed: " + name)
            with np.load(path, allow_pickle=False) as z:
                zi, zx, zy = z["ids"], z["x"], z["y"]
                if zi.tolist() != ids or array_hash(zx) != receipt["x_sha256"] or array_hash(zy) != receipt["y_sha256"]:
                    raise ValueError("saved generation chunk contents changed: " + name)
                chunks.append((zi.copy(), zx.copy(), zy.copy()))
            print(f"[prepare skip] {number + 1}/10 verified {name}", flush=True)
            continue

        x = initial(ids, n=256)
        y = trajectories(x, TIMES, dt=0.0005, length=2.0, nu=0.01, rho=5.0)
        if not np.array_equal(y[:, 0], x) or not np.isfinite(y).all():
            raise ValueError("generated chunk has invalid t0 or nonfinite values")
        _save_chunk(path, ids, x, y)
        receipt = dict(ids=ids, file_sha256=file_hash(path), x_sha256=array_hash(x), y_sha256=array_hash(y))
        progress["chunks"][name] = receipt
        atomic_json(progress_path, progress)
        log_event("generation_chunk_complete", chunk=number, ids=ids,
                  trajectories=len(ids), file_sha256=receipt["file_sha256"],
                  x_sha256=receipt["x_sha256"], y_sha256=receipt["y_sha256"])
        chunks.append((np.asarray(ids, dtype=np.int64), x, y))
        print(f"[prepare {number + 1}/10] generated IDs {ids[0]}..{ids[-1]}", flush=True)

    ids = np.concatenate([c[0] for c in chunks])
    x = np.concatenate([c[1] for c in chunks], axis=0)
    y = np.concatenate([c[2] for c in chunks], axis=0)
    if ids.tolist() != p["confirmation_ids"] or x.shape != (200, 256) or y.shape != (200, 5, 256):
        raise ValueError("assembled confirmation dataset shape or order changed")
    if np.intersect1d(ids, p["old_trajectory_ids"]).size or np.intersect1d(ids, p["solver_qa_ids"]).size:
        raise ValueError("assembled confirmation IDs overlap old/QA IDs")
    dataset_path = OUT / "confirmation_data.npz"
    if dataset_path.exists():
        receipt_path = OUT / "prepared.json"
        if not receipt_path.exists() or file_hash(dataset_path) != read_json(receipt_path).get("data_file_sha256"):
            raise ValueError("existing confirmation dataset has no matching immutable receipt")
    else:
        atomic_npz(dataset_path, ids=ids, x=x, y=y, times=TIMES, grid=GRID)
    prepared = dict(
        plan_sha256=digest(p),
        data_file="results/allen_cahn_fresh_confirmation_v1/confirmation_data.npz",
        data_file_sha256=file_hash(dataset_path),
        data_ids=ids.tolist(),
        data_count=int(len(ids)),
        input_sha256=array_hash(x),
        solution_sha256=array_hash(y),
        times=TIMES.tolist(),
        grid_sha256=array_hash(GRID),
        chunk_receipts=progress["chunks"],
        solver_qa_sha256=file_hash(qa_path),
        solver_qa_ids=p["solver_qa_ids"],
        old_pool_disjoint=True,
        solver_qa_disjoint=True,
        model_evaluation_started=False,
    )
    atomic_json(OUT / "prepared.json", prepared)
    atomic_json(OUT / "progress.json", dict(plan_sha256=digest(p), phase="prepared",
                generated_trajectories=len(ids), evaluated_models=0, total_models=24,
                evaluation_started=False))
    log_event("confirmation_data_prepared", data_sha256=prepared["data_file_sha256"],
              count=len(ids), first_id=int(ids[0]), last_id=int(ids[-1]))
    return prepared


def load_training_initials(p):
    """Load only the old training cache; never read its validation chunks."""
    pointwise_plan = read_json(MODELS_ROOT / "plan.json")
    if digest(pointwise_plan) != p["parent_plan_sha256"]:
        raise ValueError("parent training plan changed")
    by_id = {}
    for name, sha in p["training_cache_hashes"].items():
        path = ROOT / name
        if file_hash(path) != sha:
            raise ValueError("training cache hash changed: " + name)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj["y"].shape[1:] != (5, 256):
            raise ValueError("unexpected old training cache field shape")
        for j, case_id in enumerate(obj["ids"]):
            if case_id in by_id:
                raise ValueError("duplicate old training cache ID")
            by_id[int(case_id)] = (np.asarray(obj["x"][j], dtype=np.float64),
                                   np.asarray(obj["y"][j], dtype=np.float64))
    ids = pointwise_plan["train_ids"]
    if set(ids) != set(by_id):
        raise ValueError("old training cache IDs do not match the audited train split")
    x = np.stack([by_id[i][0] for i in ids])
    y = np.stack([by_id[i][1] for i in ids])
    return x, y


def _predict(model, raw_x, checkpoint, device):
    norm = checkpoint["normalization"]
    scale = float(checkpoint["scale"])
    x = (raw_x - norm["mean"]) / norm["std"]
    coords = torch.tensor(GRID, dtype=torch.float32, device=device)
    times = torch.tensor(TIMES, dtype=torch.float32, device=device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = torch.tensor(x[start:start + BATCH_SIZE], dtype=torch.float32, device=device)
            b, n = xb.shape
            pred = model(xb.repeat_interleave(len(TIMES), dim=0), times.repeat(b), coords)
            pred = pred.reshape(b, len(TIMES), n).cpu().numpy().astype(np.float64)
            pred = pred * scale + raw_x[start:start + b].mean(axis=1)[:, None, None]
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def _columns(condition):
    return [1, 2, 3, 4] if condition == "dense" else [1, 2, 3]


def _evaluate_model(key, entry, ids, x, y, train_x, train_y, device, plan_sha, data_sha):
    result_path = ROOT / entry["result_path"]
    checkpoint_path = ROOT / entry["checkpoint_path"]
    result = read_json(result_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = result["identity"]["config"]
    if (config != entry["config"] or checkpoint["normalization"] != entry["normalization"] or
            float(checkpoint["scale"]) != entry["target_scale"] or
            model_hash(checkpoint["model"]) != entry["weights_sha256"]):
        raise ValueError("frozen checkpoint contents changed: " + key)
    train_columns = [0, 1, 2, 3, 4] if entry["condition"] == "dense" else [0, 4]
    expected_scale = float(train_y[:, train_columns].std())
    expected_norm = dict(mean=float(train_x.mean()), std=float(train_x.std()))
    if checkpoint["normalization"] != expected_norm or float(checkpoint["scale"]) != expected_scale:
        raise ValueError("frozen checkpoint normalization does not match audited training data")

    model = PositionTimeFNO1d(
        config["width"], config["modes"], config["layers"],
        position=config["position"], length=config["length"],
        position_center=config["position_center"], position_scale=config["position_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    pred = _predict(model, x, checkpoint, device)
    err = trajectory_errors(pred, y)
    cols = _columns(entry["condition"])
    common = [1, 2, 3]
    per_case_primary = err[:, cols].mean(axis=1)
    train_initial = train_x
    diagnostics = dict(
        primary=band_errors(pred, y, train_initial, 2.0, cols),
        common_interior=band_errors(pred, y, train_initial, 2.0, common),
        all_future=band_errors(pred, y, train_initial, 2.0, [1, 2, 3, 4]),
    )
    model_out = dict(
        plan_sha256=plan_sha,
        data_sha256=data_sha,
        model_key=key,
        condition=entry["condition"], arm=entry["arm"], seed=entry["seed"],
        result_sha256=entry["result_sha256"], checkpoint_sha256=entry["checkpoint_sha256"],
        weights_sha256=entry["weights_sha256"],
        trajectory_ids=ids,
        field=dict(
            primary_columns=cols,
            primary=float(per_case_primary.mean()),
            per_case_primary=per_case_primary.tolist(),
            per_case_by_time=err.tolist(),
            by_time=err.mean(axis=0).tolist(),
            common_interior=float(err[:, common].mean()),
            all_future=float(err[:, 1:].mean()),
            initial=float(err[:, 0].mean()),
        ),
        diagnostics=diagnostics,
        prediction_sha256=array_hash(pred),
    )
    return model_out


def evaluate():
    p = read_json(OUT / "plan.json")
    verify_plan(p)
    prepared = read_json(OUT / "prepared.json")
    if prepared.get("plan_sha256") != digest(p):
        raise ValueError("prepared data receipt does not match locked protocol")
    data_path = OUT / "confirmation_data.npz"
    if file_hash(data_path) != prepared["data_file_sha256"]:
        raise ValueError("confirmation data file hash changed")
    with np.load(data_path, allow_pickle=False) as z:
        ids, x, y = z["ids"], z["x"], z["y"]
        times, grid = z["times"], z["grid"]
    if (ids.tolist() != p["confirmation_ids"] or x.shape != (200, 256) or y.shape != (200, 5, 256) or
            not np.array_equal(times, TIMES) or not np.array_equal(grid, GRID) or
            array_hash(x) != prepared["input_sha256"] or array_hash(y) != prepared["solution_sha256"]):
        raise ValueError("confirmation data contents do not match prepared receipt")
    train_x, train_y = load_training_initials(p)
    if train_x.shape != (100, 256) or train_y.shape != (100, 5, 256):
        raise ValueError("old training references have unexpected shape")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    log_event("evaluation_started", device=str(device), data_sha256=prepared["data_file_sha256"],
              model_count=len(p["models"]))
    progress_path = OUT / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    if progress.get("plan_sha256") not in (None, digest(p)):
        raise ValueError("evaluation progress belongs to a different plan")
    consumed_path = OUT / "evaluation_started.json"
    consumed_receipt = dict(plan_sha256=digest(p), data_sha256=prepared["data_file_sha256"],
                            confirmation_ids=p["confirmation_ids"],
                            note="The confirmation set is consumed from the first model evaluation onward.")
    if consumed_path.exists():
        if read_json(consumed_path) != consumed_receipt:
            raise ValueError("confirmation-consumption receipt does not match locked inputs")
    else:
        atomic_json(consumed_path, consumed_receipt)
    completed = set()
    for index, (key, entry) in enumerate(p["models"].items(), start=1):
        out_path = OUT / "runs" / f"{entry['condition']}__{entry['arm']}__s{entry['seed']}.json"
        if out_path.exists():
            saved = read_json(out_path)
            if (saved.get("plan_sha256") != digest(p) or
                    saved.get("data_sha256") != prepared["data_file_sha256"] or
                    saved.get("model_key") != key or
                    saved.get("weights_sha256") != entry["weights_sha256"] or
                    saved.get("trajectory_ids") != ids.tolist()):
                raise ValueError("saved confirmation result does not match locked inputs: " + key)
            completed.add(key)
            print(f"[evaluate skip {index}/24] verified {key}", flush=True)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            saved = _evaluate_model(key, entry, ids.tolist(), x, y, train_x, train_y, device,
                                    digest(p), prepared["data_file_sha256"])
            saved["inference_seconds"] = float(time.perf_counter() - started)
            atomic_json(out_path, saved)
            completed.add(key)
            log_event("model_evaluated", key=key, completed=len(completed), total=24,
                      primary=saved["field"]["primary"],
                      prediction_sha256=saved["prediction_sha256"],
                      inference_seconds=saved["inference_seconds"])
            print(f"[evaluate {len(completed)}/24] {key} primary={saved['field']['primary']:.8f}", flush=True)
        atomic_json(progress_path, dict(plan_sha256=digest(p), phase="evaluating",
                    completed_models=sorted(completed), completed_count=len(completed), total_models=24,
                    current_model=None, confirmation_consumed=True,
                    data_sha256=prepared["data_file_sha256"]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(completed) != 24:
        raise ValueError("not all frozen models were evaluated")
    atomic_json(progress_path, dict(plan_sha256=digest(p), phase="evaluation_complete",
                completed_models=sorted(completed), completed_count=24, total_models=24,
                confirmation_consumed=True, data_sha256=prepared["data_file_sha256"]))
    log_event("evaluation_complete", data_sha256=prepared["data_file_sha256"], models=24)


def summarize():
    p = read_json(OUT / "plan.json")
    verify_plan(p)
    prepared = read_json(OUT / "prepared.json")
    if file_hash(OUT / "confirmation_data.npz") != prepared["data_file_sha256"]:
        raise ValueError("confirmation data changed before summary")
    records = {}
    for key, entry in p["models"].items():
        path = OUT / "runs" / f"{entry['condition']}__{entry['arm']}__s{entry['seed']}.json"
        record = read_json(path)
        if (record["model_key"] != key or record["plan_sha256"] != digest(p) or
                record["data_sha256"] != prepared["data_file_sha256"] or
                record["weights_sha256"] != entry["weights_sha256"]):
            raise ValueError("run result identity mismatch: " + key)
        records[key] = record
    if len(records) != 24:
        raise ValueError("all 24 frozen model evaluations are required")

    conditions = {}
    for condition in CONDITIONS:
        means = {}
        seed_means = {}
        for arm in ARMS:
            rows = [records[f"{condition}/{arm}_s{seed}"] for seed in SEEDS]
            seed_means[arm] = [float(row["field"]["primary"]) for row in rows]
            means[arm] = dict(
                mean=float(np.mean(seed_means[arm])),
                per_seed=seed_means[arm],
                initial_mean=float(np.mean([row["field"]["initial"] for row in rows])),
                common_interior_mean=float(np.mean([row["field"]["common_interior"] for row in rows])),
                all_future_mean=float(np.mean([row["field"]["all_future"] for row in rows])),
                by_time_mean=np.mean([row["field"]["by_time"] for row in rows], axis=0).tolist(),
                gradient_relative_mean=float(np.mean([row["diagnostics"]["primary"]["gradient"]["relative_mean"] for row in rows])),
                curvature_relative_mean=float(np.mean([row["diagnostics"]["primary"]["curvature"]["relative_mean"] for row in rows])),
                field_relative_mean=float(np.mean([row["diagnostics"]["primary"]["field"]["relative_mean"] for row in rows])),
            )

        interval_cases = np.mean([np.asarray(records[f"{condition}/interval_uniform_s{seed}"]["field"]["per_case_primary"])
                                  for seed in SEEDS], axis=0)
        point_cases = np.mean([np.asarray(records[f"{condition}/pointwise_nodes_s{seed}"]["field"]["per_case_primary"])
                               for seed in SEEDS], axis=0)
        delta = interval_cases - point_cases
        rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if condition == "dense" else 1))
        indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
        boot = delta[indices].mean(axis=1)
        alpha = (1.0 - p["primary_comparison"]["two_sided_per_condition_confidence"]) / 2.0
        ci = np.quantile(boot, [alpha, 1.0 - alpha])
        conditions[condition] = dict(
            arms=means,
            primary_interval_minus_pointwise=dict(
                mean=float(delta.mean()),
                per_seed=[seed_means["interval_uniform"][i] - seed_means["pointwise_nodes"][i] for i in range(3)],
                paired_trajectory_std=float(delta.std(ddof=1)),
                fraction_interval_lower=(float(np.mean(delta < 0.0))),
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
                bootstrap_ci_confidence=float(p["primary_comparison"]["two_sided_per_condition_confidence"]),
                bootstrap_ci=[float(ci[0]), float(ci[1])],
                mean_percent_interval_vs_pointwise=float(100.0 * delta.mean() /
                    max(float(point_cases.mean()), 1e-12)),
                inference_unit="200 paired trajectories after averaging each frozen model seed's error; conditional on the three fitted seeds",
            ),
            secondary_descriptive=dict(
                interval_minus_none=float(means["interval_uniform"]["mean"] - means["none"]["mean"]),
                interval_minus_full=float(means["interval_uniform"]["mean"] - means["full"]["mean"]),
                pointwise_minus_none=float(means["pointwise_nodes"]["mean"] - means["none"]["mean"]),
                pointwise_minus_full=float(means["pointwise_nodes"]["mean"] - means["full"]["mean"]),
            ),
        )
    result = dict(
        plan_sha256=digest(p),
        data_file_sha256=prepared["data_file_sha256"],
        confirmation_ids=p["confirmation_ids"],
        confirmation_count=len(p["confirmation_ids"]),
        old_pool_disjoint=True,
        solver_qa_disjoint=True,
        frozen_models_evaluated=24,
        new_model_training_steps=0,
        conditions=conditions,
        records={key: str((OUT / "runs" / f"{entry['condition']}__{entry['arm']}__s{entry['seed']}.json").relative_to(ROOT).as_posix())
                 for key, entry in p["models"].items()},
        interpretation="Independent-trajectory confirmation of the frozen models on the declared initial-condition distribution; no model, objective, weight, or threshold was selected using this set.",
        confirmation_consumed=True,
    )
    atomic_json(OUT / "summary.json", result)
    atomic_json(OUT / "progress.json", dict(plan_sha256=digest(p), phase="summarized",
                completed_models=24, total_models=24, confirmation_consumed=True,
                data_sha256=prepared["data_file_sha256"]))
    log_event("summary_complete", data_sha256=prepared["data_file_sha256"], contrasts={
        c: conditions[c]["primary_interval_minus_pointwise"]["mean"] for c in CONDITIONS})
    print(json.dumps(conditions, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["plan", "prepare", "evaluate", "summarize", "run"])
    stage = parser.parse_args().stage
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if stage == "plan":
        p = plan()
        print(json.dumps(dict(plan_sha256=digest(p), frozen_models=len(p["models"]),
                              trajectories=len(p["confirmation_ids"]), ids=[p["confirmation_ids"][0], p["confirmation_ids"][-1]]), indent=2))
    elif stage == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif stage == "evaluate":
        evaluate()
    elif stage == "summarize":
        summarize()
    else:
        if not (OUT / "plan.json").exists():
            raise ValueError("run the locked plan stage and commit it before generation")
        prepare()
        evaluate()
        summarize()


if __name__ == "__main__":
    main()
