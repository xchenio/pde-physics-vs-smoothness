"""D4：诊断驱动的干预选择（协议见 docs/INTERVENTION_SELECTION_PROTOCOL_V1.md）。

**可续跑**：每个 (setting, arch, seed, strategy) 完成即落盘到 `<out>/runs/<key>.json`，
重跑自动跳过已完成项。机器休眠/中断后重跑同一条命令即可继续。

用法：
    # 先准备各设置的数据（只读已有 HDF5，不需要下载）
    D:/miniconda3/envs/pytorch/python.exe scripts/intervention_selection.py --prepare
    # 跑实验（--limit 可分批，方便随时中断）
    D:/miniconda3/envs/pytorch/python.exe scripts/intervention_selection.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fno import FNO1d                                          # noqa: E402
from src.models import DeepONet                                    # noqa: E402
from src.public_benchmark import prepare, atomic_json, file_hash   # noqa: E402
from src.utils import code_fingerprint, env_fingerprint, set_seed  # noqa: E402
from scripts.run_public_benchmark import metrics                   # noqa: E402
from scripts.readout_attribution import (                          # noqa: E402
    lb_basis, lb_pointwise, readout_features, rel_fro)

# ------------------------------------------------------------------ 协议常量（冻结）

SOURCE = ROOT / "external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5"
SETTINGS = {
    "T1": dict(time_index=100, stride=4, held_out=False),
    "T2": dict(time_index=50, stride=4, held_out=True),
    "T3": dict(time_index=150, stride=4, held_out=True),
    "T4": dict(time_index=100, stride=2, held_out=True),
    "T5": dict(time_index=100, stride=8, held_out=True),
}
ARCHS = ("deeponet_mlp", "fno")
SEEDS = (0, 1, 2)
BASE_STEPS = 1500
BATCH = 32
LR = 1e-3
INTERVENTIONS = ("I0", "I1", "I2", "I3", "I5")
# 协议 §4 阈值（来自 V1 的独立观测，不在本实验上调）
TAU_OPT, TAU_LB, TAU_GEN = 0.30, 0.40, 0.15


def setting_folder(sid: str) -> Path:
    s = SETTINGS[sid]
    return ROOT / f"external_data/pdebench/isel_{sid}_t{s['time_index']}_s{s['stride']}"


# ------------------------------------------------------------------ 模型

def build(arch: str, in_dim: int, intervention: str, device: torch.device):
    width = {"deeponet_mlp": 64, "fno": 32}[arch]
    trunk_type = "mlp"
    modes = 16
    if intervention == "I1":
        if arch == "fno":
            modes = 32
        else:
            trunk_type = "fourier"
    elif intervention == "I2":
        width *= 2
    if arch == "fno":
        return FNO1d(width=width, modes=modes, layers=4).to(device)
    return DeepONet(branch_in=in_dim, trunk_in=1, width=width,
                    branch_hidden=128, trunk_hidden=128, branch_layers=3,
                    trunk_layers=3, trunk_type=trunk_type).to(device)


def steps_for(intervention: str) -> int:
    return BASE_STEPS * 2 if intervention == "I3" else BASE_STEPS


def weight_decay_for(intervention: str) -> float:
    return 1e-3 if intervention == "I5" else 0.0


def train_and_eval(arch, intervention, arrays, protocol, device, seed, steps=None):
    """训练一个干预，返回 (记录, model, 归一化张量)。"""
    steps = steps if steps is not None else steps_for(intervention)
    norm = protocol["normalization"]
    xmean, xstd = norm["input_mean"], max(norm["input_std"], 1e-6)
    ymean, ystd = norm["target_mean"], max(norm["target_std"], 1e-6)
    xs = {k: torch.tensor((arrays[k + "_x"] - xmean) / xstd, device=device)
          for k in ("train", "val", "test")}
    ys = torch.tensor((arrays["train_y"] - ymean) / ystd, device=device)
    coords = torch.tensor(arrays["x"][:, None], device=device, dtype=torch.float32)

    set_seed(seed + 2000)
    model = build(arch, xs["train"].shape[1], intervention, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=weight_decay_for(intervention))
    rng = np.random.default_rng(seed + 3000)
    model.train()
    for _ in range(steps):
        idx = rng.choice(len(ys), size=min(BATCH, len(ys)), replace=False)
        pred = model(xs["train"][idx], coords)
        loss = ((pred - ys[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    rec = {"architecture": arch, "intervention": intervention, "seed": seed,
           "steps": steps,
           "real_parameters": int(sum(p.numel() * (2 if p.is_complex() else 1)
                                      for p in model.parameters()))}
    with torch.no_grad():
        raw, feats = {}, {}
        for split in ("train", "val", "test"):
            z = torch.cat([model(v, coords) for v in xs[split].split(64)])
            raw[split] = z.cpu().numpy().astype(np.float64)
            psi, phi = readout_features(model, arch, xs[split], coords)
            feats[split] = (None if psi is None else psi.cpu().numpy().astype(np.float64),
                            None if phi is None else phi.cpu().numpy().astype(np.float64))
        for split in ("train", "val", "test"):
            pred = raw[split] * ystd + ymean
            m = metrics(pred, arrays[split + "_y"])
            rec[f"{split}_mean_rel_l2"] = m["mean_relative_l2"]
            yhat = (arrays[split + "_y"].astype(np.float64) - ymean) / ystd
            rec[f"{split}_rel_fro"] = rel_fro(raw[split], yhat)
            psi, phi = feats[split]
            rec[f"{split}_lb"] = (lb_basis(phi, yhat).get("lb") if phi is not None
                                  else lb_pointwise(psi, yhat).get("lb"))
    return rec, model


# ------------------------------------------------------------------ 规则（冻结）

def rule_select(diag: dict) -> tuple:
    r_lb = diag["lb_train"] / max(diag["train_rel"], 1e-12)
    r_opt = (diag["train_rel"] - diag["lb_train"]) / max(diag["train_rel"], 1e-12)
    r_gen = (diag["val_rel"] - diag["train_rel"]) / max(diag["train_rel"], 1e-12)
    if r_opt >= TAU_OPT:
        choice = "I3"
    elif r_lb >= TAU_LB:
        choice = "I1"
    elif r_gen >= TAU_GEN:
        choice = "I5"
    else:
        choice = "I0"
    return choice, dict(r_lb=r_lb, r_opt=r_opt, r_gen=r_gen)


# ------------------------------------------------------------------ 主流程

def load_setting(sid: str):
    folder = setting_folder(sid)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(folder / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError(f"{sid}: pair cache changed")
    with np.load(folder / "pairs.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--out", default="results/intervention_selection_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--limit", type=int, default=0, help="本轮最多跑多少个新任务，0=不限")
    args = ap.parse_args()

    if args.prepare:
        for sid, s in SETTINGS.items():
            if sid not in args.settings.split(","):
                continue
            folder = setting_folder(sid)
            if (folder / "protocol.json").exists():
                print("[skip]", sid, flush=True)
                continue
            info = prepare(SOURCE, folder, sizes=(1000, 200, 200), seed=20260922,
                           time_index=s["time_index"], stride=s["stride"])
            print("[prepared]", sid, folder, info["cache_sha256"][:12], flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)
    atomic_json(out / "plan.json",
                dict(settings=SETTINGS, archs=ARCHS, seeds=SEEDS,
                     base_steps=BASE_STEPS, thresholds=[TAU_OPT, TAU_LB, TAU_GEN],
                     code_hash=code_fingerprint(ROOT), env=env_fingerprint(str(device))))

    tasks = []
    for sid in args.settings.split(","):
        for arch in ARCHS:
            for seed in SEEDS:
                for strategy in ("baseline", "rule", "fix", "rand", "val"):
                    repeats = 3 if strategy == "rand" else 1
                    for r in range(repeats):
                        tasks.append((sid, arch, seed, strategy, r))

    done = 0
    for sid, arch, seed, strategy, rep in tasks:
        key = f"{sid}_{arch}_s{seed}_{strategy}" + (f"_r{rep}" if rep else "")
        target = out / "runs" / f"{key}.json"
        if target.exists():
            continue
        arrays, protocol = load_setting(sid)
        started = time.time()
        try:
            if strategy == "baseline":
                rec, _ = train_and_eval(arch, "I0", arrays, protocol, device, seed)
                diag = {"lb_train": rec["train_lb"], "train_rel": rec["train_rel_fro"],
                        "val_rel": rec["val_rel_fro"]}
                chosen, ratios = rule_select(diag)
                rec.update(strategy="baseline", chosen_by_rule=chosen, ratios=ratios)
            elif strategy == "rule":
                base = json.loads((out / "runs" / f"{sid}_{arch}_s{seed}_baseline.json"
                                   ).read_text(encoding="utf-8"))
                chosen = base["chosen_by_rule"]
                if chosen == "I0":
                    rec = dict(base)
                    rec["strategy"] = "rule"
                    rec["consumed_steps"] = BASE_STEPS
                else:
                    rec, _ = train_and_eval(arch, chosen, arrays, protocol, device, seed)
                    rec["strategy"] = "rule"
                    rec["consumed_steps"] = rec["steps"]
            elif strategy == "fix":
                rec, _ = train_and_eval(arch, "I3", arrays, protocol, device, seed)
                rec["strategy"] = "fix"
                rec["consumed_steps"] = rec["steps"]
            elif strategy == "rand":
                pick = INTERVENTIONS[int(np.random.default_rng(seed * 17 + rep).integers(
                    0, len(INTERVENTIONS)))]
                rec, _ = train_and_eval(arch, pick, arrays, protocol, device, seed)
                rec["strategy"] = "rand"
                rec["consumed_steps"] = rec["steps"]
            else:  # val：预算内验证集搜索
                cands = ("I1", "I3", "I5")
                short = {}
                for c in cands:
                    r, _ = train_and_eval(arch, c, arrays, protocol, device, seed,
                                          steps=BASE_STEPS // 3)
                    short[c] = r["val_mean_rel_l2"]
                best = min(cands, key=lambda c: short[c])
                rec, _ = train_and_eval(arch, best, arrays, protocol, device, seed)
                rec["strategy"] = "val"
                rec["short_probes"] = short
                rec["consumed_steps"] = BASE_STEPS + BASE_STEPS  # 3×(1/3) 探测 + 全量
            rec["setting"] = sid
            rec["held_out"] = SETTINGS[sid]["held_out"]
            rec["status"] = "ok"
        except Exception as exc:                                   # noqa: BLE001
            rec = {"setting": sid, "architecture": arch, "seed": seed,
                   "strategy": strategy, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        rec["seconds"] = round(time.time() - started, 2)
        atomic_json(target, rec)
        done += 1
        print(f"[{done}] {key} status={rec['status']} "
              f"test={rec.get('test_mean_rel_l2', float('nan')):.4f} "
              f"({rec['seconds']}s)", flush=True)
        if args.limit and done >= args.limit:
            print("[limit reached] re-run to continue", flush=True)
            break

    rows = sorted((out / "runs").glob("*.json"))
    atomic_json(out / "progress.json", dict(n_runs=len(rows)))
    print(f"completed runs: {len(rows)}")


if __name__ == "__main__":
    main()
