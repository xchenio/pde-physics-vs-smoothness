"""D5 的 H1：跨粘度 ν 外推时，物理残差是否有独特增益。

协议见 docs/PHYSICS_AS_SMOOTHNESS_V1.md §2。动机：平滑正则只约束"粗糙度"，
而 Burgers 的耗散强度**依赖 ν**（PDEBench 生成器的扩散系数是 Nu/π）。
因此若把模型训练在 ν=0.001 上、测试在 ν=0.01 / 0.1 上，
平滑正则**没有任何关于 ν 的信息**，而 PDE 残差至少编码了 ν 依赖的耗散结构。

待检验的问题不是"跨 ν 泛化好不好"，而是：

    PDE − best_smoothness 在**外推集**上是否显著 < 0（PDE 更好）

训练只用 ν=0.001；惩罚里的扩散系数也用训练 ν（不偷看测试 ν）。
归一化同样只用训练集的统计量。

数据需求（需先下载，见 docs/PHYSICS_AS_SMOOTHNESS_V1.md §4）：
    external_data/pdebench/1D_Burgers_Sols_Nu0.01.hdf5
    external_data/pdebench/1D_Burgers_Sols_Nu0.1.hdf5

用法（可续跑）：
    D:/miniconda3/envs/pytorch/python.exe scripts/physics_vs_smoothness_h1.py --prepare
    D:/miniconda3/envs/pytorch/python.exe scripts/physics_vs_smoothness_h1.py --device cuda
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

from src.fno import FNO1d                                         # noqa: E402
from src.public_benchmark import prepare, atomic_json, file_hash  # noqa: E402
from src.utils import code_fingerprint, env_fingerprint, set_seed  # noqa: E402
from scripts.run_public_benchmark import metrics                   # noqa: E402
from scripts.physics_vs_smoothness_h2h3 import (                   # noqa: E402
    penalties, _grad_norm, BAND, STEPS, BATCH, LR, WARMUP)

DATA_DIR = ROOT / "external_data/pdebench"
SOURCES = {
    "nu0.001": DATA_DIR / "1D_Burgers_Sols_Nu0.001.hdf5",   # 训练
    "nu0.01": DATA_DIR / "1D_Burgers_Sols_Nu0.01.hdf5",     # 外推测试
    "nu0.1": DATA_DIR / "1D_Burgers_Sols_Nu0.1.hdf5",       # 外推测试
}
TRAIN_NU = "nu0.001"
EXTRAP = ("nu0.01", "nu0.1")
TIMES = (100, 150)          # u_t 用两个时刻的有限差分
STRIDE = 4
SEEDS = (0, 1, 2)
STRENGTHS = (0.03, 0.1, 0.3)
FAMILIES = ("none", "physics", "gradient", "curvature")


def folder_for(nu: str, ti: int) -> Path:
    return DATA_DIR / f"h1_{nu}_t{ti}_s{STRIDE}"


def load(nu: str, ti: int):
    d = folder_for(nu, ti)
    protocol = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(d / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError(f"{nu}@{ti}: cache changed")
    with np.load(d / "pairs.npz") as z:
        return {k: z[k].copy() for k in z.files}, protocol


def train_one(data, protocol_train, dt, nu_train, device, seed,
              family, strength, steps=STEPS):
    arr_a, arr_b = data[TRAIN_NU]                    # (t=100, t=150) 的训练集
    norm = protocol_train["normalization"]
    xmean, xstd = norm["input_mean"], max(norm["input_std"], 1e-6)
    ymean, ystd = norm["target_mean"], max(norm["target_std"], 1e-6)
    coords_x = np.asarray(arr_a["x"], dtype=np.float64)

    xs = {s: torch.tensor((arr_a[s + "_x"] - xmean) / xstd, device=device)
          for s in ("train", "val", "test")}
    xtr = xs["train"]
    ytr = torch.tensor(
        np.stack([(arr_a["train_y"] - ymean) / ystd,
                  (arr_b["train_y"] - ymean) / ystd], axis=1), device=device)
    coords = torch.tensor(arr_a["x"][:, None], device=device, dtype=torch.float32)

    set_seed(seed + 2000)
    model = FNO1d(width=32, modes=16, layers=4).to(device)
    model.project[-1] = torch.nn.Conv1d(32, 2, 1).to(device)
    torch.nn.init.zeros_(model.project[-1].weight)
    torch.nn.init.zeros_(model.project[-1].bias)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed + 3000)
    model.train()

    lam = 0.0
    started = time.time()
    for step in range(steps):
        if family != "none" and step == WARMUP:
            sel0 = rng.choice(len(xtr), size=min(BATCH, len(xtr)), replace=False)
            pred0 = model(xtr[sel0], coords)
            gd = _grad_norm(model, ((pred0 - ytr[sel0]) ** 2).mean())
            gp = _grad_norm(model, penalties(pred0, coords_x, dt, nu_train, ystd, BAND)[family])
            lam = float(strength * gd / max(gp, torch.tensor(1e-30, device=gd.device)))
        sel = rng.choice(len(xtr), size=min(BATCH, len(xtr)), replace=False)
        pred = model(xtr[sel], coords)
        loss = ((pred - ytr[sel]) ** 2).mean()
        if family != "none" and step >= WARMUP:
            loss = loss + lam * penalties(pred, coords_x, dt, nu_train, ystd, BAND)[family]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    rec = {"seed": seed, "family": family, "strength": strength,
           "gradient_ratio": strength, "lambda": lam, "steps": steps,
           "seconds": round(time.time() - started, 2)}
    with torch.no_grad():
        for nu in (TRAIN_NU,) + EXTRAP:
            a, b = data[nu]
            xs_nu = torch.tensor((a["test_x"] - xmean) / xstd, device=device)
            p = torch.cat([model(v, coords) for v in xs_nu.split(64)])
            p = p.cpu().numpy() * ystd + ymean
            for j, tag in enumerate(("t1", "t2")):
                rec[f"{nu}_{tag}_rel_l2"] = metrics(
                    p[:, j], a["test_y"] if j == 0 else b["test_y"])["mean_relative_l2"]
            rec[f"{nu}_mean_rel_l2"] = 0.5 * (rec[f"{nu}_t1_rel_l2"] + rec[f"{nu}_t2_rel_l2"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--out", default="results/physics_vs_smoothness_h1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.prepare:
        for nu, src in SOURCES.items():
            if not src.exists():
                print(f"[missing] {src.name}  (download it first)", flush=True)
                continue
            for ti in TIMES:
                d = folder_for(nu, ti)
                if (d / "protocol.json").exists():
                    print("[skip]", nu, ti, flush=True)
                    continue
                info = prepare(src, d, sizes=(1000, 200, 200), seed=20260922,
                               time_index=ti, stride=STRIDE)
                print(f"[prepared] {nu} t={ti} Nu={info.get('file_Nu')} "
                      f"{info['cache_sha256'][:12]}", flush=True)
        return

    for nu in (TRAIN_NU,) + EXTRAP:
        if not SOURCES[nu].exists():
            raise SystemExit(f"missing data file: {SOURCES[nu]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)

    data, protocols = {}, {}
    for nu in (TRAIN_NU,) + EXTRAP:
        a, pa = load(nu, TIMES[0])
        b, pb = load(nu, TIMES[1])
        if pa["sample_ids"] != pb["sample_ids"]:
            raise ValueError(f"{nu}: time slices use different sample ids")
        data[nu] = (a, b)
        protocols[nu] = pa
    dt = float(pb["target_time"] - pa["target_time"])
    nu_train = float(protocols[TRAIN_NU]["file_Nu"])

    # 记录各 ν 的初始场统计，用来判断"外推"里混了多少输入分布偏移
    stats = {}
    for nu in (TRAIN_NU,) + EXTRAP:
        x = np.asarray(data[nu][0]["test_x"], dtype=np.float64)
        stats[nu] = dict(input_mean=float(x.mean()), input_std=float(x.std()))
    atomic_json(out / "plan.json",
                dict(train_nu=TRAIN_NU, extrapolate=list(EXTRAP), times=list(TIMES),
                     strengths=STRENGTHS, band=BAND, steps=STEPS, seeds=SEEDS,
                     nu_in_penalty=nu_train,
                     input_stats=stats,
                     note="normalization and penalty nu both come from the training split only",
                     code_hash=code_fingerprint(ROOT), env=env_fingerprint(str(device))))

    done = 0
    for fam in FAMILIES:
        for st in ((0.0,) if fam == "none" else STRENGTHS):
            for seed in SEEDS:
                key = f"{fam}_{st}_s{seed}"
                target = out / "runs" / f"{key}.json"
                if target.exists():
                    continue
                try:
                    rec = train_one(data, protocols[TRAIN_NU], dt, nu_train,
                                    device, seed, fam, st)
                    rec["status"] = "ok"
                except Exception as exc:                          # noqa: BLE001
                    rec = {"seed": seed, "family": fam, "strength": st,
                           "status": "error",
                           "error": f"{type(exc).__name__}: {exc}"}
                atomic_json(target, rec)
                done += 1
                msg = " ".join(f"{nu}={rec.get(nu + '_mean_rel_l2', float('nan')):.4f}"
                               for nu in (TRAIN_NU,) + EXTRAP)
                print(f"[{done}] {key} status={rec['status']} {msg}", flush=True)
                if args.limit and done >= args.limit:
                    print("[limit reached] re-run to continue", flush=True)
                    return
    print(f"total runs: {len(list((out / 'runs').glob('*.json')))}")


if __name__ == "__main__":
    main()
