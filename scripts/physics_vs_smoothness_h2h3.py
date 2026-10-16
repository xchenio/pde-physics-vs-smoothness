"""D5 的 H2（分辨率）与 H3（样本量）：物理残差 vs 匹配平滑正则。

协议见 docs/PHYSICS_AS_SMOOTHNESS_V1.md（实验前冻结）。核心问题：
在"粗糙度信息不足以确定解"的条件下，PDE 残差是否出现**相对于平滑正则的独特增益**。

## 为什么需要两个时刻

fixed-t1 数据只有单个时刻，算不出 u_t，因此 PDE 残差无法构造。
本脚本用**同一批样本的两个时刻**（t=100 与 t=150）做有限差分：

    u_t ≈ (u(t2) - u(t1)) / (t2 - t1)

两个设置的 split 用相同 seed 与相同 total，故 sample_ids 一致，可直接配对
（脚本会断言）。

⚠️ PDEBench 的 1D Burgers 使用 **Nu/π** 作为扩散系数（见 protocol.json 的
physics_note: "upstream generator diffusion uses epsilon/pi"），不是 Nu 本身。

## 惩罚（沿用 FNO_REGULARIZATION_V1 的固定频段 0..16 与强度网格）

    physics   : ||P_low (u_t + ∂x(u²/2) - (Nu/π) ∂xx u)||²
    gradient  : ||P_low ∂x u||²
    curvature : ||P_low ∂xx u||²

P_low 是低通到 0..16 模态的投影。三者都用**同一频段**，所以对比是匹配的。

## 判定（协议 §3）

`PDE − best_smoothness` 的配对 bootstrap（10k，Bonferroni），门槛 1% 且上界 < 0。
区间上界 ≥ 0 即判该假设不成立，记为负结果，**不重试、不调强度**。

用法（可续跑，每个任务完成即落盘）：
    D:/miniconda3/envs/pytorch/python.exe scripts/physics_vs_smoothness_h2h3.py --prepare
    D:/miniconda3/envs/pytorch/python.exe scripts/physics_vs_smoothness_h2h3.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.time_fno import coefficient_energy                       # noqa: E402
from scripts.run_public_benchmark import metrics                   # noqa: E402

SOURCE = ROOT / "external_data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5"

# 每个分辨率需要两个时刻；(stride, time_index)
NEEDED = {
    "R4_a": (4, 100), "R4_b": (4, 150),
    "R8_a": (8, 100), "R8_b": (8, 150),
    "R16_a": (16, 100), "R16_b": (16, 150),
}
# 与 intervention_selection.py 的目录命名规则一致，以便复用已 prepare 的数据
ALIAS = {"R4_a": "T1", "R4_b": "T3", "R8_a": "T5"}

SEEDS = (0, 1, 2)
STRENGTHS = (0.03, 0.1, 0.3)
FAMILIES = ("none", "physics", "gradient", "curvature")
BAND = 16
STEPS, BATCH, LR = 1500, 32, 1e-3
WARMUP = 50          # 纯数据损失预热步数，之后才加入惩罚并做梯度校准

# H2：分辨率（全量样本）；H3：样本量（stride=4）
H2_RESOLUTIONS = ("R4", "R8")
H3_SIZES = (20, 50, 100)


def folder_for(sid: str) -> Path:
    stride, ti = NEEDED[sid]
    name = ALIAS.get(sid, sid)
    return ROOT / f"external_data/pdebench/isel_{name}_t{ti}_s{stride}"


def load(sid: str):
    d = folder_for(sid)
    protocol = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(d / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError(f"{sid}: cache changed")
    with np.load(d / "pairs.npz") as z:
        return {k: z[k].copy() for k in z.files}, protocol


# ------------------------------------------------------------------ 惩罚

def _lowpass(u: torch.Tensor, band: int) -> torch.Tensor:
    return torch.fft.rfft(u, norm="forward")[..., : band + 1]


def penalties(pred: torch.Tensor, coords_x: np.ndarray, dt: float,
              nu: float, scale: float, band: int) -> dict:
    """pred: (B, 2, X)，通道 0 = t1，通道 1 = t2。"""
    u1, u2 = pred[:, 0], pred[:, 1]                       # (B, X)
    n = u1.shape[-1]
    length = float(coords_x[-1] - coords_x[0]) * n / (n - 1)   # 周期长度
    # 与 torch.fft.fft 的输出顺序一致的角频率 [0,1,...,n/2-1,-n/2,...,-1]
    k = 2 * math.pi * torch.fft.fftfreq(n, d=length / n,
                                        device=u1.device, dtype=u1.dtype)
    # 谱微分（周期）
    u1f = torch.fft.fft(u1, norm="forward")
    ux = torch.fft.ifft(1j * k * u1f, norm="forward").real
    uxx = torch.fft.ifft(-(k ** 2) * u1f, norm="forward").real
    ut = (u2 - u1) / dt

    resid = ut + 0.5 * _d_dx(u1 * u1, k) - (nu / math.pi) * uxx
    return {
        "physics": coefficient_energy(_lowpass(resid, band)).mean() / scale ** 2,
        "gradient": coefficient_energy(_lowpass(ux, band)).mean() / scale ** 2,
        "curvature": coefficient_energy(_lowpass(uxx, band)).mean() / scale ** 2,
    }


def _d_dx(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    uf = torch.fft.fft(u, norm="forward")
    return torch.fft.ifft(1j * k * uf, norm="forward").real


# ------------------------------------------------------------------ 训练

def _grad_norm(model, loss):
    """梯度平方和的开方。FNO 的谱卷积权重是复数，复数梯度要用模平方。"""
    gs = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                             retain_graph=True)
    total = None
    for g in gs:
        g = g.detach()
        term = (g.abs() ** 2).sum() if g.is_complex() else (g ** 2).sum()
        total = term if total is None else total + term
    return torch.sqrt(total)


def train_one(arrays_pair, protocol_a, dt, nu, device, seed,
              family, strength, n_train, steps=STEPS):
    arr_a, arr_b = arrays_pair
    norm = protocol_a["normalization"]
    xmean, xstd = norm["input_mean"], max(norm["input_std"], 1e-6)
    ymean, ystd = norm["target_mean"], max(norm["target_std"], 1e-6)
    coords_x = np.asarray(arr_a["x"], dtype=np.float64)

    idx = np.arange(n_train)                       # 取前 n_train 个（固定顺序，可复现）
    xs = {s: torch.tensor((arr_a[s + "_x"] - xmean) / xstd, device=device)
          for s in ("train", "val", "test")}
    xtr = xs["train"][idx]
    # 两个时刻共用**同一**归一化，否则 u_t = (u2-u1)/dt 的尺度不一致
    ytr = torch.tensor(
        np.stack([(arr_a["train_y"][idx] - ymean) / ystd,
                  (arr_b["train_y"][idx] - ymean) / ystd], axis=1),
        device=device)                              # (n, 2, X)
    coords = torch.tensor(arr_a["x"][:, None], device=device, dtype=torch.float32)

    set_seed(seed + 2000)
    model = FNO1d(width=32, modes=16, layers=4).to(device)
    model.project[-1] = torch.nn.Conv1d(32, 2, 1).to(device)
    torch.nn.init.zeros_(model.project[-1].weight)
    torch.nn.init.zeros_(model.project[-1].bias)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed + 3000)
    model.train()

    # 惩罚从 WARMUP 步之后才加入（沿用 FNO_REGULARIZATION_V1 的
    # "data-only warm start + continuation" 结构），并在那时做梯度校准。
    #
    # 为什么必须先 warm up：最后一层是零初始化，模型初始输出恒为 0；
    # 而 u≡0 **恰好满足 Burgers**，残差在该点附近极其平坦，惩罚梯度 → 0，
    # 于是 λ = strength·‖∇data‖/‖∇penalty‖ 会爆炸，把解彻底压成零场
    # （实测 test rel L2 = 1.000）。先跑一段纯数据损失再校准才有意义。
    lam = 0.0
    started = time.time()
    for step in range(steps):
        if family != "none" and step == WARMUP:
            sel0 = rng.choice(len(xtr), size=min(BATCH, len(xtr)), replace=False)
            pred0 = model(xtr[sel0], coords)
            gd = _grad_norm(model, ((pred0 - ytr[sel0]) ** 2).mean())
            gp = _grad_norm(model, penalties(pred0, coords_x, dt, nu, ystd, BAND)[family])
            lam = float(strength * gd / max(gp, torch.tensor(1e-30, device=gd.device)))
        sel = rng.choice(len(xtr), size=min(BATCH, len(xtr)), replace=False)
        pred = model(xtr[sel], coords)
        loss = ((pred - ytr[sel]) ** 2).mean()
        if family != "none" and step >= WARMUP:
            loss = loss + lam * penalties(pred, coords_x, dt, nu, ystd, BAND)[family]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    rec = {"seed": seed, "family": family, "strength": strength,
           "gradient_ratio": strength, "lambda": lam,
           "n_train": int(n_train), "steps": steps,
           "seconds": round(time.time() - started, 2)}
    with torch.no_grad():
        for s in ("val", "test"):
            p = torch.cat([model(v, coords) for v in xs[s].split(64)])
            p = p.cpu().numpy() * ystd + ymean           # (n, 2, X)
            for j, tag in enumerate(("t1", "t2")):
                rec[f"{s}_{tag}_rel_l2"] = metrics(p[:, j], arr_a[s + "_y"] if j == 0
                                                   else arr_b[s + "_y"])["mean_relative_l2"]
            rec[f"{s}_mean_rel_l2"] = 0.5 * (rec[f"{s}_t1_rel_l2"] + rec[f"{s}_t2_rel_l2"])
    return rec


# ------------------------------------------------------------------ 主流程

def tasks():
    out = []
    for res in H2_RESOLUTIONS:                       # H2：分辨率，全量样本
        for fam in FAMILIES:
            for st in ((0.0,) if fam == "none" else STRENGTHS):
                for seed in SEEDS:
                    out.append(dict(kind="H2", res=res, n_train=1000,
                                    fam=fam, st=st, seed=seed))
    for n in H3_SIZES:                               # H3：样本量，stride=4
        for fam in FAMILIES:
            for st in ((0.0,) if fam == "none" else STRENGTHS):
                for seed in SEEDS:
                    out.append(dict(kind="H3", res="R4", n_train=n,
                                    fam=fam, st=st, seed=seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--out", default="results/physics_vs_smoothness_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.prepare:
        for sid, (stride, ti) in NEEDED.items():
            d = folder_for(sid)
            if (d / "protocol.json").exists():
                print("[skip]", sid, d.name, flush=True)
                continue
            info = prepare(SOURCE, d, sizes=(1000, 200, 200), seed=20260922,
                           time_index=ti, stride=stride)
            print("[prepared]", sid, d.name, info["cache_sha256"][:12], flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = ROOT / args.out
    (out / "runs").mkdir(parents=True, exist_ok=True)

    data = {}
    for res in ("R4", "R8"):
        a, pa = load(f"{res}_a")
        b, pb = load(f"{res}_b")
        if pa["sample_ids"] != pb["sample_ids"]:
            raise ValueError(f"{res}: the two time slices use different sample ids")
        data[res] = (a, b, pa, float(pb["target_time"] - pa["target_time"]),
                     float(pa["file_Nu"]))
    atomic_json(out / "plan.json",
                dict(H2=H2_RESOLUTIONS, H3=H3_SIZES, strengths=STRENGTHS,
                     band=BAND, steps=STEPS, seeds=SEEDS,
                     nu_note="PDEBench Burgers diffusion is Nu/pi",
                     code_hash=code_fingerprint(ROOT), env=env_fingerprint(str(device))))

    done = 0
    for t in tasks():
        res = t["res"]
        arr_a, arr_b, pa, dt, nu = data[res]
        key = f"{t['kind']}_{res}_n{t['n_train']}_{t['fam']}_{t['st']}_s{t['seed']}"
        target = out / "runs" / f"{key}.json"
        if target.exists():
            continue
        try:
            rec = train_one((arr_a, arr_b), pa, dt, nu, device,
                            t["seed"], t["fam"], t["st"], t["n_train"])
            rec["status"] = "ok"
        except Exception as exc:                                # noqa: BLE001
            rec = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        rec.update({k: t[k] for k in ("kind", "res", "n_train", "fam", "st", "seed")})
        atomic_json(target, rec)
        done += 1
        print(f"[{done}] {key} status={rec['status']} "
              f"test={rec.get('test_mean_rel_l2', float('nan')):.4f}", flush=True)
        if args.limit and done >= args.limit:
            print("[limit reached] re-run to continue", flush=True)
            break
    print(f"total runs: {len(list((out / 'runs').glob('*.json')))}")


if __name__ == "__main__":
    main()
