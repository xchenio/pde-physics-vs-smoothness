"""D1：把"表示 / 优化 / 泛化"三项归因从 DeepONet 推广到 FNO。

## 统一的 LB 定义（readout-projection lower bound）

许多算子学习架构的最后一步都是**共享的线性 readout**：

    DeepONet :  out(u)(y) = <b(u), phi(y)> + bias       phi 只依赖 y，系数逐样本
    FNO1d    :  out(u)(x) = <W, z_u(x)>  + bias         z 依赖 u，W 跨样本与位置共享

两者都能写成"把当前参数下的 readout 特征投影到目标上"的最小二乘问题，
差别只在**允许重优化哪些系数**：

| family    | 可重优化的自由度        | 约束含义                       |
|-----------|-------------------------|--------------------------------|
| basis     | 每个样本独立的系数 B    | 解必须落在 trunk 张成的固定低维函数空间里 |
| pointwise | 全局共享的 (W, b)       | 同一组通道权重必须同时服务所有样本         |

于是定义（归一化空间、相对 Frobenius）：

    LB(split) = min_{readout 允许的自由参数} ||pred_hat - Y_hat||_F / ||Y_hat||_F

它**恒 ≤ 该 split 上任何实际模型的误差**（实际模型只是该族中的一个元素），
所以三项分解

    test_rel = LB + opt_gap + gen_gap,   opt_gap = train_rel - LB,  gen_gap = test_rel - train_rel

里 opt_gap ≥ 0 是**数学保证**的，不是经验近似。

同时对 pointwise family 额外报告 `lb_refit`：只在 train 上重拟合 readout、
再用到目标 split——这是"真的可以做到的干预"所能达到的误差（≥ lb_oracle）。

## 可续跑

每完成一个 run 立即 atomic 写 `<out>/runs/<name>.json` 并更新 `progress.json`；
重跑时已完成的 run 自动跳过。机器休眠/中断后直接重跑同一条命令即可。

用法：
    D:/miniconda3/envs/pytorch/python.exe scripts/readout_attribution.py
    D:/miniconda3/envs/pytorch/python.exe scripts/readout_attribution.py --device cpu
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

from src.fno import FNO1d                                     # noqa: E402
from src.models import DeepONet                               # noqa: E402
from src.public_benchmark import atomic_json, file_hash       # noqa: E402
from src.utils import code_fingerprint, env_fingerprint       # noqa: E402
from scripts.run_public_benchmark import metrics              # noqa: E402

FAMILY = {"fno": "pointwise", "deeponet_mlp": "basis", "deeponet_fourier": "basis"}


# ------------------------------------------------------------ LB 核心

def lb_basis(phi: np.ndarray, y: np.ndarray) -> dict:
    """family=basis：每个样本独立地在 span([phi, 1]) 上投影。"""
    x, p = phi.shape
    a = np.column_stack([np.asarray(phi, dtype=np.float64), np.ones(x)])
    q, _ = np.linalg.qr(a)
    y = np.asarray(y, dtype=np.float64)
    res = y - (y @ q) @ q.T
    denom = float(np.linalg.norm(y))
    return {"lb": float(np.linalg.norm(res) / max(denom, 1e-300)),
            "residual_fro": float(np.linalg.norm(res)),
            "readout_dof": f"{p + 1} per sample", "family": "basis"}


def lb_pointwise(psi: np.ndarray, y: np.ndarray,
                 train_psi=None, train_y=None) -> dict:
    """family=pointwise：全局共享 (W, b)，一次最小二乘。

    psi: (n, X, p) readout 特征；y: (n, X) 目标（同一归一化空间）。
    """
    n, x, p = psi.shape
    a = np.concatenate([psi.reshape(-1, p), np.ones((n * x, 1))], axis=1)
    a = np.asarray(a, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    coef, *_ = np.linalg.lstsq(a, yy, rcond=None)
    res = yy - a @ coef
    denom = float(np.linalg.norm(yy))
    out = {"lb": float(np.linalg.norm(res) / max(denom, 1e-300)),
           "residual_fro": float(np.linalg.norm(res)),
           "readout_dof": f"{p + 1} total (shared)",
           "family": "pointwise"}
    if train_psi is not None and train_y is not None:
        m = train_psi.shape[0] * train_psi.shape[1]
        ta = np.concatenate([train_psi.reshape(-1, p), np.ones((m, 1))], axis=1)
        tcoef, *_ = np.linalg.lstsq(np.asarray(ta, dtype=np.float64),
                                    np.asarray(train_y, dtype=np.float64).reshape(-1),
                                    rcond=None)
        ra = np.concatenate([psi.reshape(-1, p), np.ones((n * x, 1))], axis=1)
        rres = yy - np.asarray(ra, dtype=np.float64) @ tcoef
        out["lb_refit_on_train"] = float(np.linalg.norm(rres) / max(denom, 1e-300))
    return out


def rel_fro(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true))
                 / max(np.linalg.norm(np.asarray(true)), 1e-300))


# ------------------------------------------------------------ 特征提取

def build_model(arch: str, in_dim: int, device: torch.device):
    if arch == "fno":
        return FNO1d(width=32, modes=16, layers=4).to(device)
    return DeepONet(branch_in=in_dim, trunk_in=1, width=64,
                    branch_hidden=128, trunk_hidden=128, branch_layers=3,
                    trunk_layers=3,
                    trunk_type="fourier" if arch == "deeponet_fourier" else "mlp"
                    ).to(device)


def predict(model, arch, inputs, coords):
    return model(inputs, coords)


def readout_features(model, arch, inputs, coords):
    """返回 (psi, phi)：pointwise family 用 psi (n,X,p)；basis family 用 phi (X,p)。"""
    if arch == "fno":
        x = coords.flatten().expand(inputs.shape[0], 1, -1)
        h = model.lift(torch.cat([inputs[:, None, :], x], dim=1))
        for spec, local in zip(model.spectral, model.local):
            h = torch.nn.functional.gelu(spec(h) + local(h))
        z = model.project[:2](h)                      # (n, p, X)
        return z.transpose(1, 2), None                # (n, X, p)
    phi = model.trunk_features(coords)                # (X, p)
    return None, phi


# ------------------------------------------------------------ 单个 run

def process(run_dir: Path, arrays: dict, protocol: dict, device: torch.device) -> dict:
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    cfg = res["identity"]["configuration"]
    arch = cfg["architecture"]
    cp = torch.load(run_dir / "checkpoints" / "latest.pt",
                    map_location=device, weights_only=False)
    norm = protocol["normalization"]
    xmean, xstd = norm["input_mean"], max(norm["input_std"], 1e-6)
    ymean, ystd = norm["target_mean"], max(norm["target_std"], 1e-6)

    xs = {k: torch.tensor((arrays[k + "_x"] - xmean) / xstd, device=device)
          for k in ("train", "val", "test")}
    coords = torch.tensor(arrays["x"][:, None], device=device, dtype=torch.float32)
    model = build_model(arch, xs["train"].shape[1], device)
    model.load_state_dict(cp["model"])
    model.eval()

    out = {"run": run_dir.name, "architecture": arch, "seed": cfg["seed"],
           "family": FAMILY[arch], "steps": cfg["steps"]}
    with torch.no_grad():
        raw, feats = {}, {}
        for split in ("train", "val", "test"):
            z = torch.cat([predict(model, arch, v, coords)
                           for v in xs[split].split(64)])
            raw[split] = z.cpu().numpy().astype(np.float64)
            psi, phi = readout_features(model, arch, xs[split], coords)
            feats[split] = (None if psi is None else psi.cpu().numpy().astype(np.float64),
                            None if phi is None else phi.cpu().numpy().astype(np.float64))

    # 1) 复现既有指标（必须与存档一致，否则说明口径不同）
    replay = {}
    for split in ("train", "val", "test"):
        pred = raw[split] * ystd + ymean
        replay[split] = metrics(pred, arrays[split + "_y"])
    out["replayed_mean_relative_l2"] = {s: replay[s]["mean_relative_l2"]
                                        for s in replay}
    out["archive_mean_relative_l2"] = {s: res["metrics"][s]["mean_relative_l2"]
                                       for s in ("train", "val", "test")}
    out["replay_max_abs_diff"] = max(
        abs(out["replayed_mean_relative_l2"][s] - out["archive_mean_relative_l2"][s])
        for s in ("train", "val", "test"))

    # 2) 归一化空间的三项归因
    yhat = {s: (arrays[s + "_y"].astype(np.float64) - ymean) / ystd for s in xs}
    lb = {}
    for split in ("train", "val", "test"):
        psi, phi = feats[split]
        if FAMILY[arch] == "basis":
            lb[split] = lb_basis(phi, yhat[split])
        else:
            lb[split] = lb_pointwise(psi, yhat[split],
                                     train_psi=feats["train"][0],
                                     train_y=yhat["train"])
    rel = {s: rel_fro(raw[s], yhat[s]) for s in xs}
    out["relative_frobenius"] = rel
    out["lb"] = {s: lb[s]["lb"] for s in lb}
    out["lb_detail"] = {s: {k: v for k, v in lb[s].items() if k != "lb"} for s in lb}

    # 严格可加的两侧分解。LB 必须与同一侧的训练误差比，才能保证 opt_gap >= 0。
    opt_gap_train = rel["train"] - lb["train"]["lb"]
    gen_gap = rel["test"] - rel["train"]
    out["attribution"] = {
        "test_relative_fro": rel["test"],
        "LB_train": lb["train"]["lb"],
        "LB_test": lb["test"]["lb"],
        "opt_gap_train": opt_gap_train,
        "gen_gap": gen_gap,
        "test_minus_LB_test": rel["test"] - lb["test"]["lb"],
        "note": ("opt_gap_train = train_rel - LB_train >= 0 由定义保证；"
                 "test_rel - LB_test 是 test 侧离地板总距离，"
                 "等于 opt_gap_train + gen_gap + (LB_train - LB_test)"),
    }
    # 数学保证的守门检查：任何一侧出现负值都说明口径错了，必须暴露而不是抹平。
    out["guarantee_ok"] = bool(opt_gap_train >= -1e-9
                               and rel["test"] - lb["test"]["lb"] >= -1e-9)
    if "lb_refit_on_train" in lb["test"]:
        out["attribution"]["lb_refit_on_train_then_test"] = lb["test"]["lb_refit_on_train"]
    return out


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="external_data/pdebench/fixed_t1_v1")
    ap.add_argument("--runs", default="results/public_fixed_t1_v1")
    ap.add_argument("--out", default="results/readout_attribution_v1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir, runs_dir, out = ROOT / args.data_dir, ROOT / args.runs, ROOT / args.out
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(data_dir / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError("pair cache changed")
    with np.load(data_dir / "pairs.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    atomic_json(out / "protocol.json", protocol)

    run_dirs = sorted(p for p in runs_dir.iterdir()
                      if (p / "result.json").exists()
                      and (p / "checkpoints" / "latest.pt").exists())
    atomic_json(out / "plan.json",
                dict(created=time.time(), runs=[p.name for p in run_dirs],
                     data=str(data_dir), code_hash=code_fingerprint(ROOT),
                     env=env_fingerprint(str(device))))

    results = []
    for i, rd in enumerate(run_dirs, 1):
        target = out / "runs" / f"{rd.name}.json"
        if target.exists():
            results.append(json.loads(target.read_text(encoding="utf-8")))
            print(f"[skip {i}/{len(run_dirs)}] {rd.name}", flush=True)
            continue
        started = time.time()
        try:
            rec = process(rd, arrays, protocol, device)
            rec["status"] = "ok"
            rec["seconds"] = round(time.time() - started, 2)
        except Exception as exc:                                  # noqa: BLE001
            rec = {"run": rd.name, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(target, rec)
        results.append(rec)
        atomic_json(out / "progress.json",
                    dict(done=i, total=len(run_dirs), last=rd.name))
        print(f"[{i}/{len(run_dirs)}] {rd.name} status={rec['status']} "
              f"lb={rec.get('lb', {}).get('test', float('nan')):.4f} "
              f"test={rec.get('relative_frobenius', {}).get('test', float('nan')):.4f}",
              flush=True)

    table = ["| run | family | test relFro | LB_train | LB_test | opt_gap(train) | gen_gap | replay diff |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.get("status") != "ok":
            table.append(f"| `{r['run']}` | — | — | — | — | — | — | {r.get('error','')} |")
            continue
        a = r["attribution"]
        table.append(
            f"| `{r['run']}` | {r['family']} | {a['test_relative_fro']:.4f} "
            f"| {a['LB_train']:.4f} | {a['LB_test']:.4f} | {a['opt_gap_train']:.4f} "
            f"| {a['gen_gap']:.4f} | {r['replay_max_abs_diff']:.2e} |")

    # 按架构汇总，便于直接引用
    groups = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = r["architecture"]
        groups.setdefault(key, []).append(r)
    summary_rows = []
    for key, rows in sorted(groups.items()):
        def stat(fn):
            v = np.array([fn(r) for r in rows], dtype=float)
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
        m_lbt, s_lbt = stat(lambda r: r["attribution"]["LB_train"])
        m_test, s_test = stat(lambda r: r["attribution"]["test_relative_fro"])
        m_opt, s_opt = stat(lambda r: r["attribution"]["opt_gap_train"])
        m_gen, s_gen = stat(lambda r: r["attribution"]["gen_gap"])
        summary_rows.append(dict(architecture=key, family=rows[0]["family"], n=len(rows),
                                 LB_train=[m_lbt, s_lbt], test_relFro=[m_test, s_test],
                                 opt_gap_train=[m_opt, s_opt], gen_gap=[m_gen, s_gen]))
    atomic_json(out / "group_summary.json", dict(groups=summary_rows))
    (out / "report.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    atomic_json(out / "summary.json", dict(results=results, n=len(results)))
    print("\n" + "\n".join(table))


if __name__ == "__main__":
    main()
