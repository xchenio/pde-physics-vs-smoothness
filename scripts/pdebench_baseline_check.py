"""D6：PDEBench 上先把朴素基线打过。

README 记录的硬伤之一是"基线在 PDEBench 上连空间均值基线都没打过"。
本脚本把这件事量化：

  B1 mean-field  —— 预测 = 训练集目标场的**逐点平均**（一个常数场，不训练）
  B2 input-mean  —— 预测 = 每个样本**输入场的均值**（Burgers 质量守恒 ⇒ 解的
                    均值等于初值均值，这是一个物理上成立的朴素常数预测）

再训练若干 FNO/DeepONet 配置，报告它们相对这两个朴素基线的位置。
在基线没有明确优于 B1/B2 之前，任何"表示/优化归因"的结论都缺乏说服力。

可续跑：每个 (config, seed) 完成即落盘；重跑自动跳过。

用法：
    D:/miniconda3/envs/pytorch/python.exe scripts/pdebench_baseline_check.py --device cuda
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

from src.fno import FNO1d                                       # noqa: E402
from src.models import DeepONet                                 # noqa: E402
from src.public_benchmark import atomic_json, file_hash         # noqa: E402
from src.utils import code_fingerprint, env_fingerprint, set_seed  # noqa: E402
from scripts.run_public_benchmark import metrics                # noqa: E402

CONFIGS = {
    "fno_w32_m16_s1500": dict(kind="fno", width=32, modes=16, steps=1500),
    "fno_w32_m16_s6000": dict(kind="fno", width=32, modes=16, steps=6000),
    "fno_w64_m32_s6000": dict(kind="fno", width=64, modes=32, steps=6000),
    "donet_fourier_s6000": dict(kind="deeponet_fourier", steps=6000),
    "donet_mlp_s6000": dict(kind="deeponet_mlp", steps=6000),
}
SEEDS = (0, 1, 2)
BATCH, LR = 32, 1e-3


def naive_baselines(arrays):
    """返回两个朴素基线的 test 指标（不训练）。"""
    y_train = np.asarray(arrays["train_y"], dtype=np.float64)
    y_test = np.asarray(arrays["test_y"], dtype=np.float64)
    x_test = np.asarray(arrays["test_x"], dtype=np.float64)

    out = {}
    mean_field = np.broadcast_to(y_train.mean(axis=0), y_test.shape)   # 逐点平均场
    out["B1_mean_field"] = metrics(mean_field, y_test)

    # Burgers（周期、无源）：∫u dx 守恒，故解的均值 ≈ 初值均值
    in_mean = x_test.mean(axis=1, keepdims=True)
    out["B2_input_mean_conservation"] = metrics(
        np.broadcast_to(in_mean, y_test.shape), y_test)
    return out


def build(kind, in_dim, cfg, device):
    if kind == "fno":
        return FNO1d(width=cfg.get("width", 32), modes=cfg.get("modes", 16),
                     layers=4).to(device)
    return DeepONet(branch_in=in_dim, trunk_in=1, width=64,
                    branch_hidden=128, trunk_hidden=128, branch_layers=3,
                    trunk_layers=3,
                    trunk_type="fourier" if kind == "deeponet_fourier" else "mlp"
                    ).to(device)


def train_one(name, cfg, arrays, protocol, device, seed):
    norm = protocol["normalization"]
    xmean, xstd = norm["input_mean"], max(norm["input_std"], 1e-6)
    ymean, ystd = norm["target_mean"], max(norm["target_std"], 1e-6)
    xs = {k: torch.tensor((arrays[k + "_x"] - xmean) / xstd, device=device)
          for k in ("train", "val", "test")}
    ys = torch.tensor((arrays["train_y"] - ymean) / ystd, device=device)
    coords = torch.tensor(arrays["x"][:, None], device=device, dtype=torch.float32)

    set_seed(seed + 2000)
    model = build(cfg["kind"], xs["train"].shape[1], cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed + 3000)
    model.train()
    started = time.time()
    for _ in range(cfg["steps"]):
        idx = rng.choice(len(ys), size=min(BATCH, len(ys)), replace=False)
        loss = ((model(xs["train"][idx], coords) - ys[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    rec = {"config": name, "seed": seed, "steps": cfg["steps"],
           "real_parameters": int(sum(p.numel() * (2 if p.is_complex() else 1)
                                      for p in model.parameters())),
           "seconds": round(time.time() - started, 2)}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            pred = torch.cat([model(v, coords) for v in xs[split].split(64)])
            pred = pred.cpu().numpy() * ystd + ymean
            rec[f"{split}_mean_rel_l2"] = metrics(pred, arrays[split + "_y"])["mean_relative_l2"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="external_data/pdebench/fixed_t1_v1")
    ap.add_argument("--out", default="results/pdebench_baseline_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir, out = ROOT / args.data_dir, ROOT / args.out
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    if file_hash(data_dir / "pairs.npz") != protocol["cache_sha256"]:
        raise ValueError("pair cache changed")
    with np.load(data_dir / "pairs.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)

    naive = naive_baselines(arrays)
    atomic_json(out / "naive_baselines.json", naive)
    atomic_json(out / "plan.json",
                dict(configs=CONFIGS, seeds=SEEDS,
                     code_hash=code_fingerprint(ROOT), env=env_fingerprint(str(device))))

    print("naive baselines (test mean relative L2):")
    for k, v in naive.items():
        print(f"  {k}: {v['mean_relative_l2']:.4f}")

    done = 0
    for name, cfg in CONFIGS.items():
        for seed in SEEDS:
            target = out / "runs" / f"{name}_s{seed}.json"
            if target.exists():
                continue
            try:
                rec = train_one(name, cfg, arrays, protocol, device, seed)
                rec["status"] = "ok"
            except Exception as exc:                              # noqa: BLE001
                rec = {"config": name, "seed": seed, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            atomic_json(target, rec)
            done += 1
            print(f"[{done}] {name}_s{seed} status={rec['status']} "
                  f"test={rec.get('test_mean_rel_l2', float('nan')):.4f}", flush=True)
            if args.limit and done >= args.limit:
                print("[limit reached] re-run to continue", flush=True)
                return summarize(out, naive)

    return summarize(out, naive)


def summarize(out, naive):
    b1 = naive["B1_mean_field"]["mean_relative_l2"]
    b2 = naive["B2_input_mean_conservation"]["mean_relative_l2"]
    rows = ["| config | seed | test mean rel L2 | vs B1 mean-field | vs B2 input-mean |",
            "|---|---|---:|---:|---:|"]
    for p in sorted((out / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            rows.append(f"| `{r['config']}` | {r['seed']} | ERROR | {r.get('error','')} | |")
            continue
        t = r["test_mean_rel_l2"]
        rows.append(f"| `{r['config']}` | {r['seed']} | {t:.4f} "
                    f"| {'BETTER' if t < b1 else 'WORSE'} ({(b1 - t) / b1:+.1%}) "
                    f"| {'BETTER' if t < b2 else 'WORSE'} ({(b2 - t) / b2:+.1%}) |")
    text = "\n".join(rows)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    return text


if __name__ == "__main__":
    main()
