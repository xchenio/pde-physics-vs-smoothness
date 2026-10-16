"""汇总 D5 的 H2/H3，给出协议 §3 的判定。

流程（全部事前定义，不看结果再挑）：

1. 对每个 (kind, res, n_train)，每个族**独立**按其 val 均值选一个强度
   （沿用 FNO_REGULARIZATION_V1 的 "minimum three-seed development mean"）；
   `none` 族无强度。
2. 在该强度下取 test（3 seed），比较
       PDE - best_smoothness,  best_smoothness = min(gradient, curvature)
3. 判定：配对 bootstrap（按 seed 配对，10k 次，Bonferroni 校正条件数 5），
   要求 均值差 < 0 且 校正后区间上界 < 0，且收益 > 1% 门槛。
   不满足即判该假设**不成立**（负结果），不重试、不调强度。

用法：
    D:/miniconda3/envs/pytorch/python.exe scripts/physics_vs_smoothness_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.public_benchmark import atomic_json  # noqa: E402

N_CONDITIONS = 5          # H2 两个分辨率 + H3 三个样本量
THRESHOLD = 0.01          # 最小 1% 相对增益
BOOT = 10000


def load(out):
    rows = []
    for p in sorted((out / "runs").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "ok":
            rows.append(r)
    return rows


def group(rows):
    g = {}
    for r in rows:
        key = (r["kind"], r["res"], r["n_train"])
        g.setdefault(key, {}).setdefault(r["fam"], {}).setdefault(r["st"], []).append(r)
    return g


def pick(g) -> dict:
    """每个族按 val 均值选强度，返回 {key: {fam: dict(test=[...], val=..., st=...)}}。"""
    out = {}
    for key, fams in sorted(g.items()):
        out[key] = {}
        for fam, by_st in fams.items():
            best = min(by_st.items(),
                       key=lambda kv: float(np.mean([r["val_mean_rel_l2"] for r in kv[1]])))
            tests = np.array([r["test_mean_rel_l2"] for r in best[1]], dtype=float)
            out[key][fam] = dict(st=best[0], tests=tests.tolist(),
                                 test=float(tests.mean()),
                                 val=float(np.mean([r["val_mean_rel_l2"]
                                                    for r in best[1]])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/physics_vs_smoothness_v1")
    args = ap.parse_args()
    out = ROOT / args.out
    rows = load(out)
    picked = pick(group(rows))

    rng = np.random.default_rng(20261016)
    alpha = 0.05 / N_CONDITIONS
    lines = ["| condition | family | chosen strength | test rel L2 (mean of 3 seeds) |",
             "|---|---|---|---|"]
    verdicts = {}
    for key, fams in picked.items():
        kind, res, n = key
        label = (f"H2 res={res} n=1000" if kind == "H2" else f"H3 n={n} res={res}")
        lines.append(f"| **{label}** | | | |")
        for fam in ("none", "physics", "gradient", "curvature"):
            if fam not in fams:
                continue
            v = fams[fam]
            lines.append(f"| | {fam} | {v['st']} | {v['test']:.4f} |")

        if "physics" not in fams:
            continue
        smooth = [fam for fam in ("gradient", "curvature") if fam in fams]
        if not smooth:
            continue
        best_fam = min(smooth, key=lambda f: fams[f]["test"])
        p = np.asarray(fams["physics"]["tests"], dtype=float)
        s = np.asarray(fams[best_fam]["tests"], dtype=float)
        n0 = np.asarray(fams["none"]["tests"], dtype=float) if "none" in fams else None
        # d > 0 表示 PDE 比对照更好（相对增益）
        d = (s - p) / s
        n_p = len(d)
        idx = rng.integers(0, n_p, size=(BOOT, n_p))
        boot = d[idx].mean(axis=1)
        lo, hi = float(np.percentile(boot, 100 * alpha / 2)), \
            float(np.percentile(boot, 100 * (1 - alpha / 2)))
        gain = float(d.mean())
        # 假设成立 = PDE 相对平滑正则**显著更好**：增益超门槛且校正后下界 > 0。
        # （区间上界 < 0 说明的是"PDE 显著更差"，那是证伪，不是成立。）
        holds = bool(gain > THRESHOLD and lo > 0)

        # 附带对照：PDE 相对"无惩罚"。这是正面结论（物理是否有用），
        # 与"是否优于平滑正则"是两个不同的问题，不能混为一谈。
        vs_none = None
        if n0 is not None:
            d0 = (n0 - p) / n0
            b0 = d0[rng.integers(0, n_p, size=(BOOT, n_p))].mean(axis=1)
            vs_none = dict(gain=float(d0.mean()),
                           ci=[float(np.percentile(b0, 100 * alpha / 2)),
                               float(np.percentile(b0, 100 * (1 - alpha / 2)))])
        verdicts[label] = dict(best_smoothness_family=best_fam,
                               best_smoothness_strength=fams[best_fam]["st"],
                               physics_strength=fams["physics"]["st"],
                               gain_pde_over_smoothness=gain,
                               boot_ci_corrected=[lo, hi],
                               gain_pde_over_none=(vs_none or {}).get("gain"),
                               ci_pde_over_none=(vs_none or {}).get("ci"),
                               n_pairs=n_p,
                               hypothesis_holds=holds)
        lines.append(
            f"| | **PDE - {best_fam}** | | gain={gain:+.4f} "
            f"CI={lo:+.4f}..{hi:+.4f} → "
            f"{'holds' if holds else 'does NOT hold (negative result)'} |")
        if vs_none is not None:
            lines.append(
                f"| | **PDE - none** | | gain={vs_none['gain']:+.4f} "
                f"CI={vs_none['ci'][0]:+.4f}..{vs_none['ci'][1]:+.4f} |")

    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    atomic_json(out / "summary.json",
                dict(picked={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in picked.items()},
                     verdicts=verdicts, n_runs=len(rows),
                     bonferroni_alpha=alpha, threshold=THRESHOLD))
    print(text)
    print()
    for k, v in verdicts.items():
        print(f"{k}: gain={v['gain_pde_over_smoothness']:+.4f} "
              f"CI={v['boot_ci_corrected']} holds={v['hypothesis_holds']}")


if __name__ == "__main__":
    main()
