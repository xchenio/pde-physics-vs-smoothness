"""汇总干预选择实验，给出协议 §5 的三个指标与 §8 的失败判定。

口径（与 docs/INTERVENTION_SELECTION_PROTOCOL_V1.md 一致）：

    gain(strategy) = (test(I0) - test(strategy)) / test(I0)        # 相对收益
    wrong(strategy) = 1[test(strategy) > test(I0)]                 # 错误选择
    net(strategy)   = gain / (consumed_steps / BASE_STEPS)         # 每单位预算的收益

`rand` 有 3 次重复：先按 (setting, arch, seed) 内部取均值，再进入跨设置统计。
主统计只用**未见设置**（held_out=True，即 T2–T5）；T1 只作校准对照，不进主统计。

统计检验：对 S-RULE 与 S-RAND 做**配对** bootstrap（按 setting×arch×seed 配对重抽样，
10k 次），报告差异均值与 2.5%/97.5% 分位；不宣称"显著"，只给区间。

用法：
    D:/miniconda3/envs/pytorch/python.exe scripts/intervention_selection_report.py
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

SETTINGS = ("T1", "T2", "T3", "T4", "T5")
STRATS = ("rule", "fix", "rand", "val")


def load_runs(out: Path):
    runs = []
    for p in sorted((out / "runs").glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        if r.get("status") == "ok":
            runs.append(r)
    return runs


def index(runs):
    """(setting, arch, seed) -> {strategy: [rec, ...]}，并给出 baseline test。"""
    table = {}
    for r in runs:
        key = (r["setting"], r["architecture"], r["seed"])
        table.setdefault(key, {"baseline": None, "strats": {}})
        if r["strategy"] == "baseline":
            table[key]["baseline"] = r
        else:
            table[key]["strats"].setdefault(r["strategy"], []).append(r)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/intervention_selection_v1")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    out = ROOT / args.out
    runs = load_runs(out)
    table = index(runs)

    per_case = []
    for key, entry in sorted(table.items()):
        sid, arch, seed = key
        base = entry["baseline"]
        if base is None:
            continue
        b = base["test_mean_rel_l2"]
        row = {"setting": sid, "arch": arch, "seed": seed,
               "held_out": sid != "T1", "baseline_test": b,
               "ratios": base.get("ratios"), "chosen_by_rule": base.get("chosen_by_rule")}
        for s in STRATS:
            recs = entry["strats"].get(s)
            if not recs:
                continue
            tests = np.array([r["test_mean_rel_l2"] for r in recs], dtype=float)
            steps = np.array([r.get("consumed_steps", 1500) for r in recs], dtype=float)
            gain = (b - tests) / b
            row[f"{s}_gain"] = float(gain.mean())
            row[f"{s}_wrong"] = float((tests > b).mean())
            row[f"{s}_net"] = float((gain / (steps / 1500.0)).mean())
            row[f"{s}_steps"] = float(steps.mean())
            row[f"{s}_test"] = float(tests.mean())
        per_case.append(row)

    def agg(rows, field):
        v = np.array([r[field] for r in rows if field in r], dtype=float)
        if len(v) == 0:
            return None                      # 没数据就别造 NaN，JSON 里也拒绝 NaN
        return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))

    groups = {}
    scope = {"held_out(T2-T5)": [r for r in per_case if r["held_out"]],
             "calibration(T1)": [r for r in per_case if not r["held_out"]],
             "all": per_case}
    for name, rows in scope.items():
        if not rows:
            continue
        groups[name] = {}
        for arch in sorted({r["arch"] for r in rows}):
            sub = [r for r in rows if r["arch"] == arch]
            groups[name][arch] = {}
            for s in STRATS:
                for field in (f"{s}_gain", f"{s}_wrong", f"{s}_net"):
                    got = agg(sub, field)
                    if got is not None:
                        m, sd, n = got
                        groups[name][arch][field] = [m, sd, n]

    # 配对 bootstrap：rule vs rand（在未见设置上，按 case 配对）
    ho = [r for r in per_case if r["held_out"] and "rule_gain" in r and "rand_gain" in r]
    pairs = np.array([[r["rule_gain"], r["rand_gain"],
                       r.get("rule_net", 0.0), r.get("rand_net", 0.0)] for r in ho])
    boot = {"n_pairs": int(len(pairs))}
    if len(pairs) > 1:
        rng = np.random.default_rng(20261016)
        idx = rng.integers(0, len(pairs), size=(args.bootstrap, len(pairs)))
        dg = pairs[idx, 0].mean(1) - pairs[idx, 1].mean(1)
        dn = pairs[idx, 2].mean(1) - pairs[idx, 3].mean(1)
        boot["gain_rule_minus_rand"] = [float(np.mean(dg)),
                                        float(np.percentile(dg, 2.5)),
                                        float(np.percentile(dg, 97.5))]
        boot["net_rule_minus_rand"] = [float(np.mean(dn)),
                                       float(np.percentile(dn, 2.5)),
                                       float(np.percentile(dn, 97.5))]
        boot["p_rule_better_than_rand"] = float((dg > 0).mean())
        # rule vs val
        pv = np.array([[r.get("rule_gain", np.nan), r.get("val_gain", np.nan)]
                       for r in ho
                       if "rule_gain" in r and "val_gain" in r])
        if len(pv) > 1:
            j = rng.integers(0, len(pv), size=(args.bootstrap, len(pv)))
            dv = pv[j, 0].mean(1) - pv[j, 1].mean(1)
            boot["gain_rule_minus_val"] = [float(np.mean(dv)),
                                           float(np.percentile(dv, 2.5)),
                                           float(np.percentile(dv, 97.5))]

    # 失败判定（协议 §8）
    rule_wrong = agg([r for r in per_case if r["held_out"]], "rule_wrong")
    verdict = {
        "F1_rule_better_than_rand": bool(boot.get("p_rule_better_than_rand", 0) > 0.95),
        "F2_rule_better_than_val": bool(boot.get("gain_rule_minus_val", [0, 0, 0])[1] > 0),
        "F3_wrong_rate_over_one_third": bool(
            rule_wrong is not None and rule_wrong[0] > 1.0 / 3),
        "rule_wrong_rate_held_out": rule_wrong,
        "note": "任一 F1/F2 不成立或 F3 成立即判为失败（协议 §8）",
    }

    lines = ["| scope | arch | strategy | gain (mean±sd) | wrong rate | net/step |",
             "|---|---|---|---|---|---|"]
    for name, archs in groups.items():
        for arch, stats in archs.items():
            for s in STRATS:
                g = stats.get(f"{s}_gain"); w = stats.get(f"{s}_wrong")
                if not g:
                    continue
                net = stats.get(f"{s}_net", [float("nan")])[0]
                lines.append(f"| {name} | {arch} | {s} | {g[0]:+.4f} ± {g[1]:.4f} "
                             f"(n={g[2]}) | {w[0]:.2f} | {net:+.4f} |")
    report = "\n".join(lines)
    (out / "selection_report.md").write_text(report + "\n", encoding="utf-8")
    atomic_json(out / "selection_summary.json",
                dict(per_case=per_case, groups=groups, bootstrap=boot,
                     verdict=verdict, n_runs=len(runs)))
    print(report)
    print("\nbootstrap:", json.dumps(boot, ensure_ascii=False))
    print("verdict:", json.dumps(verdict, ensure_ascii=False))


if __name__ == "__main__":
    main()
