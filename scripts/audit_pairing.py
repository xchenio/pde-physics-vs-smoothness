"""D2：逐样本配对契约的**项目内审计**（含注入缺陷与传统 QA 对照）。

产出：`results/audit_pairing_v1/audit_pairing.json`（版本化目录，历史结果不动）。

三类 target：
  1. `pipeline:*`  —— 用 `check_pipeline` 把整个 `sample_points` 当黑盒检查；
  2. `evaluator:*` —— 用 `check_evaluator` 只检查 `input_at` 求值器；
  3. `fd_reference` —— 传统 QA 代理：证明在共享网格（非 batch）路径上，
     fixed 与 legacy 两种实现**逐位相同**，因此传统检查对该缺陷无鉴别力。

`legacy` 是对修复前实现的忠实复刻：把逐样本坐标一律当作共享网格来求值。
用法：

    D:/miniconda3/envs/pytorch/python.exe scripts/audit_pairing.py
    D:/miniconda3/envs/pytorch/python.exe scripts/audit_pairing.py --out results/audit_pairing_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.pdes import make_problem                      # noqa: E402
from src.utils import code_fingerprint, env_fingerprint, save_json  # noqa: E402
from pairing_check import (                            # noqa: E402
    check_evaluator, check_pipeline, format_report)

DEFAULT_PDES = ["burgers", "burgers2d", "dr", "darcy"]
SEED = 20261016
BATCH = 4
N_PTS = 32


# ------------------------------------------------------------ legacy 注入

def _legacy_wrapper(orig):
    """复刻修复前的调用约定：coords 无条件当作**共享网格**处理。

    这正是 §6.5 的缺陷来源：调用方传的是 (B, N[, d]) 的逐样本坐标，
    实现却对每个 coef[b] 在所有行上求值，再 reshape 成 (B, B, N)。
    """
    def wrapper(coef, coords, *args, **kwargs):
        coords = np.asarray(coords, dtype=np.float64)
        # 只有当输入"看起来像逐样本坐标"（首维 == batch）时才走错误的共享网格分支；
        # 真正的共享网格调用（如求解器内部的 grid）与历史实现完全一致——
        # 这正是"传统 QA 查不出该缺陷"的机理，见 fd_reference_probe。
        if coords.ndim >= 2 and coords.shape[0] == coef.shape[0]:
            n = coef.shape[0]
            flat = (coords.reshape(-1, coords.shape[-1]) if coords.ndim >= 3
                    else coords.reshape(-1))
            out = orig(coef, flat, *args, **kwargs)
            # 每个样本都对全部 B*N 行求值 -> (B, B*N)；畸形 reshape 后即 (B, B, N[, ...])
            target = ((n, n, *coords.shape[1:-1]) if coords.ndim >= 3
                      else (n, n, coords.shape[1]))
            if isinstance(out, (list, tuple)):
                shaped = [np.asarray(o).reshape(target) for o in out]
                return type(out)(shaped)
            return np.asarray(out).reshape(target)
        return orig(coef, coords, *args, **kwargs)

    return wrapper


def inject_legacy(p):
    """就地 monkeypatch 所有会吃到逐样本坐标的求值方法。"""
    for name in ("input_at", "input_and_grad", "ic_values"):
        if hasattr(p, name):
            setattr(p, name, _legacy_wrapper(getattr(p, name)))
    return p


# ------------------------------------------------------------ target 构造

def _eval_coords(name, batch, n, rng):
    """input_at 的坐标形状：标量坐标问题用 (B,N)，二维问题用 (B,N,2)。"""
    if name in ("burgers", "dr"):
        return rng.uniform(0.0, 1.0, (batch, n))
    return rng.uniform(0.0, 1.0, (batch, n, 2))


def _pipeline_fn(p, name, pts_cfg, seed):
    def fn(coef):
        rng = np.random.default_rng(seed)          # 每次调用重建，保证确定性
        return p.sample_points(coef, *pts_cfg, rng)
    return fn


PIPELINE_PTS = (16, 8, 8)          # n_col, n_ic, n_bc


def run_pipeline(problem, name, seed, coef, label):
    return check_pipeline(_pipeline_fn(problem, name, PIPELINE_PTS, seed),
                          coef, label=label)


def run_evaluator(problem, name, seed, coef, coords, label):
    return check_evaluator(problem.input_at, coef, coords, label=label)


def build_targets(pde_names, batch, n_pts, seed):
    """返回 [(spec, expect)]，spec 是描述一次审计的 dict。

    expect 是**事前登记**的期望结果（不看输出再定），用于事后统计假阳性/假阴性：
    fixed 应为 pass；legacy（注入缺陷）应为 fail。
    唯一例外：`darcy/evaluator:fixed` —— `DarcyProblem.input_at` 至今缺少逐样本
    分支（只有 `input_and_grad` 有），属于**真实存在**的契约不一致，故登记为 fail。
    """
    targets = []
    for name in pde_names:
        coef = make_problem(name).sample_coeffs(batch,
                                                np.random.default_rng(seed))
        coords = _eval_coords(name, batch, n_pts, np.random.default_rng(seed + 1))
        for variant, problem in (("fixed", make_problem(name)),
                                 ("legacy", inject_legacy(make_problem(name)))):
            for kind in ("pipeline", "evaluator"):
                expect = "fail" if variant == "legacy" else "pass"
                if expect == "pass" and name == "darcy" and kind == "evaluator":
                    expect = "fail"          # 见 docstring
                targets.append(({
                    "name": name, "kind": kind, "variant": variant,
                    "problem": problem, "coef": coef, "coords": coords,
                    "seed": seed, "label": f"{name}/{kind}:{variant}",
                }, expect))
    return targets


# ------------------------------------------------------------ 传统 QA 对照

def fd_reference_probe(name, batch, seed):
    """传统 QA 代理：在**共享网格**（非 batch）路径上求值。

    返回 (identical, grid_shape, max_abs_diff)。
    identical = True 说明传统检查路径的输出不受缺陷影响 —— 即**查不出来**。
    """
    rng = np.random.default_rng(seed)
    p = make_problem(name)
    coef = p.sample_coeffs(batch, rng)
    q = inject_legacy(make_problem(name))

    grid = np.asarray(p.grid, dtype=np.float64)          # (Ng, d)，无 batch 维
    rows = grid[:: max(1, grid.shape[0] // 512)]         # 抽样以控制耗时
    a = np.asarray(p.input_at(coef, rows))
    b = np.asarray(q.input_at(coef, rows))
    if a.shape != b.shape:
        return {"identical": False, "grid_shape": tuple(rows.shape),
                "max_abs_diff": float("nan"),
                "note": "shape mismatch on the shared-grid path"}
    return {"identical": bool(np.array_equal(a, b)),
            "grid_shape": tuple(rows.shape),
            "max_abs_diff": float(np.max(np.abs(a - b)))}


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/audit_pairing_v1")
    ap.add_argument("--pdes", default=",".join(DEFAULT_PDES))
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--n-pts", type=int, default=N_PTS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    pde_names = [s.strip() for s in args.pdes.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = build_targets(pde_names, args.batch, args.n_pts, args.seed)
    results, errors = [], []
    for spec, expect in targets:
        label = spec["label"]
        try:
            if spec["kind"] == "pipeline":
                res = run_pipeline(spec["problem"], spec["name"], spec["seed"],
                                   spec["coef"], label)
            else:
                res = run_evaluator(spec["problem"], spec["name"], spec["seed"],
                                    spec["coef"], spec["coords"], label)
        except Exception as exc:                          # noqa: BLE001
            errors.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            res = {"label": label, "passed": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "checks": {}, "leaves": {}, "failing_leaves": {},
                   "n_samples": 0}
        res["expected"] = expect
        res["agree_with_expected"] = (
            res.get("error") is None
            and (("pass" if res.get("passed") else "fail") == expect))
        results.append(res)

    fd = {name: fd_reference_probe(name, args.batch, args.seed)
          for name in pde_names}

    print(format_report(results))
    print("\n[traditional-QA proxy] shared-grid path, fixed vs legacy:")
    for name, r in fd.items():
        verdict = "IDENTICAL -> traditional check cannot tell them apart" \
            if r["identical"] else "differs"
        print(f"  {name:12s} grid={r['grid_shape']} "
              f"max|diff|={r['max_abs_diff']:.3e}  {verdict}")

    agree = [r for r in results if r["agree_with_expected"]]
    print(f"\n[registered expectation] {len(agree)}/{len(results)} targets agree "
          f"with their pre-registered outcome")
    if errors:
        print("[errors]")
        for e in errors:
            print("  ", e)

    payload = {
        "created": str(np.datetime64("now")),
        "seed": args.seed,
        "batch": args.batch,
        "n_pts": args.n_pts,
        "pipeline_pts": PIPELINE_PTS,
        "pdes": pde_names,
        "code_hash": code_fingerprint(ROOT),
        "env": env_fingerprint("cpu"),
        "results": results,
        "fd_reference": fd,
        "errors": errors,
        "n_pass": int(sum(1 for r in results if r.get("passed"))),
        "n_total": len(results),
        "n_agree_with_expected": len(agree),
    }
    save_json(payload, out_dir / "audit_pairing.json")
    (out_dir / "report.md").write_text(
        format_report(results) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'audit_pairing.json'} "
          f"({payload['n_pass']}/{payload['n_total']} passed)")


if __name__ == "__main__":
    main()
