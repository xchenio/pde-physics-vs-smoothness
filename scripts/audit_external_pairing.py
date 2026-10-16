"""在**第三方代码库**上运行配对契约检查（只读，不改对方代码）。

用法：

    D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py --status
    D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py \\
        --target deepxde --out results/audit_pairing_external_v1

设计原则：本脚本**不内置任何未经本机验证的第三方 API 调用**。
目标仓库如何构造 probe，写在仓库侧的 `pairing_probes.py` 里（见下方模板），
由本脚本负责加载、跑检查、写报告。这样新增一个目标只需约 20 行适配代码，
且"有没有真正跑过"由 `status` 列如实标注，不会被静默跳过。

`pairing_probes.py` 模板（放在目标仓库根目录）：

    import numpy as np

    def probes():
        \"\"\"返回 [(label, coef, coords, fn)]，fn(coef, coords) -> 逐样本场值 (B, N)。\"\"\"
        def make_fn(...):
            def fn(coef, coords):
                ...
                return values
            return fn
        coef = ...      # (B, n_coef)
        coords = ...    # (B, N) 或 (B, N, d)
        return [("my_lib/input_fn", coef, coords, make_fn())]

约定：B 明显小于 N（建议 B=4, N>=32），否则 C1 的形状判别会退化。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.utils import code_fingerprint, env_fingerprint, save_json  # noqa: E402
from pairing_check import check_evaluator, format_report           # noqa: E402

# 目标清单。repo 未就位时状态一律为 MISSING，**不得**替换为"已通过"。
TARGETS = {
    "deepxde": {
        "desc": "DeepXDE（lululxvi/deepxde），CartesianProd DeepONet 数据管线",
        "clone": "pip install deepxde   // git clone https://github.com/lululxvi/deepxde",
    },
    "pi_deeponet": {
        "desc": "PI-DeepONet 官方（PredictiveIntelligenceLab/Physics-informed-DeepONets）",
        "clone": "git clone https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets",
    },
    "pino": {
        "desc": "PINO 官方（neuraloperator/physics_informed）",
        "clone": "git clone https://github.com/neuraloperator/physics_informed",
    },
}


def probe_path(repo_dir: Path, name: str | None = None) -> Path:
    """适配层优先放在 `scripts/audit_probes/<name>_probes.py`。

    放在仓库自己的目录里，而不是目标仓库内——外部源码是**只读快照**（已登记
    snapshot SHA-256），往里写文件会破坏指纹，也会让人分不清哪些是我们加的。
    """
    if name:
        local = ROOT / "scripts" / "audit_probes" / f"{name}_probes.py"
        if local.exists():
            return local
    return Path(repo_dir) / "pairing_probes.py"


def load_probes(module_path: Path):
    spec = importlib.util.spec_from_file_location("pairing_probes", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load probe module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "probes"):
        raise AttributeError(f"{module_path} 必须定义 probes()")
    return mod.probes()


def run_target(name, repo_dir: Path, out_root: Path, batch_limit: int | None = None):
    pp = probe_path(repo_dir, name)
    if not repo_dir.exists():
        return {"target": name, "status": "MISSING_REPO",
                "clone": TARGETS[name]["clone"], "results": []}
    if not pp.exists():
        return {"target": name, "status": "MISSING_PROBE",
                "note": f"需要适配层：{pp}", "results": []}

    entries = load_probes(pp)
    if batch_limit:
        entries = entries[:batch_limit]
    results = []
    for label, coef, coords, fn in entries:
        try:
            res = check_evaluator(fn, coef, coords, label=f"{name}/{label}")
        except Exception as exc:                             # noqa: BLE001
            res = {"label": f"{name}/{label}", "passed": False, "checks": {},
                   "leaves": {}, "failing_leaves": {}, "error": repr(exc)}
        results.append(res)
    status = "OK" if results else "EMPTY_PROBES"
    return {"target": name, "status": status, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=None,
                    help="要跑的目标名；省略则跑全部")
    ap.add_argument("--repo-root", action="append", default=[],
                    help="形如 NAME=DIR，指定目标仓库路径")
    ap.add_argument("--out", default="results/audit_pairing_external_v1")
    ap.add_argument("--status", action="store_true", help="只打印目标就位情况")
    args = ap.parse_args()

    repos = dict(TARGETS)
    for item in args.repo_root:
        name, _, dir_ = item.partition("=")
        if name in repos:
            repos[name]["dir"] = Path(dir_)
    # 默认位置：external/<name>
    for name, meta in repos.items():
        meta.setdefault("dir", ROOT / "external" / name)

    if args.status:
        for name, meta in repos.items():
            d = meta["dir"]
            exists = d.exists()
            probe = probe_path(d).exists()
            state = ("repo+probe ready" if (exists and probe)
                     else "repo present, probe needed" if exists else "MISSING repo")
            print(f"{name:14s} {state:28s} {d}")
            print(f"{'':14s} {meta['clone']}")
        return

    names = args.target or list(repos)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results, blocks = [], []
    for name in names:
        block = run_target(name, repos[name]["dir"], out_root)
        blocks.append(block)
        all_results.extend(block["results"])
        print(f"[{name}] status={block['status']} n={len(block['results'])}")

    if all_results:
        print("\n" + format_report(all_results))

    payload = {
        "created": str(np.datetime64("now")),
        "code_hash": code_fingerprint(ROOT),
        "env": env_fingerprint("cpu"),
        "targets": {b["target"]: b["status"] for b in blocks},
        "results": all_results,
        "n_pass": int(sum(1 for r in all_results if r.get("passed"))),
        "n_total": len(all_results),
    }
    save_json(payload, out_root / "audit_external.json")
    (out_root / "report.md").write_text(
        format_report(all_results) + "\n" if all_results
        else "# no external target was available on this machine\n",
        encoding="utf-8")
    print(f"\nwrote {out_root / 'audit_external.json'}")


if __name__ == "__main__":
    main()
