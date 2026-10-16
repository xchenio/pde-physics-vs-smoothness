"""逐样本配对契约检查器（per-sample pairing contract checker）。

仅依赖 numpy，可整体复制到任意 DeepONet / PI-DeepONet / PINO 代码库中独立使用。

## 背景

算子学习中，输入函数（系数）是按 batch 组织的：`coef[b]` 描述第 b 个 PDE 实例。
因此初值、源项、扩散系数、边界目标等**逐样本量**必须同时满足两条契约：

    (形状) fn(coef)[...] 的第一维是 B；
    (配对) 第 b 行只依赖 coef[b]，不依赖其它样本。

若实现沿用了"共享网格"求值约定（`out.reshape(n, *x.shape)`），而调用方传入的是
**逐样本坐标** `(B, N[, d])`，numpy 会静默广播成 `(B, B, N)`：
第 b 个样本用的是"另一个样本"的物理量。它不抛异常、损失曲线看起来正常，
并且常用的 QA（**参考解的有限差分残差**）查不出来——两者不在同一条代码路径上。

本模块提供三条**不需要训练**的检验：

| 检验 | 判据 | 适用场合 | 能抓什么 |
|---|---|---|---|
| C1 shape         | 叶子数组首维 == B，且不出现 `(B, B, N)` 形疑似广播 | 全部 | 共享网格约定被误用 |
| C2 contamination | 替换其它样本的行，`第 b 行`输出不得改变 | 全部 | 跨样本混配、意外池化 |
| C3 pairing       | 循环置换 coef 的行，输出的第 i 行必须等于原 `第 perm[i] 行` | **仅当坐标由外部固定**（`check_evaluator`） | 逐样本配对被破坏 |

C3 的适用边界（实测确定，勿忽视）：

- 它**不能**用于"内部随机采样坐标"的黑盒管线（`check_pipeline`）：那种管线里
  coords 本身就是逐样本的随机变量，置换 coef 后第 i 行的坐标也跟着换，
  比较双方讲的不是同一组点，必然假 FAIL。因此 `check_pipeline` 默认只跑 C1 + C2。
- 坐标由调用方显式给出时（`check_evaluator`），置换不改变坐标，C3 才有效。

已知的**检验盲区**（针对纯 `(B,B,N)` 广播）：C2 会**漏**——
`arr[i, i]` 恰好就是"第 i 个样本在第 i 行坐标上"的正确值，
污染检验看它完全正常。这种情况下 C1 是唯一的指纹。
所以三条检验是互补关系，任一单条都不足够。

典型用法（检查一个 `(coef, coords) -> 场值` 的求值器）：

    from pairing_check import check_evaluator
    res = check_evaluator(my_input_fn, coef[:4], coords[:4])
    print(res["passed"])

约定：探测时请让 **B 明显小于配点数 N**（例如 B=4, N>=32），
否则 C1 的 `(B, N, d)` 与 `(B, B, N)` 判别会退化。
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "check_evaluator",
    "check_pipeline",
    "format_report",
    "iter_leaves",
]


# ---------------------------------------------------------------- 叶子遍历

def iter_leaves(obj: Any, prefix: str = "") -> Iterable[tuple[str, np.ndarray]]:
    """深度遍历 dict / list / tuple，产出 (名字, ndarray) 叶子。

    非数组叶子（标量、字符串、None）被跳过。名字用于报告定位，例如 "col/forcing"。
    """
    if isinstance(obj, np.ndarray):
        yield (prefix or "out"), obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_leaves(v, f"{prefix}[{i}]" if prefix else f"[{i}]")
    # 其它类型（标量 / None / Tensor 之外的对象）不参与检查


def _nominal_row(arr: np.ndarray, i: int) -> np.ndarray:
    """取"名义上第 i 个样本"的行：正确实现是 arr[i]，`(B,B,N)` 广播是 arr[i, i]。"""
    if arr.ndim >= 3 and arr.shape[0] == arr.shape[1]:
        return arr[i, i]
    return arr[i]


# ---------------------------------------------------------------- 核心检查

def _run(fn: Callable, coef: np.ndarray) -> tuple[dict, str | None]:
    try:
        return dict(iter_leaves(fn(coef))), None
    except Exception as exc:                                   # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"


def _check_core(fn: Callable, coef: np.ndarray, *, label: str,
                alt_coef: np.ndarray | None = None, rtol: float = 1e-5,
                atol: float = 1e-6,                 allow_global: Sequence[str] = (),
                singleton_call: Callable[[int], Any] | None = None,
                enable_c3: bool = True) -> dict:
    """singleton_call：给定样本下标 i，返回"只对该样本求值"的结果。

    用于那些**输入本身也是逐样本的**场合（如 fn(coef, coords)）——
    C3 必须同时把坐标裁到同一行，否则比较的是不同坐标上的值，会假 FAIL。

    enable_c3=False 时 C3 记为 n/a（内部随机采样坐标的黑盒管线适用，见模块 docstring）。
    """
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim < 2:
        raise ValueError("coef 必须是 (B, n_coef)")
    B = coef.shape[0]
    if B < 2:
        raise ValueError("batch 至少为 2，否则无法检测跨样本污染")

    if alt_coef is None:
        # 与 coef 完全不同的一组系数：循环移位即可，无需知道其分布。
        alt_coef = np.roll(coef, 1, axis=0).copy()
        same = np.all(np.isclose(alt_coef, coef, rtol=rtol, atol=atol), axis=1)
        if np.any(same):                       # 极小概率：位移后仍相同，加扰动
            idx = np.where(same)[0]
            alt_coef[idx] += 1.0

    base, err = _run(fn, coef)
    if err is not None:
        return {"label": label, "n_samples": B, "passed": False, "error": err,
                "leaves": {}, "checks": {}}

    leaves = {}
    for name, arr in base.items():
        if any(g in name for g in allow_global):
            leaves[name] = {"shape": tuple(arr.shape), "skipped": True,
                            "suspected_broadcast": False, "c1": True,
                            "c2": True, "c3": True, "max_diff": 0.0}
            continue

        arr = np.asarray(arr)
        batch_shaped = arr.ndim >= 1 and arr.shape[0] == B
        suspected = (arr.ndim == 3 and arr.shape[0] == B and arr.shape[1] == B)
        c1 = bool(batch_shaped and not suspected)

        # C2 contamination：逐个保留第 i 行，其余换成 alt_coef
        max_diff, c2_diff = 0.0, 0.0
        c2 = True
        if batch_shaped:
            for i in range(B):
                mixed = alt_coef.copy()
                mixed[i] = coef[i]
                out_i, _ = _run(fn, mixed)
                cur = out_i.get(name)
                if cur is None or np.asarray(cur).shape != arr.shape:
                    c2 = False
                    break
                a = _nominal_row(np.asarray(cur), i)
                b = _nominal_row(arr, i)
                if a.shape != b.shape:
                    c2 = False
                    break
                d = float(np.max(np.abs(a - b)))
                max_diff = max(max_diff, d)
                c2_diff = max(c2_diff, d)
                scale = float(np.max(np.abs(b)))
                if d > atol + rtol * max(scale, 1.0):
                    c2 = False
                    break

        # C3：pairing。默认用**循环置换**（保持 batch 大小，随机采样可复现）；
        # 若提供了 singleton_call（坐标由外部显式给出），则用 singleton 检验。
        c3 = True
        c3_diff = 0.0
        if batch_shaped and not enable_c3:
            c3 = None                      # n/a：本 target 不适用该检验
        if batch_shaped and enable_c3:
            if singleton_call is not None:
                for i in range(min(B, 3)):
                    out_1, _ = _run(lambda _c, i=i: singleton_call(i), coef[i: i + 1])
                    arr1 = out_1.get(name)
                    if arr1 is None:
                        c3 = False
                        break
                    arr1 = np.asarray(arr1)
                    if arr1.shape[0] != 1:
                        c3 = False
                        break
                    a, b = arr1[0], _nominal_row(arr, i)
                    if a.shape != b.shape:
                        c3 = False
                        break
                    d = float(np.max(np.abs(a - b)))
                    c3_diff = max(c3_diff, d)
                    scale = float(np.max(np.abs(b)))
                    if d > atol + rtol * max(scale, 1.0):
                        c3 = False
                        break
            else:
                perm = np.roll(np.arange(B), 1)              # 无不动点的循环置换
                out_p, _ = _run(fn, coef[perm])
                arr_p = out_p.get(name)
                if arr_p is None or np.asarray(arr_p).shape != arr.shape:
                    c3 = False
                else:
                    arr_p = np.asarray(arr_p)
                    for i in range(B):
                        a = _nominal_row(arr_p, i)
                        b = _nominal_row(arr, perm[i])
                        if a.shape != b.shape:
                            c3 = False
                            break
                        d = float(np.max(np.abs(a - b)))
                        c3_diff = max(c3_diff, d)
                        scale = float(np.max(np.abs(b)))
                        if d > atol + rtol * max(scale, 1.0):
                            c3 = False
                            break

        leaves[name] = {
            "shape": tuple(arr.shape),
            "skipped": not batch_shaped,
            "suspected_broadcast": bool(suspected),
            "c1": c1,
            "c2": c2 and batch_shaped,
            "c3": (None if (c3 is None or not batch_shaped) else bool(c3)),
            "c2_diff": c2_diff,
            "c3_diff": c3_diff,
            "max_diff": max_diff,
        }

    c3_values = [v["c3"] for v in leaves.values()]
    checks = {
        "C1_shape": all(v["c1"] for v in leaves.values()),
        "C2_contamination": all(v["c2"] for v in leaves.values()),
        "C3_pairing": (None if all(v is None for v in c3_values)
                       else all(v for v in c3_values if v is not None)),
    }
    failing = {
        "C1_shape": [k for k, v in leaves.items() if not v["c1"]],
        "C2_contamination": [k for k, v in leaves.items() if not v["c2"]],
        "C3_pairing": [k for k, v in leaves.items() if v["c3"] is False],
    }
    return {
        "label": label,
        "n_samples": B,
        "rtol": rtol,
        "atol": atol,
        "leaves": leaves,
        "checks": checks,
        "failing_leaves": failing,
        "passed": all(v for v in checks.values() if v is not None),
        "error": None,
    }


def check_pipeline(fn: Callable[[np.ndarray], Any], coef: np.ndarray, *,
                   label: str = "pipeline", alt_coef: np.ndarray | None = None,
                   rtol: float = 1e-5, atol: float = 1e-6,
                   allow_global: Sequence[str] = (),
                   enable_c3: bool = False) -> dict:
    """检查"数据管线" fn(coef) -> dict/list/array。

    fn 必须**确定性**（同样的 coef 返回同样的结果）；若内部有随机采样，
    请在闭包里固定随机源，否则 C2 会被采样差异本身污染。

    默认**不启用 C3**：黑盒管线的坐标通常是随机逐样本采样出来的，
    置换/单例检验对它结构性不适用（会产生假阳性），详见模块 docstring。
    若你的管线坐标是外部固定的，可传 `enable_c3=True`。
    """
    return _check_core(fn, coef, label=label, alt_coef=alt_coef, rtol=rtol,
                       atol=atol, allow_global=allow_global, enable_c3=enable_c3)


def check_evaluator(fn: Callable[[np.ndarray, np.ndarray], Any],
                    coef: np.ndarray, coords: np.ndarray, *,
                    label: str = "evaluator", **kw) -> dict:
    """检查"求值器" fn(coef, coords) -> 场值。

    coords: (B, N) 或 (B, N, d) 的**逐样本**坐标。
    期望输出：(B, N)。
    """
    coef = np.asarray(coef, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] != coef.shape[0]:
        raise ValueError("coords 第 0 维必须等于 batch size")

    # 坐标由调用方显式给定，故 C3 适用；注意 C2 用**完整**坐标（只替换 coef），
    # C3 用**对应行的坐标**——两者不可混用。
    enable_c3 = kw.pop("enable_c3", True)
    return _check_core(lambda c: fn(c, coords),
                       coef, label=label,
                       singleton_call=lambda i: fn(coef[i: i + 1], coords[i: i + 1]),
                       enable_c3=enable_c3, **kw)


# ---------------------------------------------------------------- 报告

_STATUS = {True: "PASS", False: "FAIL", None: "n/a"}


def format_report(results: Sequence[dict]) -> str:
    """把若干次检查结果渲染成 markdown 表，便于直接贴进审计记录。"""
    lines = [
        "| target | C1 shape | C2 contamination | C3 pairing | verdict | failing leaves | max |diff| |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        err = r.get("error")
        c = r.get("checks") or {}
        if err:
            lines.append(f"| `{r['label']}` | — | — | — | **ERROR** | {err} | — |")
            continue
        failing = {k: v for k, v in (r.get("failing_leaves") or {}).items() if v}
        worst = max((d for leaf in r["leaves"].values()
                     for d in (leaf["c2_diff"], leaf["c3_diff"])), default=0.0)
        if not r["leaves"] or all(v["skipped"] for v in r["leaves"].values()):
            verdict = "NO BATCH-SHAPED LEAF"
        else:
            verdict = "**PASS**" if r["passed"] else "**FAIL**"
        bad = "; ".join(f"{k}: {', '.join(v)}" for k, v in failing.items()) or "--"
        lines.append(
            f"| `{r['label']}` | {_STATUS[c.get('C1_shape', False)]} "
            f"| {_STATUS[c.get('C2_contamination', False)]} "
            f"| {_STATUS.get(c.get('C3_pairing', None), 'n/a')} | {verdict} "
            f"| {bad} | {worst:.3e} |")
    return "\n".join(lines)


if __name__ == "__main__":
    # 最小自检：正确实现 PASS，被广播污染的实现 FAIL。
    rng = np.random.default_rng(0)

    def fixed(coef, x):
        k = np.arange(1, 5)
        return np.einsum("bk,bnk->bn", coef, np.sin(x[..., None] * k))

    def buggy(coef, x):
        k = np.arange(1, 5)
        rows = x.reshape(-1)
        vals = coef @ np.sin(np.outer(rows, k)).T          # (B, B*N)
        return vals.reshape(coef.shape[0], *x.shape)       # (B, B, N) ← 静默广播

    coef = rng.normal(0, 1, (4, 4))
    x = rng.uniform(0, 1, (4, 32))
    res = [check_evaluator(fixed, coef, x, label="fixed"),
           check_evaluator(buggy, coef, x, label="broadcast-contaminated")]
    print(format_report(res))
