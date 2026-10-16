"""配对契约检查器的单元测试。

运行：D:/miniconda3/envs/pytorch/python.exe -m unittest tests.test_pairing_check -q

每个用例都对应一类**真实可能出现在 DeepONet 代码里**的实现：
"correct" 必须三项全 PASS，四类缺陷必须被至少一条检验抓到。
这类测试的价值在于：检查器本身的正例/负例敏感度是被代码锁住的，
不会随时间悄悄退化成"永远 PASS 的摆设"。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from pairing_check import (  # noqa: E402
    check_evaluator, check_pipeline, format_report, iter_leaves)

B, N, K = 4, 32, 5
K_VEC = np.arange(1, K + 1)


def _basis(x):
    """x: (...,) -> (..., K)"""
    return np.sin(np.asarray(x)[..., None] * K_VEC)


def correct(coef, x):
    """逐样本配对：第 b 行只用 coef[b]。"""
    return np.einsum("bk,bnk->bn", coef, _basis(x))


def broadcast(coef, x):
    """修复前的共享网格约定 -> (B, B, N)。"""
    rows = np.asarray(x).reshape(-1)
    vals = coef @ _basis(rows).T                       # (B, B*N)
    return vals.reshape(coef.shape[0], *np.asarray(x).shape)


def pooled(coef, x):
    """输出在每个系数的**全部坐标块**上取平均：形状正确、无跨样本系数泄漏，
    但第 i 行混入了"别人的坐标"。这种缺陷只有 C3 抓得到（C2 判不出，它确实
    没有污染，只是算错了对象）。"""
    rows = np.asarray(x).reshape(-1)
    vals = coef @ _basis(rows).T                       # (B, B*N)
    per = vals.reshape(B, B, N)
    return per.mean(axis=1)                            # (B, N)


def rolled(coef, x):
    """索引错位：第 i 行装的是 coef[(i+1)%B] 的结果。"""
    return correct(np.roll(coef, -1, axis=0), x)


def batch_normed(coef, x):
    """误用跨 batch 归一化：第 i 行依赖了**其它样本**的系数 -> C2 必抓。"""
    u = correct(coef, x)
    return u - u.mean(axis=0, keepdims=True)


def nested(coef, x):
    """返回值是嵌套结构（dict 里套 list），用于验证叶子遍历。"""
    u = correct(coef, x)
    return {"ic": {"target": u, "terms": [u, 2.0 * u]}, "meta": "not-an-array"}


class TestPairingCheck(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.coef = rng.normal(0, 1, (B, K))
        self.x = rng.uniform(0, 1, (B, N))

    # ---------------- 正例 ----------------
    def test_correct_evaluator_passes_all_checks(self):
        r = check_evaluator(correct, self.coef, self.x, label="correct")
        self.assertIsNone(r["error"])
        self.assertTrue(r["passed"], msg=str(r["checks"]))
        self.assertTrue(all(r["checks"].values()))

    def test_correct_pipeline_passes(self):
        fn = lambda c: {"ic": {"target": correct(c, self.x)}, "meta": None}  # noqa: E731
        r = check_pipeline(fn, self.coef, label="pipeline")
        self.assertTrue(r["passed"], msg=str(r["checks"]))
        self.assertIsNone(r["checks"]["C3_pairing"])          # pipeline 默认不启用 C3

    # ---------------- 负例 ----------------
    def test_broadcast_is_caught_by_shape_check(self):
        r = check_evaluator(broadcast, self.coef, self.x, label="broadcast")
        self.assertFalse(r["passed"])
        self.assertFalse(r["checks"]["C1_shape"])
        self.assertIn("out", r["failing_leaves"]["C1_shape"])
        self.assertTrue(r["leaves"]["out"]["suspected_broadcast"])
        self.assertEqual(r["leaves"]["out"]["shape"], (B, B, N))

    def test_pooled_is_caught_only_by_pairing(self):
        """坐标块被平均：不污染系数（C2 PASS），但 singleton 结果不一致（C3 FAIL）。"""
        r = check_evaluator(pooled, self.coef, self.x, label="pooled")
        self.assertFalse(r["passed"])
        self.assertFalse(r["checks"]["C3_pairing"])
        self.assertTrue(r["checks"]["C1_shape"])
        self.assertTrue(r["checks"]["C2_contamination"])

    def test_batch_norm_is_caught_by_contamination(self):
        r = check_evaluator(batch_normed, self.coef, self.x, label="batchnorm")
        self.assertFalse(r["passed"])
        self.assertFalse(r["checks"]["C2_contamination"])
        self.assertGreater(r["leaves"]["out"]["c2_diff"], 0.0)

    def test_rolled_is_caught_by_pairing_and_contamination(self):
        r = check_evaluator(rolled, self.coef, self.x, label="rolled")
        self.assertFalse(r["passed"])
        self.assertFalse(r["checks"]["C3_pairing"])
        self.assertFalse(r["checks"]["C2_contamination"])

    def test_broadcast_in_pipeline_is_caught(self):
        fn = lambda c: {"col": {"forcing": broadcast(c, self.x)}}  # noqa: E731
        r = check_pipeline(fn, self.coef, label="pipeline-broadcast")
        self.assertFalse(r["passed"])
        self.assertFalse(r["checks"]["C1_shape"])
        self.assertIn("col/forcing", r["failing_leaves"]["C1_shape"])

    # ---------------- 结构支持 ----------------
    def test_nested_leaves_are_traversed(self):
        names = dict(iter_leaves(nested(self.coef, self.x)))
        self.assertIn("ic/target", names)
        self.assertIn("ic/terms[0]", names)
        self.assertIn("ic/terms[1]", names)
        self.assertNotIn("meta", names)                    # 字符串叶子被跳过

    def test_nested_structures_are_checked(self):
        r = check_evaluator(nested, self.coef, self.x, label="nested")
        self.assertTrue(r["passed"])
        self.assertEqual(len(r["leaves"]), 3)

    def test_allow_global_skips_known_global_leaf(self):
        fn = lambda c: {"global_grid": np.zeros((N, 2)), "ic": correct(c, self.x)}  # noqa: E731
        r = check_pipeline(fn, self.coef, label="with-global",
                           allow_global=("global_grid",))
        self.assertTrue(r["passed"])
        self.assertTrue(r["leaves"]["global_grid"]["skipped"])

    # ---------------- 报告 ----------------
    def test_report_renders_all_results(self):
        results = [check_evaluator(correct, self.coef, self.x, label="ok"),
                   check_evaluator(broadcast, self.coef, self.x, label="bad")]
        text = format_report(results)
        self.assertIn("ok", text)
        self.assertIn("bad", text)
        self.assertIn("**PASS**", text)
        self.assertIn("**FAIL**", text)

    def test_error_is_reported_not_raised(self):
        def boom(coef, x):
            raise RuntimeError("not implemented")

        r = check_evaluator(boom, self.coef, self.x, label="boom")
        self.assertFalse(r["passed"])
        self.assertIn("RuntimeError", r["error"])


if __name__ == "__main__":
    unittest.main()
