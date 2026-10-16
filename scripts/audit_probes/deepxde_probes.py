"""DeepXDE 的配对契约适配层（只读审计，不安装、不执行对方代码）。

审计对象：`external/deepxde/deepxde/nn/pytorch/deeponet.py`

    class DeepONet.merge_branch_trunk            -> einsum("bi,bi->b")   逐样本
    class DeepONetCartesianProd.merge_branch_trunk -> einsum("bi,ni->bn") 笛卡尔积
    class PODDeepONet.forward                    -> einsum("bi,ni->bn")  笛卡尔积

两条路径的接口意图是清楚的（非笛卡尔 = 逐样本每样本一个点；笛卡尔 = 共享网格），
文档也写明了。**但 einsum 的下标不区分"样本维 b"与"网格维 n"**：
当笛卡尔路径被喂进**逐样本坐标**（每个样本自己的一组点）时，
trunk 会把输入第一维当成"点数"，于是 (B, P) 与 (B, P) 得到 (B, B)，
**不抛异常、不报警**，形状只是"看起来小了一号"。

本适配层用 numpy 复刻这两个 einsum 的**形状语义**（不依赖 deepxde 包），
三个 probe：
  1. cartesian_shared_grid   —— 笛卡尔路径的正确用法（共享网格），预期 PASS；
  2. cartesian_per_sample    —— 同一段 einsum，喂逐样本坐标，预期 FAIL；
  3. per_sample_einsum       —— 逐样本配对的正确写法，预期 PASS。
"""
from __future__ import annotations

import numpy as np

P = 16          # branch/trunk 特征宽度（对应 DeepONet 最后一层的宽度）
M_SENSOR = 64   # branch 输入（输入函数在传感器上的采样）


def _weights(rng: np.random.Generator, coord_dim: int = 32):
    """(branch 权重, trunk 权重(标量坐标 -> P), trunk 权重(被误当成"坐标维"的 N -> P))。

    第二种 trunk 权重的输入维是 N：因为笛卡尔路径把传入矩阵的第一维当作"点数"、
    第二维当作坐标维，于是 (B, N) 被解读成"B 个点、每点 N 个坐标分量"。
    """
    return (rng.normal(0, 1, (M_SENSOR, P)) / np.sqrt(M_SENSOR),
            rng.normal(0, 1, (1, P)),
            rng.normal(0, 1, (coord_dim, P)) / np.sqrt(coord_dim))


_WB, _WT1, _WT2 = _weights(np.random.default_rng(7))


def _branch(coef: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(coef) @ _WB)          # (B, P)


# ------------------------------------------------------------------ probes

def cartesian_shared_grid(coef, coords):
    """`einsum("bi,ni->bn")` 的正确用法：coords 是被所有样本共享的网格。

    这里传入的 coords 每行相同（共享网格语义），trunk 只用其中一行。
    """
    grid = np.asarray(coords)[0][:, None]           # (N, 1)
    tf = np.tanh(grid @ _WT1)                       # (N, P)
    return np.einsum("bi,ni->bn", _branch(coef), tf)   # (B, N)


def cartesian_per_sample(coef, coords):
    """同一句 `einsum("bi,ni->bn")`，但 coords 是**逐样本**坐标 (B, N)。

    trunk 把 (B, N) 当成"B 个点、每点 N 维"，输出 (B, P)；
    于是 einsum 的 b 与 n 都等于 B，静默产出 (B, B)。
    """
    tf = np.tanh(np.asarray(coords) @ _WT2)         # (B, P) —— 把 B 当点数
    return np.einsum("bi,ni->bn", _branch(coef), tf)   # (B, B)  ← 静默错位


def per_sample_einsum(coef, coords):
    """逐样本配对的正确写法：trunk 对每个 (样本, 点) 都算一次特征。"""
    bf = _branch(coef)                                     # (B, P)
    tf = np.tanh(np.asarray(coords)[..., None] @ _WT1)     # (B, N, P)
    return np.einsum("bi,bni->bn", bf, tf)                 # (B, N)


def probes():
    rng = np.random.default_rng(20261016)
    coef = rng.normal(0, 1, (4, M_SENSOR))
    grid = rng.uniform(0, 1, 32)
    shared = np.tile(grid, (coef.shape[0], 1))     # (B, N) 每行相同 = 共享网格
    per_sample = rng.uniform(0, 1, (coef.shape[0], 32))
    return [
        ("cartesian_prod(shared grid, correct)", coef, shared, cartesian_shared_grid),
        ("cartesian_prod(per-sample coords)", coef, per_sample, cartesian_per_sample),
        ("per-sample einsum(correct pairing)", coef, per_sample, per_sample_einsum),
    ]
