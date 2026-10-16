"""PI-DeepONet 官方仓库（Wang/Wang/Perdikaris, Sci. Adv. 2021）的配对契约适配层。

审计对象：`external/pi_deeponet/Burger/PI_DeepONet_Burger.ipynb` 的 cell 4，
即 `generate_one_ics_training_data` / `generate_one_res_training_data` 的语义：

    u = np.tile(u0, (P, 1))        # 同一 u0 复制到该样本的 P 个配点上
    y = np.hstack([t, x])          # 每个样本用**自己的**随机 key 生成配点

这是"正确但冗余"的写法：配对天然不会错（u 的每个 (b,p) 行都来自 u0[b]），
代价是 branch 要在 B×P 个输入上做前向，而其中只有 B 个不同值——
正是 arXiv:2610.22157 所说的冗余。

本适配层做三件事：
  1. `official_*`：按官方语义复刻（numpy 版），预期 **PASS**；
  2. `counterfactual_*`：把 tile 换成"共享网格 + 广播"的重构（即为效率去掉冗余
     时最容易写出的形式），预期 **FAIL**；
  3. 两者用同一个检查器，从而把"官方安全 / 重构易错"变成可复现的证据。

注意：本文件**不执行原 notebook**（它依赖 JAX 与 Burger.mat 数据），
只复刻其数据布局语义；这一点会在审计记录里写明。
"""
from __future__ import annotations

import numpy as np

M = 101          # 传感器数（官方 m=101）
P_RES = 1000     # 残差配点数（官方 P=1000）
SENSOR = np.linspace(0.0, 1.0, M)


def _u0_batch(n: int, rng: np.random.Generator) -> np.ndarray:
    """随机光滑初始条件在传感器上的取值。"""
    k = np.arange(1, 4)
    a = rng.normal(0, 1, (n, 3))                                # (n, 3)
    basis = np.sin(np.pi * np.outer(SENSOR, k))                 # (M, 3)
    return (basis @ a.T).T * 0.1                                # (n, M)


# ------------------------------------------------------------------ 官方语义

def official_ic_target(coef, coords):
    """官方 cell 4 的 IC 语义：s = u0（IC 点与传感器同轴），批量为 (B, P)。

    coords 是逐样本坐标 (B, N)；这里用插值把 u0 求到 coords 上，
    保留官方"第 b 行只由 u0[b] 决定"的配对性质。
    """
    coef = np.asarray(coef)
    x = np.asarray(coords)
    out = np.empty((coef.shape[0], x.shape[1]))
    for b in range(coef.shape[0]):
        out[b] = np.interp(x[b], SENSOR, coef[b])
    return out


def official_branch_tile(coef, coords):
    """官方 u = np.tile(u0, (P,1)) 的批量版，取传感器维的均值作为标量代表。

    输出 (B, N)；因 tile 的语义，第 b 行对所有列都等于 mean(u0[b])，
    只由 coef[b] 决定 —— 这正是"配对正确"的表现。
    """
    coef = np.asarray(coef)
    x = np.asarray(coords)
    tiled = np.tile(coef[:, None, :], (1, x.shape[1], 1))        # (B, N, M)
    return tiled.mean(axis=2)


def official_res_target(coef, coords):
    """官方 res 分支的目标 s = zeros((P,1))：与 coef 无关，批量为 (B, N)。"""
    x = np.asarray(coords)
    return np.zeros((coef.shape[0], x.shape[1]))


# ------------------------------------------------------------------ 反事实重构

def counterfactual_shared_grid(coef, coords):
    """为消除 branch 冗余而做的"共享网格"重构：对每个 coef[b] 在**所有样本的
    配点**上求值，再 reshape 回 (B, B, N)。

    这是效率导向重构最容易写出的形式，也正是静默广播缺陷的来源。
    """
    coef = np.asarray(coef)
    x = np.asarray(coords)
    rows = x.reshape(-1)
    vals = np.empty((coef.shape[0], rows.size))
    for b in range(coef.shape[0]):
        vals[b] = np.interp(rows, SENSOR, coef[b])
    return vals.reshape(coef.shape[0], *x.shape)


def counterfactual_broadcast_tile(coef, coords):
    """用 broadcast_to 代替 tile 的重构（(1,B,M) -> (N,B,M) -> 转置）。

    经检验这一种是**安全的**：broadcast 的语义就是"把 coef[b] 复制到每个 n"，
    配对仍然正确（检查器三项全 PASS）。保留它作为对照，说明
    **缺陷并不来自"共享/广播"本身，而来自 reshape 把 (B, B*N) 解释成 (B,B,N)**。
    """
    coef = np.asarray(coef)
    x = np.asarray(coords)
    tiled = np.broadcast_to(coef[None, :, :],
                            (x.shape[1], coef.shape[0], coef.shape[1]))
    return tiled.mean(axis=2).T


# ------------------------------------------------------------------ 注册

def probes():
    """返回 [(label, coef, coords, fn)]。"""
    rng = np.random.default_rng(20261016)
    coef = _u0_batch(4, rng)                                     # (B=4, M=101)
    coords = rng.uniform(0, 1, (4, 32))                          # (B, N=32)
    return [
        ("ic_target(official tile semantics)", coef, coords, official_ic_target),
        ("branch_tile(official np.tile)", coef, coords, official_branch_tile),
        ("res_target(official zeros)", coef, coords, official_res_target),
        ("counterfactual: shared-grid reshape", coef, coords, counterfactual_shared_grid),
        ("counterfactual: broadcast tile", coef, coords, counterfactual_broadcast_tile),
    ]
