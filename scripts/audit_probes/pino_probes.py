"""PINO（Li et al., ICLR 2023）的配对契约适配层（只读审计）。

审计对象：`external/pino/baselines/model.py` 的 `DeepONetCP.forward`：

    def forward(self, u0, grid):
        a = self.branch(u0)      # batchsize x width
        b = self.trunk(grid)     # N x width
        return torch.einsum('bi,ni->bn', a, b)

与 DeepXDE 的 `DeepONetCartesianProd` **逐字同款** einsum 惯用法
（两个独立仓库出现同一句，说明这是社区的标准写法）。
注释明确 grid 是共享网格（N 个点），接口意图清楚 —— 但 einsum 的下标
同样不区分 b 与 n，因此"把逐样本坐标喂给这段"会静默产出 (B, B)。

其余主干（FNO / fourier1d-3d、tfno）不涉及"输入函数在配点上求值"这一模式：
`train_utils/losses.py: darcy_loss(u, a)` 里 u 与 a 同形状逐元素运算，
且 `reshape(batchsize, size, size)` 有元素数保护（形状不对会直接报错）。
因此**该缺陷类别在 PINO 的 FNO 主干上不适用**，本适配层对此给出显式说明，
并用同一个检查器验证 darcy 系数场的逐样本配对。
"""
from __future__ import annotations

import numpy as np

P = 16
M_SENSOR = 64
N_GRID = 32

_rng = np.random.default_rng(11)
_WB = _rng.normal(0, 1, (M_SENSOR, P)) / np.sqrt(M_SENSOR)
_WT1 = _rng.normal(0, 1, (1, P))          # grid: (N, 1) -> (N, P)
_WT2 = _rng.normal(0, 1, (N_GRID, P)) / np.sqrt(N_GRID)   # 被误当作"坐标维"的 N


def _branch(coef):
    return np.tanh(np.asarray(coef) @ _WB)


def deeponet_cp_shared_grid(coef, coords):
    """PINO DeepONetCP 的正确用法：grid 是所有样本共享的网格。"""
    grid = np.asarray(coords)[0][:, None]                  # (N, 1)
    b = np.tanh(grid @ _WT1)                               # (N, P)
    return np.einsum("bi,ni->bn", _branch(coef), b)        # (B, N)


def deeponet_cp_per_sample(coef, coords):
    """同一句 einsum，喂进逐样本坐标 (B, N) -> 静默产出 (B, B)。"""
    b = np.tanh(np.asarray(coords) @ _WT2)                 # (B, P)，B 被当成点数
    return np.einsum("bi,ni->bn", _branch(coef), b)        # (B, B)


def darcy_coefficient_field(coef, coords):
    """复刻 `darcy_loss(u, a)` 中系数场的语义：a 与 u 同形状、逐元素配对。

    coef 是系数场的参数化（传感器上的取值），在逐样本坐标上插值得到 (B, N)。
    """
    coef = np.asarray(coef)
    x = np.asarray(coords)
    sensor = np.linspace(0.0, 1.0, coef.shape[1])
    out = np.empty((coef.shape[0], x.shape[1]))
    for i in range(coef.shape[0]):
        out[i] = np.interp(x[i], sensor, coef[i])
    return np.exp(out)                                     # Darcy 的 a = exp(g) > 0


def probes():
    rng = np.random.default_rng(20261016)
    coef = rng.normal(0, 1, (4, M_SENSOR))
    grid = rng.uniform(0, 1, N_GRID)
    shared = np.tile(grid, (coef.shape[0], 1))
    per_sample = rng.uniform(0, 1, (coef.shape[0], N_GRID))
    return [
        ("DeepONetCP(shared grid, correct)", coef, shared, deeponet_cp_shared_grid),
        ("DeepONetCP(per-sample coords)", coef, per_sample, deeponet_cp_per_sample),
        ("darcy coefficient field(pairing)", coef, per_sample, darcy_coefficient_field),
    ]
