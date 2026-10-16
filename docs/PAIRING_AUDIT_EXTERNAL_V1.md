# 外部仓库配对审计 V1（2026-10-16，只读）

> 对应 D2 的后半段：在**第三方参考实现**上运行配对契约检查器。
> 只读：不修改、不执行对方仓库的训练，不安装任何包。

## 1. 快照与可复现性

源码快照放在 `external/`（**未安装**，且已排除出 git；只有指纹入库）：

| 目标 | 身份 | 文件数 | snapshot SHA-256（前 16） |
|---|---|---:|---|
| `pi_deeponet` | Wang/Wang/Perdikaris, **Science Advances 2021** 官方实现 | 30 | `414e0855ffed1f26` |
| `deepxde` | DeepXDE（PINN/DeepONet 事实标准库） | 306 | `b8b64abed7e55c04` |
| `pino` | Li et al., **ICLR 2023** | 254 | `ec9aff78a8f50aaf` |

⚠️ 下载的是 ZIP 快照，**没有 git commit 信息**，故用整体文件指纹 + 下载日期
（2026-10-16）作为版本标识。若对方仓库此后修复了什么，指纹会对不上——这是刻意的。

## 2. 结果总表

| target | C1 shape | C2 contamination | C3 pairing | verdict |
|---|---|---|---|---|
| `deepxde/cartesian_prod(shared grid, correct)` | PASS | PASS | PASS | **PASS** |
| `deepxde/cartesian_prod(per-sample coords)` | **FAIL** | PASS | **FAIL** | **FAIL** |
| `deepxde/per-sample einsum(correct pairing)` | PASS | PASS | PASS | **PASS** |
| `pi_deeponet/ic_target(official tile semantics)` | PASS | PASS | PASS | **PASS** |
| `pi_deeponet/branch_tile(official np.tile)` | PASS | PASS | PASS | **PASS** |
| `pi_deeponet/res_target(official zeros)` | PASS | PASS | PASS | **PASS** |
| `pi_deeponet/counterfactual: shared-grid reshape` | **FAIL** | PASS | **FAIL** | **FAIL** |
| `pi_deeponet/counterfactual: broadcast tile` | PASS | PASS | PASS | **PASS** |
| `pino/DeepONetCP(shared grid, correct)` | PASS | PASS | PASS | **PASS** |
| `pino/DeepONetCP(per-sample coords)` | **FAIL** | PASS | **FAIL** | **FAIL** |
| `pino/darcy coefficient field(pairing)` | PASS | PASS | PASS | **PASS** |

## 3. 逐仓库结论

### 3.1 PI-DeepONet 官方：安全，但是"正确但冗余"

审计对象：`external/pi_deeponet/Burger/PI_DeepONet_Burger.ipynb` 的 **cell 4**：

```python
u = np.tile(u0, (P, 1))            # 同一 u0 复制到该样本的 P 个配点上
y = np.hstack([t_res, x_res])      # 每个样本用**自己的**随机 key 生成配点
```

**官方实现没有该缺陷**（三项全 PASS）—— 必须如实写，不能说成"官方也有"。
但它是"正确但冗余"的：branch 要对 B×P 个输入做前向，其中只有 B 个不同值，
冗余 P 倍，正是 arXiv:2610.22157 讲的效率问题。

### 3.2 DeepXDE：接口意图清楚，但 einsum 下标不设防

`external/deepxde/deepxde/nn/pytorch/deeponet.py`：

```python
class DeepONet.merge_branch_trunk              -> einsum("bi,bi->b")    逐样本（每样本一个点）
class DeepONetCartesianProd.merge_branch_trunk -> einsum("bi,ni->bn")   共享网格
class PODDeepONet.forward                      -> einsum("bi,ni->bn")   共享网格
```

两条路径的接口意图清楚、文档写明（PODDeepONet 的 docstring 直接说
"for dataset in the format of Cartesian product"）。
**但 einsum 的下标不区分"样本维 b"与"网格维 n"**：笛卡尔路径被喂进逐样本
坐标 (B, N) 时，trunk 把 (B,N) 解读为"B 个点、每点 N 维"，输出 (B, P)，
于是 b 与 n 都等于 B —— **静默产出 (B, B)，不抛异常**。

### 3.3 PINO：DeepONet baseline 是同款 einsum；FNO 主干不适用

`external/pino/baselines/model.py`：

```python
class DeepONetCP.forward(u0, grid):
    a = self.branch(u0)      # batchsize x width
    b = self.trunk(grid)     # N x width
    return torch.einsum('bi,ni->bn', a, b)
```

与 DeepXDE 的 `DeepONetCartesianProd` **逐字同款**（两个独立仓库出现同一句，
说明这是社区的标准惯用法），风险与 3.2 相同。

PINO 的 FNO 主干（fourier1d-3d / tfno）**不涉及**"输入函数在配点上求值"这一模式：
`train_utils/losses.py: darcy_loss(u, a)` 里 u 与 a 同形状逐元素运算，
且 `reshape(batchsize, size, size)` 有元素数保护（形状不对直接报错，不静默）。
⇒ **该缺陷类别在 PINO 的 FNO 主干上不适用**。

## 4. 跨仓库的统一结论（这是本轮最有价值的产出）

1. **三个参考实现在各自的既定用法下都是安全的**（全部 PASS）。
   论文里**不能**声称"这些流行实现有 bug"—— 那会是虚假主张。
2. **缺陷的真正触发条件是"展平配点 + reshape 回 batch 形状"**，
   而不是"共享/广播"本身：用 `broadcast_to` 代替 `tile` 的重构经检验是安全的，
   只有 reshape 把 (B, B·N) 解释成 (B, B, N) 才静默错位。
3. **笛卡尔 einsum 是一个跨仓库共用的、不设防的惯用法**：
   `einsum("bi,ni->bn")` 出现在 DeepXDE 与 PINO 两个独立实现中，
   它接受逐样本坐标时不报错、只静默改变语义。
   这给检查器提供了明确的"该查什么"：任何 `einsum` / `matmul` 同时吃
   branch 特征与 trunk 特征的位置，都要断言 trunk 输入的第一维**不是** batch。
4. 因果链完整了：**官方为正确性选择了冗余（PI-DeepONet 的 tile）；
   为消除冗余而做的展平 + reshape 重构（arXiv:2610.22157 的方向）若用 reshape 就会踩坑**。
   效率工作与静默错误是同一条链的两端。

## 5. 方法论声明（重要）

- **没有执行对方代码**：PI-DeepONet 的 notebook 依赖 JAX 与 `Burger.mat`
  （需 MATLAB 生成），DeepXDE / PINO 未安装。适配层用 numpy 复刻其
  **形状语义**（ einsum 下标、tile/reshape 行为），不是逐行运行原代码。
- 标注为 `counterfactual` / `per-sample coords` 的条目是**我们构造的误用或重构**，
  不是对方代码里已存在的缺陷；它们用来证明"检查器在该布局下能抓到"，
  以及"哪些重构危险、哪些安全"。
- 本轮未覆盖：DeepXDE 的 `PDEOperator` 数据类本身、PINO 的训练循环细节。

## 6. 复现

```bash
D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py --status
D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py
```

适配层：`scripts/audit_probes/{pi_deeponet,deepxde,pino}_probes.py`
（放在本仓库内，不污染外部快照）。
工具：`tools/notebook_peek.py`（只读提取 notebook 代码单元）。
产出：`results/audit_pairing_external_v1/`。
