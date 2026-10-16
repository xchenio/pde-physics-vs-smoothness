# 外部仓库配对审计 V1（2026-10-16，只读）

> 对应 D2 的后半段：在**第三方参考实现**上运行配对契约检查器。
> 只读：不修改、不执行对方仓库的训练，不安装任何包。

## 1. 快照与可复现性

源码快照放在 `external/`（**未安装**），整体指纹登记在 `external/snapshots.json`
（按相对路径排序后逐文件拼接再取 SHA-256）：

| 目标 | 身份 | 文件数 | snapshot SHA-256（前 16） |
|---|---|---:|---|
| `pi_deeponet` | Wang/Wang/Perdikaris, **Science Advances 2021** 官方实现 | 30 | `414e0855ffed1f26` |
| `deepxde` | DeepXDE（PINN/DeepONet 事实标准库） | 306 | `b8b64abed7e55c04` |
| `pino` | Li et al., **ICLR 2023** | 254 | `ec9aff78a8f50aaf` |

⚠️ 下载的是 ZIP 快照，**没有 git commit 信息**；因此用整体文件指纹 + 下载日期
（2026-10-16）作为版本标识。若对方仓库此后修复了什么，指纹会对不上——这是刻意的，
便于日后核对。

## 2. 已审计：PI-DeepONet 官方（Burgers）

审计对象：`external/pi_deeponet/Burger/PI_DeepONet_Burger.ipynb` 的 **cell 4**，
即 `generate_one_ics_training_data` / `generate_one_res_training_data`：

```python
u = np.tile(u0, (P, 1))            # 同一 u0 复制到该样本的 P 个配点上
y = np.hstack([t_res, x_res])      # 每个样本用**自己的**随机 key 生成配点
```

**结果**（`scripts/audit_probes/pi_deeponet_probes.py`，同一个检查器）：

| target | C1 shape | C2 contamination | C3 pairing | verdict |
|---|---|---|---|---|
| 官方 IC 目标语义 | PASS | PASS | PASS | **PASS** |
| 官方 `np.tile` branch 输入 | PASS | PASS | PASS | **PASS** |
| 官方 res 目标（zeros） | PASS | PASS | PASS | **PASS** |
| 反事实：共享网格 + reshape | **FAIL** | PASS | **FAIL** | **FAIL** |
| 反事实：`broadcast_to` 代替 tile | PASS | PASS | PASS | **PASS** |

### 结论

1. **官方实现没有该静默缺陷。** 它用 `np.tile` 把每个样本的 u0 显式复制到自己的
   配点上，且每个样本的配点由自己的随机 key 生成，配对天然成立。
   这一点必须如实写进论文——**不能**把缺陷说成"PI-DeepONet 官方也有"。
2. **但它是"正确但冗余"的**：branch 要对 B×P 个输入做前向，而其中只有 B 个不同值，
   冗余 P 倍。这正是 arXiv:2610.22157（2026-10-15）所讲的效率问题。
3. **缺陷的真正来源是 reshape，不是"共享/广播"本身。**
   上表最后一行很关键：用 `broadcast_to` 代替 `tile` 的重构**是安全的**（三项全 PASS）；
   只有把 (B, B·N) 的中间结果 `reshape` 成 (B, B, N) 才会静默错位。
   这是一条对实践者有用的**精确**警示：优化 branch 冗余是安全的，
   但"展平所有配点再 reshape 回 batch 形状"这个动作是危险的。
4. 因此我们的贡献定位更清晰了：**官方为了正确性选择了冗余；一旦有人为了效率
   改用"展平 + reshape"的写法，就会踩到静默广播**。这两篇工作（效率 + 静默错误）
   是同一条因果链的两端。

### 方法论声明（重要）

- **没有执行原 notebook**：它依赖 JAX 与 `Burger.mat`（需 MATLAB 生成），本机没有。
  适配层复刻的是其**数据布局语义**（tile + 逐样本随机配点），不是逐行运行原代码。
  审计结论针对该语义；若原 notebook 在别处另有求值路径，本轮未覆盖。
- 反事实条目是**我们构造的重构**，不是对方代码里存在的缺陷，标注清楚以免误读。

## 3. 尚未审计

| 目标 | 状态 | 下一步 |
|---|---|---|
| `deepxde` | 快照已就位，**未审计** | 审计 `deepxde/nn/*/deeponet.py` 与 `deepxde/data/pde_operator.py` 的 CartesianProd 分支 |
| `pino` | 快照已就位，**未审计** | 审计 `models/` 与 `train_utils/` 中输入函数在配点上的求值路径 |

在此之前，本文**不构成**对 DeepXDE / PINO 的任何评价。

## 4. 复现

```bash
D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py --status
D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py --target pi_deeponet
```

工具：`tools/notebook_peek.py`（只读提取 notebook 代码单元，便于在审计记录里
引用"第几个 cell 写了什么"）。产出：`results/audit_pairing_external_v1/`。
