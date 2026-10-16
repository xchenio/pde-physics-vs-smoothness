# 逐样本配对契约审计 V1（2026-10-16）

> 对应 `PUBLICATION_STRATEGY_20261016.md` 的 **D2**：把"静默广播缺陷 + 检测配方"
> 做成可在他人代码上运行的检查器。本轮完成：检查器、单元测试、项目内审计、
> 注入缺陷对照、传统 QA 对照。**外部第三方仓库尚未审计**（需要下载，见 §7）。

## 1. 为什么现在做

arXiv:2610.22157（2026-10-15）《The importance of tensor handling and implementation
of physics-informed DeepONets》把"PI-DeepONet 的张量处理/实现"摆上了台面。
它讲的是**效率**（branch 在配点上的冗余计算，2D 加速 3.75×）；
我们这条线讲的是**静默错误与检测**。议题相邻、结论互补，但存在**优先级的时间窗口**，
因此先把检查器本体和 프로젝 내검증 落地。

## 2. 检查配方（`tools/pairing_check.py`）

零依赖（仅 numpy）单文件，可整体复制到任意仓库独立使用。
三条检验都**不需要训练**：

| 检验 | 判据 | 适用 | 能抓什么 |
|---|---|---|---|
| C1 shape | 叶子数组首维 == B，且不出现 `(B, B, N)` 形疑似广播 | 全部 | 共享网格约定被误用 |
| C2 contamination | 替换其它样本的行，第 b 行输出不得改变 | 全部 | 跨样本混配、意外池化/归一化 |
| C3 pairing | 循环置换 coef 的行，第 i 行必须等于原 `第 perm[i] 行` | **仅当坐标由外部固定** | 坐标块错位、索引错位 |

**两条实测确定的边界**（写进理由是它们会导致假 FAIL，误用过一次）：

1. C3 不能用于"内部随机采样坐标"的黑盒管线。那种管线里 coords 本身就是逐样本的
   随机量，置换 coef 后第 i 行的坐标也跟着换，比较双方讲的不是同一组点。
   因此 `check_pipeline` 默认**只跑 C1 + C2**，C3 记为 `n/a`。
2. **已知盲区**：对纯 `(B, B, N)` 广播，C2 会漏——`arr[i, i]` 恰好就是
   "第 i 个样本在第 i 行坐标上"的正确值，污染检验看它完全正常。
   此时 **C1 是唯一指纹**。三条检验是互补关系，任何单条都不足够。

单元测试 `tests/test_pairing_check.py`（12 项）把这五类实现的判定锁住：
correct（全 PASS）、broadcast（C1）、pooled（**只有 C3**）、rolled（C2+C3）、
batch-normed（C2）。它们是防止检查器退化成"永远 PASS 的摆设"的回归网。

## 3. 审计目标与事前登记

对每个 PDE 跑四个 target：`{pipeline,evaluator} × {fixed,legacy}`。
`expect` 是**事前登记**的期望结果（不看输出再定）：fixed 应为 pass，legacy 应为 fail。

唯一例外：`darcy/evaluator:fixed` 登记为 **fail**——`DarcyProblem.input_at`
至今缺少逐样本分支（只有 `input_and_grad` 有），见 §5。

## 4. 结果（4 PDE × 4 target = 16）

| target | C1 shape | C2 contamination | C3 pairing | verdict | failing leaves | max \|diff\| |
|---|---|---|---|---|---|---|
| `burgers/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `burgers/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `burgers/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: ic/target | 0.000e+00 |
| `burgers/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `burgers2d/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `burgers2d/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `burgers2d/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: ic/target | 0.000e+00 |
| `burgers2d/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `dr/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `dr/evaluator:fixed` | PASS | PASS | PASS | **PASS** | -- | 0.000e+00 |
| `dr/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: col/forcing | 0.000e+00 |
| `dr/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `darcy/pipeline:fixed` | PASS | PASS | n/a | **PASS** | -- | 0.000e+00 |
| `darcy/evaluator:fixed` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |
| `darcy/pipeline:legacy` | FAIL | PASS | n/a | **FAIL** | C1_shape: col/a, col/ax, col/ay | 0.000e+00 |
| `darcy/evaluator:legacy` | FAIL | PASS | FAIL | **FAIL** | C1_shape: out; C3_pairing: out | 0.000e+00 |

**16/16 与事前登记的期望一致**（假阳性 0、假阴性 0，在本轮这批实现上）。

## 5. 检查器抓到的真实问题（不是注入的）

`DarcyProblem.input_at(coef, coords)` 对 `(B, N, 2)` 的逐样本坐标返回 `(B, B, N)`：
它只有"共享网格"一种约定，逐样本分支至今只加在了 `input_and_grad` 上。
当前训练路径（`sample_points` → `input_and_grad`）不受影响，因此**没有造成错误结果**；
但只要有人用 `input_at` 做逐样本求值（例如新增一种损失项），就会静默踩同一个坑。

这是检查器给出的第一个**非注入**发现，也支持"该缺陷类别是反复出现的，而非一次性笔误"。
建议后续给 `input_at` 补逐样本分支（或至少加形状断言）。

## 6. 传统 QA 对照：为什么"检查真解的 FD 残差"查不出来

对每个 PDE 在**共享网格**路径（求解器与参考检查实际走的那条路）上求值，
比较 fixed 与 legacy 实现：

| PDE | grid 采样 | max \|diff\| | 结论 |
|---|---|---|---|
| burgers | (525, 2) | 0.000e+00 | 逐位相同 |
| burgers2d | (512, 3) | 0.000e+00 | 逐位相同 |
| dr | (518, 2) | 0.000e+00 | 逐位相同 |
| darcy | (561, 2) | 0.000e+00 | 逐位相同 |

两者在同一条代码路径上（ historically 的共享网格分支本身没变），
缺陷**只存在于"逐样本坐标"这条调用路径上**，而传统参考检查从不走这条路。
这是对该限制的结构性证据，而不是又一次数值相近的巧合。

⚠️ 这是**代理**论证：我们比对的是"gird 上的输入函数取值"，
不是完整 PDE 残差本身。要把它升级为完整论证，还需要做
"用监督训练好的模型算 autograd 残差"（README §6.5 第 2 条）。这条留作下一步。

## 7. 局限（必须一起读）

1. **legacy 是复刻，不是历史代码**。本仓库的 git 起始于 2026-09-22，
   晚于缺陷修复日（2026-09-16），修复前的代码不在版本库中。
   `inject_legacy()` 是按 README §6.5 的描述复刻的，**不得**称为"在历史缺陷版本上运行的结果"。
2. **外部仓库一行未审计**。`scripts/audit_external_pairing.py --status` 如实显示
   三个目标均为 `MISSING repo`。在完成之前，本文不构成对任何第三方代码库的评价。
3. C2 对纯 `(B, B, N)` 广播漏检（§2），C3 对随机坐标管线假阳性（§2）——
   两者都已用单元测例固定下来。
4. 本轮只检查**输入侧**（初值/源项/系数），未检查 branch 侧或损失计算中的同类广播。

## 8. 复现命令

```bash
D:/miniconda3/envs/pytorch/python.exe tools/pairing_check.py            # 最小自检
D:/miniconda3/envs/pytorch/python.exe -m unittest tests.test_pairing_check -q
D:/miniconda3/envs/pytorch/python.exe scripts/audit_pairing.py          # -> results/audit_pairing_v1/
D:/miniconda3/envs/pytorch/python.exe scripts/audit_external_pairing.py --status
```

产出：`results/audit_pairing_v1/audit_pairing.json`（含 code_hash / env 指纹）、`report.md`。

## 9. 下一步（D2 剩余部分）

- [ ] 下载并安装外部目标（详见 `scripts/audit_external_pairing.py` 的 TARGETS.clone）：
      DeepXDE（`pip install deepxde` 即可）、PI-DeepONet 官方、PINO 官方
- [ ] 为每个目标写 ~20 行 `pairing_probes.py` 适配层（模板见脚本文档字符串）
- [ ] 跑 `scripts/audit_external_pairing.py`，把结果补进本文 §4 之后的新一节
- [ ] 补"监督模型 autograd 残差"这条第二防线（把 §6 的代理升级为直接论证）
- [ ] 视情况给 `DarcyProblem.input_at` 补逐样本分支

## 10. 回归测试

全量 `python -m unittest discover -s tests -q`：**106 项通过**（含新增 12 项）。
