# 跨架构的三项误差归因 V1（2026-10-16）

> 对应 `PUBLICATION_STRATEGY_20261016.md` 的 **D1**：把原先只在 DeepONet 上成立的
> "表示 / 优化 / 泛化"三项归因，推广成一个对 **FNO 也成立**的统一定义，
> 并在公开数据上给出对照。**没有重新训练任何模型**（复用 `public_fixed_t1_v1` 的 15 个存档）。

## 1. 统一的 LB 定义：readout-projection lower bound

许多算子学习架构的最后一步都是**共享的线性 readout**：

```
DeepONet :  out(u)(y) = <b(u), phi(y)> + bias     phi 只依赖 y；系数逐样本
FNO1d    :  out(u)(x) = <W, z_u(x)>  + bias       z 依赖 u；W 跨样本、跨位置共享
```

两者都能写成"把当前参数下的 readout 特征投影到目标上"的最小二乘问题，
差别只在**允许重优化哪些系数**：

| family | 可重优化的自由度 | 约束含义 |
|---|---|---|
| `basis`（DeepONet） | 每个样本独立的系数（p+1 per sample） | 解必须落在 trunk 张成的固定低维函数空间里 |
| `pointwise`（FNO） | 全局共享的 (W, b)（p+1 total） | 同一组通道权重必须同时服务所有样本与位置 |

定义（归一化空间、相对 Frobenius）：

```
LB(split) = min_{readout 允许的自由参数} ||pred_hat - Y_hat||_F / ||Y_hat||_F
```

它**恒 ≤ 该 split 上任何实际模型的误差**（实际模型只是该族中的一个元素），
所以 `opt_gap_train = train_rel - LB_train >= 0` 是**数学保证**，不是经验近似。
脚本对每个 run 都跑这个守门检查（`guarantee_ok`），出现负值会直接标出来而不是抹平。

> 修正记录：第一版用 `train_rel - LB_test` 作为 opt_gap，在 FNO 上出现**负值**
> （-0.043 ~ -0.025）。原因是 LB 用了 test 目标、误差用了 train 目标，两者不同侧。
> 已改为两侧各算各的：`opt_gap_train = train_rel - LB_train`，
> 另报 `test_rel - LB_test` 作为"test 侧离地板的总距离"。

## 2. 设置

- 数据：PDEBench 1D Burgers（`external_data/pdebench/fixed_t1_v1`，SHA-256 已 pin），
  1000/200/200 划分，输入=初始场，输出=固定未来时刻的解。
- 模型：3 架构 × 5 seed = 15 个已完成的 run（1500 步，Adam，batch 32）：
  `deeponet_mlp`、`deeponet_fourier`（width 64，trunk_in 1）、`fno`（width 32 / modes 16 / 4 层）。
- **未重训**；指标为从存档 checkpoint 复算，`replay diff` 列是与存档 `mean_relative_l2`
  的最大绝对差：DeepONet ~1e-8，FNO ~1e-5（CPU/GPU 浮点差，相对误差 < 0.03%）。

## 3. 结果（逐 run）

| run | family | test relFro | LB_train | LB_test | opt_gap(train) | gen_gap | replay diff |
|---|---|---|---|---|---|---|---|
| `deeponet_fourier_s0` | basis | 0.2624 | 0.0288 | 0.0280 | 0.1973 | 0.0363 | 1.42e-08 |
| `deeponet_fourier_s1` | basis | 0.2545 | 0.0294 | 0.0287 | 0.1825 | 0.0425 | 1.04e-08 |
| `deeponet_fourier_s2` | basis | 0.2646 | 0.0354 | 0.0346 | 0.2011 | 0.0282 | 1.41e-08 |
| `deeponet_fourier_s3` | basis | 0.2415 | 0.0305 | 0.0296 | 0.1791 | 0.0319 | 2.57e-08 |
| `deeponet_fourier_s4` | basis | 0.2463 | 0.0343 | 0.0334 | 0.1741 | 0.0379 | 7.87e-09 |
| `deeponet_mlp_s0` | basis | 0.2559 | 0.1180 | 0.1161 | 0.1281 | 0.0098 | 1.97e-08 |
| `deeponet_mlp_s1` | basis | 0.2407 | 0.1180 | 0.1156 | 0.1127 | 0.0100 | 4.25e-09 |
| `deeponet_mlp_s2` | basis | 0.2497 | 0.1184 | 0.1163 | 0.1181 | 0.0132 | 1.51e-08 |
| `deeponet_mlp_s3` | basis | 0.2454 | 0.1159 | 0.1140 | 0.1211 | 0.0084 | 1.43e-08 |
| `deeponet_mlp_s4` | basis | 0.2426 | 0.1173 | 0.1152 | 0.1144 | 0.0108 | 3.79e-09 |
| `fno_s0` | pointwise | 0.1459 | 0.0992 | 0.1432 | 0.0015 | 0.0452 | 4.61e-06 |
| `fno_s1` | pointwise | 0.1476 | 0.1067 | 0.1392 | 0.0076 | 0.0333 | 3.11e-05 |
| `fno_s2` | pointwise | 0.1424 | 0.0995 | 0.1400 | 0.0014 | 0.0415 | 1.29e-05 |
| `fno_s3` | pointwise | 0.1520 | 0.1012 | 0.1414 | 0.0083 | 0.0425 | 1.59e-05 |
| `fno_s4` | pointwise | 0.1512 | 0.1097 | 0.1472 | 0.0010 | 0.0405 | 5.66e-06 |

分组（均值 ± 样本标准差，n=5）：

| 架构 | family | LB_train | test relFro | opt_gap(train) | gen_gap | opt_gap / train_rel |
|---|---|---|---|---|---|---|
| deeponet_fourier | basis | 0.0317 ± 0.0029 | 0.2539 ± 0.0100 | **0.1868 ± 0.0118** | 0.0354 ± 0.0055 | **0.855** |
| deeponet_mlp | basis | 0.1175 ± 0.0010 | 0.2469 ± 0.0061 | **0.1189 ± 0.0061** | 0.0105 ± 0.0017 | **0.503** |
| fno | pointwise | 0.1033 ± 0.0047 | 0.1478 ± 0.0039 | **0.0040 ± 0.0037** | 0.0406 ± 0.0045 | **0.037** |

## 4. 三条结论

**① 修复表示 ≠ 提升精度（在公开数据、跨架构上复现）。**
把 MLP trunk 换成 Fourier trunk，LB_train 从 0.1175 降到 0.0317（**降低 3.7 倍**），
但 test 相对误差几乎不动（0.2469 → 0.2539，甚至略差）。
原因在表里：opt_gap 同时从 0.1189 **涨到** 0.1868——表示这一侧的富余被优化这一侧的退步吃掉了。
这与 README §6.7 在自建数据上得到的结论一致，现在有了公开数据版本。

**② 两种架构的瓶颈相反，且这是"换骨干还成立吗"的直接答案。**
- DeepONet：LB 只占 test 误差的 12%（0.0317/0.2539），
  而"离地板的相对距离" `opt_gap / train_rel` 高达 **0.50–0.86**
  → **优化受限**：训练连自己能表示的最优都远没达到。
- FNO：`opt_gap / train_rel` 只有 **0.037**，训练误差几乎贴在 readout 地板上
  → **不是优化受限**；它的 test 误差里地板本身（0.103）与泛化差距（0.041）主导。
- 同一个诊断框架在两种骨干上给出**相反**的瓶颈判定，正说明"表示受限 vs 优化受限"
  是**随架构而变**的性质，而不是某个模型的固有标签。

**③ 该方法可直接回答"下一步算力该花在哪"。**
- DeepONet → 花在优化器/训练配方上（换 trunk 没用）；
- FNO → 花在表示（更宽/更多模态，抬高 readout 地板）与泛化（正则/数据）上。

## 5. 局限（必须一起读）

1. **跨 family 的 LB 数值不可直接比较**。`basis` 允许每样本独立系数，
   `pointwise` 只允许全局共享 (W, b)，后者的约束更紧、LB 天然更大。
   可比的是**同一 family 内**的差异，以及归一化的 `opt_gap / train_rel`。
2. 只有 **1 个 PDE、1 个时刻、1 个分辨率**（PDEBench Burgers，fixed t1）；
   没有多分辨率/多时刻/第二 PDE 的外推。
3. 训练预算只有 1500 步（既有 run 的固定设置），
   "FNO 的 opt_gap ≈ 0"可能只是这个预算下的结论；加步数后 LB 与 train 可能一起下降。
4. FNO 的指标复算差 ~1e-5（DeepONet 是 ~1e-8），跨设备浮点差异，已如实记录。
5. `LB_test` 是**oracle readout**（在 test 上重优化最后一层）得到的最松下界，
   不是可达的干预；可达干预是 `lb_refit_on_train_then_test`（只报未用）。
6. 未覆盖 CViT / Transolver 类骨干（本机没有对应实现与 checkpoint）。

## 6. 复现

```bash
D:/miniconda3/envs/pytorch/python.exe scripts/readout_attribution.py --device cpu
```

- 输出：`results/readout_attribution_v1/`（`plan.json`、`runs/<name>.json`、
  `progress.json`、`group_summary.json`、`report.md`）。
- **可续跑**：每个 run 完成即落盘，重跑自动跳过已完成的；休眠中断后重跑同一条命令即可。

## 7. 下一步（D1 剩余 + 供 D4 使用）

- [ ] 把预算拉长（6k/20k/50k）看 opt_gap 是否随预算下降（结论 3 的前提）
- [ ] 加 width / modes 扫描，看 FNO 的 LB 是否随容量抬升（验证"FNO 卡在表示"）
- [ ] 补一个 CViT/Transolver 类骨干（若有现成实现）
- [ ] 这些 LB/opt/gen 三元组正是 D4"干预选择规则"的输入特征
