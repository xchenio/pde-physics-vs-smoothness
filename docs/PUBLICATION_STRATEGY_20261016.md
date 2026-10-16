# 投稿策略与后续方向（2026-10-16）

> 本文是 `PUBLICATION_ASSESSMENT.md` 的**执行层补充**：前者回答"能不能投"，本文回答
> "投哪里、怎么投、接下来做什么"。
> 约束条件由用户给定：**独立作者、已离开高校、版面费尽量为零或很低**。
> 写作原则与本项目一致：可主张 / 不可主张分开写，未核实的事实一律标注"待核实"。

---

## 0. 一分钟结论

| 问题 | 结论 |
|---|---|
| 现在能投什么？ | **TMLR**（首选）；同时把预印本放 arXiv 抢占优先权 |
| 最有希望的"第一篇"是什么？ | **诊断/归因框架 + 可复现性审计**（不是新算子架构） |
| 最便宜的第二篇？ | **JOSS** 软件论文（把诊断工具包单独发），零费用 |
| 零费用的会议路径？ | NeurIPS / ICLR / ICML 的 **AI4Science 类 workshop**（无 APC，但需主会注册或差旅；ML4PS 2026 明确不要求 NeurIPS 注册且可线上参与——今年截稿已过，关注 2027） |
| 绝不该选的？ | 任何 APC ≥ 2000 美元/瑞郎的金色 OA 刊（Nature 系、PLOS、MDPI、IEEE Access、Heliyon） |
| 提高上限的关键一步？ | 把"诊断"升级为**可预登记的干预选择规则**（见 D4），并把基线在 **PDEBench** 上打到能打 |

---

## 1. 项目可发表资产盘点（实事求是）

### 1.1 站得住的四块（按发表价值排序）

1. **静默广播缺陷 + 检测配方**（`docs/CURRENT_CONCLUSIONS.md` 第 1–2 条）
   共享网格求值器配逐样本坐标时把目标从 `(B, N)` 广播成 `(B, B, N)`，
   在 Burgers 初值、扩散-反应源项、Darcy 系数上均发生；损失曲线看起来正常，
   **真解的有限差分残差检查不出来**，只有"监督模型的 autograd 残差 + 显式形状断言"能暴露。
   → 这是**社区级**贡献：DeepONet/PI-DeepONet 类代码几乎都会踩。

2. **免训练的表示误差下界与三项归因**（`LB = sqrt(1-captured)`；`test = LB + opt_gap_train + gen_gap`）
   不需要训练到收敛即可把误差拆成"表示 / 优化 / 泛化"三部分；
   1D Burgers 是**优化主导**（LB 0.22–0.39，opt 0.45–0.52），2D 是**表示主导**（LB 0.85），
   `gen_gap ≈ 0`。→ 可做"下一单位算力该花在哪"的判据。

3. **一批被证伪的流行假设**（负面结果，价值被低估）
   因果加权在本设定单调变差；Adam+L-BFGS 精调在算子学习上失效（需要代表整个算子族的固定集）；
   解析谱先验对齐不优于零成本的随机 Fourier trunk；无目标的去相关正则在 2D 完全无效；
   "物理打赢监督"常是**监督没训够**的假象（2D 监督 6k 步 0.868 → 30k 步 0.637）。

4. **可复现性基础设施**（`src/utils.py:code_fingerprint/env_fingerprint`、封存确认集、
   逐检查点重放审计、σ_repeat = 0 的逐位一致性验证）
   → 独立作者最稀缺的就是可信度，这套东西是最强的背书。

### 1.2 已被自己撤回 / 不能主张的

- 基坍缩是 PI-DeepONet 失败的**根因**（修 bug 后基线自己能训到 captured 0.947）；
- 对齐是必要的、最优的或与 Fourier 特征互补；
- 增益 ∝ 基线跨种子标准差的线性律（5 seed × n=8 后 r 从 0.985 掉到 0.349）；
- 任何单 seed 的 20k/50k 结果构成方法排序。

### 1.3 三个硬伤（审稿人一定会问）

| 硬伤 | 现状 | 最低修复 |
|---|---|---|
| **基线竞争力弱** | PDEBench 上 MLP/Fourier DeepONet 0.394/0.412，**连"初始空间均值"基线 0.376 都没打过**，FNO 0.228 | 按 Lu et al. FAIR comparison（arXiv:2111.05512）把 POD-DeepONet、归一化、预算对齐补齐，先做到"不弱于均值基线" |
| **数据全是自建合成** | Burgers 1D/2D、DR、Darcy 均为仓库内实现；Allen-Cahn 是自生成 | 至少补一个公开基准（PDEBench Burgers/NS）+ 明确说明自生成数据的语义 |
| **精度区间偏低** | relL2 多在 0.6–0.8 / 0.003–0.04 | 论文定位必须是"诊断/归因"，**不能**包装成"高精度新方法" |

---

## 2. 相关领域最新论文（2025–2026）与定位对照

> 检索方式：arXiv 全文检索 + Web 检索（2026-10-16）。arXiv API 在本机被限流，
> 结果通过 arXiv 搜索页与摘要页核对。以下条目均给出编号以便自行复核。

### 2.1 必须引用并与之"划清界限"的工作

| 工作 | 出处 | 与本项目关系 | 写法建议 |
|---|---|---|---|
| **Diagnosing Failure Modes of Neural Operators Across Diverse PDE Families** arXiv:2601.11428 | **TMLR**（2026），17 页；**独立作者** Lennon Shikhman；750 个训练模型、3 架构 × 5 PDE 族、压力测试 | ⭐**最直接的成功先例**：证明"神经算子失败模式诊断"这类工作能被 TMLR 接收，且独立作者可行 | 作为首要对标：他们做**部署时的分布偏移压力测试**，我们做**训练内的误差归因与实现正确性**，互补而非竞争 |
| **The importance of tensor handling and implementation of physics-informed DeepONets** arXiv:2610.22157 | arXiv 2026-10-15 | ⚠️**与我们贡献①相邻**：同一议题（PI-DeepONet 的张量处理/实现）正在成为独立可发表主题 | 他们讲**效率**（去冗余、2D 加速 3.75×），我们讲**静默错误与检测**。**必须引用并明确区分**，且要抢时间 |
| **What do physics-informed DeepONets learn?** arXiv:2411.18459 → *J. Comput. Phys.* 2026 | JCP | trunk 基函数视角的先行工作 | 我们的下界是**可证**的投影恒等式，他们的诊断是观察性的；写清"从观察性诊断到可证下界"的增量 |
| **Characterizing possible failure modes in PINNs**（Krishnapriyan 等, NeurIPS 2021） | NeurIPS 2021 | 失败模式研究的开山 | 对标"从 PINN 到算子学习"的迁移，并指出算子学习多了"逐样本配对"这一新失效面 |
| **FAIR comparison of neural network architectures**（Lu 等, arXiv:2111.05512） | arXiv 2022 | POD-DeepONet / DeepONet-FNO 公平比较 | 补基线时的操作手册 |
| Lee & Shin 两阶段 + 正交化（arXiv:2309.01020） | arXiv 2023 | 两阶段训练与正交化已被做过 | 不能把"固定 trunk 后训 branch、QR/SVD"写成首创 |

### 2.2 领域正在往哪走（决定我们的叙事）

| 趋势 | 证据 | 对我们的含义 |
|---|---|---|
| **"诊断/负结果"已是可发表的独立品类** | 2601.11428 进 TMLR；2610.07811（物理约束在粗粒化下失效）被 NeurIPS 2026 STODY workshop 接收；2610.10972（神经 PDE 求解器的状态可迁移性）投 ICLR 2027 | 我们的定位**正好在风口**，不需要伪装成新方法 |
| **实现/数值可靠性开始被单独讨论** | 2610.22157（张量处理）；本项目已发现"FD 参考步长敏感性 12–44%" | 贡献①有窗口期，越早挂预印本越好 |
| **强基线碾压新架构** | 本项目自身证据：FNO 0.313 vs 最佳混合 0.442（差 41%）；2601.11428 也指出 FNO 分布内最优但压力下退化最严重 | 别再堆模块；把"强基线 + 受控对照"当卖点 |
| **Sim-to-real 与真实数据基准** | RealPDEBench（ICLR 2026 Oral, arXiv:2601.01829）、PDEBench 持续更新 | 用公开基准补"外部效度"是性价比最高的一步 |

---

## 3. 投稿场所清单（按"独立作者 + 零/低费用"过滤）

> ⚠️ **费用与规则均以官网最新说明为准**，投稿前逐条核对下表"待核实"项。

### 3.1 第一梯队：完全免费、开放获取、接受独立作者

| 场所 | 费用 | 契合度 | 备注 |
|---|---|---|---|
| **TMLR**（Transactions on Machine Learning Research） | 作者不付费（**待核实**，以 tmlr.net / OpenReview 官方说明为准） | ⭐⭐⭐⭐⭐ | OpenReview 开放评审、无固定截稿期、看重"意义与可读性"而非 novelty。已有同类型独立作者工作录用（2601.11428）。**主推** |
| **JMLR** | 作者不付费（**待核实**） | ⭐⭐⭐ | 偏好方法论/理论，纯诊断类略吃亏；可作为 TMLR 之后的第二选择 |
| **JOSS**（Journal of Open Source Software） | 免费 | ⭐⭐⭐⭐（针对工具包） | 只发软件论文，5–10 页短稿，独立作者友好。可把诊断/审计工具包单独发一篇（见 D3） |
| **arXiv + Zenodo** | 免费 | ⭐⭐⭐⭐⭐ | 不是发表，但**立即建立优先权**（尤其针对 2610.22157），且是独立作者最省钱的"时间戳" |

### 3.2 第二梯队：混合期刊的**订阅制（非 OA）**路径 —— 作者零费用

选"传统订阅模式"出版时作者通常不付 APC；若选金色 OA 则需付费（Elsevier/Springer/IOP 的 APC 通常 2000–3500 美元量级，**需逐刊核实**）。

| 期刊 | 主题契合 | 门槛 |
|---|---|---|
| **Machine Learning: Science and Technology**（IOP） | ⭐⭐⭐⭐⭐ 科学机器学习专刊，近年大量 SciML 复现/诊断类论文 | 中 |
| **Journal of Computational Physics**（Elsevier） | ⭐⭐⭐⭐ | 高（需要更强的问题动机与精度） |
| **Computer Methods in Applied Mechanics and Engineering**（Elsevier） | ⭐⭐⭐⭐ | 高 |
| **Neurocomputing** / *Journal of Computational Science* / *Engineering with Computers*（Elsevier/Springer） | ⭐⭐⭐ | 中低，诊断类更容易过 |
| **Journal of Scientific Computing** / *Advances in Computational Mathematics*（Springer） | ⭐⭐⭐⭐ | 中高 |
| **SIAM J. Scientific Computing**（SISC） | ⭐⭐⭐ | 很高 |

> 提示：混合期刊的"非 OA"选项在投稿系统里通常叫 **"Subscription / Standard publication route"**，
> 不是默认绿色 OA。提交前确认选了哪条路，并保留版权协议副本。

### 3.3 第三梯队：中文期刊（版面费低，国际能见度低）

《计算力学学报》《力学学报》《空气动力学学报》《数值计算与计算机应用》《计算机研究与发展》等，
版面费通常为**数百至数千元人民币**，部分免收（**需查各刊官网最新稿约**）。
优势：写作与审稿沟通成本低、对国内独立作者友好；劣势：对后续国际投稿的"已发表"约束要留意
（一稿多投与再发表的版权问题必须先问清楚）。

### 3.4 会议 / Workshop

| 场所 | 费用 | 状态 |
|---|---|---|
| **ML4PS**（Machine Learning & the Physical Sciences） | 无 APC；**明确不要求 NeurIPS 注册**，可线上参与 | 2026 年截稿 9/19（已过），workshop 12/11 在 Georgia Tech；关注 2027 版 |
| NeurIPS / ICLR / ICML 的 AI4Science、STODY、ML4PDE 等 workshop | 无 APC，但通常需主会注册（数百至上千美元）+ 差旅 | 2610.07811 即 NeurIPS 2026 STODY 接收案例；ICLR 2027 workshop 多在 2027 年 1–2 月截稿 |
| NeurIPS / ICLR 主会 | 无 APC，但注册费 + 差旅，且本项目当前完成度不够 | ❌ 不建议作为第一目标 |

### 3.5 明确排除

任何 APC ≥ 2000 美元/瑞郎的金色 OA：Nature Communications、Scientific Reports、PLOS ONE、
MDPI 全系、IEEE Access、Heliyon、AIP/APL 系的部分 OA 选项等。

---

## 4. 三条投稿路线（可直接执行）

### 路线 A（推荐，3–4 个月）：TMLR 诊断论文

标题方向：
> *Where Should the Next Unit of Compute Go? Training-Free Error Attribution and a Silent
> Pairing Pitfall in Physics-Informed Operator Learning*

结构建议（对应四块资产）：

1. **Pitfall**：`(B,N)` → `(B,B,N)` 广播缺陷，最小复现 + 检测配方（监督模型 autograd 残差 + sampler 形状断言）；
2. **Attribution**：`LB = sqrt(1-captured)` 的可证下界 + `test = LB + opt + gen` 三项分解；
3. **Evidence**：1D 优化主导 / 2D 表示主导的对照，步数扫描（6k/20k/50k）说明"结论是预算条件下的"；
4. **Falsifications**：因果加权、L-BFGS、对齐 vs Fourier、ortho 在 2D 无效、"物理打赢监督"假象；
5. **Tooling**：代码指纹、封存确认集、重放审计、σ_repeat=0；
6. **Retractions**：明确列出撤回项（这是加分项，不是减分项）。

**必须补齐（否则大概率被拒）**：
- 在 **FNO 骨干**上复现三项归因（项目已开始 FNO 侧工作，需把它做成正式实验）；
- 在 **PDEBench** 至少一个子集上把基线调到不弱于"空间均值"基线；
- 每格 ≥ 5 seed，报告区间而非点估计。

### 路线 B（1–2 个月，零成本）：JOSS 软件论文 + arXiv 预印本

把 `src/diagnostics.py` + `src/utils.py` 的指纹 + sampler 契约断言抽成独立包
（例如 `pinode-diag`），投 JOSS：LB/三项归因诊断、形状契约检查、代码指纹、报告生成。
同时把路线 A 的预印本挂 arXiv（先占时间戳）。

### 路线 C（6–12 个月）：把诊断升级为方法

见 §5 的 D4。这是唯一能把"案例分析"变成"方法论文"的路径，也是冲击更好期刊的前提。

---

## 5. 后续研究方向（按投入产出比分档）

> 排序依据：是否需要大模型/大算力（独立作者劣势）→ 是否依赖现有资产（项目优势）→ 社区争点热度。

### 🥇 第一档：6–9 个月，直接提高录用概率

**D1. 三项归因跨架构验证（DeepONet → FNO / CViT）**
- 动机：现在的归因只在 DeepONet 上成立，审稿人会说"换个骨干还成立吗"。
- 做法：把 `LB` 的定义推广到 FNO（用 trunk 等价物：最后一层特征 / 模态系数张成的子空间），
  在已有 FNO 实验（expW / expS20k / FNO_PROJECTION 系列）上复算三项分解。
- 产出：路线 A 的 §3 章节。风险：低（大部分代码与数据已在）。

**D2. 静默配对缺陷的社区审计（抢时间）**
- 动机：arXiv:2610.22157 已把"PI-DeepONet 的张量处理"摆上台面（虽然讲效率）。
- 做法：把检测配方做成**可在他人代码库上跑的检查器**，在 DeepXDE、
  `Physics-informed-DeepONets` 官方实现、`physics_informed`（PINO）上做**只读审计**，
  报告"哪些模式会触发、现有代码是否有防护"。**只报告事实，不做指摘**，措辞需极度克制。
- 产出：一篇独立的"实现可靠性"短文（TMLR 短轨 / workshop / JOSS 配套），或并入路线 A 作为 §1 的外部效度。
- 风险：中（涉及对他人代码的评价，需附可复现脚本并保持中性）。**时间窗口有限，优先于 D1。**

**D3. 诊断工具包 → JOSS**
- 见路线 B。产出：一篇零费用发表 + 一个可引用的 DOI。风险：低。

### 🥈 第二档：9–18 个月，决定能否上更好的场所

**D4. 诊断驱动的干预选择（把案例升级为方法）**
- 问题：给定固定总预算，**只用训练/验证期能看到的信息**，能否提前判断该改善表示、
  稳定优化、还是加数据/正则，并选到有效的干预？
- 做法：预登记一个规则（例如：LB 不降 → 换 Fourier trunk；opt_gap_train 主导 → 加步数/改优化器；
  gen_gap > 阈值 → 加数据）。对照三方：**固定默认方案 / 随机选干预 / 预算内验证集搜索**。
  在**未见设置**（新 PDE 或新分辨率）上评估"选中方案的收益"与"错误选择率"。
  **选干预的试验成本必须计入预算**；规则只能用当时允许看到的标签。
- 产出：如果规则的样本外选择价值成立 → 方法论文（TMLR/MLST，甚至 AI4Science 主会 workshop）；
  若不成立 → 诚实写成负结果，并入路线 A。
- 风险：中。**这是本项目唯一能突破"诊断"天花板的路径，值得押注。**

**D5. "物理损失何时退化为平滑正则"**
- 动机：本项目已发现 PDE 惩罚并不优于等强度的曲率/梯度平滑（确认集上 PDE 1.376% vs 曲率 2.561%，
  且 PDE-vs-curvature 优越性门槛失败）；arXiv:2610.07811 又从"粗粒化下非线性残差不闭合"
  给出理论侧证据。社区正在争论"物理信息到底带来了什么"。
- 做法：固定架构与预算，扫描 **数据预算 × 离散化分辨率 × 标签噪声**，
  量化 PDE 损失相对"等强度平滑正则"的**边际信息量**（用配对区间而非点估计）。
- 产出：高话题度论文，适合 MLST / NeurIPS·ICLR AI4Science workshop。
- 风险：中。需要把"等强度"定义清楚（本项目此前正则强度最优点都在搜索边界 0.3，需扩大搜索范围）。

**D6. 公开基准上的外部效度**
- 用 PDEBench（Burgers / 反应扩散 / NS）+ 条件允许时 RealPDEBench 的 sim-to-real 子集
  （真实数据量小、物理预训练+少量真实微调契合本项目叙事）。
- **前提**：先把基线调到能打（对齐 FAIR comparison 的 POD-DeepONet 与预算），
  否则"基线太弱"会一票否决。
- 风险：中（算力），但这是所有路线共同的必做项。

### 🥉 第三档：不建议作为独立作者主线

- 算子预训练规模律、OOD 泛化界紧化、高维/张量低秩框架——都需要大算力与合作网络，
  独立作者与团队正面竞争不占优。可作为 D4/D5 的"讨论与展望"，不必作为主攻。

---

## 6. 独立作者的实操注意事项

1. **arXiv 背书**：无机构邮箱时 cs.LG / math.NA 可能需要 endorsement。
   提前注册 ORCID、Zenodo DOI；如被要求背书，找一位相关领域的已发表作者，
   或先投有编辑审查的场所再反向挂预印本。
2. **Affiliation**：多数期刊接受 "Independent Researcher, <城市>, <国家>"；
   少数要求机构邮箱，投稿前先写信确认，避免白跑一轮。
3. **可信度信号**（本项目已具备，务必在投稿信和 README 里显式列出）：
   预登记规则、封存确认集（含"已消耗"清单）、代码指纹、逐位一致的重放审计、
   明确的撤回声明。**这是独立作者最好的名片。**
4. **费用核查清单**（投稿前逐项核对官网）：
   - 选的是**订阅制**还是**金色 OA**；
   - 是否收取审稿费 / 超页费 / 彩图费；
   - 版权协议是否允许把预印本挂在 arXiv（多数允许，但要看具体条款）。
5. **时间成本**：TMLR/JMLR 无固定截稿期，适合边做边投；workshop 有硬截稿，
   建议以 ICLR 2027 workshop 周期（通常 2027 年初截稿，**待核实**）为第一个会议目标。

---

## 7. 建议的 90 天执行顺序

| 周 | 动作 | 产出 |
|---|---|---|
| 1–2 | 挂 arXiv 预印本（路线 A 的当前版本，标注"working draft"）+ Zenodo DOI | 时间戳 |
| 1–4 | D2：检测配方做成可移植检查器，在 2–3 个开源实现上做只读审计 | 外部效度表 + 时窗口保护 |
| 3–6 | D1：三项归因搬到 FNO 骨干，复算 LB/opt/gen | 路线 A 的 §3 |
| 4–8 | D6 前半：PDEBench 上按 FAIR comparison 补齐 POD-DeepONet 与预算对齐 | 基线不弱于均值基线 |
| 6–10 | D3：把诊断工具包独立成包 + JOSS 投稿 | 零费用发表 |
| 8–12 | 路线 A 完稿 → 投 TMLR | 主投稿件 |
| 并行 | 关注 ICLR 2027 / NeurIPS 2027 的 AI4Science workshop 截稿 | 会议备份 |

---

## 8. 待核实清单（投稿前逐条确认，不要凭本文下结论）

- [ ] TMLR / JMLR 是否向作者收费（官方 Author Guidelines）
- [ ] 各混合期刊（JCP / CMAME / MLST / JSC / Neurocomputing）当年是否仍为非 OA 路线免费
- [ ] ML4PS 2027、ICLR 2027 / NeurIPS 2027 AI4Science 类 workshop 的截稿与注册要求
- [ ] 目标期刊对"独立作者 / 无机构邮箱"的具体规定
- [ ] 目标期刊对预印本（arXiv）的政策
- [ ] 中文期刊（若选）的版面费与"已发表后再投国际刊"的版权条款
