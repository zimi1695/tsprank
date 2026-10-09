# TSPRank 股票选股系统

> **2026 THU大数据挑战赛** 个人实验工作区  
> 作者：li177  
> 基于 TSPRank (KDD 2025) × DoubleEnsemble W4 的 CSI300 每周 Top-5 选股系统  
> **当前状态：v2.2 TSPRank 实验已完成。Walk-forward 严格评估（52周）显示 TSPRank 无净贡献（Sharpe 0.015 vs Backbone 1.19），Rolling 模式的高收益（Sharpe 4.72）完全来自数据泄漏。
> 当前生产方案：Backbone baseline（52周 Sharpe 1.19, +22.84%）。下一步：接入小组 pred.pkl 提升 backbone IC，探索 LambdaMART 排序学习。**

---

## 目录

1. [项目定位与背景](#1-项目定位与背景)
2. [比赛任务说明](#2-比赛任务说明)
3. [论文来源：TSPRank 详解](#3-论文来源tsprank-详解)
4. [小组仓库源代码的使用](#4-小组仓库源代码的使用)
5. [本项目如何结合两者](#5-本项目如何结合两者)
6. [系统架构全览](#6-系统架构全览)
7. [各模块详细说明](#7-各模块详细说明)
8. [实验结果与诚实评估](#8-实验结果与诚实评估)
9. [已知问题清单](#9-已知问题清单)
10. [目录结构](#10-目录结构)
11. [环境依赖](#11-环境依赖)
12. [快速开始](#12-快速开始)
13. [参数说明](#13-参数说明)
14. [后续计划](#14-后续计划)
15. [参考文献](#15-参考文献)
    - [6] 论文原文逐项核查记录

---

## 1. 项目定位与背景

### 这个仓库是什么

`~/tsprank/` 是 li177 在小组项目之外开辟的**个人实验工作区**，独立于小组主仓库
`~/quant-stock-forecast/qlib-csi-300/`。

### 小组主仓库结构

```
~/quant-stock-forecast/qlib-csi-300/
├── lib/                          # 核心代码库（qlib pipeline 封装）
│   ├── train.py                  # 模型训练（调用 qlib model.fit）
│   ├── predict.py                # 信号生成 + IC/ICIR 计算 + 比赛评估
│   ├── evaluate.py               # 最终评估（生成 metrics.json + HTML 图表）
│   ├── contest_record.py         # 比赛公式评估（周度 Top-K 收益 vs 基准）
│   ├── contest_topk.py           # Top-K 配置工具
│   ├── contracts.py              # 领域契约（portfolio 构建、IC 计算、数据验证）
│   ├── evaluation_metrics.py     # 30+ 量化指标（beat_rate/excess/MDD/CVaR/NDCG...）
│   ├── model_assets.py           # 模型资产准备
│   ├── runtime.py                # RuntimeContext 数据类
│   ├── monitoring.py             # 监控计时
│   ├── rolling.py                # 滚动训练
│   └── utils.py                  # 工具函数
├── experiments/                  # 实验目录
│   ├── round1/                   # Round1：13 个模型 × 3 个 seed = 39 个实验
│   ├── round2/                   # Round2：6 个模型 × 5 个窗口 = 30 个实验
│   └── ex1~ex12/                 # 早期探索实验
└── resources/
    └── score.csv                 # 沪深300基准周收益率（比赛评估用）
```

### 实验命名规则

以本项目基于的实验为例：`r2_c06_densemble_alpha158_w4_sim2019_s20260508`

| 字段 | 含义 |
|------|------|
| `r2` | Round 2（第二轮实验） |
| `c06` | Candidate 06（第 6 号候选方案） |
| `densemble` | DoubleEnsemble 模型（ICDM 2020） |
| `alpha158` | Alpha158 因子集（158 维特征） |
| `w4` | Window 4（训练窗口方案 4） |
| `sim2019` | 训练截止 2019 年（模拟 2019 年前市场状态） |
| `s20260508` | 随机种子 20260508 |

Round2 的研究问题：**基于 CSI300 状态相似度选择历史训练窗口，检验相似窗口路线**。
W4（sim2019）的核心假设：当前市场（2025-2026）与 2019 年反弹行情存在状态相似性。

### 两个仓库的关系

| 维度 | 小组主仓库 (qlib-csi-300) | 本仓库 (tsprank) |
|------|--------------------------|------------------|
| 完整性 | 完整 qlib pipeline + mlflow | 独立运行，不依赖 mlflow |
| 实验数 | 95+ 个正式实验 | 单一实验性质 |
| 最佳模型 | DoubleEnsemble W4 | 复用 W4 超参 + 新增 TSPRank 排序层 |
| 评估框架 | lib/evaluate.py（30+ 指标） | score_self.py（比赛固定接口） |
| 评估基准 | 沪深300（score.csv） | 绝对收益（vs 空仓） |
| 价格字段 | 复权开盘价 `$open*$adjclose/$close` | 原始开盘价（与 score_self.py 一致） |
| 数据 | ~/quant-stock-forecast/data/ | ~/tsprank/data/（本地副本） |

**本仓库的目标**：将 KDD 2025 论文 TSPRank 的排序学习方法叠加在已验证的
DoubleEnsemble W4 方案之上，探索排序学习是否能进一步提升选股质量。

**⚠️ v2.2 更新记录**：
  v2.0 存在的四项实现偏差已在 v2.2 中修复：
Global Loss	Structured max-margin + solver-in-loop（Eq.6-7）	✅ v2.2 已对齐
  ② 缺少 bias 项 b：已添加 `nn.Parameter(torch.zeros(1))`（论文 Eq.2）✅
  ③ Sector 分组：已实现按行业分组训练+推理（论文 Section 5.4，N=5~25）✅
  ④ cost 矩阵归一化：TSPSolver.solve() 内置归一化，训练-推理行为一致 ✅
  新增 SCIP 精确 MILP 求解器（pyscipopt，免费替代 Gurobi）✅
  新增 PIT（point-in-time）行业映射，避免前视偏差 ✅

**⚠️ 历史纠偏**：
  此前版本将未收敛归因于"A股市场变化大""分布漂移"，这是错误的。
  论文在 stock ranking 实验上验证有效（Table 1: Kendall Tau=0.0447）；
  根本原因是实现偏差，不是方法本身或市场因素。

## 2. 比赛任务说明

### 评分规则

THU-BDC 2026 每周提交一次，从 CSI300 成分股中选出**至多 5 支股票**并分配权重：
提交格式 (`output/result.csv`): `股票代码,权重`，如 `600000,0.2 601318,0.2`

约束条件：
- 股票数 ≤ 5
- 所有权重之和 ∈ [0.0, 1.0]
- 股票代码为纯数字 6 位（`score_self.py` 按此格式匹配）

最终得分 = Σ(股票_i 当周收益率 × 权重_i)

收益率 = (当周最后交易日开盘价 − 当周第一交易日开盘价) / 当周第一交易日开盘价

评分代码固定在 `score_self.py`，不可修改。

### 评估周期

- 评估起始: 2025-02-24
- 评估截止: 2026-04-30
- 仅计入完整交易周 (`complete_week_only=True`，严格 5 个交易日，周一到周五)
- 最近 12 周参考 (`recent_weeks=12`)

---

## 3. 论文来源：TSPRank 详解

### 3.1 论文信息

- **标题**: TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear Travelling Salesman Model
- **作者**: Weilong Li et al.
- **会议**: KDD 2025
- **arXiv**: https://arxiv.org/abs/2411.12064
- **代码**: https://github.com/waylonli/TSPRank-KDD2025

### 3.2 论文要解决的问题

排序学习（Learning to Rank）存在两种主流范式的矛盾：

| 范式 | 代表方法 | 优势 | 劣势 |
|------|---------|------|------|
| **Pairwise** | RankNet, LambdaRank, **LambdaMART** | 建模两两比较，训练信号丰富 | 忽略全局排列结构 |
| **Listwise** | ListNet, Rankformer | 考虑整体排列，全局最优 | 计算复杂，对规模敏感 |

> ⚠️ **注意**：LambdaMART 在论文中被明确归类为 **Pairwise** 方法（作为 baseline
> 与 TSPRank 对比），不是 Listwise 方法。

TSPRank 的核心贡献是**同时利用两者的优点**：
- 用 Row-wise Cross Entropy 提供局部训练信号（Local Loss）
- 用 TSP 路径 loss 引入全局排列约束（Global Loss，工程近似版）
- 推理时用 TSP 求解器得到近优排列

### 3.3 论文原文逐节精读（基于 arXiv:2411.12064v2 逐字阅读）

以下内容基于论文原文 24 页逐字阅读，不跳过任何公式和关键段落。

#### 3.3.1 Section 3.1 — 排序建模为 TSP

论文将排序问题建模为 **open-loop TSP**（无闭环的旅行商问题）。

**公式 Eq.1**：

    π* = argmax_π Σ_{i=1}^{N-1} s(e_{π(i)}, e_{π(i+1)})

其中 π(i) 是排列中第 i 个位置的实体。
s(e_{π(i)}, e_{π(i+1)}) 的语义是：**从位置 i 的实体走到位置 i+1 的实体的得分**。

这是路径上的 **successor 关系**——不是"任意两个实体谁优谁劣"，
而是"谁紧接在谁后面"。论文原文：
> "Given a set of entities e1, e2, ..., eN and all the pairwise score values 
> s(ei, ej) indicating the gain of ranking entity ej immediately after ei, 
> we define a complete graph G = (V, W)"

把 N 个实体看成 N 个城市，边权重 = pairwise score，
找总路径得分最高的 open-loop tour = 最优排名。

#### 3.3.2 Section 3.2 — 评分模型（Scoring Model）

评分模型包含两部分：**optional encoder** + **trainable bilinear model**。

**Encoder**：
论文原文："The default encoder used is the Transformer encoder block, 
but it can be replaced with any other encoder suited to the specific task and dataset. 
Additionally, **using an encoder is optional if sufficiently robust backbone models are available**."

→ 在 stock ranking 实验中，论文直接用 Feng et al. 的 relational embedding 层输出，
**不做 Transformer encoder**。本项目用 MLP projector 替代是合理的工程选择。

**双线性模型（公式 Eq.2）**：

    s(e_i, e_j) = e_i^T · W · e_j + b

其中：
- W ∈ R^{d×d} 是可学习参数矩阵
- **b 是可学习标量偏置（scalar bias）**
- W 和 b 与 encoder 参数一起优化

**✅ v2.2 已修复**：已添加 `nn.Parameter(torch.zeros(1))` 作为可学习标量偏置 b（见 `tsprank_model.py` BilinearTSPLayer）。
**邻接矩阵（公式 Eq.3）**：

    A_ij = s(e_i, e_j),  对所有 i, j ∈ {1, ..., N}

**⚠️ 论文中 A 就是原始得分矩阵 s(e_i, e_j)，不是 sigmoid 输出**。
当前实现中 A = sigmoid(S) 只用于诊断，loss 统一用 S，这与论文一致。


**本项目的工程说明**：
- S 不做 L2 归一化：L2 归一化 + 正交 W → Cauchy-Schwarz 硬约束 |S[i,j]| ≤ 1，
  对 N=100 做 softmax 永远 ≈ log(100)=4.605，梯度消失
- Projector 末层 LayerNorm 已控制 H 量级，无需额外 L2 归一化
- W 正交初始化 → 训练初期 S≈0，A≈0.5
#### 3.3.3 Section 3.3 — 推理（Inference）

论文原文："Given the scale of the ranking problem and the desired ranking accuracy, 
**we prefer to obtain an exact solution** for the TSP rather than an approximate one. 
Therefore, we formulate the TSP as a Mixed Integer Linear Programming (MILP) problem"

使用 Gurobi 或 CPLEX 求解精确 MILP。

决策变量（公式 Eq.4）：

    x_ij = 1  if entity e_j is ranked immediately after e_i
    x_ij = 0  otherwise

**Appendix A 给出了完整的 MILP 公式**（Eq.10-15）：
- 目标函数：max Σ_i Σ_j s_ij · x_ij（最大化路径总分）
- 约束 Eq.11-12：每个节点最多一个前驱、一个后继
- 约束 Eq.13：总边数 = N - 1
- 约束 Eq.14-15：MTZ subtour elimination（防止出现子环路）

**与当前实现的差异**：
- 论文用 Gurobi/CPLEX 精确 MILP → 当前用贪心最近邻 + Or-opt 启发式
- 精确求解保证全局最优 → 启发式只保证局部近优
- 论文 Section 8 明确说 TSPRank 适合 **N < 30** 的小规模问题（N=100 时 solver 占 99.8% 推理时间）


**本项目的工程说明**：
- 用 Or-opt 而非 2-opt 的原因：标准 2-opt 反转子路径在无向 TSP 有效，
  但在有向非对称 TSP 中反转会改变内部边方向，生成无效路径。
  Or-opt（节点重插入）对有向图天然有效。
- Cost 矩阵：C = -(S - S.mean()) / S.std()，归一化后取反（最大 S 路径 = 最小 cost 路径）
- 贪心最近邻选 row-sum 最小的节点为起点，逐步选最近未访问节点
#### 3.3.4 Section 4.1 — Local Learning（公式 Eq.5）

**目标邻接矩阵 A^t** 的定义：
    a^t_ij = 1  当且仅当 e_j 紧接在 e_i 后面（successor），否则 0

**公式 Eq.5**（逐符号解释）：

    L_local(A^p, A^t) = -Σ_{i=1}^{N} y_k · log(exp(A^p_ik) / Σ_j exp(A^p_ij))
    其中 k = argmax_j A^t_ij

逐符号理解：
- i 遍历所有实体（按 gold 排列顺序 π(1), π(2), ..., π(N)）
- k = argmax_j A^t_ij = π(i+1)，即实体 i 的真实 successor
- log(exp(A^p_ik) / Σ_j exp(A^p_ij)) = log_softmax(A^p 的第 i 行)[k]
  → 从 N 个候选中识别 successor 的多分类 cross-entropy
- **y_k 是权重**，论文原文："We weight the loss by the true ranking label y, 
  allowing the penalty to vary according to different ranking positions. 
  Additionally, y can be adjusted to N+1-y depending on whether y 
  represents ascending or descending order."

**权重 y_k 的含义**：
- y 是排序标签（rank label），不是固定公式
- 对于降序排列（收益最高的排第一），权重 = N+1-position
- 当前实现用 w_k = N - k（即 N+1-(k+1)），对应降序场景，**与论文一致**

**与小组实现的差异**：
- 小组 lib/tsprank.py 用 **all-pair BCE**（所有 i>j 对做二分类），不是 row-wise CE
- 论文 Eq.5 明确是 row-wise：对每行只识别真实 successor，其余 N-1 个是负类
- 当前实现的 row-wise weighted CE **与论文 Eq.5 一致**，小组实现反而偏离了


**本项目实现（代码）**：

    perm    = torch.argsort(returns, descending=True)   # 真实排列
    weights = torch.arange(N, 0, -1, dtype=torch.float32)  # [N, N-1, ..., 1]
    src, dst = perm[:-1], perm[1:]                      # 相邻对
    log_prob = F.log_softmax(S[src], dim=1)             # 用 S 而非 A
    loss     = -(weights[:N-1] * log_prob[range(N-1), dst]).sum() / weights[:N-1].sum()

权重 w_k = N-k 即 N+1-(k+1)，对应论文说的"降序排列"场景（收益最高的权重最大）。
#### 3.3.5 Section 4.2 — Global Learning（公式 Eq.6-8）

**这是论文最核心的部分，也是当前实现偏差最大的地方。**

**公式 Eq.6**（margin 约束）：

    x^t · A^p ≥ x · A^p + Δ(x^t, x),  对所有 x

含义：真实路径 x^t 在预测矩阵 A^p 上的得分，必须比任何其他合法路径 x 的得分高出至少 Δ(x^t, x)。

**公式 Eq.7**（max-margin loss）：

    L_glob(A^p, x^t) = max(0, max_{x≠x^t} [Δ(x, x^t) + x · A^p] - x^t · A^p)

逐符号理解：
- x^t · A^p = 真实路径在预测矩阵上的总分
- **max_{x≠x^t} [Δ(x,x^t) + x·A^p]** = 找一个不同于真实路径的合法 TSP 路径 x，
  使得 Δ(x,x^t) + x·A^p 最大 → **这本身就是一个 TSP 求解问题！**
- Loss = max(0, 最违反约束的路径得分 - 真实路径得分)

**公式 Eq.8**（structured score）：

    Δ(x, x^t) = Σ_i Σ_j max(0, x_ij - x^t_ij)

含义：统计 x 中有多少边是 x^t 没有的（错误边的数量），作为 margin。

**论文原文关键句**：
> "This approach eliminates the need to manually define the target adjacency matrix. 
> Instead, it uses the output from the **black-box optimiser** to directly guide 
> the updates of the model parameters."

> "Relying solely on the global loss requires significantly more epochs to converge. 
> Therefore, during training, we **alternate batches** between the global loss 
> and the local loss, ensuring faster and more efficient model convergence."

**✅ v2.2 已修复**：已对齐论文 Eq.6-7，实现 structured max-margin + SCIP solver-in-loop。

**v2.3 实验结论**：52 周 walk-forward 严格评估显示 TSPRank 无净贡献（Sharpe 0.015 vs Backbone 1.19）。
详见第 8 节。
#### 3.3.6 Section 5.4 — Stock Ranking 实验设置

**论文原文逐字引用**：

> "Considering that TSPRank specialises in **small-scale ranking problems**, 
> we implement rankings **within individual sectors**, 
> categorising stocks into groups based on their sectors. 
> We exclude all sector groups containing three or fewer stocks 
> and remove stocks with unidentified sector information."

**这是设计决策，不是可选细节**。论文明确说 TSPRank 适合小规模排序问题，
所以按行业分组，每组 5~20 只股票。

论文 stock ranking 数据集：NASDAQ 1026 只 + NYSE 1737 只，
按行业分成 50+70=120 个小组，组内排序。
过滤后平均每组约 9~11 只股票（见 Appendix E 的 Table 7-16）。

> "we use outputs from the **relational embedding layer** introduced by Feng et al., 
> replacing the original prediction layer with the other benchmark models"

直接用 Feng et al. backbone 的 embedding 输出，不做 Transformer encoder。
embedding 维度来自 Feng et al. 的 relational embedding layer。

> "All benchmark models and TSPRank with local learning are trained for **100 epochs** 
> for each sector. We find that TSPRank with global learning generally requires 
> more epochs to converge; therefore, TSPRank-global is trained for 
> an additional **50 epochs** to ensure convergence."

每个 sector 独立训练：Local 100 epochs，Global 150 epochs。

**Appendix B 超参数**（Table 5）：
- lr = 1e-4, weight_decay = 1e-5
- tf_layer = 1, tf_nheads = 8, tf_dim_ff = 128
- batch_size = 128
- LambdaMART: n_estimators = 10,000, XGBoost 实现

**与当前实现的设置级差异**：

| 维度 | 论文 | 当前实现 |
|------|------|----------|
| 排序范围 | sector 内 5~20 只 | 300 只 CSI300 混排 |
| 训练粒度 | 每个 sector 独立训练 | 全部股票一起训练 |
| N 的规模 | 5~20（精确 TSP 可行） | 100~300（精确 TSP 不可行）|
| 训练数据 | 2013-2017（Feng et al. 数据集） | 2013-2019 |
| backbone | Feng et al. relational embedding | DoubleEnsemble W4 (LightGBM) |

#### 3.3.7 论文实验结果（Table 1）

NASDAQ stock ranking（50 个 sector 平均）：

| 模型 | Kendall Tau | IRR@5 | SR@5 |
|------|-------------|-------|------|
| Feng et al. MLP (Original) | 0.0093 | 0.1892 | 0.9682 |
| LambdaMART | 0.0071 | 0.0505 | 0.2678 |
| Rankformer | 0.0110 | 0.2309 | 1.0943 |
| **TSPRank-Local** | **0.0291** | **0.2537** | **1.2623** |
| **TSPRank-Global** | **0.0447** | **0.2937** | **1.4331** |

TSPRank-Global 在所有指标上显著超越所有 baseline。
Kendall Tau 从 backbone 的 0.0093 提升到 0.0447（**4.8 倍**）。

**结论（v2.3）**：v2.2 已修复所有实现偏差。但 52 周 walk-forward 严格评估显示 TSPRank 无净贡献（Sharpe 0.015 vs Backbone 1.19）。根本原因：backbone IC 太弱（0.099）+ sector N 太小（5-25），successor 信噪比极低。

### 3.4 论文实现对照表（v2.2/v2.3 最终状态）

| 论文组件 | 论文原版 | 本项目实现 | 状态 |
|---------|---------|----------|------|
| Bilinear Score (Eq.2) | s = e_i^T W e_j + b | S = (H@W)@H.T + bias | ✅ 已对齐 |
| Encoder | Transformer（可选） | 3层MLP Projector | ✅ 论文说可选 |
| Local Loss (Eq.5) | Row-wise weighted CE | 同论文 | ✅ 一致 |
| Global Loss (Eq.6-7) | structured max-margin + solver-in-loop | SCIP solver-in-loop | ✅ 已对齐 |
| TSP 求解器 | Gurobi/CPLEX 精确 MILP | SCIP 精确（N≤25）+ Or-opt（N>25） | ✅ 已对齐 |
| Sector 分组 (Sec 5.4) | N=5~20/行业 | N=5~25/行业，PIT映射 | ✅ 已对齐 |
| 训练超参 (Appx B) | lr=1e-4, wd=1e-5 | lr=3e-4, wd=1e-4 | ⚠️ 3x/10x |
| Backbone | Feng et al. (IC高) | DE W4 (IC=0.099) | ⚠️ 信号弱 |
| 数据集 | NASDAQ+NYSE 2800+ | CSI300 300只 | ⚠️ 规模不同 |

## 4. 小组仓库源代码的使用

### 4.1 源实验：r2_c06_densemble_alpha158_w4_sim2019_s20260508

本项目基于小组仓库的 **Round2 C06 实验**，其完整配置位于：
`~/quant-stock-forecast/qlib-csi-300/experiments/round2/r2_c06_densemble_alpha158_w4_sim2019_s20260508/config.yaml`

### 3.4 论文实现对照表（v2.2/v2.3 最终状态）

| 论文组件 | 论文原版 | 本项目实现 | 状态 |
|---------|---------|----------|------|
| Bilinear Score (Eq.2) | s = e_i^T W e_j + b | S = (H@W)@H.T + bias | ✅ 已对齐 |
| Encoder | Transformer（可选） | 3层MLP Projector | ✅ 论文说可选 |
| Local Loss (Eq.5) | Row-wise weighted CE | 同论文 | ✅ 一致 |
| Global Loss (Eq.6-7) | structured max-margin + solver-in-loop | SCIP solver-in-loop | ✅ 已对齐 |
| TSP 求解器 | Gurobi/CPLEX 精确 MILP | SCIP 精确（N≤25）+ Or-opt（N>25） | ✅ 已对齐 |
| Sector 分组 (Sec 5.4) | N=5~20/行业 | N=5~25/行业，PIT映射 | ✅ 已对齐 |
| 训练超参 (Appx B) | lr=1e-4, wd=1e-5 | lr=3e-4, wd=1e-4 | ⚠️ 3x/10x |
| Backbone | Feng et al. (IC高) | DE W4 (IC=0.099) | ⚠️ 信号弱 |
| 数据集 | NASDAQ+NYSE 2800+ | CSI300 300只 | ⚠️ 规模不同 |
W4（sim2019）以 Top5 胜率 60.8%、累计超额 +0.520 胜出，被选为小组推荐方案。

**从 config.yaml 移植的内容**：

| 移植项 | config.yaml 原值 | 本项目值 | 变动说明 |
|--------|------------------|----------|----------|
| learning_rate | 0.2 | 0.05 | CSZScoreNorm 后 label 方差≈1，高 lr 导致第 1 轮就最优 |
| epochs/n_estimators | 28 | 200 | 配合降低的 lr |
| num_leaves | 210 | 210 | 不变 |
| max_depth | 8 | 8 | 不变 |
| colsample_bytree | 0.8879 | 0.8879 | 不变 |
| subsample | 0.8789 | 0.8789 | 不变 |
| lambda_l1 | 205.6999 | 205.6999 | 不变 |
| lambda_l2 | 580.9768 | 580.9768 | 不变 |
| seed | 20260508 | 20260508 | 不变 |
| num_models | 3 | 3 | 不变 |
| sample_ratios | [0.8,0.7,0.6,0.5,0.4] | 同左 | 不变 |

### 4.2 数据分割（对齐 config.yaml）

```python
# 直接从 config.yaml 的 segments 段移植
QLIB_CONFIG = {
    "train_start": "2013-01-04",    # config.yaml: segments.train[0]
    "train_end":   "2019-11-04",    # config.yaml: segments.train[1]
    "valid_start": "2019-11-12",    # config.yaml: segments.valid[0]
    "valid_end":   "2021-12-31",    # config.yaml: segments.valid[1]
    "test_start":  "2025-02-21",    # config.yaml: segments.test[0]
    "test_end":    "2026-04-30",    # config.yaml: segments.test[1]
    "eval_start":  "2025-02-24",    # config.yaml: contest_eval_config.start_time
}

# config.yaml 的 label 定义（复权开盘价 5 日收益率）
# 原文：Ref($open * $adjclose / $close, -5) / Ref($open * $adjclose / $close, -1) - 1
# 本项目简化为（使用原始开盘价，与 score_self.py 一致）：
LABEL_EXPR = "Ref($open,-5)/Ref($open,-1)-1"
```

### 4.3 小组 lib 代码库概览

小组仓库的 `lib/` 目录实现了完整的 qlib pipeline，本项目**未直接复用** lib 代码，
而是独立实现了 backbone 训练和回测逻辑。以下是 lib 的关键模块对照：

| lib 模块 | 职责 | 本项目对应 |
|----------|------|-----------|
| `train.py` | 调用 `model.fit(dataset)` 训练 | `backbone.py` 的 `fit()` 方法 |
| `predict.py` | 生成信号 + IC/ICIR + 比赛评估 | 无直接对应（本项目用 score_self.py） |
| `evaluate.py` | 生成 metrics.json + HTML 图表 | `evaluator.py`（简化版，Sharpe/MDD/Calmar） |
| `contest_record.py` | 周度 Top-K vs 基准评估 | `backtest.py`（但无基准对比） |
| `contracts.py` | portfolio 构建、IC 计算、数据验证 | 无直接对应 |
| `evaluation_metrics.py` | 30+ 量化指标 | `evaluator.py`（仅 7 个指标） |

**关键差异**：
- 小组用**复权开盘价** `$open * $adjclose / $close`，本项目用**原始开盘价**（与 score_self.py 一致）
- 小组评估**对比沪深300基准**（beat_rate），本项目对比**绝对收益**（vs 空仓）
- 小组有完整的**数据验证**（segment overlap/label leakage），本项目缺失

### 4.4 W4 的研究背景

核心假设：当前市场（2025-2026）与2019年反弹行情存在状态相似性，
          以2019年前数据训练的模型在当前市场具有更好的泛化能力。

W4 在小组 Round2 评估：Top5胜率 60.8%，累计超额 +0.520，
在6个候选窗口中表现最优，被选为小组当前推荐方案。

> ⚠️ **注意**：W4 选择 2019 年前数据是基于 backbone（DoubleEnsemble）的泛化假设，
> 与 TSPRank 排序层的训练是两个独立问题。
> TSPRank 未收敛的根本原因是实现偏差（Global Loss、bias、sector 分组等），
> 不是 W4 数据窗口的选择。即使换用 2025 年数据训练 TSPRank，
> 如果上述实现偏差不修复，模型仍然不会收敛。

### 4.5 小组 tsprank.py（lib/tsprank.py）与本项目实现的对比

小组仓库 `~/quant-stock-forecast/qlib-csi-300/lib/tsprank.py`（733行）是另一个 TSPRank 实现，
基于 Qlib Model contract。以下逐项对比：

| 维度 | 小组 lib/tsprank.py | 本项目 tsprank_core | 论文原版 |
|------|---------------------|---------------------|----------|
| **Local Loss** | All-pair BCE（所有 i>j 对做二分类） | Row-wise weighted CE（每行只识别真实下一位） | Row-wise weighted CE（Eq.5） |
Global Loss	Structured max-margin + solver-in-loop（Eq.6-7）	✅ v2.2 已对齐
| **TSP 求解器** | SCIP 精确求解 + greedy fallback | 贪心 + Or-opt 启发式 | Gurobi/CPLEX 精确 MILP |
| **方向偏置** | direction_bias: (h_i - h_j)·d | 无 | 无（论文无此项） |
| **bias 项** | 无显式 bias | 无 | 有（Eq.2: b 可学习标量） |
| **输入归一化** | z-score（input_mean/std） | 无 | 未明确说明 |
| **candidate_topn** | 30（组内排序，接近论文 sector 分组） | 随机采样 100 | Sector 分组（5~20 只） |
| **特征维度** | Alpha158 + base_score_z + base_rank_pct（160维） | Alpha158 + rank_pct（159维） | Backbone embedding |
| **集成方式** | Qlib Model contract（标准 pipeline） | 独立 pipeline | — |
| **Encoder** | 2层 MLP（Linear→ReLU→Linear→ReLU） | 3层 MLP + LayerNorm + GELU | Transformer（stock 实验中 optional） |
| **训练方式** | Per-sample SGD，每个 sample 独立 backward | Batch training | — |

**关键结论**：
Global Loss	Structured max-margin + solver-in-loop（Eq.6-7）	✅ v2.2 已对齐
    小组实现了 hinge margin：当 solver 找到的路径得分 > 真实路径得分 + margin 时产生额外惩罚，
    这是论文 structured max-margin loss 的核心思想。
  - 小组的 candidate_topn=30 比本项目的随机采样 100 更接近论文的 sector 分组设置。
  - 小组的 Local Loss（all-pair BCE）不是论文 Eq.5 的 row-wise CE，但至少覆盖了所有 pairwise 关系。
  - 小组有 direction_bias 项（论文没有），增加了非对称性，可能有助于 successor 建模。
  - 两个实现都缺少论文 Eq.2 的 bias 项 b。

> ⚠️ **重要**：小组的实现虽然也不完美（Local Loss 不是 row-wise CE，缺少 bias），
> 但其 Global Loss 的 solver-in-loop 设计是正确的方向。
> 本项目如果要继续在 TSPRank 方向上探索，应优先修复 Global Loss（加 solver-in-loop），
> 而不是在 path likelihood 框架下继续调参。
## 5. 本项目如何结合两者


小组仓库验证的结论:
  DoubleEnsemble W4 在 CSI300 上有效 (IC≈0.10，胜率60.8%)
  Alpha158 是有效的因子集合
         +
TSPRank 论文的方法:
  将排序建模为TSP，用 successor 关系监督，联合优化 local+global 目标
         ↓
本项目的结合方式（combined 模式）:
  1. 用 W4 相同超参和数据窗口训练 LightGBM backbone
  2. backbone 预测得分做截面 rank 百分位（1维，[-1,1]）
  3. 拼接到 Alpha158 特征（158维）→ 159维输入
  4. TSPRank 在 159维输入上学习排序
  5. TSP solver 输出全局近优排列，取 Top-5
为什么用 combined 而不是 leaf embedding？

旧文档使用 LightGBM 叶节点 index（600维）作为 TSPRank 输入，存在两个问题：

leaf index 是整数离散量，不同叶节点编号之间没有连续几何语义
叶节点编号经归一化后信息密度低，MLP 难以从中提取有效信息
v2.0 已删除 leaf 模式，只保留 feature（Alpha158，158维）和 combined（Alpha158 + backbone rank，159维）。 combined 模式通过在 Projector 输入端注入 backbone 信号， 而不是在 Bilinear 层内部加 pointwise 偏置项（后者会破坏 successor 语义的纯粹性）。

## 6. 系统架构全览


┌─────────────────────────────────────────────────────────┐
│                    数据层                                │
│  data/qlib_bin/ → Alpha158因子(158维) → train/valid切分   │
│  (覆盖2013-2026，parquet缓存，290个测试截面)              │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            DoubleEnsemble Backbone (W4超参)              │
│  3个 LightGBM 子模型 (SR样本重加权 + FS特征选择)          │
│  输出: pred_score(N,)                                    │
│  当前状态：v2.2 重新训练，IC=0.0987，正IC率80%                    │
└──────────────────────────┬──────────────────────────────┘
                           │ combined模式: Alpha158(158) + rank_pct(1) = 159维
┌──────────────────────────▼──────────────────────────────┐
│                  TSPRank 排序层（v2.2）                   │
│  EmbeddingProjector: 159 → 256 → 128 → 64               │
│  BilinearTSPLayer:   S[i,j] = h_i^T W h_j + b           │
│  Local Loss: row-wise weighted CE（论文 Eq.5）           │
│  Global Loss: structured max-margin + SCIP solver-in-loop│
│  训练: sector 分组（N=5~25/组），SCIP 精确求解            │
│  推理: sector 内 SCIP 排名 + backbone 全局混合            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    选股与评分                            │
│  Top-5 stocks, 等权 0.2 → output/result.csv             │
│  score_self.py → temp/tmp.csv（官方评分接口）             │
└─────────────────────────────────────────────────────────┘
## 7. 各模块详细说明

### tsprank_core/data_loader.py

职责：数据加载（v2.2）

  ① Alpha158 加载 + parquet 缓存 + 截面提取
  ② _normalize_code：统一 SH600000 / sh.600000 / 600000.SH 等格式为 6 位数字
  ③ load_industry_mapping()：静态行业映射（最新快照，推理/回测用）
  ④ load_industry_df()：读取全历史行业快照（供 PIT 用）
  ⑤ build_sector_dict()：逐日行业映射
     - 传 DataFrame → PIT 版（按预测日回溯最近快照）
     - 传 Dict → 静态版（向后兼容）
  ⑥ make_result_csv：支持 top_k 截断，返回 DataFrame
### tsprank_core/backbone.py


职责：复现 DoubleEnsemble 核心机制

关键修正（v2.0）：
  ① 不使用 early_stopping
     原因：CSZScoreNorm 后 label 方差≈1，valid L2≈1.0，
     early_stopping 会在第1轮触发，只输出1棵树
  ② n_trees_per_model 直接赋值 N_ESTIMATORS，不信 best_iteration
  ③ load() 兼容旧格式 pkl
     旧格式缺失 feat_weights_list，用单位权重补全（IC从0.10退化到0.06）
  ④ 不再提供 predict_leaf（leaf 模式已废弃）
### tsprank_core/tsprank_model.py

职责：TSPRank 核心模型（v2.2）

  ① BilinearTSPLayer：s(e_i, e_j) = e_i^T W e_j + b（含 bias 项，论文 Eq.2）
  ② TSPSolver：贪心最近邻 + Or-opt 启发式，solve() 内置 cost 矩阵归一化
  ③ SCIPSolver：基于 pyscipopt 的精确 MILP 求解器（MTZ 公式）
     MAX_EXACT_N=25（训练），MAX_INFER_N=50（推理），超限降级为 Or-opt
  ④ Global Loss：structured max-margin + solver-in-loop（论文 Eq.6-7）
     增广矩阵 S_aug = S + (1 - A_t)，SCIP 找最违反约束路径，straight-through 梯度
  ⑤ Local Loss：row-wise weighted CE（论文 Eq.5）
  ⑥ "both" 模式：奇数步 local+global，偶数步 local-only（论文 Section 4.2 交替训练）
  ⑦ 编码器为 MLP Projector（论文 stock ranking 实验中 encoder 为 optional）

### tsprank_core/tsprank_trainer.py

职责：TSPRank 训练流程管理（v2.2）

  ① _sample_stocks()：分层采样（替代纯随机），保证覆盖收益率全范围
  ② _one_day_sector()：按行业分组计算 loss，N<5 跳过，N>25 用 backbone 降采样
  ③ train_sector()：sector 分组完整训练循环，接受 PIT sector_dict（按日期查行业）
  ④ train() / train_rolling() / train_walk_forward()：非 sector 模式训练接口
  ⑤ _reset_model()：从存储参数重建模型（不再硬编码超参）
### tsprank_core/backtest.py

职责：周度回测引擎（v2.2）

  ① _strip_prefix：剥离 SH/SZ 前缀，对齐 score_self.py 的纯数字格式
  ② _score_inline：手动 groupby 计算周收益率
  ③ 多样性惩罚：后处理规则，报告中单独标注
  ④ 集成 StrategyEvaluator：自动输出 Sharpe/MDD/Calmar/净值曲线
  ⑤ 四种推理模式：
     - backbone：直接 Top-K（推荐）
     - qlib_pred：接入小组 pred.pkl
     - tsprank：全截面 TSP 路径
     - tsprank_sector（v2.2）：行业内 SCIP 精确排名 + backbone 全局混合
### tsprank_core/evaluator.py


职责：完整量化评估指标模块（P3）

对齐小组仓库 lib/evaluation_metrics.py 的指标体系：
  ① Sharpe Ratio（年化，无风险利率=0）
  ② 最大回撤 (MDD) + 回撤区间标注
  ③ Calmar Ratio（年化收益 / MDD）
  ④ 胜率（周收益 > 0 的比例）
  ⑤ 平均盈利周 / 平均亏损周
  ⑥ 净值曲线图（PNG 输出）
  ⑦ 与沪深300基准对比（若提供基准收益率）
### tsprank_core/qlib_pred_loader.py


职责：接入小组 pred.pkl（P1）

从 mlflow artifacts 中读取 DoubleEnsemble W4 的完整预测分数：
  ① 自动搜索 mlruns 目录下的 pred.pkl
  ② 标准化 MultiIndex(datetime, instrument) 格式
  ③ 剥离 SH/SZ 前缀，对齐本项目股票代码格式
## 8. 实验结果与诚实评估

### 8.1 Backbone 质量

IC=0.0987, IR=0.91, 正IC率 80%（20天）。子模型 IC 分别为 0.045, -0.034, 0.032，集成后显著提升。

### 8.2 TSPRank 实验结论（52 周严格评估）

| 方案 | 累计收益 | Sharpe | MDD | 胜率 | 备注 |
|------|---------|--------|-----|------|------|
| Backbone baseline | +22.84% | 1.19 | 10.00% | 59.6% | 生产方案 |
| Rolling blend=0.5 | +90.63% | 4.72 | 4.28% | 73.1% | ⚠️ 数据泄漏 |
| **Walk-forward blend=0.5** | **+1.47%** | **0.015** | **10.94%** | **46.2%** | **无泄漏，结论性** |

**结论**：Rolling 模式的 Sharpe 4.72 完全是数据泄漏造成的假象。Walk-forward（无泄漏）下 TSPRank Sharpe 0.015，基本是随机水平。TSPRank 在当前条件下（backbone IC=0.099, sector N=5-25）无净贡献。

根本原因：backbone IC 太弱（0.099）+ sector 分组后 N 太小（5-25），successor 关系信噪比极低，神经网络学不到有意义的排序。
## 9. 已知问题清单

**已修复（v2.2）**：Global Loss 偏差、bias 项缺失、cost 归一化、sector 分组、采样分布偏移、SCIP 求解器。

### 问题1：Backbone 超参不完全对齐 W4 ⚠️

lr=0.05（原 0.2）、n_estimators=200（原 28）。原因：CSZScoreNorm 归一化后高 lr 导致 early_stopping 在第 1 轮触发。
修复方向：接入小组 pred.pkl（IC≥0.10）绕过此问题。

### 问题2：TSPRank 无净贡献 ✅ 已确认（v2.3 walk-forward）

52 周 walk-forward 评估：Sharpe 0.015（vs Backbone 1.19）。根本原因：backbone IC 太弱 + sector N 太小。
## 10. 目录结构

~/tsprank/
│
├── main.py                         # 统一入口（支持 --train_mode standard/rolling/walk_forward）
├── score_self.py                   # ⚠️ 比赛评分接口（固定，不可修改）
├── 论文原文.pdf                     # TSPRank 论文原文 arXiv:2411.12064v2（24页）
├── README.md                       # 简洁版快速上手（新）
├── README_detailed.md              # 本文档（完整技术文档）
├── requirements.txt                # Python 依赖列表
├── .gitignore                      # Git 忽略规则
│
├── tsprank_core/                   # 核心代码库
│   ├── __init__.py
│   ├── data_loader.py              # 数据加载：Alpha158/parquet缓存/截面提取/价格数据
│   ├── backbone.py                 # DoubleEnsemble 骨干网络（W4超参，旧pkl兼容）
│   ├── tsprank_model.py            # TSPRank 核心：BilinearTSPLayer/TSPSolver/Local&Global Loss
│   ├── tsprank_trainer.py          # TSPRank 训练器：standard/rolling/walk_forward 三种模式
│   ├── backtest.py                 # 周度回测引擎：三种推理模式/多样性惩罚/归因标注
│   ├── evaluator.py                # 完整量化指标：Sharpe/MDD/Calmar/净值曲线
│   └── qlib_pred_loader.py         # P1：接入小组 pred.pkl（IC≈0.10）
│   └── qlib_pred_loader.py         # P1: 接入小组 pred.pkl（IC≈0.10）
│
├── fix/                            # ⚠️ 历史修复存档（已合并到 tsprank_core/，可删除）
│
├── data/                           # 所有数据文件
│   ├── stock_data.csv              # 原始价格数据（300只股票，2020-2026）
│   ├── industry.csv                # 行业分类（baostock，26期快照，300只/42行业）
│   ├── fetch_industry.py           # 行业数据采集脚本（断点续传）
│   ├── test.csv                    # 当周价格数据（backtest时动态生成）
│   └── qlib_bin/                   # qlib 二进制数据（calendars/features/instruments）
│
├── checkpoints/                    # 模型权重
│   ├── backbone.pkl                # v2.2 重新训练，IC=0.0987
│   └── tsprank.pt                  # v2.2 训练中，val loss 未改善
│
├── cache/                          # parquet 缓存（首次加载 qlib 后自动生成）
│   ├── alpha158_dataset_train.parquet  # 1661个训练截面
│   ├── alpha158_dataset_valid.parquet  # 522个验证截面
│   └── alpha158_dataset_test.parquet   # 290个测试截面
│
├── output/                         # 输出文件
│   ├── result.csv                  # 选股结果（score_self.py读取）
│   ├── backtest_results.csv        # 回测详细记录
│   └── backtest_results_equity_curve.png  # 净值曲线图
│
├── temp/
│   └── tmp.csv                     # 评分结果（score_self.py输出）
│
└── logs/                           # 历史运行日志（归档）
    ├── run_v2.log                  # v2.0 早期运行记录
    ├── run_v3.log
    ├── run_v4.log
    └── run_v5.log
## 11. 环境依赖

Python   3.11  (conda env: bigdata)
torch    2.8.0
lightgbm 4.6.0
numpy    2.0.2
pandas   2.3.3
pyqlib   0.9.7   (Alpha158特征加载)
pyarrow  20.0.0  (parquet缓存)
scipy    1.13.1  (Spearman相关系数)
matplotlib 3.7+  (净值曲线绘图)
pyscipopt 6.2.1  (SCIP MILP solver，sector 分组训练用)

完整依赖列表见 requirements.txt
## 12. 快速开始
Bash

conda activate bigdata
cd ~/tsprank

# ── 验证数据层（30秒）─────────────────────────────────
python -c "
from tsprank_core.data_loader import load_stock_data, get_eval_weeks
df = load_stock_data()
print('stocks:', df['股票代码'].nunique(), 'dates:', df['日期'].nunique())
weeks = get_eval_weeks(stock_df=df, recent_weeks=12)
print('eval weeks:', len(weeks))
"

# ── Backbone baseline（生产方案）──────────────────
python main.py --mode backtest --use_cache --embed_mode combined --infer_mode backbone

# ── 完整 pipeline（standard 训练）──────────────────
python main.py --mode full \
               --use_cache \
               --embed_mode combined \
               --loss_mode local \
               --train_mode standard \
               --max_epochs 100 \
               --patience 25

# ── TSPRank 滚动窗口训练 ────────────────────────
python main.py --mode train_tsprank --use_cache \
               --embed_mode combined \
               --loss_mode local \
               --train_mode rolling

# ── v2.2：sector 分组训练 + SCIP + 回测 ─────────────
python main.py --mode full \
               --sector_mode \
               --loss_mode both \
               --train_mode rolling \
               --infer_mode tsprank_sector \
               --embed_mode combined \
               --max_sector_n 25 \
               --sector_blend 0.5 \
               --use_cache

# ── Walk-forward 严格无泄露 ─────────────────────
python main.py --mode train_tsprank --use_cache \
               --embed_mode combined \
               --loss_mode local \
               --train_mode walk_forward

# ── 仅预测最新周 ──────────────────────────────────────
python main.py --mode predict --use_cache --embed_mode combined

# ── 官方评分 ──────────────────────────────────────────
python score_self.py && cat temp/tmp.csv

# ── 诊断 backbone IC ──────────────────────────────────
python -c "
import numpy as np
from scipy.stats import spearmanr
from tsprank_core.data_loader import (
    load_alpha158_dataset, extract_cross_sections, TSPRANK_ROOT
)
from tsprank_core.backbone import DEBackbone
_, _, test_df = load_alpha158_dataset(
    cache_path=str(TSPRANK_ROOT/'cache'/'alpha158_dataset'), use_cache=True
)
_, X, y, _ = extract_cross_sections(test_df)
bb = DEBackbone().load('checkpoints/backbone.pkl')
dates = sorted(X.keys())
ics = [spearmanr(bb.predict_score(X[d]), y[d])[0] for d in dates[:50]]
ics = [x for x in ics if not np.isnan(x)]
print(f'IC={np.mean(ics):.4f}  IR={np.mean(ics)/np.std(ics):.4f}')
"
## 13. 参数说明
参数	默认值	可选值	说明
--mode	full	full/train_backbone/train_tsprank/backtest/predict	运行模式
--use_cache	False	—	使用 parquet 缓存
--embed_mode	combined	feature/combined	feature=Alpha158(158维), combined=Alpha158+rank(159维)
--loss_mode	local	local/global/both	local=row-wise CE (Eq.5), global=structured max-margin (Eq.6-7), both=交替训练
--train_mode	standard	standard/rolling/walk_forward	训练模式: standard=2013-2019, rolling=P2近期数据, walk_forward=严格无泄露
--emb_dim	64	32/64/128	TSPRank 内部 embedding 维度
--max_epochs	100	—	TSPRank 最大训练轮数
--patience	25	—	早停 patience
--max_stocks	100	—	每日采样股票数（N²OOM工程妥协）
--top_k	5	—	选股数量
--total_weight	1.0	—	总权重
--recent_weeks	12	—	回测最近 N 周
--diversity_penalty	0.0	—	多样性惩罚（⚠️后处理规则，非TSPRank组成部分）
--penalty_window	4	—	惩罚回溯周数
--no_or_opt	False	—	禁用 Or-opt（N>150自动禁用）
--infer_mode	tsprank	backbone/qlib_pred/tsprank/tsprank_sector	推理模式
--qlib_pred_pkl	None	—	P1模式下pred.pkl路径（不指定则自动搜索mlruns）
--no_equity_curve	False	—	不保存净值曲线图
--sector_mode	False	—	启用行业分组训练+推理（论文 Section 5 设置）
--industry_csv	None	—	行业分类 CSV 路径（默认 data/industry.csv）
--max_sector_n	25	—	训练时每行业最大股票数（超限用 backbone 降采样）
--sector_blend	0.5	0~1	推理时 sector_rank 权重（0=纯 backbone，1=纯 sector）
--no_scip	False	—	禁用 SCIP，改用 Or-opt（调试用）
## 14. 后续计划

TSPRank 实验已结束。结论：在 backbone IC=0.099 + sector N=5-25 条件下，TSPRank 无净贡献。

### P0（立即）：接入小组 pred.pkl

把 backbone IC 从 0.099 提到 0.10+，再测一次纯 backbone 的 52 周表现。成本低，可能有增量。
```bash
python main.py --mode backtest --use_cache --embed_mode combined --infer_mode qlib_pred --recent_weeks 52
```

### P1（如时间充裕）：LambdaMART

LightGBM 原生支持 lambdarank objective，Alpha158 直接输入，不需要 SCIP 和神经网络。
小 N 场景下 GBDT 比神经网络稳健得多。论文中 LambdaMART 是最强 baseline。

### 已完成事项

- v2.2 Global Loss 对齐论文 Eq.6-7（SCIP solver-in-loop）
- v2.2 bias 项、sector 分组、PIT 行业映射
- v2.3 52 周 walk-forward 严格评估（TSPRank 结论性实验）
- v2.3 诊断文件与代码审计
## 15. 参考文献

[1] TSPRank: Bridging Pairwise and Listwise Methods with a Bilinear
    Travelling Salesman Model
    Weilong Li, Yanan Cao et al., KDD 2025
    arXiv: https://arxiv.org/abs/2411.12064
    官方代码: https://github.com/waylonli/TSPRank-KDD2025

[2] DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting
    and Feature Selection for Financial Data Analysis
    Chuheng Zhang, Yuqing Yang et al., ICDM 2020
    arXiv: https://arxiv.org/abs/2010.01265

[3] Qlib: An AI-oriented Quantitative Investment Platform
    Microsoft Research Asia
    https://github.com/microsoft/qlib

[4] 小组实验配置（W4超参来源）:
    r2_c06_densemble_alpha158_w4_sim2019_s20260508
    ~/quant-stock-forecast/qlib-csi-300/experiments/round2/
    config.yaml: DoubleEnsemble, Alpha158, train=2013-2019, valid=2019-2021, test=2025-2026

[5] Alpha158 因子文档:
    https://qlib.readthedocs.io/en/latest/advanced/alpha158.html

[6] 论文原文逐项核查记录（2026-05-31）:
    基于 arXiv:2411.12064v2（24页）逐项对比
    发现关键脱节：Global Loss 根本性偏差、缺少 bias 项、backtest cost 矩阵未归一化
    详见第3.4节对照表和第9节问题清单
