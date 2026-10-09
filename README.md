# TSPRank 股票选股系统

> 2026 THU 大数据挑战赛 · 个人实验工作区
> CSI300 每周 Top-5 选股：DoubleEnsemble W4 骨干 + TSPRank 排序层（探索已证伪）
> 当前状态：v2.3

论文逐节精读、公式推导、实现对照表、完整实验记录见 [README_detailed.md](./README_detailed.md)。

---

## 1. 项目在做什么

每周从沪深 300 成分股中选**至多 5 支**股票并分配权重，提交 `output/result.csv`。

- 计分公式：`得分 = Σ(股票周收益率 × 权重)`，权重和 ∈ [0, 1]
- 周收益率 = `(当周最后交易日开盘价 − 第一交易日开盘价) / 第一交易日开盘价`
- 评分接口 [score_self.py](./score_self.py) 固定，不可修改
- 评估期：2025-02-24 ~ 2026-04-30，仅计完整交易周

整体思路：先复现小组验证过的 **DoubleEnsemble W4 骨干**（给股票打预测分），再叠加 **TSPRank 排序层**（KDD 2025 论文方法），探索排序学习能否进一步提升选股质量。

## 2. 概念解读

### 2.1 TSP 是什么，和选股有什么关系

**TSP（旅行商问题）**：一个商人要依次访问 N 个城市各一次，找一条总路程最短的路线。

TSPRank 的核心洞见：**给 N 只股票排名，等价于在 N 个"城市"之间找一条最优路径。**

- 每只股票 = 一个城市
- 边 `i → j` 的权重 = "股票 j 紧跟在股票 i 后面"的得分（successor 关系，而非"i 和 j 谁更好"）
- 最优排名 = 总得分最高的**开环路径**（走完所有节点，不要求回到起点）

于是"排序"问题转化成了可以用成熟 TSP 求解器求解的组合优化问题。

### 2.2 TSPRank 的四个组件

| 组件 | 做法 | 作用 |
|------|------|------|
| ① 双线性打分 | `s(eᵢ,eⱼ) = eᵢᵀ·W·eⱼ + b` | 用股票 embedding 算任意两股间的 successor 得分（W、b 可学习） |
| ② Local Loss | Row-wise 加权交叉熵（Eq.5） | 对每只股票，从 N 个候选中识别"真实的下一位"，按排名位置加权 |
| ③ Global Loss | Structured max-margin + solver-in-loop（Eq.6-7） | 用 TSP 求解器找出"最违反约束"的路径并惩罚 —— 把全局排列结构写进 loss |
| ④ 推理 | MILP 精确求解 TSP（Eq.10-15） | 直接解出近优排列，取 Top-K |

论文效果（Table 1，NASDAQ）：Kendall Tau 从 backbone 的 0.0093 提升到 0.0447（**4.8 倍**）。

### 2.3 为什么需要 TSPRank：排序学习的两大流派

| 范式 | 代表方法 | 优点 | 缺点 |
|------|---------|------|------|
| Pairwise | RankNet / **LambdaMART** | 两两比较，训练信号丰富 | 忽略全局排列结构 |
| Listwise | ListNet / Rankformer | 考虑整体排列 | 计算复杂、对规模敏感 |

TSPRank 试图兼得两者：② 提供 pairwise 式的局部信号，③ 引入 listwise 式的全局约束。

### 2.4 骨干模型：DoubleEnsemble W4

3×LightGBM 集成。训练时做 **SR 样本重加权**（按预测残差动态调整样本权重）+ **FS 特征选择**（筛选有效特征）。输入为 Alpha158（158 维量价因子）。本仓库复现版 IC=0.0987 / IR=0.91 / 正 IC 率 80%。W4 = 训练截止 2019-11 的窗口方案，来自小组 Round2 的推荐配置。

## 3. 系统架构（本项目）

```
data/qlib_bin → Alpha158 因子(158维) → train/valid/test 切分
       │
       ▼
DoubleEnsemble 骨干（W4 超参，3×LightGBM）→ 预测得分 + 截面 rank 百分位
       │
       ▼  combined 模式：Alpha158(158) + backbone_rank(1) = 159 维
TSPRank 排序层
  Projector：159 → 256 → 128 → 64（替代论文的 Transformer encoder）
  Bilinear TSP：S[i,j] = hᵢᵀ·W·hⱼ + b
  Local Loss（row-wise weighted CE）+ Global Loss（SCIP solver-in-loop）
  推理：行业内 SCIP 精确排名 + backbone 全局排名混合（sector_blend）
       │
       ▼
Top-5 等权 0.2 → output/result.csv → score_self.py 评分
```

## 4. 代码的创新尝试（相对论文原版）

| # | 尝试 | 说明 | 结果 |
|---|------|------|------|
| 1 | **combined embedding** | 把 backbone 预测分的截面 rank 拼接到 Alpha158 之后（158→159 维），在 Projector 输入端注入排序先验；而非在 Bilinear 层内加 pointwise 偏置（会破坏 successor 语义的纯粹性） | 已实现 |
| 2 | **SCIP 免费精确 MILP** | 论文用商业求解器 Gurobi/CPLEX，本项目改用开源 pyscipopt（MTZ 公式）。训练 N≤25 / 推理 N≤50 精确求解，超限自动降级 Or-opt | 已实现 |
| 3 | **Or-opt 启发式** | 有向非对称 TSP 中，标准 2-opt 反转子路径会改变内部边方向、生成无效路径；改用节点重插入（Or-opt）天然适配有向图 | 已实现 |
| 4 | **PIT 行业映射** | sector 分组用 point-in-time 行业快照（按预测日回溯最近快照），避免用未来行业信息造成前视偏差 | 已实现 |
| 5 | **sector 分组训练 + 推理混合** | 对齐论文 Section 5.4 设置：组内 SCIP 精确排名，推理时与 backbone 全局排名按 `sector_blend` 加权混合 | 已实现 |
| 6 | **walk-forward 严格评估** | 逐周只用"当时可得数据"训练+推理，抓出了 rolling 模式高收益纯属数据泄漏（Sharpe 4.72 → 0.015） | 已实现，结论性 |
| 7 | **v2.2 实现级修复** | 补齐论文 bias 项、Global Loss solver-in-loop（Eq.6-7）、cost 矩阵归一化、分层采样（覆盖收益率全范围） | 已对齐论文 |

## 5. 实验结论与失败原因

| 方案 | 52周累计 | Sharpe | MDD | 胜率 | 结论 |
|------|---------|--------|-----|------|------|
| **Backbone baseline** | **+22.84%** | **1.19** | 10.0% | 59.6% | ✅ 生产方案 |
| Rolling blend=0.5 | +90.63% | 4.72 | 4.3% | 73.1% | ⚠️ 数据泄漏造成的假象 |
| Walk-forward blend=0.5 | +1.47% | 0.015 | 10.9% | 46.2% | ❌ 严格无泄漏，TSPRank 结论性实验 |

**失败原因分析**（按重要性排序）：

1. **骨干信号太弱（根因）**：本项目 backbone IC≈0.099；论文用的是 Feng et al. relational embedding（IC 高得多）。successor 是"谁紧接在谁后面"的二阶关系，建立在噪声较大的预测分上时信噪比极低。
2. **分组后样本太小**：论文按行业分组（N=5~20，平均 9~11 只/组）；本项目 sector N=5~25，每天每组只有个位数条有效 successor 监督信号，神经网络学不到稳定模式。
3. **一次重要教训**：早期把不收敛归因于"A股市场变化大 / 分布漂移"，属误判；实际是实现偏差（Global Loss 等）。v2.2 修复全部偏差后严格评估，确认方法在当前条件下无增量。**先修评估框架（防泄漏），再谈模型调优。**
4. 次要问题：复现的 backbone 超参未完全对齐 W4（lr=0.05 vs 0.2、n_estimators=200 vs 28），IC 有提升空间但受限。

> 论文能成功而本项目不行：论文是 N<30 小规模 + 强 backbone + 2800 只股票（120 个行业组）；本项目是 300 只 CSI300 + 弱 backbone。

## 6. 下一步方向

- **P0（立即）：接入小组 pred.pkl**，把 backbone IC 从 0.099 提到 0.10+，重测 52 周基线：

  ```bash
  python main.py --mode backtest --use_cache --embed_mode combined --infer_mode qlib_pred --recent_weeks 52
  ```

- **P1（如时间充裕）：LambdaMART** —— LightGBM 原生 lambdarank objective，Alpha158 直接输入，不需要 SCIP 和神经网络；小 N 场景下 GBDT 比神经网络稳健，且是论文中最强 baseline。
- **若重启 TSPRank 方向**：前提是骨干 IC 显著提升（≥0.12+），否则 successor 信噪比不足以支撑排序学习。

## 7. 快速开始

```bash
conda activate bigdata
cd ~/tsprank

# ① Backbone 回测（52 周，生产方案）
python main.py --mode backtest --use_cache --embed_mode combined --infer_mode backbone --recent_weeks 52

# ② 预测最新一周 → output/result.csv
python main.py --mode predict --use_cache --embed_mode combined

# ③ 官方评分
python score_self.py; cat temp/tmp.csv
```

> 首次运行需先不带宽 `--use_cache` 跑一次生成 parquet 缓存；之后均可加 `--use_cache` 加速。

- 运行模式（`--mode`）：`full` / `train_backbone` / `train_tsprank` / `backtest` / `predict`
- 推理模式（`--infer_mode`）：`backbone`（推荐）/ `qlib_pred`（P0）/ `tsprank` / `tsprank_sector`（后两者已证伪）
- 常用参数：`--use_cache` / `--embed_mode combined|feature` / `--recent_weeks N` / `--top_k 5`。完整参数表见 README_detailed.md 第 13 节

## 8. 目录结构与核心模块

```
tsprank/
├── main.py               # 统一入口（5 种 mode）
├── score_self.py         # 比赛评分接口（固定，不可修改）
├── tsprank_core/         # 核心代码库
│   ├── data_loader.py    # Alpha158 / parquet 缓存 / 截面提取 / PIT 行业映射
│   ├── backbone.py       # DoubleEnsemble 骨干（W4 超参）
│   ├── tsprank_model.py  # Bilinear 打分 / SCIP+Or-opt 求解 / Local&Global Loss
│   ├── tsprank_trainer.py# standard / rolling / walk_forward / sector 训练
│   ├── backtest.py       # 周度回测引擎（4 种推理模式）
│   ├── evaluator.py      # Sharpe / MDD / Calmar / 胜率 / 净值曲线
│   └── qlib_pred_loader.py  # 接入小组 pred.pkl（P0）
├── data/                 # stock_data.csv、industry.csv、qlib_bin/
├── cache/                # Alpha158 parquet 缓存（自动生成）
├── checkpoints/          # backbone.pkl（IC=0.0987）、tsprank.pt
├── output/               # result.csv（提交文件）、回测记录、净值曲线
├── temp/tmp.csv          # score_self.py 评分输出
├── logs/                 # 运行日志、诊断.txt（完整实验记录）
├── README.md             # 本文档（简版）
├── README_detailed.md    # 完整技术文档
├── 论文原文.pdf           # TSPRank (KDD 2025, arXiv:2411.12064) 原文
└── requirements.txt
```

## 9. 环境

Python 3.11（conda env: `bigdata`）+ torch 2.8.0 / lightgbm 4.6.0 / pyqlib 0.9.7 /
pyarrow 20.0.0 / scipy 1.13.1 / matplotlib / pyscipopt 6.2.1。完整依赖见 `requirements.txt`。

数据分割（对齐小组 config.yaml）：train 2013-01 ~ 2019-11 / valid 2019-11 ~ 2021-12 / test 2025-02 ~ 2026-04。