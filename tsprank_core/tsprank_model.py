# tsprank_core/tsprank_model.py
"""
TSPRank 核心模型 v2.1

修复内容（vs v2.0）：
  [Fix 0.6] BilinearTSPLayer：添加可学习标量偏置 b（论文 Eq.2）
             s(e_i,e_j) = e_i^T W e_j + b
             原因：缺少 b 时，当 embedding 正交则所有 score 被迫过原点，
             模型表达力受限。b 初始化为 0，不影响训练初期行为。

  [Fix 0]   _global_loss：从 path likelihood 近似改为论文 Eq.6-7 的
             Structured Max-Margin Loss with Solver-in-Loop
             核心算法：
               1. S_aug[i,j] = S[i,j] + (1 - A^t[i,j])
                  真实路径边 +0，其余边 +1（margin 惩罚项）
               2. x_worst = argmax_{Hamiltonian path} S_aug
                  用现有 TSPSolver（greedy + Or-opt）近似求解
               3. loss = max(0, x_worst·S_aug - x_true·S)
                       = max(0, Δ(x_worst,x_true) + x_worst·S - x_true·S)
             梯度：solver 输出固定（直通估计），仅通过 S 的高级索引传播：
               ∂loss/∂S[i,j] = +1 for (i,j) in worst_tour (降低错误边)
                               = -1 for (i,j) in true_path  (提升正确边)
                               =  0 otherwise

  [Fix]     compute_loss("both" 模式)：改为逐步交替 local / (local+global)
             对齐论文 Section 4.2："alternate batches between the global loss
             and the local loss, ensuring faster and more efficient convergence"
             增加 force_local 参数，供 trainer 验证阶段强制使用 local loss
             （验证不需要调用 solver，节省开销）

  [Fix]     TSPRankModel 存储构造参数（in_dim/emb_dim/hidden_dims/dropout_p）
             供 trainer._reset_model() 正确重建模型

  [Note 0.5] 问题0.5（backtest cost矩阵未归一化）已在 TSPSolver.solve() 内置修复，
             _infer_tsprank 与 predict_rank 均通过 TSPSolver.solve() 路径，
             行为一致，backtest.py 无需改动。

v2.0 保留内容：
  ① 纯双线性 S = (H@W)@H.T + b，删除 L2 归一化
  ② forward 返回 (S, A)，loss 统一用 S
  ③ _local_loss：row-wise weighted CE（对齐论文 Eq.5）
  ④ Or-opt 替换 2-opt（天然适用有向非对称 TSP）
  ⑤ DEVICE 导出到模块级
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 模块级常量（可被外部导入）─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingProjector
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingProjector(nn.Module):
    """
    MLP Projector：in_dim → hidden_dims → emb_dim
    末层 LayerNorm 控制 H 量级，替代 L2 归一化。
    """

    def __init__(
        self,
        in_dim      : int,
        emb_dim     : int   = 64,
        hidden_dims : tuple = (256, 128),
        dropout     : float = 0.1,
    ):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [
            nn.Linear(prev, emb_dim),
            nn.LayerNorm(emb_dim),   # 末层 LayerNorm：控制 H 量级，防止 S 爆炸
        ]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (N, emb_dim)


# ─────────────────────────────────────────────────────────────────────────────
# BilinearTSPLayer
# ─────────────────────────────────────────────────────────────────────────────

class BilinearTSPLayer(nn.Module):
    """
    论文 Eq.2：s(e_i, e_j) = e_i^T · W · e_j + b

    [Fix 0.6] 增加可学习标量偏置 b（self.bias）
    - 初始化为 0，不影响训练初期行为（W 正交初始化，S ≈ 0）
    - 作用：让所有 score 不必过原点，提升模型表达力
    - 论文原文：b 是全局标量，不是 per-entity bias

    设计原则（保留自 v2.0）：
    - 不做 L2 归一化（Projector 末层 LayerNorm 已控制量级）
    - 不加 u·h_i / v·h_j（pointwise 信息从输入端注入）
    - W 正交初始化 → 训练初期 S ≈ b ≈ 0，A ≈ sigmoid(0) = 0.5
    """

    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W    = nn.Parameter(torch.empty(emb_dim, emb_dim))
        # [Fix 0.6] 论文 Eq.2 的可学习标量偏置 b
        self.bias = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(p=dropout)
        nn.init.orthogonal_(self.W)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H : (N, emb_dim)
        返回 S : (N, N) 原始双线性得分（无界）
        s(e_i, e_j) = h_i^T W h_j + b
        """
        W_eff = self.drop(self.W) if self.training else self.W
        # [Fix 0.6] + self.bias（标量广播到 N×N）
        return (H @ W_eff) @ H.T + self.bias


# ─────────────────────────────────────────────────────────────────────────────
# TSPSolver（Or-opt，天然适用有向非对称图）
# ─────────────────────────────────────────────────────────────────────────────

class TSPSolver:
    """
    轻量级有向 TSP 求解器：贪心最近邻 + Or-opt 局部搜索。

    Or-opt vs 2-opt：
    - 2-opt 反转子路径 → 在有向图中改变内部边方向 → 无效
    - Or-opt 节点重插入 → 天然保持方向性 → 适用有向非对称 TSP

    在 Global Loss 的 solver-in-loop 中，此求解器被用于寻找"最违反约束"路径。
    虽然非精确（论文用 Gurobi MILP），但方向正确：任何比真实路径得分更高的
    替代路径都能产生有效的 max-margin 梯度信号。
    """

    @staticmethod
    def greedy_tour(cost: np.ndarray) -> np.ndarray:
        n       = len(cost)
        row_sum = np.where(np.isinf(cost), 0.0, cost).sum(axis=1)
        start   = int(np.argmin(row_sum))
        visited = np.zeros(n, dtype=bool)
        tour    = [start]
        visited[start] = True
        for _ in range(n - 1):
            c = cost[tour[-1]].copy()
            c[visited] = np.inf
            nxt = int(np.argmin(c))
            tour.append(nxt)
            visited[nxt] = True
        return np.array(tour, dtype=np.int64)

    @staticmethod
    def _path_cost(tour: np.ndarray, cost: np.ndarray) -> float:
        return float(sum(cost[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    @classmethod
    def or_opt(
        cls,
        tour    : np.ndarray,
        cost    : np.ndarray,
        max_iter: int = 300,
    ) -> np.ndarray:
        """
        单节点重插入（first-improvement）。
        每轮找到第一个改进就接受并重新扫描。
        """
        best      = tour.copy()
        best_cost = cls._path_cost(best, cost)

        for _ in range(max_iter):
            improved = False
            n        = len(best)
            for k in range(n):
                node      = best[k]
                remaining = np.concatenate([best[:k], best[k + 1:]])

                for pos in range(len(remaining) + 1):
                    new_tour = np.concatenate([
                        remaining[:pos], [node], remaining[pos:]
                    ])
                    new_cost = cls._path_cost(new_tour, cost)
                    if new_cost < best_cost - 1e-10:
                        best      = new_tour
                        best_cost = new_cost
                        improved  = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    @classmethod
    def solve(
        cls,
        S         : np.ndarray,   # (N, N) 原始双线性得分（非 sigmoid 输出）
        use_or_opt: bool = True,
        max_n     : int  = 150,
    ) -> np.ndarray:
        """
        S 矩阵 → 排名路径索引。

        cost 矩阵：归一化后取反（最大 S 路径 = 最小 cost 路径）
        [Note 0.5] 此处内置归一化修复了 README 问题0.5：
        backtest._infer_tsprank 与 predict_rank 均经过此函数，行为一致。
        """
        N      = len(S)
        s_mean = S.mean()
        s_std  = S.std() + 1e-8
        C      = -(S - s_mean) / s_std
        np.fill_diagonal(C, np.inf)

        tour = cls.greedy_tour(C)
        if use_or_opt and N <= max_n:
            tour = cls.or_opt(tour, C)
        return tour


# ─────────────────────────────────────────────────────────────────────────────
# SCIPSolver（精确 MILP，pyscipopt + MTZ 公式）
# ─────────────────────────────────────────────────────────────────────────────

class SCIPSolver:
    """
    基于 SCIP + MTZ 公式的精确最大权重 Hamiltonian 路径求解器。

    对齐论文设置（vs 本项目之前的 greedy+Or-opt 近似）：
      论文用 Gurobi MILP → 本项目用 SCIP（pyscipopt 6.2.1，已装好）
      N≤25 时 SCIP 精确求解速度 ~10-100ms，完全满足行业分组训练需求。

    MILP 公式（最大化权重 Hamiltonian 路径）：
      变量：x[i,j] ∈ {0,1}（路径边），u[i] ∈ [1,N]（位置，MTZ消环用）
      目标：max Σ_{i≠j} S[i,j] · x[i,j]
      约束：
        (1) 出度 ≤ 1：Σ_j x[i,j] ≤ 1  ∀i
        (2) 入度 ≤ 1：Σ_i x[i,j] ≤ 1  ∀j
        (3) 边数 = N-1（Hamiltonian 路径）：Σ x[i,j] = N-1
        (4) MTZ 消除子回路：u[i] - u[j] + N·x[i,j] ≤ N-1  ∀i≠j

    N 阈值策略：
      训练时 MAX_EXACT_N=25 → C39(N=46) 由 backbone Top-25 降采样后再求解
      推理时 MAX_INFER_N=50，time_limit=5s → C39(N=46) 直接 SCIP
      超限时自动降级为 TSPSolver.greedy+Or-opt
    """

    MAX_EXACT_N : int = 25   # 训练时精确求解上限
    MAX_INFER_N : int = 50   # 推理时精确求解上限（允许更大 N + 更长时限）

    @classmethod
    def solve_max_path(
        cls,
        S_np      : np.ndarray,
        time_limit: float = 5.0,
        for_infer : bool  = False,
    ) -> np.ndarray:
        """
        求最大权重 Hamiltonian 路径。

        参数
        ----
        S_np       : (N, N) 得分矩阵，最大化路径总得分
        time_limit : SCIP 时间上限（秒）；超时返回当前最优可行解
        for_infer  : True=推理模式，允许 N≤MAX_INFER_N；False=训练，N≤MAX_EXACT_N

        返回
        ----
        tour : (N,) int64 排名索引，tour[0] 排第一
        """
        N = len(S_np)
        if N <= 1:
            return np.arange(N, dtype=np.int64)

        max_n = cls.MAX_INFER_N if for_infer else cls.MAX_EXACT_N
        if N > max_n:
            return cls._fallback(S_np)

        try:
            from pyscipopt import Model, quicksum
        except ImportError:
            return cls._fallback(S_np)

        try:
            m = Model()
            m.hideOutput()
            m.setParam("limits/time", time_limit)
            m.setParam("display/verblevel", 0)

            # 决策变量 x[i,j] ∈ {0,1}
            x = {
                (i, j): m.addVar(vtype="B", name=f"x_{i}_{j}")
                for i in range(N) for j in range(N) if i != j
            }
            # 位置变量（MTZ）
            u = {
                i: m.addVar(vtype="C", lb=1.0, ub=float(N), name=f"u_{i}")
                for i in range(N)
            }

            # 目标：最大化路径总得分
            m.setObjective(
                quicksum(float(S_np[i, j]) * x[i, j]
                         for (i, j) in x),
                sense="maximize",
            )

            # 约束1：出度 ≤ 1
            for i in range(N):
                m.addCons(
                    quicksum(x[i, j] for j in range(N) if j != i) <= 1,
                    name=f"out_{i}",
                )
            # 约束2：入度 ≤ 1
            for j in range(N):
                m.addCons(
                    quicksum(x[i, j] for i in range(N) if i != j) <= 1,
                    name=f"in_{j}",
                )
            # 约束3：恰好 N-1 条边
            m.addCons(
                quicksum(x[i, j] for (i, j) in x) == N - 1,
                name="path_length",
            )
            # 约束4：MTZ 消除子回路
            for (i, j) in x:
                m.addCons(
                    u[i] - u[j] + N * x[i, j] <= N - 1,
                    name=f"mtz_{i}_{j}",
                )

            m.optimize()

            if m.getNSols() > 0:
                sol = m.getBestSol()

                # 提取有向边集合
                succ: dict = {}
                for (i, j) in x:
                    if m.getSolVal(sol, x[i, j]) > 0.5:
                        succ[i] = j

                # 找起点（无入边）
                has_incoming = set(succ.values())
                starts       = [i for i in range(N) if i not in has_incoming]
                start        = starts[0] if starts else 0

                # 按链表顺序构建路径
                tour: list = []
                visited: set = set()
                cur = start
                while cur is not None and cur not in visited:
                    tour.append(cur)
                    visited.add(cur)
                    cur = succ.get(cur)

                # 补全未被访问节点（timelimit 解可能不完整）
                missing = [i for i in range(N) if i not in visited]
                tour.extend(missing)
                return np.array(tour[:N], dtype=np.int64)

        except Exception as e:
            import warnings
            warnings.warn(
                f"[SCIPSolver] 求解失败（N={N}），降级为 Or-opt。原因：{e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls._fallback(S_np)

    @staticmethod
    def _fallback(S_np: np.ndarray) -> np.ndarray:
        """N 超限或 SCIP 异常时，降级为 greedy+Or-opt。"""
        mean_ = S_np.mean()
        std_  = S_np.std() + 1e-8
        C     = -(S_np - mean_) / std_
        np.fill_diagonal(C, np.inf)
        tour = TSPSolver.greedy_tour(C)
        if len(S_np) <= 150:
            tour = TSPSolver.or_opt(tour, C, max_iter=200)
        return tour


# ─────────────────────────────────────────────────────────────────────────────
# TSPRankModel
# ─────────────────────────────────────────────────────────────────────────────

class TSPRankModel(nn.Module):
    """
    TSPRank 完整模型 v2.1

    forward(x) → (S, A)
      S : 原始双线性得分（loss 用）
      A : sigmoid(S)（诊断用）

    训练：compute_loss(x, returns, force_local=False)
    推理：predict_rank(x) → TSP 路径（生产选股用）
         predict_score(x) → row-sum 得分（仅诊断）
    """

    def __init__(
        self,
        in_dim       : int,
        emb_dim      : int   = 64,
        hidden_dims  : tuple = (256, 128),
        dropout      : float = 0.1,
        loss_mode    : str   = "local",
        global_weight: float = 0.3,
    ):
        super().__init__()
        # [Fix] 存储构造参数，供 trainer._reset_model() 正确重建
        self.in_dim       = in_dim
        self.emb_dim      = emb_dim
        self.hidden_dims  = hidden_dims
        self.dropout_p    = dropout

        self.projector     = EmbeddingProjector(in_dim, emb_dim, hidden_dims, dropout)
        self.bilinear      = BilinearTSPLayer(emb_dim, dropout=dropout)
        self.loss_mode     = loss_mode
        self.global_weight = global_weight

        # [Fix] 训练步计数器，用于 "both" 模式的逐步交替（论文推荐 alternate batches）
        self._train_step   = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (S, A)：S 原始得分，A = sigmoid(S)"""
        H = self.projector(x)
        S = self.bilinear(H)
        A = torch.sigmoid(S)
        return S, A

    # ── Local Loss：Row-wise Weighted Cross-Entropy（论文 Eq.5）─────────────

    def _local_loss(
        self,
        S      : torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        对真实排列 π（收益率降序）中每个位置 k：
          - 从 N 个候选中识别 π(k+1)（真实下一节点）
          - 权重 w_k = N - k（靠前位置权重更大，对齐论文 y_k = N+1-position）
        使用 S 做 log_softmax（不是 A，避免梯度消失）
        """
        N    = len(returns)
        perm = torch.argsort(returns, descending=True)
        w    = torch.arange(N, 0, -1, dtype=torch.float32, device=S.device)
        src  = perm[:-1]
        dst  = perm[1:]

        log_prob        = F.log_softmax(S[src], dim=1)           # (N-1, N)
        target_log_prob = log_prob[torch.arange(N - 1, device=S.device), dst]
        loss            = -(w[:N - 1] * target_log_prob).sum() / (w[:N - 1].sum() + 1e-8)
        return loss

    # ── Global Loss：Structured Max-Margin + Solver-in-Loop（论文 Eq.6-7）──

    def _global_loss(
        self,
        S        : torch.Tensor,
        returns  : torch.Tensor,
        use_scip : bool = False,
    ) -> torch.Tensor:
        """
        论文 Eq.6-7：Structured Max-Margin Loss with Solver-in-Loop

        use_scip=True  → SCIPSolver 精确 MTZ（N≤25，行业分组训练用）
        use_scip=False → greedy+Or-opt 近似（N 较大时降级方案）

        L_glob(A^p, x^t) = max(0, max_{x} [Δ(x,x^t) + x·A^p] - x^t·A^p)

        其中：
          Δ(x, x^t) = 路径 x 中不属于真实路径 x^t 的边数（margin 惩罚）
          x · A^p   = 路径 x 在预测矩阵 A^p（即 S）上的总得分

        实现步骤：
          1. 构造增广矩阵：S_aug[i,j] = S[i,j] + (1 - A^t[i,j])
             - 真实路径边 A^t[i,j]=1 → +0（保持 S 值）
             - 其他边 A^t[i,j]=0 → +1（加入 margin 项）
          2. 找最违反约束路径：x_worst = argmax_{Hamiltonian path} S_aug
             （TSPSolver 近似，最大化 = 最小化 -S_aug）
          3. loss = max(0, x_worst·S_aug - x_true·S)
             等价于：max(0, Δ(x_worst,x_true) + x_worst·S - x_true·S)

        梯度传播（直通估计）：
          solver 输出 x_worst 固定，不参与反向传播
          梯度仅流经 S 的高级索引：
            ∂loss/∂S[i,j] = +1  if (i,j) in x_worst and loss > 0
            ∂loss/∂S[i,j] = -1  if (i,j) in x_true  and loss > 0
            ∂loss/∂S[i,j] =  0  otherwise
          效果：降低最违反路径上的 S 值，提升真实路径上的 S 值。

        与论文的差异（工程妥协）：
          - 论文用 Gurobi MILP（精确解）
          - 本实现用 greedy + Or-opt（近似，但对 N≤100 效果足够）
          - 任何比真实路径得分更高的替代路径都可产生有效梯度信号
        """
        N      = len(returns)
        device = S.device

        # 真实路径：收益率降序排列
        perm     = torch.argsort(returns, descending=True)
        true_src = perm[:-1]   # 真实路径边的起点
        true_dst = perm[1:]    # 真实路径边的终点

        # 构造目标邻接矩阵 A^t（稀疏：N-1 条边为 1，其余为 0）
        # 注意：A_t 不参与梯度计算（它是常数，仅依赖于 returns）
        A_t = torch.zeros(N, N, device=device)
        A_t[true_src, true_dst] = 1.0

        # 增广矩阵：S_aug = S + (1 - A_t)
        # (1 - A_t) 是常数张量，无梯度；梯度通过 S 传播
        S_aug = S + (1.0 - A_t)

        # ── 找最违反约束路径（solver 在 detached 增广矩阵上运行）─────────────
        S_aug_np = S_aug.detach().cpu().numpy().astype(np.float64)

        if use_scip:
            # SCIP 精确 MTZ：直接在 S_aug 上求最大权重路径
            # N≤MAX_EXACT_N=25 时约 10-100ms，行业分组后满足此条件
            worst_tour = SCIPSolver.solve_max_path(S_aug_np, time_limit=2.0)
        else:
            # greedy+Or-opt 近似（N 较大时或 SCIP 不可用时的降级方案）
            aug_mean = S_aug_np.mean()
            aug_std  = S_aug_np.std() + 1e-8
            C_aug    = -(S_aug_np - aug_mean) / aug_std
            np.fill_diagonal(C_aug, np.inf)
            worst_tour = TSPSolver.greedy_tour(C_aug)
            if N <= 100:
                worst_tour = TSPSolver.or_opt(worst_tour, C_aug, max_iter=50)

        # ── 可微分的得分计算（梯度流经 S 的高级索引）─────────────────────────
        worst_src = torch.tensor(worst_tour[:-1], dtype=torch.long, device=device)
        worst_dst = torch.tensor(worst_tour[1:],  dtype=torch.long, device=device)

        # x_worst · S_aug = Δ(x_worst, x_true) + x_worst · S
        worst_score_aug = S_aug[worst_src, worst_dst].sum()
        # x_true · S
        true_score      = S[true_src, true_dst].sum()

        # loss = max(0, Δ + x_worst·S - x_true·S)
        return F.relu(worst_score_aug - true_score)

    # ── 总损失 ────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        x          : torch.Tensor,
        returns    : torch.Tensor,
        force_local: bool = False,
        use_scip   : bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失。

        Parameters
        ----------
        x           : (N, in_dim) 输入特征
        returns     : (N,) 收益率标签
        force_local : True 时强制使用 local loss（供验证阶段使用，跳过 solver 开销）
        use_scip    : True 时 global loss 使用 SCIP 精确求解（行业分组训练推荐）
                      False 时使用 greedy+Or-opt 近似

        Loss modes
        ----------
        "local"  : 仅 row-wise weighted CE（论文 Eq.5）
        "global" : 仅 structured max-margin（论文 Eq.6-7）
        "both"   : hybrid 交替训练（论文推荐）
                   奇数步：local + global；偶数步：local only
                   对齐论文 Section 4.2 "alternate batches" 设计
        """
        S, _  = self.forward(x)
        ll    = self._local_loss(S, returns)
        info  = {"local_loss": ll.item()}

        # force_local=True 时（验证阶段）直接返回，不调用 solver
        if force_local or self.loss_mode == "local":
            info["total_loss"] = ll.item()
            return ll, info

        if self.loss_mode == "global":
            gl    = self._global_loss(S, returns, use_scip=use_scip)
            total = gl
            info["global_loss"]  = gl.item()
            info["total_loss"]   = total.item()
            return total, info

        # "both" → hybrid 交替（论文 Section 4.2）
        # 奇数步：local + global（引入全局约束）
        # 偶数步：local only（加速收敛，节省 solver 开销）
        self._train_step += 1
        if self._train_step % 2 == 1:
            gl    = self._global_loss(S, returns, use_scip=use_scip)
            total = ll + self.global_weight * gl
            info["global_loss"] = gl.item()
        else:
            total = ll

        info["total_loss"] = total.item()
        return total, info

    # ── 推理 ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_rank(
        self,
        x         : torch.Tensor,
        use_or_opt: bool = True,
    ) -> np.ndarray:
        """TSP 路径推理 → 排名索引（生产选股用）"""
        self.eval()
        S, _ = self.forward(x)
        return TSPSolver.solve(S.cpu().numpy().astype(np.float64), use_or_opt=use_or_opt)

    @torch.no_grad()
    def predict_score(self, x: torch.Tensor) -> np.ndarray:
        """Row-sum 得分（仅诊断用，不用于生产）"""
        self.eval()
        S, _ = self.forward(x)
        N    = S.shape[0]
        mask = 1.0 - torch.eye(N, device=S.device)
        return (S * mask).sum(dim=1).cpu().numpy().astype(np.float32)
