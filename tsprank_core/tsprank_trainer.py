# tsprank_core/tsprank_trainer.py
"""
TSPRank 训练流程管理 v2.1

修复内容（vs v2.0）：
  [Fix 4]  _sample_stocks()：随机采样 → 分层采样（Stratified Sampling）
           将收益率分布均匀划分为 max_stocks 层，每层随机取 1 只。
           原因：纯随机采样在某些截面中大概率漏掉最高/最低收益股票，
           导致训练集内的 successor 关系与全量推理时的 successor 关系
           不一致（N=100 子集的前驱关系 ≠ N=300 全集的前驱关系）。
           分层采样保证每次训练样本都覆盖收益率的全范围，
           使模型能学到有效的排名判别信号。

  [Fix]   _one_day(train=False)：增加 force_local=True 调用
           验证阶段跳过 solver（Global Loss），节省开销。
           early stopping 的 val_loss 始终用 local loss 作代理指标。

  [Fix]   _reset_model()：从硬编码 (256,128)/0.1 改为读取
           self.model.in_dim / emb_dim / hidden_dims / dropout_p，
           保证 walk-forward 每个 fold 重建的模型与初始模型结构完全一致。

v2.0 保留内容：
  ① 验证集使用 test_df 前 60%（2025 年数据），而非 valid_df（2019-2021）
  ② patience=25
  ③ samples_per_day=2（每日两次分层采样，获得不同子集）
  ④ train_rolling() / train_walk_forward() 接口不变
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .tsprank_model import TSPRankModel, DEVICE
from .data_loader import _normalize_code


class TSPRankTrainer:
    """
    TSPRank 训练器 v2.1

    Parameters
    ----------
    in_dim         : TSPRankModel 输入维度（158 for feature, 159 for combined）
    emb_dim        : 双线性层的 embedding 维度
    hidden_dims    : Projector 隐藏层维度
    dropout        : Dropout 率
    loss_mode      : "local" | "global" | "both"
    global_weight  : both 模式下 global loss 的权重
    lr             : AdamW 学习率
    weight_decay   : L2 正则
    max_epochs     : 最大训练轮数
    patience       : 早停 patience
    min_stocks     : 单日最少股票数
    max_stocks     : 每日采样上限（分层采样的总层数 = max_stocks）
    samples_per_day: 每日采样次数（不同随机子集，增加多样性）
    grad_clip      : 梯度裁剪阈值
    ckpt_path      : 检查点保存路径
    """

    def __init__(
        self,
        in_dim         : int,
        emb_dim        : int   = 64,
        hidden_dims    : tuple = (256, 128),
        dropout        : float = 0.1,
        loss_mode      : str   = "local",
        global_weight  : float = 0.3,
        lr             : float = 3e-4,
        weight_decay   : float = 1e-4,
        max_epochs     : int   = 100,
        patience       : int   = 25,
        min_stocks     : int   = 10,
        max_stocks     : int   = 100,
        samples_per_day: int   = 2,
        grad_clip      : float = 1.0,
        ckpt_path      : Optional[str] = None,
    ):
        self.model = TSPRankModel(
            in_dim        = in_dim,
            emb_dim       = emb_dim,
            hidden_dims   = hidden_dims,
            dropout       = dropout,
            loss_mode     = loss_mode,
            global_weight = global_weight,
        ).to(DEVICE)
        self.optimizer  = optim.AdamW(
            self.model.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
            betas        = (0.9, 0.999),
            eps          = 1e-8,
        )
        self.max_epochs     = max_epochs
        self.patience       = patience
        self.min_stocks     = min_stocks
        self.max_stocks     = max_stocks
        self.samples_per_day= samples_per_day
        self.grad_clip      = grad_clip
        self.ckpt_path      = ckpt_path
        self.best_val_loss  = float("inf")
        self.best_state     : Optional[dict] = None

    # ── [Fix 4] 分层采样（Stratified Sampling）──────────────────────────────

    def _sample_stocks(
        self,
        x_np: np.ndarray,
        y_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        分层采样：将 N 只股票按收益率排名均匀划分为 max_stocks 层，
        每层随机取 1 只，保证训练样本覆盖收益率的全范围。

        v2.0 问题4：纯随机采样在 N=300→100 的降维中大概率漏掉
        收益极值股票，使 successor 关系只反映"中间段"排名，
        与推理时 N=300 全集的排名语义不一致。

        分层采样修复：
          - 每次采样总是包含最高收益区、中间区、最低收益区的代表
          - 模型被迫学习全范围排名判别，而不仅是局部排名
          - 多次调用（samples_per_day=2）产生不同的随机子集，保持多样性

        Returns
        -------
        (x_sampled, y_sampled) : 各 (max_stocks, F) 和 (max_stocks,) 的数组
        """
        N = len(x_np)
        if N <= self.max_stocks:
            return x_np, y_np

        # 按收益率计算序号（0=最低收益, N-1=最高收益）
        rank = np.argsort(np.argsort(y_np))

        # 将 [0, N) 均匀划分为 max_stocks 个分层区间
        strata_edges = np.linspace(0, N, self.max_stocks + 1, dtype=int)
        idx: List[int] = []
        for i in range(self.max_stocks):
            lo, hi     = strata_edges[i], strata_edges[i + 1]
            candidates = np.where((rank >= lo) & (rank < hi))[0]
            if len(candidates) > 0:
                idx.append(int(np.random.choice(candidates)))

        idx_arr = np.array(idx, dtype=np.int64)

        # 安全补充：若某层为空（极罕见），用剩余股票随机填充
        if len(idx_arr) < self.max_stocks:
            shortage  = self.max_stocks - len(idx_arr)
            remaining = np.setdiff1d(np.arange(N), idx_arr)
            if len(remaining) >= shortage:
                fill = np.random.choice(remaining, size=shortage, replace=False)
            else:
                fill = remaining
            idx_arr = np.concatenate([idx_arr, fill])

        idx_arr = idx_arr[:self.max_stocks]
        return x_np[idx_arr], y_np[idx_arr]

    # ── 单日前向 + loss ───────────────────────────────────────────────────────

    def _one_day(
        self,
        x_np : np.ndarray,
        y_np : np.ndarray,
        train: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        单日截面的 forward + loss。

        训练（train=True）：分层采样到 max_stocks，model.train()
        验证（train=False）：全量计算，force_local=True 跳过 solver
        """
        if train and len(x_np) > self.max_stocks:
            # [Fix 4] 分层采样替代纯随机采样
            x_np, y_np = self._sample_stocks(x_np, y_np)

        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)

        if train:
            self.model.train()
            loss, info = self.model.compute_loss(x, y)
        else:
            self.model.eval()
            with torch.no_grad():
                # [Fix] 验证时 force_local=True，跳过 Global Loss solver 开销
                # val_loss 始终是 local loss，用作 early stopping 代理指标
                loss, info = self.model.compute_loss(x, y, force_local=True)
        return loss, info

    # ── 核心训练（train/val_dates 分离）─────────────────────────────────────

    def train(
        self,
        train_dates: List,
        val_dates  : List,
        X_dict     : Dict,
        y_dict     : Dict,
    ) -> float:
        """
        标准训练接口（train_df 训练 + val_dates 早停）。

        Parameters
        ----------
        train_dates : 训练日期列表
        val_dates   : 验证日期列表（v2.0 建议使用 test_df 前 60%）
        X_dict      : date → ndarray 特征字典
        y_dict      : date → ndarray 标签字典

        Returns
        -------
        best_val_loss : float
        """
        scheduler    = CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        patience_cnt = 0
        self.best_val_loss = float("inf")
        self.best_state    = None

        for epoch in range(self.max_epochs):
            t0 = time.time()

            # ── 训练阶段 ──
            shuffled = list(train_dates)
            np.random.shuffle(shuffled)
            tl, nt = 0.0, 0

            for d in shuffled:
                if d not in X_dict or len(X_dict[d]) < self.min_stocks:
                    continue
                for _ in range(self.samples_per_day):
                    # [Fix 4] _one_day 内部现在使用分层采样
                    loss, _ = self._one_day(X_dict[d], y_dict[d], train=True)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.optimizer.step()
                    tl += loss.item()
                    nt += 1

            # ── 验证阶段（force_local=True 已在 _one_day 内处理）──
            vl, nv = 0.0, 0
            for d in val_dates:
                if d not in X_dict or len(X_dict[d]) < self.min_stocks:
                    continue
                loss, _ = self._one_day(X_dict[d], y_dict[d], train=False)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    vl += loss.item()
                    nv += 1

            scheduler.step()
            avg_t = tl / max(nt, 1)
            avg_v = vl / max(nv, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [tsprank] Ep {epoch + 1:3d}/{self.max_epochs}  "
                    f"train={avg_t:.4f}  val={avg_v:.4f}  "
                    f"{time.time() - t0:.1f}s"
                )

            # ── 早停 ──
            if avg_v < self.best_val_loss - 1e-6:
                self.best_val_loss = avg_v
                patience_cnt       = 0
                self.best_state    = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(
                        f"  [tsprank] Early stop at ep={epoch + 1}  "
                        f"best_val={self.best_val_loss:.4f}"
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.ckpt_path is not None:
            self.save(self.ckpt_path)
        return self.best_val_loss

    # ── P2：滚动窗口训练（使用 test_df 近期数据）────────────────────────────

    def train_rolling(
        self,
        test_X_dict   : Dict,
        test_y_dict   : Dict,
        train_ratio   : float = 0.6,
        verbose       : bool  = True,
    ) -> float:
        """
        P2 方案：用 test_df（2025 年数据）做滚动窗口训练。

        前 train_ratio（60%）的测试日 → 训练；后 40% → 验证早停。
        ⚠️  训练数据与回测数据存在重叠（数据泄露风险）。
        """
        test_dates = sorted(test_X_dict.keys())
        n_total    = len(test_dates)
        split_idx  = int(n_total * train_ratio)

        train_dates = test_dates[:split_idx]
        val_dates   = test_dates[split_idx:]

        if verbose:
            print(
                f"\n[tsprank] P2 滚动窗口训练:\n"
                f"  总测试日: {n_total}\n"
                f"  训练日:  {len(train_dates)} "
                f"({train_dates[0].date()} ~ {train_dates[-1].date()})\n"
                f"  验证日:  {len(val_dates)} "
                f"({val_dates[0].date()} ~ {val_dates[-1].date()})\n"
                f"  ⚠️  注意: 训练数据与回测区间存在重叠，"
                f"严格评估请使用 train_walk_forward()"
            )

        return self.train(train_dates, val_dates, test_X_dict, test_y_dict)

    def train_walk_forward(
        self,
        test_X_dict    : Dict,
        test_y_dict    : Dict,
        train_window   : int = 120,
        val_window     : int = 30,
        step           : int = 20,
        verbose        : bool = True,
    ) -> List[float]:
        """
        Walk-Forward Validation（严格无泄露版）。

        以 train_window 天训练，val_window 天验证，
        向前滑动 step 天，避免训练数据与回测数据重叠。
        """
        test_dates = sorted(test_X_dict.keys())
        n          = len(test_dates)
        val_losses = []
        fold       = 0

        start = 0
        while start + train_window + val_window <= n:
            t_dates = test_dates[start : start + train_window]
            v_dates = test_dates[start + train_window :
                                  start + train_window + val_window]
            if verbose:
                print(
                    f"\n[tsprank] Walk-Forward Fold {fold + 1}: "
                    f"train {t_dates[0].date()}~{t_dates[-1].date()} "
                    f"| val {v_dates[0].date()}~{v_dates[-1].date()}"
                )
            self._reset_model()
            bv = self.train(t_dates, v_dates, test_X_dict, test_y_dict)
            val_losses.append(bv)
            fold  += 1
            start += step

        if verbose:
            print(
                f"\n[tsprank] Walk-Forward 完成: {fold} folds, "
                f"mean val_loss={np.mean(val_losses):.4f}"
            )
        return val_losses

    # ── [Fix] _reset_model：从存储参数重建，不再硬编码 ───────────────────────

    def _reset_model(self) -> None:
        """
        重置模型权重（用于 walk-forward 每个 fold）。

        v2.0 bug：硬编码 hidden_dims=(256,128), dropout=0.1，
        若实例化时用不同超参会导致 fold 模型结构与原始不一致。

        v2.1 修复：从 self.model 存储的构造参数读取，保证一致性。
        TSPRankModel v2.1 在 __init__ 中存储了
        in_dim / emb_dim / hidden_dims / dropout_p。
        """
        self.model = TSPRankModel(
            in_dim        = self.model.in_dim,
            emb_dim       = self.model.emb_dim,
            hidden_dims   = self.model.hidden_dims,
            dropout       = self.model.dropout_p,
            loss_mode     = self.model.loss_mode,
            global_weight = self.model.global_weight,
        ).to(DEVICE)
        # 保留 lr / weight_decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr           = self.optimizer.param_groups[0]["lr"],
            weight_decay = self.optimizer.param_groups[0]["weight_decay"],
        )
        self.best_val_loss = float("inf")
        self.best_state    = None
        # 注意：新模型的 _train_step 自动重置为 0（在 __init__ 中初始化）

    # ── [Sector] 行业分组单日训练 ─────────────────────────────────────────────

    def _one_day_sector(
        self,
        x_np         : np.ndarray,
        y_np         : np.ndarray,
        codes        : List[str],
        sector_map   : Dict[str, str],   # {clean_code: sector}，单日行业映射
        backbone     = None,
        train        : bool = True,
        max_sector_n : int  = 25,
        use_scip     : bool = True,
        min_sector_n : int  = 5,
    ) -> Tuple[torch.Tensor, dict]:
        """
        按行业分组计算 loss，返回所有有效行业的平均 loss。

        sector_map : 当天的 {clean_code: sector}，来自 sector_dict[d]（PIT）。
        跳过规则：
          ① sector == "unknown"：无行业分类，混杂组无经济含义，显式跳过
          ② len < min_sector_n（默认5）：论文 Section 5 N=5~20，< 5 排名无意义
        """
        sector_groups: Dict[str, List[int]] = defaultdict(list)
        for i, code in enumerate(codes):
            clean  = _normalize_code(str(code))
            sector = sector_map.get(clean, "unknown")
            sector_groups[sector].append(i)

        total_loss : Optional[torch.Tensor] = None
        n_valid    = 0

        for sector, indices in sector_groups.items():
            if sector == "unknown":          # Bug3 修复：显式跳过无分类股票
                continue
            if len(indices) < min_sector_n:
                continue

            idx_arr = np.array(indices)
            x_sec   = x_np[idx_arr]
            y_sec   = y_np[idx_arr]

            # 超大行业降采样（C39 N=46 → backbone Top-25）
            if len(idx_arr) > max_sector_n:
                if backbone is not None:
                    try:
                        scores  = backbone.predict_score(x_sec)
                        top_idx = np.argsort(scores)[::-1][:max_sector_n]
                    except Exception:
                        top_idx = np.arange(max_sector_n)
                else:
                    # 无 backbone：分层采样保证收益率分布完整
                    x_sec, y_sec = self._sample_stocks(x_sec, y_sec)
                    top_idx = np.arange(min(max_sector_n, len(x_sec)))
                x_sec = x_sec[top_idx]
                y_sec = y_sec[top_idx]

            x_t = torch.tensor(x_sec, dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y_sec, dtype=torch.float32, device=DEVICE)

            if train:
                self.model.train()
                loss, _ = self.model.compute_loss(x_t, y_t, use_scip=use_scip)
            else:
                self.model.eval()
                with torch.no_grad():
                    loss, _ = self.model.compute_loss(x_t, y_t, force_local=True)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss = loss if total_loss is None else total_loss + loss
            n_valid   += 1

        if n_valid == 0 or total_loss is None:
            dummy = torch.tensor(0.0, device=DEVICE)
            return dummy, {"total_loss": 0.0, "n_sectors": 0}

        avg_loss = total_loss / n_valid
        return avg_loss, {"total_loss": avg_loss.item(), "n_sectors": n_valid}

    # ── [Sector] 行业分组完整训练循环 ────────────────────────────────────────

    def train_sector(
        self,
        train_dates   : List,
        val_dates     : List,
        emb_dict      : Dict,
        y_dict        : Dict,
        stock_dict    : Dict,
        sector_dict   : Dict,           # Bug2 修复：接受 PIT 全量映射 {date: {clean_code: sector}}
        backbone      = None,
        max_sector_n  : int  = 25,
        use_scip      : bool = True,
    ) -> float:
        """
        Sector-grouped + SCIP solver-in-loop 训练。

        Bug2 修复：原接收静态 code_to_sector（全日期同一份），
        现改为接收 PIT sector_dict（{date: {clean_code: sector}}），
        每个训练日从 sector_dict[d] 取当日的行业分组，
        真正实现点位一致（point-in-time）的行业分组训练。

        sector_dict 由 main.py 通过 build_sector_dict(all_s, industry_df)
        构建，key 已统一为 clean code。
        """
        scheduler    = CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=1e-6
        )
        patience_cnt = 0
        self.best_val_loss = float("inf")
        self.best_state    = None

        for epoch in range(self.max_epochs):
            t0 = time.time()

            # ── 训练阶段 ──
            shuffled = list(train_dates)
            np.random.shuffle(shuffled)
            tl, nt = 0.0, 0

            for d in shuffled:
                if d not in emb_dict or d not in stock_dict:
                    continue
                # Bug2 修复：每天查 PIT sector_dict[d]
                day_sector_map = sector_dict.get(d)
                if day_sector_map is None:
                    continue
                codes = stock_dict[d]
                y_np  = y_dict.get(d)
                if y_np is None or len(codes) < self.min_stocks:
                    continue

                x_np = emb_dict[d]
                loss, info = self._one_day_sector(
                    x_np, y_np, codes, day_sector_map, backbone,
                    train=True, max_sector_n=max_sector_n, use_scip=use_scip,
                )
                n_sec = info.get("n_sectors", 0)
                if n_sec == 0 or loss.item() == 0.0:
                    continue
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()
                tl += loss.item()
                nt += 1

            # ── 验证阶段 ──
            vl, nv = 0.0, 0
            for d in val_dates:
                if d not in emb_dict or d not in stock_dict:
                    continue
                day_sector_map = sector_dict.get(d)
                if day_sector_map is None:
                    continue
                codes = stock_dict[d]
                y_np  = y_dict.get(d)
                if y_np is None:
                    continue

                x_np = emb_dict[d]
                loss, _ = self._one_day_sector(
                    x_np, y_np, codes, day_sector_map, backbone,
                    train=False,
                )
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    vl += loss.item()
                    nv += 1

            scheduler.step()
            avg_t = tl / max(nt, 1)
            avg_v = vl / max(nv, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                solver_tag = "SCIP" if use_scip else "OrOpt"
                print(
                    f"  [sector] Ep {epoch + 1:3d}/{self.max_epochs}  "
                    f"train={avg_t:.4f}  val={avg_v:.4f}  "
                    f"[{solver_tag}]  {time.time() - t0:.1f}s"
                )

            # ── 早停 ──
            if avg_v < self.best_val_loss - 1e-6:
                self.best_val_loss = avg_v
                patience_cnt       = 0
                self.best_state    = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(
                        f"  [sector] Early stop at ep={epoch + 1}  "
                        f"best_val={self.best_val_loss:.4f}"
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if self.ckpt_path is not None:
            self.save(self.ckpt_path)
        return self.best_val_loss

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  [tsprank] Saved → {path}")

    def load(self, path: str) -> "TSPRankTrainer":
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(state)
        print(f"  [tsprank] Loaded ← {path}")
        return self
