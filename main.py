# main.py
"""
TSPRank Stock Selection — 统一入口 v2.2

修复内容（vs v2.0）：
  [Fix] _predict_latest() line 581：
        make_result_csv(top5_codes, weights, top_k=top_k)
        v2.0 中 make_result_csv 不接受 top_k 参数且返回 None，
        导致 TypeError（unexpected keyword argument）和
        AttributeError（NoneType has no attribute 'to_string'）。
        v2.1 修复：data_loader.make_result_csv 已更新接口，
        此处调用无需改动，兼容性由 data_loader.py 保证。

  [Fix] _predict_latest() 的 DEVICE 导入：简化为直接
        from tsprank_core.tsprank_model import TSPSolver, DEVICE，
        替代 v2.0 中的 __import__(...).DEVICE 写法。

v2.0 保留内容（不变）：
  - 全部 CLI 参数
  - 所有 step_* 函数
  - train_mode: standard / rolling / walk_forward
  - infer_mode: tsprank / backbone / qlib_pred

用法:
  python main.py --mode full            # 完整 pipeline
  python main.py --mode train_backbone  # 仅训练 backbone
  python main.py --mode train_tsprank   # 仅训练 TSPRank（需先有 backbone）
  python main.py --mode backtest        # 仅回测
  python main.py --mode predict         # 仅预测最新周

推荐组合:
  # P0（立即可用）：backbone 直接 Top-5
  python main.py --mode backtest --use_cache --embed_mode combined --infer_mode backbone

  # P1：接入小组 pred.pkl
  python main.py --mode backtest --use_cache --infer_mode qlib_pred \\
      --qlib_pred_pkl ~/quant-stock-forecast/.../artifacts/pred.pkl

  # P2：滚动窗口训练 TSPRank（用近期数据 + Global Loss）
  python main.py --mode train_tsprank --use_cache \\
      --embed_mode combined --loss_mode both --train_mode rolling
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from tsprank_core.data_loader import (
    QLIB_CONFIG,
    TSPRANK_ROOT,
    DATA_DIR,
    extract_cross_sections,
    load_alpha158_dataset,
    load_stock_data,
    get_eval_weeks,
    build_test_csv,
    make_result_csv,
    load_industry_mapping,
    load_industry_df,
    build_sector_dict,
)
from tsprank_core.backbone       import DEBackbone
from tsprank_core.tsprank_model  import TSPRankModel, TSPSolver, DEVICE
from tsprank_core.tsprank_trainer import TSPRankTrainer
from tsprank_core.backtest        import run_backtest

CKPT_DIR      = TSPRANK_ROOT / "checkpoints"
CACHE_DIR     = TSPRANK_ROOT / "cache"
BACKBONE_PKL  = CKPT_DIR / "backbone.pkl"
TSPRANK_PT    = CKPT_DIR / "tsprank.pt"
DATASET_CACHE = CACHE_DIR / "alpha158_dataset"
SEED          = 20260508


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[main] Seed={seed}")


def parse_args():
    p = argparse.ArgumentParser(description="TSPRank 选股系统 v2.1")

    # ── 运行模式 ──
    p.add_argument(
        "--mode", default="full",
        choices=["full", "train_backbone", "train_tsprank", "backtest", "predict"],
        help="运行模式",
    )
    p.add_argument(
        "--use_cache", action="store_true",
        help="使用 parquet 缓存（首次运行必须先不带此参数生成缓存）",
    )

    # ── 特征模式 ──
    p.add_argument(
        "--embed_mode", default="combined",
        choices=["feature", "combined"],
        help="feature=Alpha158(158维), combined=Alpha158+backbone_rank(159维)",
    )

    # ── 推理模式 ──
    p.add_argument(
        "--infer_mode", default="tsprank",
        choices=["tsprank", "backbone", "qlib_pred", "tsprank_sector"],
        help=(
            "tsprank        : 全截面 TSP 路径（Or-opt）\n"
            "backbone       : P0 直接用 backbone Top-K（推荐）\n"
            "qlib_pred      : P1 接入小组 pred.pkl（IC≈0.10）\n"
            "tsprank_sector : v2.2 行业分组 SCIP 精确推理（论文对齐）"
        ),
    )
    p.add_argument(
        "--qlib_pred_pkl", default=None, type=str,
        help="P1 模式下 pred.pkl 的路径（若不指定则自动搜索）",
    )

    # ── TSPRank 训练参数 ──
    p.add_argument("--loss_mode",  default="local",
                   choices=["local", "global", "both"],
                   help="local=row-wise CE, global=structured max-margin, both=hybrid交替")
    p.add_argument("--emb_dim",    type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience",   type=int, default=25)
    p.add_argument(
        "--train_mode", default="standard",
        choices=["standard", "rolling", "walk_forward"],
        help=(
            "standard    : 用 train_df(2013-2019) 训练\n"
            "rolling     : P2 用 test_df 前60%%训练（推荐与 loss_mode=both 配合）\n"
            "walk_forward: P2 严格无泄露版"
        ),
    )

    # ── 回测参数 ──
    p.add_argument("--recent_weeks",      type=int,   default=12)
    p.add_argument("--top_k",             type=int,   default=5)
    p.add_argument("--total_weight",      type=float, default=1.0)
    p.add_argument("--diversity_penalty", type=float, default=0.0,
                   help="多样性惩罚强度（0=关闭，推荐 0.1~0.2）")
    p.add_argument("--penalty_window",    type=int,   default=4)
    p.add_argument("--no_or_opt",         action="store_true",
                   help="禁用 Or-opt（N>150 自动禁用）")
    p.add_argument("--no_equity_curve",   action="store_true",
                   help="不保存净值曲线图")

    # ── 行业分组（v2.2 新增）──
    p.add_argument(
        "--sector_mode", action="store_true",
        help=(
            "启用行业分组训练+推理（论文 Section 5 设置）\n"
            "训练：每行业独立计算 SCIP Global Loss\n"
            "推理：行业内 SCIP 精确排名 + backbone 全局混合"
        ),
    )
    p.add_argument(
        "--industry_csv", default=None, type=str,
        help="industry.csv 路径（默认 data/industry.csv）",
    )
    p.add_argument(
        "--max_sector_n", type=int, default=25,
        help="训练时每行业最大股票数（超限用 backbone Top-N 降采样，默认 25）",
    )
    p.add_argument(
        "--sector_blend", type=float, default=0.5,
        help="推理时 sector_rank 权重（0=纯 backbone，1=纯 sector，默认 0.5）",
    )
    p.add_argument(
        "--no_scip", action="store_true",
        help="行业分组模式中禁用 SCIP，改用 Or-opt（调试用）",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 步骤函数
# ─────────────────────────────────────────────────────────────────────────────

def step_load_data(args):
    """Step 1+2：加载 Alpha158 并提取截面。"""
    print("\n[Step 1] Loading Alpha158...")
    cache = DATASET_CACHE if (args.use_cache or args.mode != "train_backbone") else None
    train_df, valid_df, test_df = load_alpha158_dataset(
        cache_path=cache,
        use_cache=args.use_cache,
    )

    print("[Step 2] Extracting cross-sections...")
    tr_d, tr_X, tr_y, tr_s = extract_cross_sections(train_df)
    va_d, va_X, va_y, va_s = extract_cross_sections(valid_df)
    te_d, te_X, te_y, te_s = extract_cross_sections(test_df)

    all_X = {**tr_X, **va_X, **te_X}
    all_y = {**tr_y, **va_y, **te_y}
    all_s = {**tr_s, **va_s, **te_s}

    return (
        train_df, valid_df, test_df,
        tr_d, tr_X, tr_y, tr_s,
        va_d, va_X, va_y, va_s,
        te_d, te_X, te_y, te_s,
        all_X, all_y, all_s,
    )


def step_train_backbone(tr_X, tr_y, va_X, va_y):
    """Step 3：训练 backbone 并保存。"""
    print("\n[Step 3] Training backbone...")
    X_tr = np.vstack(list(tr_X.values()))
    y_tr = np.concatenate(list(tr_y.values()))
    X_va = np.vstack(list(va_X.values()))
    y_va = np.concatenate(list(va_y.values()))

    backbone = DEBackbone()
    backbone.fit(X_tr, y_tr, X_va, y_va)
    backbone.save(str(BACKBONE_PKL))
    return backbone


def step_load_backbone(args, embed_mode: str):
    """Step 3（加载）：从 checkpoint 加载 backbone。"""
    if BACKBONE_PKL.exists():
        bb = DEBackbone().load(str(BACKBONE_PKL))
        return bb
    else:
        if embed_mode == "combined":
            print(
                "[main] ⚠️  backbone.pkl 不存在，combined 模式降级为 feature 模式"
            )
        return None


def step_build_emb(all_X, all_y, backbone, embed_mode):
    """
    Step 4：构建 TSPRank 输入 embedding 字典。
    """
    print(f"\n[Step 4] Building embeddings (embed_mode={embed_mode})...")
    emb_dict = {}
    total = len(all_X)

    for i, (date, x_np) in enumerate(all_X.items()):
        if embed_mode == "combined" and backbone is not None:
            try:
                pred_score = backbone.predict_score(x_np)
                rank_raw   = np.argsort(np.argsort(pred_score)).astype(np.float32)
                n          = len(rank_raw)
                rank_pct   = (rank_raw / max(n - 1, 1) - 0.5) * 2
                emb_dict[date] = np.concatenate(
                    [x_np.astype(np.float32), rank_pct[:, None]], axis=1
                )
            except Exception as e:
                print(f"  [warn] embed failed at {date}: {e}, fallback to feature")
                emb_dict[date] = x_np.astype(np.float32)
        else:
            emb_dict[date] = x_np.astype(np.float32)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total}  emb_dim={emb_dict[date].shape[1]}")

    sample  = emb_dict[next(iter(emb_dict))]
    in_dim  = sample.shape[1]
    print(f"  embedding dim={in_dim}")
    return emb_dict, in_dim


def step_train_tsprank(args, emb_dict, all_y, te_X, te_y, tr_d, va_d, in_dim):
    """
    Step 5：训练 TSPRank。

    train_mode:
      standard    → 用 train_df 日期（2013-2019）训练
      rolling     → P2：用 test_df 前 60% 训练（推荐，近期数据）
      walk_forward→ P2：严格无泄露 walk-forward
    """
    print(
        f"\n[Step 5] Training TSPRank "
        f"(loss={args.loss_mode}, emb_dim={args.emb_dim}, "
        f"train_mode={args.train_mode})..."
    )

    trainer = TSPRankTrainer(
        in_dim         = in_dim,
        emb_dim        = args.emb_dim,
        hidden_dims    = (256, 128),
        dropout        = 0.1,
        loss_mode      = args.loss_mode,
        global_weight  = 0.3,
        lr             = 3e-4,
        weight_decay   = 1e-4,
        max_epochs     = args.max_epochs,
        patience       = args.patience,
        min_stocks     = 10,
        max_stocks     = 100,
        samples_per_day= 2,
        ckpt_path      = str(TSPRANK_PT),
    )

    if args.train_mode == "rolling":
        te_emb = {d: emb_dict[d] for d in te_X if d in emb_dict}
        bv = trainer.train_rolling(
            test_X_dict = te_emb,
            test_y_dict = te_y,
            train_ratio = 0.6,
            verbose     = True,
        )
        print(
            f"\n  [P2] rolling 训练完成，best_val={bv:.4f}\n"
            f"  ⚠️  训练数据与回测区间存在重叠，评估结果仅供参考"
        )

    elif args.train_mode == "walk_forward":
        te_emb = {d: emb_dict[d] for d in te_X if d in emb_dict}
        val_losses = trainer.train_walk_forward(
            test_X_dict  = te_emb,
            test_y_dict  = te_y,
            train_window = 120,
            val_window   = 30,
            step         = 20,
            verbose      = True,
        )
        print(
            f"\n  [P2 walk-forward] {len(val_losses)} folds, "
            f"mean val_loss={np.mean(val_losses):.4f}"
        )
        # 最终模型：用最近一段数据重训一次
        te_dates = sorted(te_emb.keys())
        n        = len(te_dates)
        split    = max(0, n - 150)
        final_tr = te_dates[split : max(split + 120, n - 30)]
        final_va = te_dates[max(split + 120, n - 30):]
        if len(final_tr) > 10 and len(final_va) > 5:
            trainer._reset_model()
            bv = trainer.train(final_tr, final_va, te_emb, te_y)
            print(f"  最终模型 best_val={bv:.4f}")

    else:
        # standard：train_df 日期训练，val 用 test_df 前 60%
        te_dates_sorted = sorted(te_X.keys())
        split_idx       = int(len(te_dates_sorted) * 0.6)
        val_dates_v2    = te_dates_sorted[:split_idx]

        tr_emb       = {d: emb_dict[d] for d in tr_d if d in emb_dict}
        va_emb       = {d: emb_dict[d] for d in val_dates_v2 if d in emb_dict}
        train_y_std  = {d: all_y[d] for d in tr_d if d in all_y}
        val_y_std    = {d: te_y[d]  for d in val_dates_v2 if d in te_y}

        combined_X      = {**tr_emb, **va_emb}
        combined_y      = {**train_y_std, **val_y_std}
        train_dates_std = [d for d in tr_d if d in tr_emb]
        val_dates_std   = [d for d in val_dates_v2 if d in va_emb]

        print(
            f"  [standard] 训练日={len(train_dates_std)}  "
            f"验证日={len(val_dates_std)}（test_df 前60%）"
        )
        bv = trainer.train(
            train_dates_std, val_dates_std, combined_X, combined_y
        )
        print(f"  best_val={bv:.4f}")

    return trainer


def step_load_tsprank(in_dim, args):
    """Step 5（加载）：从 checkpoint 加载 TSPRank。"""
    trainer = TSPRankTrainer(
        in_dim      = in_dim,
        emb_dim     = args.emb_dim,
        hidden_dims = (256, 128),
        ckpt_path   = str(TSPRANK_PT),
    )
    trainer.load(str(TSPRANK_PT))
    return trainer


def step_train_tsprank_sector(
    args,
    emb_dict      : dict,
    all_y         : dict,
    all_s         : dict,
    te_X          : dict,
    te_y          : dict,
    te_s          : dict,
    tr_d          : list,
    va_d          : list,
    in_dim        : int,
    code_to_sector: dict,
    backbone,
    sector_dict   : dict,    # Bug2 修复：接受 PIT sector_dict，传给 train_sector
):
    """
    Step 5（sector 分组版）：行业分组 + SCIP solver-in-loop 训练。

    路由规则（与非 sector 版相同的 train_mode）：
      rolling     → test_df 前 60% 作训练，后 40% 验证
      walk_forward→ 严格无泄露 walk-forward（sector 版）
      standard    → train_df(2013-2019) 训练，test_df 前 60% 验证
    """
    use_scip     = not getattr(args, "no_scip", False)
    max_sector_n = getattr(args, "max_sector_n", 25)
    solver_tag   = "SCIP" if use_scip else "Or-opt"

    print(
        f"\n[Step 5-sector] Sector TSPRank 训练 "
        f"(loss={args.loss_mode}, solver={solver_tag}, "
        f"max_sector_n={max_sector_n}, train_mode={args.train_mode})"
    )

    trainer = TSPRankTrainer(
        in_dim         = in_dim,
        emb_dim        = args.emb_dim,
        hidden_dims    = (256, 128),
        dropout        = 0.1,
        loss_mode      = args.loss_mode,
        global_weight  = 0.3,
        lr             = 3e-4,
        weight_decay   = 1e-4,
        max_epochs     = args.max_epochs,
        patience       = args.patience,
        min_stocks     = 3,          # sector 模式每行业最少3只
        max_stocks     = max_sector_n,
        samples_per_day= 1,          # sector 模式每天处理一次（按行业分组已提供多样性）
        ckpt_path      = str(TSPRANK_PT),
    )

    if args.train_mode == "rolling":
        te_emb   = {d: emb_dict[d] for d in te_X  if d in emb_dict}
        te_stock = {d: all_s[d]    for d in te_emb if d in all_s}
        te_y_use = {d: te_y[d]     for d in te_emb if d in te_y}

        test_dates = sorted(te_emb.keys())
        split_idx  = int(len(test_dates) * 0.6)
        tr_dates   = test_dates[:split_idx]
        va_dates   = test_dates[split_idx:]

        print(
            f"  rolling: train={len(tr_dates)} days  val={len(va_dates)} days\n"
            f"  ⚠️  训练数据与回测区间存在重叠（数据泄露风险）"
        )
        bv = trainer.train_sector(
            train_dates  = tr_dates,
            val_dates    = va_dates,
            emb_dict     = te_emb,
            y_dict       = te_y_use,
            stock_dict   = te_stock,
            sector_dict  = sector_dict,   # Bug2 修复：传 PIT sector_dict
            backbone     = backbone,
            max_sector_n = max_sector_n,
            use_scip     = use_scip,
        )
        print(f"  rolling 完成, best_val={bv:.4f}")

    elif args.train_mode == "walk_forward":
        te_emb   = {d: emb_dict[d] for d in te_X  if d in emb_dict}
        te_stock = {d: all_s[d]    for d in te_emb if d in all_s}
        te_y_use = {d: te_y[d]     for d in te_emb if d in te_y}

        test_dates   = sorted(te_emb.keys())
        n            = len(test_dates)
        train_window = 120
        val_window   = 30
        step         = 20
        val_losses   = []
        fold         = 0
        start        = 0

        while start + train_window + val_window <= n:
            t_dates = test_dates[start : start + train_window]
            v_dates = test_dates[start + train_window : start + train_window + val_window]
            print(
                f"\n  [walk-forward fold {fold+1}] "
                f"train {t_dates[0].date()}~{t_dates[-1].date()} | "
                f"val {v_dates[0].date()}~{v_dates[-1].date()}"
            )
            trainer._reset_model()
            bv = trainer.train_sector(
                train_dates  = t_dates,
                val_dates    = v_dates,
                emb_dict     = te_emb,
                y_dict       = te_y_use,
                stock_dict   = te_stock,
                sector_dict  = sector_dict,   # Bug2 修复
                backbone     = backbone,
                max_sector_n = max_sector_n,
                use_scip     = use_scip,
            )
            val_losses.append(bv)
            fold  += 1
            start += step

        print(f"\n  walk-forward 完成: {fold} folds, mean val={np.mean(val_losses):.4f}")

        # 最终模型：最近 150 天重训
        split    = max(0, n - 150)
        final_tr = test_dates[split : max(split + 120, n - 30)]
        final_va = test_dates[max(split + 120, n - 30):]
        if len(final_tr) > 5 and len(final_va) > 3:
            trainer._reset_model()
            bv = trainer.train_sector(
                final_tr, final_va, te_emb, te_y_use,
                te_stock, sector_dict, backbone,   # Bug2 修复
                max_sector_n=max_sector_n, use_scip=use_scip,
            )
            print(f"  最终模型 best_val={bv:.4f}")

    else:
        # standard：train_df + test_df 前 60% 验证
        te_dates_sorted = sorted(te_X.keys())
        split_idx       = int(len(te_dates_sorted) * 0.6)
        val_dates_v2    = te_dates_sorted[:split_idx]

        tr_emb   = {d: emb_dict[d] for d in tr_d if d in emb_dict}
        va_emb   = {d: emb_dict[d] for d in val_dates_v2 if d in emb_dict}
        tr_stock = {d: all_s[d]    for d in tr_d if d in all_s}
        va_stock = {d: all_s[d]    for d in val_dates_v2 if d in all_s}
        tr_y_use = {d: all_y[d]    for d in tr_d if d in all_y}
        va_y_use = {d: te_y[d]     for d in val_dates_v2 if d in te_y}

        combined_emb   = {**tr_emb,   **va_emb}
        combined_y     = {**tr_y_use, **va_y_use}
        combined_stock = {**tr_stock, **va_stock}
        tr_dates_use   = [d for d in tr_d        if d in tr_emb]
        va_dates_use   = [d for d in val_dates_v2 if d in va_emb]

        print(
            f"  standard: train={len(tr_dates_use)} days  "
            f"val={len(va_dates_use)} days（test_df 前60%）"
        )
        bv = trainer.train_sector(
            train_dates  = tr_dates_use,
            val_dates    = va_dates_use,
            emb_dict     = combined_emb,
            y_dict       = combined_y,
            stock_dict   = combined_stock,
            sector_dict  = sector_dict,   # Bug2 修复
            backbone     = backbone,
            max_sector_n = max_sector_n,
            use_scip     = use_scip,
        )
        print(f"  best_val={bv:.4f}")

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    use_or_opt = not args.no_or_opt

    print(f"\n{'=' * 65}")
    print(
        f"  TSPRank v2.2 | mode={args.mode} "
        f"embed={args.embed_mode} "
        f"infer={args.infer_mode}"
    )
    if args.mode in ("train_tsprank", "full"):
        print(f"  loss={args.loss_mode}  train_mode={args.train_mode}")
    if getattr(args, "sector_mode", False):
        scip_tag = "Or-opt" if getattr(args, "no_scip", False) else "SCIP"
        print(f"  sector_mode=ON  max_sector_n={args.max_sector_n}  solver={scip_tag}")
    print(f"  device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'=' * 65}\n")

    # ── 数据加载 ────────────────────────────────────────────────────────────
    (
        train_df, valid_df, test_df,
        tr_d, tr_X, tr_y, tr_s,
        va_d, va_X, va_y, va_s,
        te_d, te_X, te_y, te_s,
        all_X, all_y, all_s,
    ) = step_load_data(args)

    embed_mode = args.embed_mode
    if embed_mode == "leaf":
        print("[main] ⚠️  leaf 模式已废弃，自动切换为 combined")
        embed_mode = "combined"

    # ── Backbone ─────────────────────────────────────────────────────────────
    backbone = None
    if args.mode in ("full", "train_backbone"):
        backbone = step_train_backbone(tr_X, tr_y, va_X, va_y)
    else:
        backbone = step_load_backbone(args, embed_mode)

    if args.mode == "train_backbone":
        print("[main] Backbone done.")
        return

    if backbone is not None and len(te_d) > 0:
        _quick_ic_check(backbone, te_X, te_y, te_d, n_check=20)

    # ── 行业映射（v2.2 sector_mode）────────────────────────────────────────
    code_to_sector: Optional[dict] = None
    sector_dict   : Optional[dict] = None
    use_sector = getattr(args, "sector_mode", False)

    if use_sector:
        print("\n[Step 2.5] Loading industry mapping (point-in-time)...")
        industry_csv = getattr(args, "industry_csv", None)

        # 点位一致版（训练用）：build_sector_dict 接受 DataFrame，按预测日回溯最近快照
        industry_df  = load_industry_df(path=industry_csv)
        all_s_merged = {**tr_s, **va_s, **te_s}
        sector_dict  = build_sector_dict(all_s_merged, industry_df)

        # 静态最新快照（推理/回测用）：仅用于当前时间点的行业查询
        code_to_sector = load_industry_mapping(path=industry_csv)

        print(
            f"  PIT sector_dict: {len(sector_dict)} 日期 | "
            f"推理用静态映射: {len(code_to_sector)} 只股票"
        )

    # ── Embeddings ───────────────────────────────────────────────────────────
    need_emb = (
        args.mode in ("full", "train_tsprank") or
        (args.mode in ("backtest", "predict") and
         args.infer_mode in ("tsprank", "tsprank_sector"))
    )
    emb_dict = {}
    in_dim   = (159 if embed_mode == "combined" and backbone is not None else 158)

    if need_emb:
        emb_dict, in_dim = step_build_emb(all_X, all_y, backbone, embed_mode)

    # ── TSPRank 训练 ─────────────────────────────────────────────────────────
    trainer = None
    if args.mode in ("full", "train_tsprank"):
        if use_sector and code_to_sector is not None:
            trainer = step_train_tsprank_sector(
                args, emb_dict, all_y, all_s, te_X, te_y, te_s,
                tr_d, va_d, in_dim, code_to_sector, backbone,
                sector_dict = sector_dict,   # Bug2 修复：传 PIT sector_dict
            )
        else:
            trainer = step_train_tsprank(
                args, emb_dict, all_y, te_X, te_y, tr_d, va_d, in_dim
            )
    elif args.mode in ("backtest", "predict"):
        if args.infer_mode in ("tsprank", "tsprank_sector"):
            if TSPRANK_PT.exists():
                trainer = step_load_tsprank(in_dim, args)
            else:
                print(
                    "[main] ⚠️  tsprank.pt 不存在，"
                    "infer_mode 自动降级为 backbone"
                )
                args.infer_mode = "backbone"

    if args.mode == "train_tsprank":
        print("[main] TSPRank done.")
        return

    # ── Backtest ─────────────────────────────────────────────────────────────
    tsprank_model = trainer.model if trainer is not None else None

    if args.mode in ("full", "backtest"):
        print(f"\n[Step 6] Backtest (recent_weeks={args.recent_weeks})...")
        run_backtest(
            backbone          = backbone,
            tsprank           = tsprank_model,
            test_df           = test_df,
            embed_mode        = embed_mode,
            infer_mode        = args.infer_mode,
            top_k             = args.top_k,
            total_weight      = args.total_weight,
            recent_weeks      = args.recent_weeks,
            diversity_penalty = args.diversity_penalty,
            penalty_window    = args.penalty_window,
            use_or_opt        = use_or_opt,
            qlib_pred_pkl     = args.qlib_pred_pkl,
            save_equity_curve = not args.no_equity_curve,
            # v2.2 行业分组参数
            code_to_sector    = code_to_sector,
            sector_blend      = getattr(args, "sector_blend", 0.5),
            use_scip          = not getattr(args, "no_scip", False),
        )
        print("[main] Backtest saved → backtest_results.csv")

    # ── Predict Latest ───────────────────────────────────────────────────────
    if args.mode in ("full", "predict"):
        print("\n[Step 7] Predicting latest week...")
        _predict_latest(
            backbone, tsprank_model, te_X, te_s, all_X, all_s,
            embed_mode, args.infer_mode,
            args.top_k, args.total_weight, use_or_opt,
            args.qlib_pred_pkl,
            code_to_sector = code_to_sector,
            sector_blend   = getattr(args, "sector_blend", 0.5),
            use_scip       = not getattr(args, "no_scip", False),
        )

    print("\n[main] All done.")


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _quick_ic_check(
    backbone: DEBackbone,
    te_X    : dict,
    te_y    : dict,
    te_d    : list,
    n_check : int = 20,
) -> None:
    """快速诊断 backbone IC（取测试集前 n_check 天）。"""
    from scipy.stats import spearmanr
    ics = []
    for d in te_d[:n_check]:
        if d not in te_X:
            continue
        ic, _ = spearmanr(backbone.predict_score(te_X[d]), te_y[d])
        if not np.isnan(ic):
            ics.append(ic)
    if ics:
        ics = np.array(ics)
        print(
            f"[main] Backbone IC check (前{n_check}天): "
            f"mean={ics.mean():.4f}  std={ics.std():.4f}  "
            f"IR={ics.mean() / (ics.std() + 1e-8):.4f}  "
            f"pos_rate={np.mean(ics > 0):.1%}"
        )
        if backbone.is_degraded():
            print(
                "[main] ⚠️  backbone 使用单位权重（旧格式），"
                "IC 从 0.10 退化到 ≈ 0.06\n"
                "     建议通过 P1 接入小组 pred.pkl 或重新训练 backbone"
            )


def _predict_latest(
    backbone      : Optional[DEBackbone],
    tsprank       : Optional[TSPRankModel],
    te_X          : dict,
    te_s          : dict,
    all_X         : dict,
    all_s         : dict,
    embed_mode    : str,
    infer_mode    : str,
    top_k         : int,
    total_weight  : float,
    use_or_opt    : bool,
    qlib_pred_pkl : Optional[str],
    code_to_sector: Optional[dict] = None,
    sector_blend  : float = 0.5,
    use_scip      : bool  = True,
) -> None:
    """预测最新一周并写入 output/result.csv。"""
    from tsprank_core.backtest import (
        _strip_prefix, _build_input, _infer_backbone,
        _infer_qlib_pred, _infer_tsprank, _infer_tsprank_sector,
    )

    dates = sorted(all_X.keys())
    if not dates:
        print("[predict] 无可用数据")
        return
    pred_date = dates[-1]
    x_np      = all_X[pred_date]
    codes     = all_s[pred_date]
    print(f"[predict] pred_date={pred_date.date()}  N={len(codes)}")

    # ── 按模式路由推理 ──────────────────────────────────────────────────────
    if infer_mode == "backbone" or backbone is None:
        if backbone is None:
            print("[predict] 无 backbone，跳过")
            return
        top5_codes = _infer_backbone(x_np, codes, backbone, top_k)

    elif infer_mode == "qlib_pred":
        try:
            from tsprank_core.qlib_pred_loader import QlibPredLoader
            loader = QlibPredLoader(
                pred_pkl_path=qlib_pred_pkl,
                mlruns_root=(
                    "~/quant-stock-forecast/qlib-csi-300/mlruns"
                    if qlib_pred_pkl is None else None
                ),
                run_name_keyword="w4",
            )
            top5_codes = _infer_qlib_pred(
                [_strip_prefix(c) for c in codes],
                pred_date, loader, top_k,
            )
        except Exception as e:
            print(f"[predict] qlib_pred 失败 ({e})，降级为 backbone")
            top5_codes = _infer_backbone(x_np, codes, backbone, top_k)

    elif infer_mode == "tsprank_sector":
        if tsprank is None:
            print("[predict] tsprank_sector 需要 tsprank 模型，降级为 backbone")
            top5_codes = _infer_backbone(x_np, codes, backbone, top_k)
        elif code_to_sector is None:
            print("[predict] tsprank_sector 需要 code_to_sector，降级为 tsprank")
            # Bug1 修复：_build_tsprank_S 从未定义，内联展开
            emb = _build_input(x_np, backbone, embed_mode)
            x_t = torch.from_numpy(emb).float().to(DEVICE)
            tsprank.eval()
            with torch.no_grad():
                S_t, _ = tsprank.forward(x_t)
            tour       = TSPSolver.solve(
                S_t.cpu().numpy().astype(np.float64), use_or_opt=use_or_opt
            )
            top5_codes = [_strip_prefix(codes[i]) for i in tour[:top_k]]
        else:
            top5_codes, _ = _infer_tsprank_sector(
                x_np, codes, backbone, tsprank,
                embed_mode, code_to_sector, top_k,
                blend_alpha=sector_blend,
                use_scip=use_scip,
            )

    elif tsprank is not None:  # "tsprank"
        emb = _build_input(x_np, backbone, embed_mode)
        x_t = torch.from_numpy(emb).float().to(DEVICE)
        tsprank.eval()
        with torch.no_grad():
            S_t, _ = tsprank.forward(x_t)
        tour       = TSPSolver.solve(
            S_t.cpu().numpy().astype(np.float64), use_or_opt=use_or_opt
        )
        top5_codes = [_strip_prefix(codes[i]) for i in tour[:top_k]]
    else:
        print("[predict] 无可用模型")
        return

    # ── 写结果 ──────────────────────────────────────────────────────────────
    w_each  = round(total_weight / max(len(top5_codes), 1), 6)
    weights = [w_each] * len(top5_codes)
    df      = make_result_csv(top5_codes, weights, top_k=top_k)

    print("[predict] 选股结果:")
    print(df.to_string(index=False))
    print("[predict] output/result.csv 已更新")


if __name__ == "__main__":
    main()
