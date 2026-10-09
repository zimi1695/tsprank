# tsprank_core/backtest.py
"""
周度回测引擎 v2.2

v2.2 新增（对齐论文 Section 5 sector grouping）：
  ① _infer_tsprank_sector()：行业分组 TSPRank 推理
     - 完整截面前向得到 S 矩阵 → 按行业提取子矩阵
     - 每个行业：SCIPSolver 精确求解内部排名
     - 跨行业：blend_alpha × sector_rank + (1-blend_alpha) × backbone_pct
     - 论文设置：sector内 N=5~20 精确排名，backbone 提供全局信号
  ② run_backtest 新增参数 code_to_sector、sector_blend
  ③ infer_mode="tsprank_sector" 路由到 _infer_tsprank_sector
  ④ 移除已过时的 TSPRank 未收敛警告（问题0/0.6 已在 v2.1/v2.2 修复）

v2.0/v2.1 保留内容：
  ① _strip_prefix：剥离 SH/SZ 前缀
  ② _score_inline：手动 groupby 循环（修复 FutureWarning）
  ③ infer_mode=backbone/qlib_pred/tsprank 三种路径
  ④ 多样性惩罚、StrategyEvaluator 集成不变
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .backbone import DEBackbone
from .data_loader import (
    QLIB_CONFIG,
    build_test_csv,
    get_eval_weeks,
    load_stock_data,
    make_result_csv,
    extract_cross_sections,
    _normalize_code,
)
from .tsprank_model import TSPRankModel, TSPSolver, SCIPSolver, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _strip_prefix(code: str) -> str:
    """qlib 格式 SH600000/SZ000001 → 纯 6 位数字字符串。"""
    c = str(code).strip()
    if c.upper().startswith(("SH", "SZ")):
        c = c[2:]
    return c.zfill(6)


def _score_inline(
    codes     : List[str],
    weights   : List[float],
    week_start: pd.Timestamp,
    week_end  : pd.Timestamp,
    stock_df  : pd.DataFrame,
) -> float:
    """
    复现 score_self.py 的评分逻辑：
    得分 = Σ (当周收益率_i × 权重_i)
    收益率 = (最后交易日开盘 - 第一交易日开盘) / 第一交易日开盘
    """
    mask = (
        (stock_df["日期"] >= week_start) &
        (stock_df["日期"] <= week_end)   &
        (stock_df["股票代码"].isin(codes))
    )
    sub  = stock_df[mask].copy()
    rets : Dict[str, float] = {}

    for code, grp in sub.groupby("股票代码"):
        s  = grp.sort_values("日期")
        s0 = s.iloc[0]["开盘"]
        s1 = s.iloc[-1]["开盘"]
        if pd.isna(s0) or s0 == 0:
            rets[str(code)] = 0.0
        else:
            rets[str(code)] = float((s1 - s0) / s0)

    return float(sum(rets.get(c, 0.0) * w for c, w in zip(codes, weights)))


def _build_input(
    x_np      : np.ndarray,
    backbone  : Optional[DEBackbone],
    embed_mode: str,
) -> np.ndarray:
    """
    构建 TSPRank 输入特征。
    "feature"  → Alpha158 (N, 158)
    "combined" → Alpha158 + backbone rank_pct (N, 159)
    """
    if embed_mode == "combined" and backbone is not None:
        pred_score = backbone.predict_score(x_np)
        rank_raw   = np.argsort(np.argsort(pred_score)).astype(np.float32)
        n          = len(rank_raw)
        rank_pct   = (rank_raw / max(n - 1, 1) - 0.5) * 2
        return np.concatenate(
            [x_np.astype(np.float32), rank_pct[:, None]], axis=1
        )
    return x_np.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 四种推理模式
# ─────────────────────────────────────────────────────────────────────────────

def _infer_backbone(
    x_np    : np.ndarray,
    codes   : List[str],
    backbone: DEBackbone,
    top_k   : int,
) -> List[str]:
    """P0：Backbone 直接 Top-K 排序。"""
    pred_score = backbone.predict_score(x_np)
    top_idx    = np.argsort(pred_score)[::-1][:top_k]
    return [_strip_prefix(codes[i]) for i in top_idx]


def _infer_qlib_pred(
    codes      : List[str],
    pred_date  : pd.Timestamp,
    qlib_loader,
    top_k      : int,
) -> List[str]:
    """P1：接入小组 pred.pkl（IC≈0.10）。"""
    return qlib_loader.get_top_k(pred_date, codes, k=top_k)


def _infer_tsprank(
    x_np             : np.ndarray,
    codes            : List[str],
    backbone         : Optional[DEBackbone],
    tsprank          : TSPRankModel,
    embed_mode       : str,
    top_k            : int,
    use_or_opt       : bool,
    diversity_penalty: float,
    history_codes    : List[List[str]],
    penalty_window   : int,
) -> Tuple[List[str], bool]:
    """全截面 TSPRank（Or-opt）推理，原 v2.0 默认模式。"""
    emb = _build_input(x_np, backbone, embed_mode)
    tsprank.eval()
    x_t = torch.from_numpy(emb).float().to(DEVICE)
    with torch.no_grad():
        S_t, _ = tsprank.forward(x_t)
    S_np = S_t.cpu().numpy()

    S_use           = S_np.copy()
    penalty_applied = False
    if diversity_penalty > 0 and len(history_codes) > 0:
        recent_hist    = history_codes[-penalty_window:]
        past_codes_set = set(c for week in recent_hist for c in week)
        for ci, code in enumerate(codes):
            if _strip_prefix(code) in past_codes_set:
                S_use[:, ci] -= diversity_penalty
                S_use[ci, :] -= diversity_penalty * 0.5
                penalty_applied = True

    tour      = TSPSolver.solve(S_use, use_or_opt=use_or_opt)
    top_codes = [_strip_prefix(codes[i]) for i in tour[:top_k]]
    return top_codes, penalty_applied


def _infer_tsprank_sector(
    x_np          : np.ndarray,
    codes         : List[str],
    backbone      : Optional[DEBackbone],
    tsprank       : TSPRankModel,
    embed_mode    : str,
    code_to_sector: Dict[str, str],
    top_k         : int,
    blend_alpha   : float = 0.5,
    use_scip      : bool  = True,
    diversity_penalty: float = 0.0,
    history_codes : List[List[str]] = (),
    penalty_window: int = 4,
) -> Tuple[List[str], bool]:
    """
    行业分组 TSPRank 推理（v2.2 新增，对齐论文 Section 5）。

    算法：
      1. 完整截面前向传播 → S 全矩阵（N×N）
      2. 按行业提取 S 子矩阵，SCIPSolver 精确求行业内排名
         C39(N=46) 用推理模式（MAX_INFER_N=50，time_limit=5s）直接 SCIP
      3. sector_score：行业内排名归一化到 [1/k, 1.0]
      4. backbone_pct：全局 backbone 排名百分位 ∈ [0, 1]
      5. final_score = blend_alpha × sector_score + (1-blend_alpha) × backbone_pct
      6. Top-K by final_score

    参数
    ----
    blend_alpha : sector 排名得分权重（默认 0.5，即 sector+backbone 各占一半）
    use_scip    : False 时行业内用 TSPSolver.solve（Or-opt）代替 SCIP

    返回
    ----
    (top_codes, penalty_applied)
    """
    N      = len(codes)
    device = DEVICE

    # ── Step 1: 完整截面前向传播 ─────────────────────────────────────────────
    emb = _build_input(x_np, backbone, embed_mode)
    tsprank.eval()
    x_t = torch.from_numpy(emb).float().to(device)
    with torch.no_grad():
        S_t, _ = tsprank.forward(x_t)
    S_np = S_t.cpu().numpy().astype(np.float64)

    # ── Step 2: Backbone 全局百分位（跨行业比较用）────────────────────────────
    if backbone is not None:
        bb_scores = backbone.predict_score(x_np)
    else:
        bb_scores = np.zeros(N)
    bb_pct = np.argsort(np.argsort(bb_scores)).astype(float) / max(N - 1, 1)

    # ── Step 3: 按行业分组，SCIP 精确求行业内排名 ─────────────────────────────
    clean_codes    = [_normalize_code(str(c)) for c in codes]
    sector_groups  : Dict[str, List[int]] = defaultdict(list)
    for i, clean in enumerate(clean_codes):
        sector = code_to_sector.get(clean, "unknown")
        sector_groups[sector].append(i)

    sector_scores = np.zeros(N)

    for sector, indices in sector_groups.items():
        if sector == "unknown":
            continue
        k = len(indices)
        # 论文：N<5 的行业跳过（排名无意义），这些股票的 sector_score 保持 0
        # 最终得分由 backbone_pct 决定，不会被错误的小行业排名干扰
        if k < 5:
            continue
        if k == 0:
            continue

        idx_arr = np.array(indices)
        S_sub   = S_np[np.ix_(idx_arr, idx_arr)]

        if use_scip:
            # 推理模式：MAX_INFER_N=50，time_limit=5s，C39(N=46) 直接 SCIP
            local_tour = SCIPSolver.solve_max_path(
                S_sub, time_limit=5.0, for_infer=True
            )
        else:
            local_tour = TSPSolver.solve(S_sub)

        # 行业内排名归一化：第1名 = 1.0，第k名 = 1/k
        for rank, local_idx in enumerate(local_tour):
            sector_scores[indices[local_idx]] = (k - rank) / k

    # ── Step 4 & 5: 混合得分 + 多样性惩罚 → Top-K ─────────────────────────────
    final_scores    = blend_alpha * sector_scores + (1.0 - blend_alpha) * bb_pct
    penalty_applied = False

    # P0.3 修复：惩罚作用于 final_scores 而非 sector_scores。
    # 原来对 sector_scores 惩罚，经 blend_alpha 混合后效果减半；
    # sector_blend=0（纯 backbone）时惩罚完全失效。
    if diversity_penalty > 0 and len(history_codes) > 0:
        recent_hist    = list(history_codes)[-penalty_window:]
        past_codes_set = set(c for week in recent_hist for c in week)
        for ci, code in enumerate(codes):
            if _strip_prefix(str(code)) in past_codes_set:
                final_scores[ci] *= (1.0 - diversity_penalty)
                penalty_applied = True

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_codes   = [_strip_prefix(str(codes[i])) for i in top_indices]

    return top_codes, penalty_applied


# ─────────────────────────────────────────────────────────────────────────────
# 主回测函数
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    backbone          : Optional[DEBackbone],
    tsprank           : Optional[TSPRankModel],
    test_df           : pd.DataFrame,
    embed_mode        : str   = "combined",
    infer_mode        : str   = "tsprank",
    top_k             : int   = 5,
    total_weight      : float = 1.0,
    recent_weeks      : int   = 12,
    diversity_penalty : float = 0.0,
    penalty_window    : int   = 4,
    use_or_opt        : bool  = True,
    qlib_pred_pkl     : Optional[str]  = None,
    result_csv        : str   = "output/result.csv",
    test_csv_path     : str   = "data/test.csv",
    save_equity_curve : bool  = True,
    # ── v2.2 新增：行业分组模式 ────────────────────────────────────────────
    code_to_sector    : Optional[Dict[str, str]] = None,
    sector_blend      : float = 0.5,
    use_scip          : bool  = True,
) -> pd.DataFrame:
    """
    周度回测引擎。

    infer_mode 支持：
      "backbone"       (P0) - Backbone Top-K，消除 TSPRank 噪声
      "qlib_pred"      (P1) - 接入小组 pred.pkl（IC≈0.10）
      "tsprank"             - 全截面 TSP 路径（Or-opt）
      "tsprank_sector" (v2.2) - 行业分组 SCIP 精确推理（论文对齐）

    Parameters（v2.2 新增）
    -----------------------
    code_to_sector : {clean_6digit: sector_label}（load_industry_mapping 返回值）
                     infer_mode="tsprank_sector" 时必须提供
    sector_blend   : sector_rank 与 backbone_pct 的混合权重（默认 0.5）
    use_scip       : sector 模式下是否使用 SCIP（True=精确，False=Or-opt）
    """
    print(
        f"\n[backtest] infer_mode={infer_mode}  "
        f"embed_mode={embed_mode}  "
        f"diversity_penalty={diversity_penalty}"
    )

    if infer_mode == "tsprank_sector":
        if code_to_sector is None:
            print(
                "  ⚠️  tsprank_sector 模式需要 code_to_sector，"
                "请通过 load_industry_mapping() 加载后传入。\n"
                "       自动降级为 tsprank 模式。"
            )
            infer_mode = "tsprank"
        else:
            solver_tag = "SCIP" if use_scip else "Or-opt"
            print(
                f"  [sector] {len(code_to_sector)} 只股票行业映射已加载，"
                f"blend_alpha={sector_blend}，solver={solver_tag}"
            )

    if backbone is not None and backbone.is_degraded():
        print(
            "  ⚠️  backbone.pkl 使用单位权重（旧格式），IC≈0.06（退化）。\n"
            "       建议通过 P1 接入小组 pred.pkl（IC≈0.10）。"
        )

    # ── 加载价格数据与评估周 ────────────────────────────────────────────────
    stock_df = load_stock_data()
    weeks    = get_eval_weeks(stock_df=stock_df, recent_weeks=recent_weeks)
    print(f"[backtest] 评估周数={len(weeks)}")

    # ── 提取测试集截面 ───────────────────────────────────────────────────────
    _, X_te, y_te, codes_te = extract_cross_sections(test_df)
    date_set_te = set(X_te.keys())

    # ── P1：初始化 QlibPredLoader ────────────────────────────────────────────
    qlib_loader = None
    if infer_mode == "qlib_pred":
        try:
            from .qlib_pred_loader import QlibPredLoader
            if qlib_pred_pkl:
                qlib_loader = QlibPredLoader(pred_pkl_path=qlib_pred_pkl)
            else:
                qlib_loader = QlibPredLoader(
                    mlruns_root="~/quant-stock-forecast/qlib-csi-300/mlruns",
                    run_name_keyword="w4",
                )
            if qlib_loader.is_loaded:
                info = qlib_loader.diagnose()
                print(
                    f"  [P1] pred.pkl 已加载: "
                    f"{info.get('n_dates')} 日, "
                    f"{info.get('n_instruments')} 股票"
                )
            else:
                print("  ⚠️  pred.pkl 加载失败，降级为 backbone")
                infer_mode = "backbone"
        except ImportError:
            print("  ⚠️  qlib_pred_loader 不存在，降级为 backbone")
            infer_mode = "backbone"

    # ── 主循环 ──────────────────────────────────────────────────────────────
    results       : List[dict] = []
    history_codes : List[List[str]] = []

    for wi, (ws, we) in enumerate(weeks):
        print(f"\n── 周 {wi + 1}/{len(weeks)}: {ws.date()} ~ {we.date()} ──")

        # 找最近预测日（严格 < ws，修复前视偏差）
        pred_date = None
        for d in sorted(date_set_te, reverse=True):
            if d < ws:
                pred_date = d
                break

        if pred_date is None or pred_date not in X_te:
            print("  [跳过] 无预测日数据")
            results.append({
                "week_start": ws, "week_end": we,
                "weekly_return": 0.0, "codes": [], "weights": [],
                "skipped": True, "infer_mode": infer_mode,
            })
            continue

        x_np  = X_te[pred_date]
        codes = codes_te[pred_date]
        N     = len(codes)
        print(f"  预测日={pred_date.date()}  截面大小={N}")

        if N < top_k:
            print(f"  [跳过] 截面太小 ({N} < {top_k})")
            results.append({
                "week_start": ws, "week_end": we,
                "weekly_return": 0.0, "codes": [], "weights": [],
                "skipped": True, "infer_mode": infer_mode,
            })
            continue

        # ── 推理 ─────────────────────────────────────────────────────────────
        penalty_applied = False

        if infer_mode == "backbone":
            if backbone is None:
                top5_codes = [_strip_prefix(codes[i]) for i in range(top_k)]
            else:
                top5_codes = _infer_backbone(x_np, codes, backbone, top_k)

        elif infer_mode == "qlib_pred":
            top5_codes = _infer_qlib_pred(
                [_strip_prefix(c) for c in codes],
                pred_date, qlib_loader, top_k,
            )

        elif infer_mode == "tsprank_sector":
            if tsprank is None:
                print("  [错误] tsprank_sector 但 tsprank 未加载，降级为 backbone")
                top5_codes = _infer_backbone(x_np, codes, backbone, top_k)
            else:
                top5_codes, penalty_applied = _infer_tsprank_sector(
                    x_np, codes, backbone, tsprank,
                    embed_mode, code_to_sector, top_k,
                    blend_alpha=sector_blend,
                    use_scip=use_scip,
                    diversity_penalty=diversity_penalty,
                    history_codes=history_codes,
                    penalty_window=penalty_window,
                )

        else:  # "tsprank"（默认）
            if tsprank is None:
                print("  [错误] infer_mode=tsprank 但 tsprank 未加载，降级为 backbone")
                top5_codes = _infer_backbone(x_np, codes, backbone, top_k)
            else:
                top5_codes, penalty_applied = _infer_tsprank(
                    x_np, codes, backbone, tsprank,
                    embed_mode, top_k, use_or_opt,
                    diversity_penalty, history_codes, penalty_window,
                )

        # ── 权重与文件写入 ────────────────────────────────────────────────────
        n_selected = len(top5_codes)
        w_each     = round(total_weight / max(n_selected, 1), 6)
        weights    = [w_each] * n_selected
        print(f"  选出: {top5_codes}  等权: {w_each:.4f}×{n_selected}")

        make_result_csv(top5_codes, weights, output_path=str(result_csv))
        build_test_csv(ws, we, stock_df=stock_df, output_path=str(test_csv_path))

        # ── 评分 ─────────────────────────────────────────────────────────────
        score = _score_inline(top5_codes, weights, ws, we, stock_df)
        print(f"  本周得分: {score:+.4f} ({score * 100:+.2f}%)")

        history_codes.append(top5_codes)
        results.append({
            "week_start"              : ws,
            "week_end"                : we,
            "weekly_return"           : score,
            "codes"                   : top5_codes,
            "weights"                 : weights,
            "pred_date"               : pred_date,
            "infer_mode"              : infer_mode,
            "diversity_penalty_applied": penalty_applied,
            "skipped"                 : False,
        })

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    _print_summary(results, diversity_penalty, infer_mode)
    result_df = pd.DataFrame(results)
    result_df.to_csv("backtest_results.csv", index=False)
    print("[backtest] 结果已保存 → backtest_results.csv")
    _run_evaluator(results, infer_mode, save_equity_curve)
    return result_df


def _print_summary(
    results          : List[dict],
    diversity_penalty: float,
    infer_mode       : str,
) -> None:
    valid = [r for r in results if not r.get("skipped", False)]
    if not valid:
        print("[backtest] 无有效周")
        return

    scores = [r["weekly_return"] for r in valid]
    n      = len(scores)

    print(f"\n{'=' * 60}")
    print(f"  回测汇总 [infer_mode={infer_mode}]")
    print(f"{'=' * 60}")
    print(f"  评估周数:     {n}")
    print(f"  均周得分:     {np.mean(scores):+.4f} ({np.mean(scores) * 100:+.2f}%)")
    print(f"  累计得分:     {np.sum(scores):+.4f} ({np.sum(scores) * 100:+.2f}%)")
    print(f"  正收益周数:   {sum(s > 0 for s in scores)}/{n} "
          f"({sum(s > 0 for s in scores) / n:.1%})")
    print(f"  最优周:       {max(scores):+.4f}")
    print(f"  最差周:       {min(scores):+.4f}")
    if diversity_penalty > 0:
        pen_n = sum(1 for r in valid if r.get("diversity_penalty_applied"))
        print(f"\n  多样性惩罚应用周数: {pen_n}/{n}")
    print(f"{'=' * 60}")


def _run_evaluator(
    results          : List[dict],
    infer_mode       : str,
    save_equity_curve: bool,
) -> None:
    try:
        from .evaluator import StrategyEvaluator
    except ImportError:
        print("[backtest] evaluator 模块不存在，跳过完整指标计算")
        return

    valid = [r for r in results if not r.get("skipped", False)]
    if len(valid) < 2:
        return

    ev = StrategyEvaluator(
        weekly_returns=[r["weekly_return"] for r in valid],
        week_labels   =[str(r["week_start"].date()) for r in valid],
        strategy_name =f"TSPRank [{infer_mode}]",
    )
    report = ev.full_report()
    ev.print_report(report)
    if save_equity_curve:
        curve_path = Path("backtest_results_equity_curve.png")
        ev.plot_cumulative(report, save_path=curve_path)
        print(f"[backtest] 净值曲线 → {curve_path}")
