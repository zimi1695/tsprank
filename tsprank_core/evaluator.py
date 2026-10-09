# tsprank_core/evaluator.py
"""
P3 方案：完整量化评估指标模块。
对齐小组仓库 lib/evaluation_metrics.py 的指标体系，
同时兼容本项目的 backtest_results.csv 格式。

指标列表：
  - 周度收益率序列
  - 累计收益率
  - 年化收益率
  - 年化波动率
  - Sharpe Ratio（年化，无风险利率=0）
  - 最大回撤 (MDD)
  - 胜率（相对空仓 > 0）
  - 平均盈利周 / 平均亏损周
  - Calmar Ratio
  - 周 IC（若提供 backbone scores）
  - 与沪深300对比（若提供基准收益率）

用法：
    from tsprank_core.evaluator import StrategyEvaluator
    ev = StrategyEvaluator.from_backtest_csv("backtest_results.csv")
    report = ev.full_report()
    ev.print_report(report)
    ev.plot_cumulative(report, save_path="equity_curve.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 交易周数/年（A股）
WEEKS_PER_YEAR = 52


# ─────────────────────────────────────────────────────────────────────────────
# 核心统计函数（无副作用，便于单元测试）
# ─────────────────────────────────────────────────────────────────────────────

def cumulative_return(returns: np.ndarray) -> np.ndarray:
    """累计净值曲线（从 1.0 开始）。"""
    return np.cumprod(1.0 + returns)


def annualized_return(returns: np.ndarray,
                      periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """年化收益率（几何平均）。"""
    n = len(returns)
    if n == 0:
        return float("nan")
    total = np.prod(1.0 + returns)
    return float(total ** (periods_per_year / n) - 1.0)


def annualized_volatility(returns: np.ndarray,
                          periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """年化波动率。"""
    if len(returns) < 2:
        return float("nan")
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: np.ndarray,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """年化 Sharpe Ratio（无风险利率可调，默认 0）。"""
    ann_ret  = annualized_return(returns, periods_per_year)
    ann_vol  = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    最大回撤。

    Returns
    -------
    (mdd, peak_idx, trough_idx)
        mdd: 最大回撤比例（正数，如 0.15 表示 -15%）
        peak_idx: 回撤起点周索引
        trough_idx: 回撤谷底周索引
    """
    if len(returns) == 0:
        return 0.0, 0, 0
    cum = cumulative_return(returns)
    peak_val   = cum[0]
    peak_idx   = 0
    mdd        = 0.0
    mdd_peak   = 0
    mdd_trough = 0
    for i, v in enumerate(cum):
        if v > peak_val:
            peak_val = v
            peak_idx = i
        dd = (peak_val - v) / peak_val
        if dd > mdd:
            mdd        = dd
            mdd_peak   = peak_idx
            mdd_trough = i
    return float(mdd), int(mdd_peak), int(mdd_trough)


def calmar_ratio(returns: np.ndarray,
                 periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """Calmar = 年化收益 / 最大回撤。"""
    ann_ret = annualized_return(returns, periods_per_year)
    mdd, _, _ = max_drawdown(returns)
    if mdd == 0:
        return float("inf") if ann_ret > 0 else float("nan")
    return ann_ret / mdd


def win_rate(returns: np.ndarray) -> float:
    """胜率（周收益 > 0 的比例）。"""
    if len(returns) == 0:
        return float("nan")
    return float(np.sum(returns > 0) / len(returns))


def excess_return_vs_benchmark(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> np.ndarray:
    """相对基准的超额周收益序列。"""
    min_len = min(len(strategy_returns), len(benchmark_returns))
    return strategy_returns[:min_len] - benchmark_returns[:min_len]


def information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = WEEKS_PER_YEAR,
) -> float:
    """Information Ratio = 年化超额收益 / 超额收益年化波动率。"""
    excess = excess_return_vs_benchmark(strategy_returns, benchmark_returns)
    ann_excess = annualized_return(excess, periods_per_year)
    ann_te     = annualized_volatility(excess, periods_per_year)
    if ann_te == 0 or np.isnan(ann_te):
        return float("nan")
    return ann_excess / ann_te


# ─────────────────────────────────────────────────────────────────────────────
# 主评估类
# ─────────────────────────────────────────────────────────────────────────────

class StrategyEvaluator:
    """
    策略评估器。
    
    Parameters
    ----------
    weekly_returns : array-like
        周度策略收益率序列（浮点数，如 0.012 表示 +1.2%）。
    week_labels : list[str] | None
        对应的周标签（如 "2026-01-05"），用于绘图。
    benchmark_returns : array-like | None
        周度基准收益率（如沪深300），长度与 weekly_returns 对齐。
    strategy_name : str
        策略名称，用于报告标题。
    """

    def __init__(
        self,
        weekly_returns: Union[np.ndarray, List[float]],
        week_labels: Optional[List[str]] = None,
        benchmark_returns: Optional[Union[np.ndarray, List[float]]] = None,
        strategy_name: str = "TSPRank Strategy",
    ):
        self.returns   = np.asarray(weekly_returns, dtype=np.float64)
        self.labels    = week_labels or [str(i) for i in range(len(self.returns))]
        self.benchmark = (np.asarray(benchmark_returns, dtype=np.float64)
                          if benchmark_returns is not None else None)
        self.name      = strategy_name

    # ── 类方法构造器 ──────────────────────────────────────────────────────────

    @classmethod
    def from_backtest_csv(
        cls,
        csv_path: Union[str, Path],
        return_col:    str = "weekly_return",
        week_col:      str = "week_start",
        benchmark_col: Optional[str] = "benchmark_return",
        strategy_name: str = "TSPRank Strategy",
        recent_weeks:  Optional[int] = None,
    ) -> "StrategyEvaluator":
        """
        从 backtest_results.csv 构造评估器。
        
        CSV 期望列：
            week_start, weekly_return[, benchmark_return, stocks, weights, ...]
        """
        df = pd.read_csv(csv_path)

        # 兼容列名变体
        for alias in ["weekly_return", "score", "return", "week_return"]:
            if alias in df.columns:
                return_col = alias
                break

        if return_col not in df.columns:
            raise ValueError(f"找不到收益率列，可用列: {df.columns.tolist()}")

        returns = df[return_col].values.astype(float)
        labels  = df[week_col].astype(str).tolist() if week_col in df.columns else None
        bench   = (df[benchmark_col].values.astype(float)
                   if benchmark_col and benchmark_col in df.columns else None)

        if recent_weeks is not None:
            returns = returns[-recent_weeks:]
            if labels:
                labels = labels[-recent_weeks:]
            if bench is not None:
                bench = bench[-recent_weeks:]

        return cls(returns, labels, bench, strategy_name)

    @classmethod
    def from_dict_list(
        cls,
        records: List[Dict],
        return_key:    str = "weekly_return",
        week_key:      str = "week_start",
        benchmark_key: Optional[str] = None,
        strategy_name: str = "TSPRank Strategy",
    ) -> "StrategyEvaluator":
        """从 backtest 返回的字典列表构造评估器（直接对接 run_backtest 输出）。"""
        returns  = np.array([r[return_key] for r in records], dtype=float)
        labels   = [str(r.get(week_key, i)) for i, r in enumerate(records)]
        bench    = (np.array([r[benchmark_key] for r in records], dtype=float)
                    if benchmark_key and benchmark_key in records[0] else None)
        return cls(returns, labels, bench, strategy_name)

    # ── 核心报告 ──────────────────────────────────────────────────────────────

    def full_report(self) -> Dict:
        """
        计算全套指标，返回字典。
        
        Returns
        -------
        dict with keys:
            n_weeks, total_return, annualized_return, annualized_volatility,
            sharpe_ratio, max_drawdown, mdd_start, mdd_end,
            calmar_ratio, win_rate, avg_win, avg_loss,
            best_week, worst_week, mean_weekly_return,
            [excess_*, ir 若有基准]
        """
        r = self.returns

        mdd, pi, ti = max_drawdown(r)
        wins  = r[r > 0]
        losses = r[r < 0]

        report = {
            "strategy_name":        self.name,
            "n_weeks":              int(len(r)),
            "total_return":         float(np.prod(1 + r) - 1),
            "annualized_return":    annualized_return(r),
            "annualized_volatility":annualized_volatility(r),
            "sharpe_ratio":         sharpe_ratio(r),
            "max_drawdown":         mdd,
            "mdd_start_week":       self.labels[pi] if self.labels else pi,
            "mdd_end_week":         self.labels[ti] if self.labels else ti,
            "calmar_ratio":         calmar_ratio(r),
            "win_rate":             win_rate(r),
            "n_win_weeks":          int(np.sum(r > 0)),
            "n_loss_weeks":         int(np.sum(r < 0)),
            "avg_win":              float(wins.mean())  if len(wins)   > 0 else float("nan"),
            "avg_loss":             float(losses.mean()) if len(losses) > 0 else float("nan"),
            "best_week":            float(r.max()) if len(r) > 0 else float("nan"),
            "worst_week":           float(r.min()) if len(r) > 0 else float("nan"),
            "mean_weekly_return":   float(r.mean()),
            "std_weekly_return":    float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        }

        # 有基准时的超额指标
        if self.benchmark is not None:
            excess = excess_return_vs_benchmark(r, self.benchmark)
            report.update({
                "excess_total_return":       float(np.prod(1 + excess) - 1),
                "excess_annualized_return":  annualized_return(excess),
                "excess_win_rate":           win_rate(excess),
                "information_ratio":         information_ratio(r, self.benchmark),
                "benchmark_total_return":    float(
                    np.prod(1 + self.benchmark[:len(r)]) - 1
                ),
            })

        return report

    def print_report(self, report: Optional[Dict] = None) -> None:
        """打印格式化报告到控制台。"""
        if report is None:
            report = self.full_report()

        print(f"\n{'='*60}")
        print(f"  策略评估报告：{report['strategy_name']}")
        print(f"{'='*60}")
        print(f"  回测周数          : {report['n_weeks']} 周")
        print(f"  累计收益率        : {report['total_return']:+.2%}")
        print(f"  年化收益率        : {report['annualized_return']:+.2%}")
        print(f"  年化波动率        : {report['annualized_volatility']:.2%}")
        print(f"  Sharpe Ratio      : {report['sharpe_ratio']:.4f}")
        print(f"  最大回撤          : {report['max_drawdown']:.2%}")
        print(f"    回撤起点        : {report['mdd_start_week']}")
        print(f"    回撤谷底        : {report['mdd_end_week']}")
        print(f"  Calmar Ratio      : {report['calmar_ratio']:.4f}")
        print(f"  胜率              : {report['win_rate']:.1%} "
              f"({report['n_win_weeks']}胜 / {report['n_loss_weeks']}负)")
        print(f"  平均盈利周        : {report['avg_win']:+.2%}")
        print(f"  平均亏损周        : {report['avg_loss']:+.2%}")
        print(f"  最佳周            : {report['best_week']:+.2%}")
        print(f"  最差周            : {report['worst_week']:+.2%}")
        print(f"  均周收益          : {report['mean_weekly_return']:+.4%} "
              f"± {report['std_weekly_return']:.4%}")

        if "excess_total_return" in report:
            print(f"\n  ── 相对基准 ──")
            print(f"  基准累计收益      : {report['benchmark_total_return']:+.2%}")
            print(f"  超额累计收益      : {report['excess_total_return']:+.2%}")
            print(f"  超额年化收益      : {report['excess_annualized_return']:+.2%}")
            print(f"  超额胜率          : {report['excess_win_rate']:.1%}")
            print(f"  Information Ratio : {report['information_ratio']:.4f}")

        print(f"{'='*60}\n")

    def plot_cumulative(
        self,
        report: Optional[Dict] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> None:
        """
        绘制累计净值曲线（可选与基准对比）。
        
        Parameters
        ----------
        save_path : str | Path | None
            若指定则保存图片（PNG），否则仅显示。
        show : bool
            是否调用 plt.show()（在服务器环境设为 False）。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")          # 非交互式后端，服务器安全
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick
        except ImportError:
            logger.warning("matplotlib 未安装，跳过绘图")
            return

        if report is None:
            report = self.full_report()

        cum_strategy = cumulative_return(self.returns)
        x = np.arange(len(cum_strategy))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"策略评估：{self.name}", fontsize=14, fontweight="bold")

        # ── 上图：累计净值 ──
        ax1 = axes[0]
        ax1.plot(x, cum_strategy, label="策略", color="#2196F3", linewidth=2)

        if self.benchmark is not None:
            bench_len = min(len(self.benchmark), len(cum_strategy))
            cum_bench = cumulative_return(self.benchmark[:bench_len])
            ax1.plot(x[:bench_len], cum_bench,
                     label="基准（沪深300）", color="#FF9800",
                     linewidth=1.5, linestyle="--")

        # 标记最大回撤区间
        _, pi, ti = max_drawdown(self.returns)
        ax1.axvspan(pi, ti, alpha=0.15, color="red", label="最大回撤区间")
        ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("累计净值")
        ax1.legend(loc="upper left")
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
        ax1.set_title(
            f"累计收益: {report['total_return']:+.2%}  "
            f"Sharpe: {report['sharpe_ratio']:.3f}  "
            f"MDD: {report['max_drawdown']:.2%}  "
            f"胜率: {report['win_rate']:.1%}"
        )

        # ── 下图：周度收益柱状图 ──
        ax2 = axes[1]
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in self.returns]
        ax2.bar(x, self.returns, color=colors, width=0.8, alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("周收益率")
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

        # x 轴刻度（每隔 4 周显示一个标签）
        tick_step = max(1, len(self.labels) // 12)
        tick_pos  = range(0, len(self.labels), tick_step)
        ax2.set_xticks(list(tick_pos))
        ax2.set_xticklabels(
            [self.labels[i] for i in tick_pos],
            rotation=30, ha="right", fontsize=8
        )
        ax1.set_xticks(list(tick_pos))
        ax1.set_xticklabels([])

        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info(f"净值曲线已保存: {save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口：直接对 backtest_results.csv 生成报告
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估回测结果")
    parser.add_argument("--csv", type=str, default="backtest_results.csv",
                        help="backtest_results.csv 路径")
    parser.add_argument("--recent", type=int, default=None,
                        help="只看最近 N 周")
    parser.add_argument("--plot", type=str, default=None,
                        help="净值曲线保存路径（PNG）")
    parser.add_argument("--name", type=str, default="TSPRank Strategy")
    args = parser.parse_args()

    ev = StrategyEvaluator.from_backtest_csv(
        args.csv,
        strategy_name=args.name,
        recent_weeks=args.recent,
    )
    report = ev.full_report()
    ev.print_report(report)

    if args.plot:
        ev.plot_cumulative(report, save_path=args.plot)
        print(f"净值曲线已保存至: {args.plot}")