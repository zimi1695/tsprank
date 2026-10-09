# tsprank_core/qlib_pred_loader.py
"""
P1 方案：从小组仓库 mlflow artifacts 中读取 pred.pkl，
直接使用 DoubleEnsemble W4 的完整预测分数（feat_weights 完整，IC≈0.10）。

用法：
    from tsprank_core.qlib_pred_loader import QlibPredLoader
    loader = QlibPredLoader(pred_pkl_path="~/quant-stock-forecast/.../pred.pkl")
    scores = loader.get_scores(pred_date=pd.Timestamp("2026-01-10"),
                               codes_in_scope=["600000", "601318", ...])
"""

from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _strip_prefix(code: str) -> str:
    """剥离 qlib 格式前缀：SH600000 / SZ000001 → 600000 / 000001"""
    if isinstance(code, str):
        for prefix in ("SH", "SZ", "sh", "sz"):
            if code.startswith(prefix):
                return code[len(prefix):]
    return str(code)


def _find_pred_pkl(mlruns_root: Union[str, Path],
                   experiment_name: Optional[str] = None,
                   run_name_keyword: Optional[str] = "w4") -> Optional[Path]:
    """
    在 mlruns 目录下自动搜索 pred.pkl。
    
    搜索策略：
      1. 若 experiment_name 指定，只在对应实验目录内搜索
      2. 否则搜索所有实验，取修改时间最新的 pred.pkl
      3. 若 run_name_keyword 指定，只考虑路径包含该关键词的 run
    
    Returns
    -------
    Path 或 None
    """
    root = Path(mlruns_root).expanduser()
    if not root.exists():
        logger.warning(f"mlruns 目录不存在: {root}")
        return None

    candidates = list(root.rglob("pred.pkl"))
    if not candidates:
        logger.warning(f"在 {root} 下未找到任何 pred.pkl")
        return None

    if run_name_keyword:
        filtered = [p for p in candidates if run_name_keyword.lower() in str(p).lower()]
        if filtered:
            candidates = filtered

    # 取修改时间最新的
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.info(f"自动选择 pred.pkl: {best}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class QlibPredLoader:
    """
    封装从 qlib mlflow artifacts pred.pkl 读取预测分数的逻辑。
    
    pred.pkl 格式（qlib 标准）：
        pd.Series 或 pd.DataFrame，
        MultiIndex: (datetime, instrument)，
        值列名: 'score' 或 0（位置索引）
    
    Parameters
    ----------
    pred_pkl_path : str | Path | None
        直接指定 pred.pkl 路径。若为 None，则用 mlruns_root 自动搜索。
    mlruns_root : str | Path | None
        mlruns 根目录，用于自动搜索。
    run_name_keyword : str
        自动搜索时过滤 run 名称的关键词（默认 "w4"）。
    """

    def __init__(
        self,
        pred_pkl_path: Optional[Union[str, Path]] = None,
        mlruns_root: Optional[Union[str, Path]] = None,
        run_name_keyword: str = "w4",
    ):
        self._pred_df: Optional[pd.DataFrame] = None
        self._pkl_path: Optional[Path] = None

        if pred_pkl_path is not None:
            self._pkl_path = Path(pred_pkl_path).expanduser()
        elif mlruns_root is not None:
            self._pkl_path = _find_pred_pkl(mlruns_root, run_name_keyword=run_name_keyword)

        if self._pkl_path is not None:
            self._load()

    # ── 内部加载 ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """加载并标准化 pred.pkl 为 MultiIndex DataFrame。"""
        if not self._pkl_path.exists():
            logger.error(f"pred.pkl 不存在: {self._pkl_path}")
            return

        logger.info(f"加载 pred.pkl: {self._pkl_path}")
        with open(self._pkl_path, "rb") as f:
            raw = pickle.load(f)

        # ── 统一转为 DataFrame ──
        if isinstance(raw, pd.Series):
            df = raw.to_frame(name="score")
        elif isinstance(raw, pd.DataFrame):
            df = raw.copy()
            # 若列名是整数位置索引，重命名第一列为 score
            if "score" not in df.columns:
                df = df.rename(columns={df.columns[0]: "score"})
        else:
            raise TypeError(f"不支持的 pred.pkl 格式: {type(raw)}")

        # ── 标准化 MultiIndex ──
        if not isinstance(df.index, pd.MultiIndex):
            # 尝试从列中恢复
            if "datetime" in df.columns and "instrument" in df.columns:
                df = df.set_index(["datetime", "instrument"])
            elif "date" in df.columns and "instrument" in df.columns:
                df = df.set_index(["date", "instrument"])
                df.index.names = ["datetime", "instrument"]
            else:
                raise ValueError("无法识别 pred.pkl 的索引结构，"
                                 "期望 MultiIndex(datetime, instrument) 或含对应列")

        df.index.names = ["datetime", "instrument"]

        # ── 标准化日期类型 ──
        lvl0 = df.index.get_level_values("datetime")
        if not pd.api.types.is_datetime64_any_dtype(lvl0):
            new_lvl0 = pd.to_datetime(lvl0)
            df.index = pd.MultiIndex.from_arrays(
                [new_lvl0, df.index.get_level_values("instrument")],
                names=["datetime", "instrument"]
            )

        # ── 标准化股票代码（剥离 SH/SZ 前缀）──
        instruments = df.index.get_level_values("instrument")
        stripped = [_strip_prefix(c) for c in instruments]
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values("datetime"), stripped],
            names=["datetime", "instrument"]
        )

        self._pred_df = df
        logger.info(
            f"pred.pkl 加载完成: {len(df)} 条记录, "
            f"日期范围 {df.index.get_level_values('datetime').min().date()} ~ "
            f"{df.index.get_level_values('datetime').max().date()}, "
            f"股票数 {df.index.get_level_values('instrument').nunique()}"
        )

    # ── 公开 API ──────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pred_df is not None

    def available_dates(self) -> pd.DatetimeIndex:
        """返回所有可用预测日期（升序）。"""
        if not self.is_loaded:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(
            sorted(self._pred_df.index.get_level_values("datetime").unique())
        )

    def get_scores(
        self,
        pred_date: Union[str, pd.Timestamp],
        codes_in_scope: List[str],
        fallback_score: float = 0.0,
        date_tolerance_days: int = 5,
    ) -> np.ndarray:
        """
        获取指定日期、指定股票池的预测分数。

        Parameters
        ----------
        pred_date : str | Timestamp
            目标日期（取 ≤ pred_date 的最近可用预测日）。
        codes_in_scope : List[str]
            股票代码列表（纯数字，已剥离前缀）。
        fallback_score : float
            在 pred.pkl 中找不到的股票使用此默认分数。
        date_tolerance_days : int
            若最近可用日期与 pred_date 相差超过此天数，发出警告。

        Returns
        -------
        np.ndarray, shape (len(codes_in_scope),), dtype float32
        """
        if not self.is_loaded:
            logger.warning("pred.pkl 未加载，返回全零分数")
            return np.zeros(len(codes_in_scope), dtype=np.float32)

        target = pd.Timestamp(pred_date)
        avail  = self.available_dates()

        # 找 ≤ target 的最近日期
        past = avail[avail <= target]
        if len(past) == 0:
            logger.warning(f"pred.pkl 中无 ≤ {target.date()} 的记录，"
                           f"最早日期: {avail[0].date()}")
            return np.full(len(codes_in_scope), fallback_score, dtype=np.float32)

        nearest = past[-1]
        gap_days = (target - nearest).days
        if gap_days > date_tolerance_days:
            logger.warning(
                f"最近预测日 {nearest.date()} 与目标日 {target.date()} "
                f"相差 {gap_days} 天，超过容忍阈值 {date_tolerance_days} 天"
            )

        # 取该日的预测
        try:
            day_series = self._pred_df.loc[nearest, "score"]
        except KeyError:
            logger.warning(f"无法取 {nearest} 的预测数据")
            return np.full(len(codes_in_scope), fallback_score, dtype=np.float32)

        # 对齐到 codes_in_scope
        scores = np.array(
            [float(day_series.get(_strip_prefix(c), fallback_score))
             for c in codes_in_scope],
            dtype=np.float32,
        )
        return scores

    def get_top_k(
        self,
        pred_date: Union[str, pd.Timestamp],
        codes_in_scope: List[str],
        k: int = 5,
        date_tolerance_days: int = 5,
    ) -> List[str]:
        """
        直接返回 Top-k 股票代码（按预测分数降序）。

        Returns
        -------
        List[str], 长度 ≤ k，股票代码（纯数字）
        """
        scores = self.get_scores(pred_date, codes_in_scope,
                                 date_tolerance_days=date_tolerance_days)
        top_idx = np.argsort(scores)[::-1][:k]
        return [_strip_prefix(codes_in_scope[i]) for i in top_idx]

    def diagnose(
        self,
        codes_sample: Optional[List[str]] = None,
        n_dates: int = 20,
    ) -> dict:
        """
        诊断 pred.pkl 质量：覆盖率、分数分布、样本日期。
        
        Returns
        -------
        dict with keys: n_dates, n_instruments, score_mean, score_std,
                        score_min, score_max, sample_dates
        """
        if not self.is_loaded:
            return {"error": "pred.pkl 未加载"}

        avail = self.available_dates()
        sample_dates = avail[-n_dates:].tolist() if len(avail) >= n_dates else avail.tolist()

        scores_all = self._pred_df["score"].values
        result = {
            "pkl_path":       str(self._pkl_path),
            "n_dates":        len(avail),
            "n_instruments":  self._pred_df.index.get_level_values("instrument").nunique(),
            "date_range":     (avail[0].date(), avail[-1].date()) if len(avail) > 0 else None,
            "score_mean":     float(np.nanmean(scores_all)),
            "score_std":      float(np.nanstd(scores_all)),
            "score_min":      float(np.nanmin(scores_all)),
            "score_max":      float(np.nanmax(scores_all)),
            "sample_dates":   [d.date() for d in sample_dates],
        }

        if codes_sample:
            last_date = avail[-1] if len(avail) > 0 else None
            if last_date is not None:
                try:
                    day = self._pred_df.loc[last_date, "score"]
                    coverage = sum(1 for c in codes_sample if _strip_prefix(c) in day.index)
                    result["sample_coverage"] = f"{coverage}/{len(codes_sample)}"
                except Exception:
                    result["sample_coverage"] = "查询失败"

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 命令行诊断入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="诊断 pred.pkl 加载状态")
    parser.add_argument("--pkl",  type=str, default=None, help="直接指定 pred.pkl 路径")
    parser.add_argument("--mlruns", type=str,
                        default="~/quant-stock-forecast/qlib-csi-300/mlruns",
                        help="mlruns 根目录（自动搜索）")
    parser.add_argument("--keyword", type=str, default="w4",
                        help="run 名称关键词过滤")
    parser.add_argument("--date", type=str, default="2026-04-17",
                        help="测试查询日期")
    args = parser.parse_args()

    loader = QlibPredLoader(
        pred_pkl_path=args.pkl,
        mlruns_root=args.mlruns if args.pkl is None else None,
        run_name_keyword=args.keyword,
    )

    info = loader.diagnose()
    print("\n=== pred.pkl 诊断报告 ===")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")

    if loader.is_loaded:
        test_date = pd.Timestamp(args.date)
        test_codes = ["600000", "601318", "600036", "000001", "000002"]
        scores = loader.get_scores(test_date, test_codes)
        print(f"\n=== {test_date.date()} 测试查询 ===")
        for c, s in zip(test_codes, scores):
            print(f"  {c}: {s:.6f}")
        top5 = loader.get_top_k(test_date, test_codes, k=5)
        print(f"\nTop-5: {top5}")