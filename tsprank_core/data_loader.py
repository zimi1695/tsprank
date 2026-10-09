# tsprank_core/data_loader.py
"""
数据加载模块 v2.1

修复内容（vs v2.0）：
  [Fix] make_result_csv()：
        1. 增加 top_k 参数（Optional[int]，默认 None），
           当传入时截断 codes/weights 至前 top_k 条。
        2. 返回值从 None 改为 pd.DataFrame，
           供 main.py _predict_latest() 打印使用。
        原因：main.py:581 调用 make_result_csv(codes, weights, top_k=top_k)
        并使用返回值 df.to_string()，v2.0 版本缺少该参数且返回 None，
        导致 TypeError（unexpected keyword argument）和 AttributeError。

v2.0 保留内容：
  ① parquet MultiIndex 卡死问题修正（reset_index 展平）
  ② 截面归一化：3σ winsorize + z-score
  ③ get_eval_weeks 对齐 W4 complete_week_only=True
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

warnings.filterwarnings("ignore")

# ============================================================
# 路径常量
# ============================================================
TSPRANK_ROOT = Path.home() / "tsprank"
DATA_DIR     = TSPRANK_ROOT / "data"

# ============================================================
# 全局配置（对齐 W4）
# ============================================================
QLIB_CONFIG = {
    "provider_uri": str(DATA_DIR / "qlib_bin"),
    "region": "cn",
    "train_start": "2013-01-04",
    "train_end":   "2019-11-04",
    "valid_start": "2019-11-12",
    "valid_end":   "2021-12-31",
    "test_start":  "2025-02-21",
    "test_end":    "2026-04-30",
    "eval_start":  "2025-02-24",
    "instruments":  "csi300",
}

LABEL_EXPR = "Ref($open*$adjfactor,-5)/Ref($open*$adjfactor,-1)-1"

# ============================================================
# qlib 初始化（幂等）
# ============================================================
_qlib_initialized = False

def _init_qlib() -> None:
    global _qlib_initialized
    if _qlib_initialized:
        return
    qlib.init(
        provider_uri=QLIB_CONFIG["provider_uri"],
        region=QLIB_CONFIG["region"],
    )
    _qlib_initialized = True


# ============================================================
# 股票代码标准化工具（供各模块共享使用）
# ============================================================

def _normalize_code(code: str) -> str:
    """
    从各种格式中提取 6 位纯数字股票代码。

    支持格式：
      SH600000  → 600000   (qlib MultiIndex 格式)
      sh.600000 → 600000   (baostock/industry.csv 格式)
      600000.SH → 600000   (Wind/Choice 格式)
      600000    → 600000   (纯数字格式)
    """
    s = str(code).strip().upper()
    # 处理 600000.SH 或 sh.600000 含点格式
    if "." in s:
        parts = s.split(".")
        for p in parts:
            if p.isdigit():
                return p.zfill(6)
        s = parts[0]  # 取点前部分继续处理
    # 去除交易所前缀
    for pfx in ("SH", "SZ", "BJ"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    return s.zfill(6) if s.isdigit() else s


# ============================================================
# 行业分类数据（industry.csv）
# ============================================================

def load_industry_mapping(
    path        : Optional[str] = None,
    use_latest  : bool = True,
) -> Dict[str, str]:
    """
    读取 industry.csv，返回 {6位股票代码: 行业标签} 字典。

    ⚠️  返回的是静态单期快照（默认最新期）。
        仅用于推理/回测的当前时间点查询。
        训练时的行业分组请改用 load_industry_df() + build_sector_dict()，
        后者会按预测日做点位一致（point-in-time）填充，避免前视偏差。
    """
    if path is None:
        path = str(DATA_DIR / "industry.csv")

    df = pd.read_csv(path, dtype=str)
    df["date"] = pd.to_datetime(df["date"], format="mixed")

    if use_latest:
        df = df[df["date"] == df["date"].max()].copy()

    df["clean_code"] = df["code"].apply(_normalize_code)
    df = df.sort_values("date").drop_duplicates("clean_code", keep="last")

    mapping = dict(zip(df["clean_code"], df["industry"]))

    n_sectors = df["industry"].nunique()
    snap_date = df["date"].max().date()
    print(
        f"[data_loader] 行业映射（静态）：{len(mapping)} 只股票 / "
        f"{n_sectors} 个行业 / 快照={snap_date}"
    )
    return mapping


def load_industry_df(path: Optional[str] = None) -> pd.DataFrame:
    """
    读取完整 industry.csv，返回解析好的 DataFrame（含所有历史快照）。

    用于 build_sector_dict() 的点位一致填充：
      对每个预测日，自动选取该日期当天或之前最近的行业快照，
      避免使用未来的行业分类信息。

    返回列：date, code, code_name, industry, clean_code
    """
    if path is None:
        path = str(DATA_DIR / "industry.csv")

    df = pd.read_csv(path, dtype=str)
    df["date"]       = pd.to_datetime(df["date"], format="mixed")
    df["clean_code"] = df["code"].apply(_normalize_code)
    df = df.sort_values("date").reset_index(drop=True)

    snap_dates = sorted(df["date"].unique())
    print(
        f"[data_loader] 行业数据（全历史）：{len(df)} 行 / "
        f"{len(snap_dates)} 个快照 / "
        f"{snap_dates[0].date()} ~ {snap_dates[-1].date()}"
    )
    return df


def build_sector_dict(
    stock_dict     : Dict,
    industry_source,           # pd.DataFrame（推荐）或 Dict[str,str]（兼容旧接口）
) -> Dict:
    """
    构建逐日行业映射：{date: {raw_code: sector_label}}。

    industry_source 两种用法：
      pd.DataFrame（来自 load_industry_df，推荐）
        → 点位一致（point-in-time）填充：
          每个预测日取该日当天或之前最近的行业快照，避免前视偏差。
          2020 Q1-Q3 等缺失快照自动用最早期（2020-10-01）向前回填；
          训练期（2013-2019）同样用最早可用快照，行业变动慢，影响可接受。

      Dict[str,str]（来自 load_industry_mapping，向后兼容）
        → 静态快照：所有日期套用同一份分类（旧行为）。

    Parameters
    ----------
    stock_dict     : date → [code_list]（来自 extract_cross_sections）
    industry_source: pd.DataFrame 或 Dict

    Returns
    -------
    {date: {raw_code: sector_label}}

    小行业说明
    ----------
    论文要求 N=5~20；N<5 的行业在训练/推理时统一跳过（由 min_sector_n=5 控制）。
    这里不过滤，保留所有映射，让上层决定是否使用。
    """
    if isinstance(industry_source, pd.DataFrame):
        return _build_sector_dict_pit(stock_dict, industry_source)
    else:
        return _build_sector_dict_static(stock_dict, industry_source)


def _build_sector_dict_pit(
    stock_dict  : Dict,
    industry_df : pd.DataFrame,
) -> Dict:
    """点位一致版：每个预测日查最近快照，key 统一存 clean code（6位数字）。"""
    snapshot_dates = sorted(industry_df["date"].unique())
    earliest       = snapshot_dates[0]

    # 快照缓存：snapshot_date → {clean_code: sector}
    _cache: Dict = {}

    def _get_snap(snap_date) -> Dict[str, str]:
        if snap_date not in _cache:
            sub = industry_df[industry_df["date"] == snap_date]
            # Key 统一用 clean code，与查询端（_normalize_code）一致
            _cache[snap_date] = dict(zip(sub["clean_code"], sub["industry"]))
        return _cache[snap_date]

    sector_dict   : Dict  = {}
    unknown_codes : set   = set()

    for d, codes in stock_dict.items():
        # 最近的 ≤ d 快照；d 在最早快照之前则用最早快照回填
        valid     = [s for s in snapshot_dates if s <= d]
        snap_date = valid[-1] if valid else earliest
        snap_map  = _get_snap(snap_date)

        day_map: Dict[str, str] = {}
        for code in codes:
            clean  = _normalize_code(str(code))   # 统一 normalize 再查
            sector = snap_map.get(clean)
            if sector is None:
                unknown_codes.add(clean)
                sector = "unknown"
            day_map[clean] = sector                # key 存 clean code
        sector_dict[d] = day_map

    if unknown_codes:
        print(
            f"[data_loader] ⚠️  {len(unknown_codes)} 个代码未在行业库找到，"
            f"归入 'unknown'（示例：{list(unknown_codes)[:5]}）"
        )

    used_snaps = set()
    for d in stock_dict:
        valid = [s for s in snapshot_dates if s <= d]
        used_snaps.add(valid[-1] if valid else earliest)
    print(
        f"[data_loader] sector_dict 构建完成：{len(sector_dict)} 个日期，"
        f"实际使用 {len(used_snaps)}/{len(snapshot_dates)} 个快照"
    )
    return sector_dict


def _build_sector_dict_static(
    stock_dict    : Dict,
    code_to_sector: Dict[str, str],
) -> Dict:
    """静态版（向后兼容）：所有日期用同一份分类，key 存 clean code。"""
    sector_dict   : Dict = {}
    unknown_codes : set  = set()

    for d, codes in stock_dict.items():
        day_map: Dict[str, str] = {}
        for code in codes:
            clean  = _normalize_code(str(code))
            sector = code_to_sector.get(clean)
            if sector is None:
                unknown_codes.add(clean)
                sector = "unknown"
            day_map[clean] = sector                # key 存 clean code
        sector_dict[d] = day_map

    if unknown_codes:
        print(
            f"[data_loader] ⚠️  {len(unknown_codes)} 个代码未找到行业映射，"
            f"归入 'unknown'（示例：{list(unknown_codes)[:5]}）"
        )
    return sector_dict


# ============================================================
# Alpha158 数据集加载（带 parquet 缓存）
# ============================================================

def _cache_exists(cache_path: str) -> bool:
    for split in ("train", "valid", "test"):
        if not Path(f"{cache_path}_{split}.parquet").exists():
            return False
    return True


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    df.reset_index().to_parquet(path, engine="pyarrow", index=False)


def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")

    if isinstance(df.index, pd.MultiIndex):
        df.index.names = ["datetime", "instrument"]
        new_levels = [
            pd.to_datetime(df.index.get_level_values(0)),
            df.index.get_level_values(1),
        ]
        df.index = pd.MultiIndex.from_arrays(new_levels,
                                              names=["datetime", "instrument"])
        return df

    date_col = "datetime"   if "datetime"   in df.columns else df.columns[0]
    inst_col = "instrument" if "instrument" in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index([date_col, inst_col])
    df.index.names = ["datetime", "instrument"]
    return df


def load_alpha158_dataset(
    cache_path: Optional[str] = None,
    use_cache:  bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回 (train_df, valid_df, test_df)
    每个 df：MultiIndex(datetime, instrument)，columns = Alpha158特征 + label
    """
    if cache_path is None:
        cache_path = str(TSPRANK_ROOT / "cache" / "alpha158_dataset")

    if use_cache and _cache_exists(cache_path):
        print("[data_loader] 命中 parquet 缓存，跳过 qlib 加载")
        train_df = _read_parquet(f"{cache_path}_train.parquet")
        valid_df = _read_parquet(f"{cache_path}_valid.parquet")
        test_df  = _read_parquet(f"{cache_path}_test.parquet")
        return train_df, valid_df, test_df

    print("[data_loader] 从 qlib 加载 Alpha158（首次约5-15分钟）...")
    _init_qlib()

    handler_config = {
        "start_time": QLIB_CONFIG["train_start"],
        "end_time":   QLIB_CONFIG["test_end"],
        "fit_start_time": QLIB_CONFIG["train_start"],
        "fit_end_time":   QLIB_CONFIG["train_end"],
        "instruments": QLIB_CONFIG["instruments"],
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna",           "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": [LABEL_EXPR, "label"],
    }

    dataset = DatasetH(
        handler={
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": handler_config,
        },
        segments={
            "train": (QLIB_CONFIG["train_start"], QLIB_CONFIG["train_end"]),
            "valid": (QLIB_CONFIG["valid_start"], QLIB_CONFIG["valid_end"]),
            "test":  (QLIB_CONFIG["test_start"],  QLIB_CONFIG["test_end"]),
        },
    )

    train_df = dataset.prepare("train", col_set=["feature", "label"])
    valid_df = dataset.prepare("valid", col_set=["feature", "label"])
    test_df  = dataset.prepare("test",  col_set=["feature", "label"])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print("[data_loader] 写入 parquet 缓存...")
    _write_parquet(train_df, f"{cache_path}_train.parquet")
    _write_parquet(valid_df, f"{cache_path}_valid.parquet")
    _write_parquet(test_df,  f"{cache_path}_test.parquet")
    print("[data_loader] 缓存写入完成")

    return train_df, valid_df, test_df


# ============================================================
# 截面提取与归一化
# ============================================================

def _cross_section_normalize(X: np.ndarray) -> np.ndarray:
    """截面归一化：3σ winsorize → z-score"""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m   = np.nanmean(col)
        s   = np.nanstd(col) + 1e-8
        col = np.clip(col, m - 3 * s, m + 3 * s)
        m2  = np.nanmean(col)
        s2  = np.nanstd(col) + 1e-8
        X[:, j] = (col - m2) / s2
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def extract_cross_sections(
    df: pd.DataFrame,
) -> Tuple[List[pd.Timestamp], Dict, Dict, Dict]:
    _label_candidates = [c for c in df.columns if "label" in str(c).lower()]
    if not _label_candidates:
        raise ValueError(f"找不到 label 列，实际列名末尾: {df.columns[-5:].tolist()}")
    label_col = _label_candidates[0]
    feat_cols  = [c for c in df.columns if c != label_col]

    dates      = sorted(df.index.get_level_values("datetime").unique())
    X_dict     = {}
    y_dict     = {}
    stock_dict = {}

    for d in dates:
        sub = df.loc[d]
        if len(sub) < 5:
            continue
        codes  = sub.index.tolist()
        X_raw  = sub[feat_cols].values.astype(np.float32)
        y_raw  = sub[label_col].values.astype(np.float32)

        valid_mask = ~np.isnan(y_raw)
        if valid_mask.sum() < 5:
            continue

        X_raw  = X_raw[valid_mask]
        y_raw  = y_raw[valid_mask]
        codes  = [c for c, v in zip(codes, valid_mask) if v]

        X_dict[d]     = _cross_section_normalize(X_raw)
        y_dict[d]     = y_raw
        stock_dict[d] = codes

    return dates, X_dict, y_dict, stock_dict


# ============================================================
# 价格数据
# ============================================================

def load_stock_data(
    path: Optional[str] = None,
) -> pd.DataFrame:
    if path is None:
        path = str(DATA_DIR / "stock_data.csv")

    df = pd.read_csv(path, dtype={"股票代码": str})
    df["日期"] = pd.to_datetime(df["日期"])
    df["股票代码"] = df["股票代码"].str.zfill(6)

    for col in ["开盘", "收盘", "最高", "最低"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 评估周提取
# ============================================================

def get_eval_weeks(
    stock_df:     Optional[pd.DataFrame] = None,
    eval_start:   str  = QLIB_CONFIG["eval_start"],
    eval_end:     str  = QLIB_CONFIG["test_end"],
    recent_weeks: int  = 12,
    min_days:     int  = 4,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    返回完整交易周列表：[(week_start, week_end), ...]，取最近 recent_weeks 周。
    """
    if stock_df is None:
        stock_df = load_stock_data()

    start_ts = pd.Timestamp(eval_start)
    end_ts   = pd.Timestamp(eval_end)

    td_df = stock_df[["日期"]].drop_duplicates()
    td_df = td_df[(td_df["日期"] >= start_ts) & (td_df["日期"] <= end_ts)].copy()
    td_df["year_week"] = td_df["日期"].dt.isocalendar().apply(
        lambda r: f"{int(r.year)}-{int(r.week):02d}", axis=1
    )

    weeks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for yw, grp in td_df.groupby("year_week"):
        days = grp["日期"].sort_values()
        if len(days) >= min_days:
            weeks.append((days.iloc[0], days.iloc[-1]))

    weeks.sort(key=lambda x: x[0])
    if recent_weeks > 0:
        weeks = weeks[-recent_weeks:]
    return weeks


# ============================================================
# data/test.csv 构建
# ============================================================

def build_test_csv(
    week_start:  pd.Timestamp,
    week_end:    pd.Timestamp,
    stock_df:    Optional[pd.DataFrame] = None,
    output_path: str = "data/test.csv",
) -> None:
    """
    将指定周价格数据写入 data/test.csv。
    ⚠️  必须按 (股票代码, 日期) 排序，score_self.py 依赖排序后的 tail(5)。
    """
    if stock_df is None:
        stock_df = load_stock_data()

    mask = (stock_df["日期"] >= week_start) & (stock_df["日期"] <= week_end)
    sub  = stock_df[mask].copy()
    sub  = sub.sort_values(["股票代码", "日期"])
    sub["日期"] = sub["日期"].dt.strftime("%Y/%m/%d")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)


# ============================================================
# result.csv 写入辅助
# ============================================================

def make_result_csv(
    codes:       List[str],
    weights:     List[float],
    output_path: str = "output/result.csv",
    top_k:       Optional[int] = None,
) -> pd.DataFrame:
    """
    写入 output/result.csv 并返回 DataFrame。

    Parameters
    ----------
    codes       : 股票代码列表
    weights     : 权重列表（与 codes 等长）
    output_path : 输出路径
    top_k       : [Fix] 若指定，截断至前 top_k 条（main.py _predict_latest 使用）

    Returns
    -------
    df : pd.DataFrame，含 ['股票代码', '权重'] 两列
         [Fix] v2.0 返回 None，main.py:581 对返回值调用 df.to_string() 导致
               AttributeError；v2.1 改为返回 DataFrame。

    Notes
    -----
    ⚠️ score_self.py 用 pd.read_csv 读取，必须有表头（pandas 默认写入）。
    格式：
        股票代码,权重
        600000,0.200000
        ...
    """
    # [Fix] 支持 top_k 截断（向后兼容：top_k=None 时不截断）
    if top_k is not None:
        codes   = list(codes)[:top_k]
        weights = list(weights)[:top_k]

    df = pd.DataFrame(
        {
            "股票代码": [str(c).zfill(6) for c in codes],
            "权重":    [round(float(w), 6) for w in weights],
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    # [Fix] 返回 DataFrame，供 main.py _predict_latest() 打印
    return df
