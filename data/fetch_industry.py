"""
从 baostock 获取 A 股行业分类数据，覆盖 stock_data.csv 的完整时间范围。

支持断点续传：
  - 每查完一个日期，立即追加写入 data/industry.csv
  - 重启后自动跳过已查询的日期
  - 进度日志打印到终端

用法:
  cd ~/tsprank
  python data/fetch_industry.py

输出:
  data/industry.csv — 列: date, code, code_name, industry, industryClassification
"""

import os
import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
STOCK_CSV   = SCRIPT_DIR / "stock_data.csv"
OUTPUT_CSV  = SCRIPT_DIR / "industry.csv"
PROGRESS_F  = SCRIPT_DIR / ".industry_progress.txt"

# ── 日期采样策略 ──────────────────────────────────────────────────────────────
# 行业分类变动不频繁，每季度采样一次足够覆盖
# 2020Q1 ~ 2026Q2 共 26 个季度
def build_sample_dates(start: str, end: str) -> list[str]:
    """生成季度采样日期列表（每季度第一个交易日附近）。"""
    dates = []
    for year in range(2020, 2027):
        for month in [1, 4, 7, 10]:
            d = f"{year}-{month:02d}-01"
            if d >= start and d <= end:
                dates.append(d)
    # 补充末尾日期
    if end not in dates:
        dates.append(end)
    return sorted(set(dates))


def load_done_dates() -> set[str]:
    """读取已完成的日期集合。"""
    if PROGRESS_F.exists():
        with open(PROGRESS_F, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def save_done_date(d: str) -> None:
    """追加一个已完成日期。"""
    with open(PROGRESS_F, "a") as f:
        f.write(d + "\n")


def fetch_one_date(d: str) -> list[dict]:
    """查询某日全市场行业分类。"""
    rs = bs.query_stock_industry(date=d)
    rows = []
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        # 返回字段: updateDate, code, code_name, industry, industryClassification
        rows.append({
            "date":                  d,
            "updateDate":            row[0] if len(row) > 0 else "",
            "code":                  row[1] if len(row) > 1 else "",
            "code_name":             row[2] if len(row) > 2 else "",
            "industry":              row[3] if len(row) > 3 else "",
            "industryClassification": row[4] if len(row) > 4 else "",
        })
    return rows


def main():
    # ── 读取 stock_data.csv 的日期范围 ───────────────────────────────────────
    if not STOCK_CSV.exists():
        print(f"[ERROR] {STOCK_CSV} 不存在")
        sys.exit(1)

    df = pd.read_csv(STOCK_CSV)
    date_min = df["日期"].min().replace("/", "-")
    date_max = df["日期"].max().replace("/", "-")
    stock_codes = set(df["股票代码"].astype(str).str.zfill(6))
    print(f"[INFO] stock_data.csv: {len(stock_codes)} 只股票, "
          f"日期 {date_min} ~ {date_max}")

    sample_dates = build_sample_dates(date_min, date_max)
    done_dates   = load_done_dates()
    todo_dates   = [d for d in sample_dates if d not in done_dates]
    print(f"[INFO] 采样日期 {len(sample_dates)} 个, "
          f"已完成 {len(done_dates)}, 待查 {len(todo_dates)}")

    if not todo_dates:
        print("[INFO] 全部日期已查询完毕")
        return

    # ── 登录 baostock ────────────────────────────────────────────────────────
    lg = bs.login()
    if lg.error_code != "0":
        print(f"[ERROR] baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    print(f"[INFO] baostock 登录成功")

    # ── 断点续传：追加模式写 CSV ─────────────────────────────────────────────
    header_written = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0

    try:
        for i, d in enumerate(todo_dates):
            t0 = time.time()
            rows = fetch_one_date(d)
            elapsed = time.time() - t0

            if rows:
                pd.DataFrame(rows).to_csv(
                    OUTPUT_CSV,
                    mode="a",
                    header=not header_written,
                    index=False,
                )
                header_written = True

            save_done_date(d)
            print(
                f"  [{i+1}/{len(todo_dates)}] {d}: "
                f"{len(rows)} 条记录, {elapsed:.1f}s"
            )
            # baostock 限流保护
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[INFO] 中断，已保存进度，下次运行自动续传")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("[INFO] 已保存进度，下次运行自动续传")
    finally:
        bs.logout()

    # ── 后处理：去重、过滤、统计 ─────────────────────────────────────────────
    if OUTPUT_CSV.exists():
        result = pd.read_csv(OUTPUT_CSV)
        # 保留 stock_data.csv 中的股票
        result["code_num"] = result["code"].str.replace(r"[a-z.]", "", regex=True)
        result = result[result["code_num"].isin(stock_codes)]
        result = result.drop(columns=["code_num"])
        result = result.drop_duplicates(subset=["date", "code"])
        result = result.sort_values(["date", "code"]).reset_index(drop=True)
        result.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[DONE] {OUTPUT_CSV}")
        print(f"  总记录: {len(result)}")
        print(f"  日期数: {result['date'].nunique()}")
        print(f"  股票数: {result['code'].nunique()}")
        print(f"  行业数: {result['industry'].nunique()}")
        print(f"  行业分布:")
        for name, cnt in result.groupby("industry").size().sort_values(ascending=False).head(10).items():
            print(f"    {name}: {cnt}")


if __name__ == "__main__":
    main()


# ── 补充脚本：查询缺失的 2020 年早期日期 ─────────────────────────────────────
def fix_missing_dates():
    """补查 2020Q1-Q3 的行业数据。"""
    missing = ["2020-01-01", "2020-04-01", "2020-07-01"]
    done = set()
    if PROGRESS_F.exists():
        with open(PROGRESS_F) as f:
            done = {l.strip() for l in f if l.strip()}
    todo = [d for d in missing if d not in done]
    if not todo:
        print("[FIX] 无缺失日期，跳过")
        return

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[ERROR] login failed: {lg.error_msg}")
        return
    header_written = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    try:
        for d in todo:
            rows = fetch_one_date(d)
            if rows:
                pd.DataFrame(rows).to_csv(
                    OUTPUT_CSV, mode="a",
                    header=not header_written, index=False,
                )
                header_written = True
            save_done_date(d)
            print(f"  [FIX] {d}: {len(rows)} 条")
            time.sleep(0.3)
    finally:
        bs.logout()
