#!/usr/bin/env python
"""
Adapt the AI4Finance FinRL-Trading dataset into the format this assignment's pipeline expects.

The dataset (https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/data) ships:

  finrl_trading.7z            -> finrl_trading.db  (SQLite: price_data, raw_payloads)
  fundamental_data_full.csv   -> quarterly fundamental ratios, already computed
  sp500_historical_constituents.csv -> point-in-time S&P 500 membership

Its column names and sector labels do not match what `fundamental_run_model.py` /
`ml_model.py` expect, so this script does the translation:

  * ticker  -> tic          (the column name the models use)
  * datadate -> date
  * filing_date -> reportdate   (when the market could actually see the numbers)
  * sector NAMES ("Technology") -> GICS-style NUMBERS (45), so the existing
    `sector{N}.xlsx` loop over range(10, 65, 5) finds its files
  * drops identifier / bookkeeping columns that would otherwise leak into the feature set
  * applies point-in-time S&P 500 membership, so delisted names stay in the sample
    until the day they actually left the index (this is what kills survivorship bias)

Usage
-----
    7z x finrl_trading.7z            # or: bsdtar -xf finrl_trading.7z

    python prepare_data.py \
        --db finrl_trading.db \
        --fundamentals fundamental_data_full.csv \
        --constituents sp500_historical_constituents.csv \
        --output-dir ./prepared \
        --start 2015-01-01 --end 2025-09-30 --trade-start 2020-09-30

`--start` is how far back the TRAINING history goes; `--trade-start` is when you want
backtested trading to begin. The script prints the `-first_trade_index` value that
turns your `--trade-start` into the models' rolling-window offset.

Outputs (into --output-dir)
---------------------------
    final_ratios.csv    the whole cross-section, one row per (tic, quarter)
    sector{N}.xlsx      the same table split by sector, one file per GICS code
    daily.csv           daily prices, using the WRDS/Compustat column names the
                        notebook's backtest cells expect (datadate, prcod, prccd,
                        ajexdi), plus ready-made open_adj / close_adj

Requires: pandas, openpyxl (for .xlsx output).
"""

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

# The dataset labels sectors the way Yahoo/FMP does. The models expect GICS-style
# integers, because `run_stock_selection` iterates over range(10, 65, 5).
SECTOR_NAME_TO_GICS = {
    "Energy": 10,
    "Basic Materials": 15,
    "Industrials": 20,
    "Consumer Cyclical": 25,
    "Consumer Defensive": 30,
    "Healthcare": 35,
    "Financial Services": 40,
    "Technology": 45,
    "Communication Services": 50,
    "Utilities": 55,
    "Real Estate": 60,
}

# Columns that are identifiers, timestamps or prices — never model features.
# They are dropped here so the default `-no_feature_column_names` in
# fundamental_run_model.py is enough to keep the feature set clean.
DROP_COLUMNS = [
    "id",
    "created_at",
    "accepted_date",
    "tradedate",
    "actual_tradedate",
    "trade_price",
]

# Kept in the table (the models need them) but never used as features.
NON_FEATURE_COLUMNS = ["date", "tic", "gsector", "gsector_name", "reportdate",
                       "adj_close_q", "y_return"]


def load_fundamentals(path, start, end):
    print(f"Reading fundamentals: {path}")
    df = pd.read_csv(path)
    print(f"  raw: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df = df.rename(columns={"ticker": "tic", "datadate": "date"})

    # `reportdate` is what the models use to time a prediction: a quarter's numbers
    # are only actionable once the filing is out, not on the quarter-end date.
    df["reportdate"] = df["filing_date"].fillna(df.get("accepted_date"))
    df["reportdate"] = pd.to_datetime(df["reportdate"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    missing_report = df["reportdate"].isna().sum()
    if missing_report:
        # Fall back to quarter end + 45 days, the statutory 10-Q deadline.
        df.loc[df["reportdate"].isna(), "reportdate"] = (
            df.loc[df["reportdate"].isna(), "date"] + pd.Timedelta(days=45)
        )
        print(f"  {missing_report:,} rows had no filing date -> quarter end + 45 days")

    df["gsector_name"] = df["gsector"]
    df["gsector"] = df["gsector_name"].map(SECTOR_NAME_TO_GICS)
    unmapped = df[df["gsector"].isna()]["gsector_name"].unique()
    if len(unmapped):
        print(f"  WARNING: unmapped sectors dropped: {list(unmapped)}")
        df = df[df["gsector"].notna()]
    df["gsector"] = df["gsector"].astype(int)

    df = df.drop(columns=[c for c in DROP_COLUMNS + ["filing_date"] if c in df.columns])

    # A row with no label cannot be trained on — this is the most recent quarter
    # of each ticker, whose next-quarter return does not exist yet.
    before = len(df)
    df = df[df["y_return"].notna()]
    print(f"  dropped {before - len(df):,} rows with no y_return label")

    df = df[(df["date"] >= start) & (df["date"] <= end)]
    print(f"  after {start.date()}..{end.date()}: {len(df):,} rows")
    return df


def apply_pit_membership(df, path):
    """Keep each row only if its ticker was in the S&P 500 on that date."""
    print(f"Applying point-in-time S&P 500 membership: {path}")
    members = pd.read_csv(path)
    members["date"] = pd.to_datetime(members["date"])
    members = members.sort_values("date").reset_index(drop=True)

    member_dates = members["date"].values
    member_sets = [set(str(t).split(",")) for t in members["tickers"]]

    # For each row, use the most recent membership snapshot at or before its date.
    idx = np.searchsorted(member_dates, df["date"].values, side="right") - 1
    keep = np.fromiter(
        (i >= 0 and tic in member_sets[i] for i, tic in zip(idx, df["tic"].values)),
        dtype=bool,
        count=len(df),
    )

    kept = df[keep].copy()
    print(f"  kept {len(kept):,} of {len(df):,} rows "
          f"({kept['tic'].nunique()} tickers, {df['tic'].nunique()} before)")
    return kept


def clean_features(df):
    """Replace inf/NaN in numeric features with 0.

    `fundamental_run_model.py` builds its feature list by dropping any numeric column
    that contains a NaN anywhere — so a single missing value silently costs you a whole
    feature. Filling here keeps the feature set intact; it is a crude choice, and
    improving it (sector medians, dropping sparse ratios) is a legitimate extension.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric = df[feature_cols].select_dtypes(include=[np.number]).columns

    n_inf = int(np.isinf(df[numeric].to_numpy(dtype=float)).sum())
    n_nan = int(df[numeric].isna().to_numpy().sum())
    df[numeric] = df[numeric].replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  cleaned {n_inf:,} inf and {n_nan:,} NaN values across {len(numeric)} features")

    non_numeric = [c for c in feature_cols if c not in numeric]
    if non_numeric:
        print(f"  non-numeric columns (ignored as features): {non_numeric}")
    return df


def export_prices(db_path, out_path, start, end, tickers=None):
    print(f"Reading prices: {db_path}")
    con = sqlite3.connect(db_path)
    try:
        px = pd.read_sql_query(
            "SELECT ticker AS tic, date, open, high, low, close, adj_close, volume "
            "FROM price_data WHERE date BETWEEN ? AND ?",
            con,
            params=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        )
    finally:
        con.close()

    px["date"] = pd.to_datetime(px["date"])
    if tickers is not None:
        px = px[px["tic"].isin(tickers)]

    # This dataset gives adj_close directly instead of an adjustment factor. The
    # notebook's backtest cells are written against WRDS/Compustat names and divide by
    # `ajexdi`, so emit a factor that makes those formulas come out right:
    #     ajexdi = close / adj_close
    #     prccd / ajexdi = adj_close                      (adjusted close)
    #     prcod / ajexdi = open * adj_close / close       (adjusted open)
    # That way the existing notebook runs against this data unmodified.
    px = px[(px["close"] > 0) & (px["adj_close"] > 0)]
    px["ajexdi"] = px["close"] / px["adj_close"]
    px["open_adj"] = px["open"] / px["ajexdi"]
    px["close_adj"] = px["adj_close"]

    px = px.rename(columns={"date": "datadate", "open": "prcod", "close": "prccd",
                            "high": "prchd", "low": "prcld"})
    px = px[["tic", "datadate", "prcod", "prccd", "prchd", "prcld", "ajexdi",
             "adj_close", "volume", "open_adj", "close_adj"]]

    px = px.sort_values(["tic", "datadate"])
    px.to_csv(out_path, index=False)
    print(f"  wrote {len(px):,} rows, {px['tic'].nunique()} tickers -> {out_path}")
    print(f"  price range: {px['datadate'].min().date()} .. {px['datadate'].max().date()}")
    return px


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="Path to finrl_trading.db")
    parser.add_argument("--fundamentals", required=True, help="Path to fundamental_data_full.csv")
    parser.add_argument("--constituents", help="Path to sp500_historical_constituents.csv")
    parser.add_argument("--output-dir", default="./prepared")
    parser.add_argument("--start", default="2015-01-01",
                        help="Earliest quarter to keep (training history)")
    parser.add_argument("--end", default="2025-09-30",
                        help="Latest quarter to keep; the price table ends 2025-10-16")
    parser.add_argument("--trade-start", default="2020-09-30",
                        help="First quarter you want to trade; the script prints the "
                             "matching -first_trade_index")
    parser.add_argument("--no-pit-filter", action="store_true",
                        help="Skip point-in-time membership filtering (you will have "
                             "survivorship bias — say so in your report if you do this)")
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    os.makedirs(args.output_dir, exist_ok=True)

    fund = load_fundamentals(args.fundamentals, start, end)

    if not args.no_pit_filter:
        if not args.constituents:
            sys.exit("--constituents is required unless you pass --no-pit-filter")
        fund = apply_pit_membership(fund, args.constituents)
    else:
        print("WARNING: skipping point-in-time filter -> survivorship bias in your sample")

    fund = clean_features(fund)
    fund = fund.sort_values(["tic", "date"]).reset_index(drop=True)

    ratios_path = os.path.join(args.output_dir, "final_ratios.csv")
    fund.to_csv(ratios_path, index=False)
    print(f"Wrote {ratios_path}  ({len(fund):,} rows x {fund.shape[1]} cols)")

    for code, group in fund.groupby("gsector"):
        name = group["gsector_name"].iloc[0]
        path = os.path.join(args.output_dir, f"sector{int(code)}.xlsx")
        group.to_excel(path, index=False)
        quarters = group["date"].nunique()
        print(f"  sector{int(code)} ({name}): {len(group):,} rows, "
              f"{group['tic'].nunique()} tickers, {quarters} quarters -> {path}")

    export_prices(args.db, os.path.join(args.output_dir, "daily.csv"),
                  start, end, tickers=set(fund["tic"].unique()))

    quarters = [pd.Timestamp(q) for q in sorted(fund["date"].unique())]
    trade_start = pd.Timestamp(args.trade_start)
    idx = next((i for i, q in enumerate(quarters) if q >= trade_start), None)

    print("\n" + "=" * 70)
    print(f"Quarters available: {len(quarters)}  "
          f"({quarters[0].date()} .. {quarters[-1].date()})")
    print()
    print("In fundamental_run_model.py, -first_trade_index N reserves the first N")
    print("quarters for training only; trading starts at quarter N+1.")
    if idx is None:
        print(f"  Your --trade-start {trade_start.date()} is after the last quarter.")
    else:
        print(f"  --trade-start {trade_start.date()}  ->  -first_trade_index {idx}")
        print(f"  That leaves {len(quarters) - idx} quarters of out-of-sample trading "
              f"({quarters[idx].date()} .. {quarters[-1].date()}),")
        print(f"  trained on {idx} quarters of history.")
    print()
    print("The default -first_trade_index 20 would start trading at "
          f"{quarters[20].date() if len(quarters) > 20 else 'a quarter this sample never reaches'}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
