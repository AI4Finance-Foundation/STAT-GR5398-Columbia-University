"""Hybrid ML + weekly factor stock selection for the GR5398 final project.

This script only edits/uses the execution layer. It imports the repo's
lightweight sklearn-based `ml_bucket_selection` logic, but does not modify the
core FinRL-Trading package. All generated files are written under
results/final_stock_selection_2/.
"""

import os
import sys
from pathlib import Path


os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

PYTHON_ENV_ROOT = Path(sys.executable).resolve().parent
PYTHON_ENV_DLL_DIRS = [
    PYTHON_ENV_ROOT / "Library" / "bin",
    PYTHON_ENV_ROOT / "Scripts",
    PYTHON_ENV_ROOT,
]
if sys.platform == "win32":
    existing_path = os.environ.get("PATH", "")
    env_paths = [str(path) for path in PYTHON_ENV_DLL_DIRS if path.exists()]
    os.environ["PATH"] = ";".join(env_paths + [existing_path])
    for path in PYTHON_ENV_DLL_DIRS:
        if path.exists():
            os.add_dll_directory(str(path))

import numpy as np
import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategies import ml_bucket_selection as ml_bucket


OUTPUT_DIR = REPO_ROOT / "results" / "final_stock_selection_2"
FUNDAMENTAL_CSV = REPO_ROOT / "data" / "fundamental_data_full.csv"

DOWNLOAD_START = "2025-09-01"
YFINANCE_END = "2026-05-01"  # yfinance end dates are exclusive.
SELECTION_START = pd.Timestamp("2026-01-01")
SELECTION_END = pd.Timestamp("2026-04-30")

TOP_N = 5
REBALANCE_FREQ = "W-FRI"

ML_WEIGHT = 0.60
TECHNICAL_WEIGHT = 0.40

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "META",
    "AMZN",
    "GOOGL",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "ADBE",
    "COST",
    "PEP",
    "QCOM",
    "CSCO",
    "TXN",
    "AMAT",
    "AMGN",
    "INTU",
    "ISRG",
]

FEATURE_COLUMNS = ["mom_20d", "mom_60d", "vol_20d", "drawdown_60d"]
Z_COLUMNS = ["z_mom_20d", "z_mom_60d", "z_vol_20d", "z_drawdown_60d"]
ML_FEATURE_COLUMNS = ml_bucket.FEATURE_COLS + ml_bucket.MOMENTUM_COLS

# Fundamental data is low-frequency. Q3 2025 is assumed available from its
# trade date (2025-12-01), so it drives Jan-Feb 2026 weekly selections. Q4
# 2025 is assumed available from 2026-03-01, so it drives Mar-Apr 2026.
ML_QUARTER_SCHEDULE = [
    {
        "ml_datadate": "2025-09-30",
        "val_cutoff": "2025-06-30",
        "effective_start": pd.Timestamp("2026-01-01"),
        "effective_end": pd.Timestamp("2026-02-28"),
    },
    {
        "ml_datadate": "2025-12-31",
        "val_cutoff": "2025-09-30",
        "effective_start": pd.Timestamp("2026-03-01"),
        "effective_end": SELECTION_END,
    },
]

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "ml_datadate",
    "ml_available_from",
    "ml_bucket",
    "ml_model",
    "ml_score",
    "ml_ensemble_score",
    "z_ml_score",
    "mom_20d",
    "mom_60d",
    "vol_20d",
    "drawdown_60d",
    "z_mom_20d",
    "z_mom_60d",
    "z_vol_20d",
    "z_drawdown_60d",
    "technical_score",
    "z_technical_score",
    "finrl_score",
    "rank",
    "selected",
]

OUTPUT_PATHS = {
    "selected": OUTPUT_DIR / "finrl_stock_selection.csv",
    "all_ranks": OUTPUT_DIR / "finrl_stock_selection_all_ranks.csv",
    "summary": OUTPUT_DIR / "finrl_stock_selection_summary.csv",
    "latest": OUTPUT_DIR / "finrl_stock_selection_latest.csv",
    "ml_scores": OUTPUT_DIR / "finrl_ml_quarterly_scores.csv",
    "ml_models": OUTPUT_DIR / "finrl_ml_model_results.csv",
}


def _validate_output_dir() -> None:
    expected_output_dir = REPO_ROOT / "results" / "final_stock_selection_2"
    if OUTPUT_DIR != expected_output_dir:
        raise ValueError(
            "Output directory must be exactly results/final_stock_selection_2/ "
            "relative to the repository root."
        )


def _assert_allowed_output_paths(paths: dict[str, Path]) -> None:
    allowed_paths = set(OUTPUT_PATHS.values())
    requested_paths = set(paths.values())
    if requested_paths != allowed_paths:
        raise ValueError("The script attempted to use an unexpected output path.")

    for path in requested_paths:
        if path.parent != OUTPUT_DIR:
            raise ValueError("The script attempted to write outside the output directory.")


def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def download_prices() -> pd.DataFrame:
    """Download adjusted close prices and clean the candidate universe."""
    data = yf.download(
        TICKERS,
        start=DOWNLOAD_START,
        end=YFINANCE_END,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("No price data downloaded from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded yfinance data does not contain Close prices.")
        close = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Downloaded yfinance data does not contain Close prices.")
        close = data[["Close"]].copy()
        close.columns = TICKERS[:1]

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index().reindex(columns=TICKERS)

    # Forward-fill uses only previously observed prices, preserving the
    # no-look-ahead boundary while handling isolated missing observations.
    close = close.ffill()
    close = close.dropna(axis=1, how="all")
    close = close.dropna(axis=0, how="all")

    if close.shape[1] < TOP_N:
        raise ValueError(f"Need at least {TOP_N} tickers with price data.")

    return close


def _rolling_max_drawdown(return_window: pd.Series) -> float:
    if return_window.isna().any() or len(return_window) < 60:
        return np.nan
    wealth = (1.0 + return_window).cumprod()
    drawdowns = wealth / wealth.cummax() - 1.0
    return float(drawdowns.min())


def build_daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build daily point-in-time technical features."""
    returns = prices.pct_change()

    # Every rolling calculation ends at date t and uses only prices or returns
    # dated t or earlier. No future return is used for ranking.
    mom_20d = prices / prices.shift(20) - 1.0
    mom_60d = prices / prices.shift(60) - 1.0
    vol_20d = returns.rolling(window=20, min_periods=20).std()
    drawdown_60d = returns.rolling(window=60, min_periods=60).apply(
        _rolling_max_drawdown,
        raw=False,
    )

    feature_frames = []
    for name, frame in [
        ("mom_20d", mom_20d),
        ("mom_60d", mom_60d),
        ("vol_20d", vol_20d),
        ("drawdown_60d", drawdown_60d),
    ]:
        stacked = frame.stack().rename(name).reset_index()
        stacked.columns = ["date", "ticker", name]
        feature_frames.append(stacked)

    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.merge(frame, on=["date", "ticker"], how="outer")

    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


def score_daily_cross_section(features: pd.DataFrame) -> pd.DataFrame:
    """Create the weekly technical score from momentum, volatility, drawdown."""
    scored = features.dropna(subset=FEATURE_COLUMNS).copy()

    for feature, z_col in zip(FEATURE_COLUMNS, Z_COLUMNS):
        scored[z_col] = scored.groupby("date", group_keys=False)[feature].apply(
            _cross_sectional_zscore
        )

    scored["technical_score"] = (
        0.40 * scored["z_mom_20d"]
        + 0.35 * scored["z_mom_60d"]
        - 0.15 * scored["z_vol_20d"]
        + 0.10 * scored["z_drawdown_60d"]
    )
    scored["z_technical_score"] = scored.groupby("date", group_keys=False)[
        "technical_score"
    ].apply(_cross_sectional_zscore)

    return scored.sort_values(["date", "ticker"]).reset_index(drop=True)


def get_weekly_rebalance_dates(price_dates: pd.Index) -> list[pd.Timestamp]:
    """Resolve weekly Friday targets to available trading dates."""
    available_dates = pd.DatetimeIndex(price_dates).sort_values().normalize()
    friday_targets = pd.date_range(
        start=SELECTION_START,
        end=SELECTION_END,
        freq=REBALANCE_FREQ,
    )

    rebalance_dates = []
    for target in friday_targets:
        eligible_dates = available_dates[available_dates <= target]
        if eligible_dates.empty:
            continue
        resolved_date = eligible_dates[-1]
        if SELECTION_START <= resolved_date <= SELECTION_END:
            rebalance_dates.append(resolved_date)

    return list(pd.DatetimeIndex(rebalance_dates).drop_duplicates())


def load_fundamental_data() -> pd.DataFrame:
    if not FUNDAMENTAL_CSV.exists():
        raise ValueError(f"Missing fundamental data CSV: {FUNDAMENTAL_CSV}")

    usecols = [
        "ticker",
        "datadate",
        "gsector",
        "adj_close_q",
        "trade_price",
        "y_return",
    ] + ml_bucket.FEATURE_COLS

    raw = pd.read_csv(FUNDAMENTAL_CSV, usecols=usecols)
    raw = raw[raw["ticker"].isin(TICKERS)].copy()

    missing_tickers = sorted(set(TICKERS) - set(raw["ticker"].unique()))
    if missing_tickers:
        raise ValueError(f"Fundamental data missing tickers: {missing_tickers}")

    return raw


def prepare_ml_dataset(raw: pd.DataFrame, max_datadate: str) -> pd.DataFrame:
    """Prepare repo ML features using data available through max_datadate."""
    df = raw.copy()
    df["_datadate_ts"] = pd.to_datetime(df["datadate"], errors="coerce")
    max_dt = pd.Timestamp(max_datadate)
    df = df[df["_datadate_ts"] <= max_dt].copy()
    df["datadate"] = df["_datadate_ts"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["_datadate_ts"])
    df = df.rename(columns={"ticker": "tic"})
    df = df.sort_values(["tic", "datadate"]).reset_index(drop=True)

    numeric_cols = ["adj_close_q", "trade_price", "y_return"] + ml_bucket.FEATURE_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # These momentum features mirror the repo's ml_bucket_selection pipeline.
    # They are computed only from each ticker's own history up to max_datadate.
    df["ret_1q"] = df.groupby("tic")["trade_price"].pct_change(1)
    df["ret_4q"] = df.groupby("tic")["trade_price"].pct_change(4)
    for src, dst in [
        ("EPS", "eps_chg"),
        ("roe", "roe_chg"),
        ("gross_margin", "gm_chg"),
        ("operating_margin", "om_chg"),
    ]:
        df[dst] = df.groupby("tic")[src].diff()

    pre_accel_features = [
        col for col in ML_FEATURE_COLUMNS if col != "ret_accel"
    ]
    for col in pre_accel_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    medians = df[pre_accel_features].median()
    df[pre_accel_features] = (
        df[pre_accel_features]
        .fillna(medians)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    for col in pre_accel_features:
        p01, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p01, upper=p99)

    df["ret_accel"] = (df["ret_1q"] - df["ret_4q"] / 4).fillna(0.0)
    p01, p99 = df["ret_accel"].quantile(0.01), df["ret_accel"].quantile(0.99)
    df["ret_accel"] = df["ret_accel"].clip(lower=p01, upper=p99)

    df["bucket"] = df["gsector"].str.lower().map(ml_bucket.SECTOR_TO_BUCKET)
    df = df[df["bucket"].notna()].copy()

    return df


def build_quarterly_ml_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repo ML bucket models for each point-in-time inference quarter."""
    raw = load_fundamental_data()
    all_predictions = []
    all_model_results = []

    for schedule in ML_QUARTER_SCHEDULE:
        ml_datadate = schedule["ml_datadate"]
        val_cutoff = schedule["val_cutoff"]

        ml_data = prepare_ml_dataset(raw, max_datadate=ml_datadate)
        ml_data = ml_data[
            (ml_data["datadate"] <= val_cutoff)
            | (ml_data["datadate"] == ml_datadate)
        ].copy()

        quarter_predictions = []
        for bucket, bucket_df in ml_data.groupby("bucket"):
            pred_df, model_results, _importance_records = ml_bucket.run_bucket(
                bucket,
                bucket_df,
                ML_FEATURE_COLUMNS,
                val_cutoff=val_cutoff,
                val_quarters=3,
            )
            if not pred_df.empty:
                quarter_predictions.append(pred_df)
            for row in model_results:
                row = row.copy()
                row["ml_datadate"] = ml_datadate
                row["val_cutoff"] = val_cutoff
                all_model_results.append(row)

        if not quarter_predictions:
            raise ValueError(f"ML model produced no predictions for {ml_datadate}.")

        quarter = pd.concat(quarter_predictions, ignore_index=True)
        quarter = quarter[quarter["datadate"] == ml_datadate].copy()
        if quarter["tic"].nunique() != len(TICKERS):
            missing = sorted(set(TICKERS) - set(quarter["tic"].unique()))
            raise ValueError(f"ML predictions missing tickers for {ml_datadate}: {missing}")

        quarter = quarter.rename(
            columns={
                "tic": "ticker",
                "bucket": "ml_bucket",
                "predicted_return": "ml_score",
                "pred_ensemble_avg": "ml_ensemble_score",
                "best_model": "ml_model",
            }
        )
        quarter["ml_datadate"] = ml_datadate
        quarter["ml_available_from"] = ml_bucket.datadate_to_tradedate(ml_datadate)
        quarter["val_cutoff"] = val_cutoff
        quarter["z_ml_score"] = _cross_sectional_zscore(quarter["ml_score"])

        keep_cols = [
            "ml_datadate",
            "ml_available_from",
            "val_cutoff",
            "ticker",
            "ml_bucket",
            "ml_model",
            "ml_score",
            "ml_ensemble_score",
            "z_ml_score",
        ]
        all_predictions.append(quarter[keep_cols])

    ml_scores = pd.concat(all_predictions, ignore_index=True)
    model_results = pd.DataFrame(all_model_results)

    return ml_scores.sort_values(["ml_datadate", "ticker"]).reset_index(drop=True), model_results


def _ml_datadate_for_rebalance(date: pd.Timestamp) -> str:
    for schedule in ML_QUARTER_SCHEDULE:
        if schedule["effective_start"] <= date <= schedule["effective_end"]:
            return schedule["ml_datadate"]
    raise ValueError(f"No ML quarter schedule covers rebalance date {date.date()}.")


def combine_ml_and_technical_scores(
    weekly_technical: pd.DataFrame,
    ml_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Attach low-frequency ML scores to weekly technical scores."""
    combined = weekly_technical.copy()
    combined["ml_datadate"] = combined["date"].apply(_ml_datadate_for_rebalance)
    combined = combined.merge(
        ml_scores,
        on=["ml_datadate", "ticker"],
        how="left",
        validate="many_to_one",
    )

    required = ["ml_score", "z_ml_score", "ml_bucket", "ml_model", "ml_available_from"]
    if combined[required].isna().any().any():
        missing = combined[combined[required].isna().any(axis=1)][["date", "ticker"]]
        raise ValueError(f"Missing ML scores for rows:\n{missing.to_string(index=False)}")

    # Hybrid score: ML captures slower fundamental expected-return information;
    # weekly technical score confirms current trend/risk conditions.
    combined["finrl_score"] = (
        ML_WEIGHT * combined["z_ml_score"]
        + TECHNICAL_WEIGHT * combined["z_technical_score"]
    )

    return combined


def _format_date_columns(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    if "date" in formatted.columns:
        formatted["date"] = pd.to_datetime(formatted["date"]).dt.strftime("%Y-%m-%d")
    return formatted


def save_outputs(
    all_ranks: pd.DataFrame,
    ml_scores: pd.DataFrame,
    model_results: pd.DataFrame,
) -> dict[str, Path]:
    """Save required selection outputs plus ML audit tables."""
    _validate_output_dir()
    _assert_allowed_output_paths(OUTPUT_PATHS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = all_ranks[all_ranks["selected"] == 1].copy()
    latest_date = selected["date"].max()
    latest = selected[selected["date"] == latest_date].copy()

    summary = (
        all_ranks.groupby("ticker", as_index=False)
        .agg(
            selected_weeks=("selected", "sum"),
            avg_rank=("rank", "mean"),
            avg_ml_score=("ml_score", "mean"),
            avg_technical_score=("technical_score", "mean"),
            avg_score=("finrl_score", "mean"),
        )
        .sort_values(["selected_weeks", "avg_score"], ascending=[False, False])
        .reset_index(drop=True)
    )

    selected = _format_date_columns(selected[OUTPUT_COLUMNS])
    all_ranks = _format_date_columns(all_ranks[OUTPUT_COLUMNS])
    latest = _format_date_columns(latest[OUTPUT_COLUMNS])

    selected.to_csv(OUTPUT_PATHS["selected"], index=False)
    all_ranks.to_csv(OUTPUT_PATHS["all_ranks"], index=False)
    summary.to_csv(OUTPUT_PATHS["summary"], index=False)
    latest.to_csv(OUTPUT_PATHS["latest"], index=False)
    ml_scores.to_csv(OUTPUT_PATHS["ml_scores"], index=False)
    model_results.to_csv(OUTPUT_PATHS["ml_models"], index=False)

    return OUTPUT_PATHS


def _validate_selection_outputs(selected: pd.DataFrame, latest: pd.DataFrame) -> None:
    if selected.empty:
        raise ValueError("There is at least one rebalance date required.")

    counts = selected.groupby("date")["selected"].sum()
    if not (counts == TOP_N).all():
        raise ValueError("Every rebalance date must have exactly 5 selected stocks.")

    selected_dates = pd.to_datetime(selected["date"])
    if ((selected_dates < SELECTION_START) | (selected_dates > SELECTION_END)).any():
        raise ValueError("A selected date is outside 2026-01-01 to 2026-04-30.")

    if len(latest) != TOP_N:
        raise ValueError("The latest selection file must contain exactly 5 stocks.")


def _print_validation_report(paths: dict[str, Path]) -> None:
    selected = pd.read_csv(paths["selected"])
    all_ranks = pd.read_csv(paths["all_ranks"])
    summary = pd.read_csv(paths["summary"])
    latest = pd.read_csv(paths["latest"])
    ml_scores = pd.read_csv(paths["ml_scores"])

    _validate_selection_outputs(selected, latest)

    counts = selected.groupby("date")["selected"].sum().astype(int)
    last_rebalance_date = selected["date"].max()

    print("\nHybrid score formula:")
    print(f"finrl_score = {ML_WEIGHT:.2f} * z_ml_score + {TECHNICAL_WEIGHT:.2f} * z_technical_score")

    print("\nML quarter schedule:")
    schedule_rows = []
    for schedule in ML_QUARTER_SCHEDULE:
        schedule_rows.append(
            {
                "ml_datadate": schedule["ml_datadate"],
                "val_cutoff": schedule["val_cutoff"],
                "effective_start": schedule["effective_start"].strftime("%Y-%m-%d"),
                "effective_end": schedule["effective_end"].strftime("%Y-%m-%d"),
                "ml_available_from": ml_bucket.datadate_to_tradedate(schedule["ml_datadate"]),
            }
        )
    print(pd.DataFrame(schedule_rows).to_string(index=False))

    print("\nNumber of rebalance dates:")
    print(counts.shape[0])

    print("\nNumber of selected stocks per rebalance date:")
    print(counts.to_string())

    print("\nLast rebalance date:")
    print(last_rebalance_date)

    print("\nLatest selected top-5 stocks:")
    cols = [
        "date",
        "ticker",
        "ml_score",
        "technical_score",
        "finrl_score",
        "rank",
        "selected",
    ]
    print(latest[cols].to_string(index=False))

    print("\nSelection summary table:")
    print(summary.to_string(index=False))

    print("\nML quarterly score preview:")
    print(
        ml_scores[
            ["ml_datadate", "ticker", "ml_bucket", "ml_model", "ml_score", "z_ml_score"]
        ]
        .sort_values(["ml_datadate", "z_ml_score"], ascending=[True, False])
        .head(10)
        .to_string(index=False)
    )

    print("\nGenerated CSV paths:")
    for path in paths.values():
        print(path.resolve())

    print("\nFirst few rows of finrl_stock_selection.csv:")
    print(selected.head().to_string(index=False))

    print("\nFirst few rows of finrl_stock_selection_all_ranks.csv:")
    print(all_ranks.head().to_string(index=False))

    print("\nFirst few rows of finrl_stock_selection_summary.csv:")
    print(summary.head().to_string(index=False))

    print("\nFirst few rows of finrl_stock_selection_latest.csv:")
    print(latest.head().to_string(index=False))


def main() -> None:
    prices = download_prices()
    features = build_daily_features(prices)
    technical_daily = score_daily_cross_section(features)
    ml_scores, model_results = build_quarterly_ml_scores()

    rebalance_dates = get_weekly_rebalance_dates(prices.index)
    if not rebalance_dates:
        raise ValueError("There is at least one rebalance date required.")

    weekly_technical = technical_daily[technical_daily["date"].isin(rebalance_dates)].copy()
    all_ranks = combine_ml_and_technical_scores(weekly_technical, ml_scores)
    all_ranks = all_ranks.sort_values(
        ["date", "finrl_score", "ticker"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    all_ranks["rank"] = all_ranks.groupby("date").cumcount() + 1
    all_ranks["selected"] = np.where(all_ranks["rank"] <= TOP_N, 1, 0)

    selected_counts = all_ranks.groupby("date")["selected"].sum()
    if selected_counts.empty:
        raise ValueError("There is at least one rebalance date required.")
    if not (selected_counts == TOP_N).all():
        raise ValueError("Every rebalance date must have exactly 5 selected stocks.")

    paths = save_outputs(all_ranks, ml_scores, model_results)
    _print_validation_report(paths)


if __name__ == "__main__":
    main()
