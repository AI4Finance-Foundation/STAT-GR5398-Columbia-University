"""
GR5398 Final Assignment

Integrated AI trading strategy combining:
- FinRL-Trading style price ranking and portfolio construction
- FinRobot style fundamental quality filtering
- FinGPT style structured sentiment gating from recent price-volume reaction

The required backtest window is 2026-01-01 through 2026-04-30.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBMISSION_DIR = Path(__file__).resolve().parent
ASSIGNMENT3_DIR = SUBMISSION_DIR.parents[1]
GROUP_ROOT = SUBMISSION_DIR.parents[2]
ASSIGNMENT1_DIR = GROUP_ROOT / "Assignment1" / "submissions" / "Assignment1_Jianfeng_Ma_jm6086"

UNIVERSE_FILE = ASSIGNMENT1_DIR / "data" / "security_daily.csv"
FUNDAMENTAL_FILE = ASSIGNMENT1_DIR / "outputs" / "final_ratios.csv"
LOCAL_PRICE_CACHE = SUBMISSION_DIR / "data" / "yahoo_2024_2026_prices.csv"
SHARED_PRICE_CACHE = ASSIGNMENT3_DIR / "data" / "yahoo_2024_2026_prices.csv"
OUTPUT_DIR = SUBMISSION_DIR / "outputs"

PRICE_START = "2024-01-01"
PRICE_END = "2026-05-01"
TEST_START = "2026-01-01"
TEST_END = "2026-04-30"

TRANSACTION_COST = 0.001
TOP_N = 10
MAX_WEIGHT = 0.12
QUALITY_MIN = 0.50
SENTIMENT_MIN = 0.60


@dataclass(frozen=True)
class StrategySpec:
    name: str
    quant_weight: float
    quality_weight: float
    sentiment_weight: float
    quality_min: float | None = None
    sentiment_min: float | None = None


STRATEGIES = [
    StrategySpec("Integrated AI Strategy", 0.75, 0.20, 0.05, QUALITY_MIN, SENTIMENT_MIN),
    StrategySpec("FinRL only", 1.00, 0.00, 0.00, None, None),
    StrategySpec("FinRL + FinRobot", 0.60, 0.40, 0.00, QUALITY_MIN, None),
    StrategySpec("FinRL + FinGPT", 0.85, 0.00, 0.15, None, SENTIMENT_MIN),
]


def rank01(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    return clean.rank(pct=True)


def zscore(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    return (clean - clean.mean()) / (clean.std(ddof=0) + 1e-9)


def load_universe() -> list[str]:
    universe = pd.read_csv(UNIVERSE_FILE, usecols=["tic"])
    return sorted(universe["tic"].dropna().unique().tolist())


def download_price_cache(symbols: list[str], cache_path: Path) -> None:
    import yfinance as yf
    import yfinance.cache as yf_cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(cache_path.parent / "yf_cache"))

    frames = []
    failed: list[str] = []
    for start in range(0, len(symbols), 8):
        batch = symbols[start : start + 8]
        print(f"Downloading {batch}")
        try:
            raw = yf.download(
                batch,
                start=PRICE_START,
                end=PRICE_END,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )
            if raw.empty:
                failed.extend(batch)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
                volume = raw["Volume"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
                volume = raw[["Volume"]].rename(columns={"Volume": batch[0]})
            frames.append(pd.concat({"close": close, "volume": volume}, axis=1))
        except Exception as exc:  # pragma: no cover - depends on network availability
            print(f"Download failed for {batch}: {exc}")
            failed.extend(batch)
        time.sleep(0.6)

    if not frames:
        raise RuntimeError("No Yahoo Finance prices were downloaded.")

    wide = pd.concat(frames, axis=1).sort_index()
    wide.columns = [f"{field}_{ticker}" for field, ticker in wide.columns]
    wide.index.name = "Date"
    wide.to_csv(cache_path)
    if failed:
        print(f"Warning: failed downloads: {failed}")


def ensure_price_cache(refresh: bool) -> Path:
    if refresh or not LOCAL_PRICE_CACHE.exists():
        if not refresh and SHARED_PRICE_CACHE.exists():
            LOCAL_PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_PRICE_CACHE.write_bytes(SHARED_PRICE_CACHE.read_bytes())
            return LOCAL_PRICE_CACHE
        symbols = sorted(set(load_universe() + ["SPY", "QQQ", "TLT", "GLD", "SHY"]))
        download_price_cache(symbols, LOCAL_PRICE_CACHE)
    return LOCAL_PRICE_CACHE


def load_prices(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ensure_price_cache(refresh)
    wide = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    close = wide.filter(like="close_").rename(columns=lambda c: c.replace("close_", "")).ffill()
    volume = wide.filter(like="volume_").rename(columns=lambda c: c.replace("volume_", "")).ffill()
    return close, volume


def latest_quality(fundamentals: pd.DataFrame, asof: pd.Timestamp, tickers: list[str]) -> pd.Series:
    available = fundamentals[fundamentals["reportdate"] <= asof]
    latest = available.sort_values(["tic", "reportdate"]).groupby("tic").tail(1).set_index("tic")

    positive_columns = ["ROE", "ROA", "OPM", "NPM", "EPS", "cur_ratio", "quick_ratio"]
    negative_columns = ["debt_ratio", "debt_to_equity", "pe", "pb", "ps"]
    components = []
    for column in positive_columns:
        if column in latest:
            components.append(zscore(latest[column]).rename(column))
    for column in negative_columns:
        if column in latest:
            components.append((-zscore(latest[column])).rename(column))

    quality = pd.concat(components, axis=1).mean(axis=1)
    return rank01(quality).reindex(tickers)


def build_monthly_signals(
    close: pd.DataFrame, volume: pd.DataFrame, tickers: list[str], fundamentals: pd.DataFrame
) -> dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]]:
    returns = close.pct_change().fillna(0.0)
    test_returns = returns.loc[TEST_START:TEST_END]
    rebalance_dates = pd.DatetimeIndex(
        [group.index[0] for _, group in test_returns.groupby(test_returns.index.to_period("M"))]
    )

    signals: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]] = {}
    for rebalance_date in rebalance_dates:
        signal_date = close.index[close.index < rebalance_date][-1]
        price_window = close.loc[:signal_date, tickers]
        volume_window = volume.loc[:signal_date, tickers]
        last = price_window.iloc[-1]

        ret_5 = last / price_window.iloc[-5] - 1
        ret_20 = last / price_window.iloc[-20] - 1
        ret_3m = last / price_window.iloc[-63] - 1
        ret_6m = last / price_window.iloc[-126] - 1
        ret_12m = last / price_window.iloc[-252] - 1
        vol_20 = price_window.pct_change().tail(20).std()
        vol_60 = price_window.pct_change().tail(60).std()
        ma_200 = price_window.tail(200).mean()
        volume_ratio = volume_window.tail(10).mean() / volume_window.tail(60).mean()

        quant_raw = (
            0.45 * rank01(ret_6m)
            + 0.25 * rank01(ret_3m)
            + 0.15 * rank01(ret_12m)
            + 0.15 * rank01(last / ma_200 - 1)
            - 0.20 * rank01(vol_60)
        )
        sentiment = 0.55 * rank01(ret_20) + 0.25 * rank01(ret_5) + 0.20 * rank01(volume_ratio)
        quality = latest_quality(fundamentals, signal_date, tickers)

        signal_frame = pd.DataFrame(
            {
                "quant": rank01(quant_raw),
                "quality": quality,
                "sentiment": sentiment,
                "ret_5": ret_5,
                "ret_20": ret_20,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "vol_20": vol_20,
                "vol_60": vol_60,
            }
        )
        signals[rebalance_date] = (signal_date, signal_frame)
    return signals


def cap_weights(raw_weights: pd.Series, max_weight: float) -> pd.Series:
    remaining = raw_weights.dropna().copy()
    if remaining.empty:
        return remaining
    fixed = pd.Series(dtype=float)
    budget = 1.0

    while len(remaining) > 0:
        scaled = remaining / remaining.sum() * budget
        over_cap = scaled > max_weight
        if not over_cap.any():
            return pd.concat([fixed, scaled]).sort_index()
        fixed = pd.concat([fixed, pd.Series(max_weight, index=scaled[over_cap].index)])
        budget -= max_weight * int(over_cap.sum())
        remaining = remaining[~over_cap]
        if budget <= 1e-12:
            return fixed.sort_index()
    return fixed.sort_index()


def select_weights(signal_frame: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    score = (
        spec.quant_weight * signal_frame["quant"]
        + spec.quality_weight * signal_frame["quality"]
        + spec.sentiment_weight * signal_frame["sentiment"]
    )
    eligible = signal_frame["vol_20"].notna()
    if spec.quality_min is not None:
        eligible &= signal_frame["quality"] >= spec.quality_min
    if spec.sentiment_min is not None:
        eligible &= signal_frame["sentiment"] >= spec.sentiment_min

    selected = score[eligible].dropna().sort_values(ascending=False).head(TOP_N)
    raw_weights = (selected.clip(lower=0.01) ** 2) / (signal_frame.loc[selected.index, "vol_20"] + 1e-6)
    return cap_weights(raw_weights, MAX_WEIGHT)


def performance_metrics(label: str, daily_returns: pd.Series) -> dict[str, float | str]:
    nav = (1 + daily_returns).cumprod()
    annual_factor = 252 / len(daily_returns)
    cumulative_return = nav.iloc[-1] - 1
    annual_return = nav.iloc[-1] ** annual_factor - 1
    annual_volatility = daily_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility else np.nan
    drawdown = nav / nav.cummax() - 1

    return {
        "portfolio": label,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "win_rate": (daily_returns > 0).mean(),
    }


def run_strategy(
    close: pd.DataFrame,
    tickers: list[str],
    signals: dict[pd.Timestamp, tuple[pd.Timestamp, pd.DataFrame]],
    spec: StrategySpec,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change().fillna(0.0)
    test_returns = returns.loc[TEST_START:TEST_END]
    rebalance_dates = pd.DatetimeIndex(signals.keys())
    weights = pd.DataFrame(0.0, index=test_returns.index, columns=tickers)
    rebalance_rows = []

    for index, rebalance_date in enumerate(rebalance_dates):
        signal_date, signal_frame = signals[rebalance_date]
        selected_weights = select_weights(signal_frame, spec)

        next_rebalance = (
            rebalance_dates[index + 1]
            if index + 1 < len(rebalance_dates)
            else test_returns.index[-1] + pd.Timedelta(days=1)
        )
        mask = (weights.index >= rebalance_date) & (weights.index < next_rebalance)
        weights.loc[mask, selected_weights.index] = selected_weights.values

        score = (
            spec.quant_weight * signal_frame["quant"]
            + spec.quality_weight * signal_frame["quality"]
            + spec.sentiment_weight * signal_frame["sentiment"]
        )
        for ticker, weight in selected_weights.sort_values(ascending=False).items():
            rebalance_rows.append(
                {
                    "strategy": spec.name,
                    "rebalance_date": rebalance_date.date().isoformat(),
                    "signal_date": signal_date.date().isoformat(),
                    "tic": ticker,
                    "weight": weight,
                    "score": score.loc[ticker],
                    "quant": signal_frame.loc[ticker, "quant"],
                    "quality": signal_frame.loc[ticker, "quality"],
                    "sentiment": signal_frame.loc[ticker, "sentiment"],
                    "ret_20": signal_frame.loc[ticker, "ret_20"],
                    "ret_3m": signal_frame.loc[ticker, "ret_3m"],
                }
            )

    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    strategy_returns = (weights.shift(1).fillna(weights) * test_returns[tickers]).sum(axis=1)
    strategy_returns = strategy_returns - TRANSACTION_COST * turnover
    return strategy_returns, weights, pd.DataFrame(rebalance_rows)


def plot_nav(nav: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for column in ["Integrated AI Strategy", "SPY", "QQQ", "FinRL only", "FinRL + FinRobot"]:
        if column in nav:
            plt.plot(nav.index, nav[column], label=column, linewidth=2 if column == "Integrated AI Strategy" else 1.4)
    plt.title("GR5398 Final Assignment Backtest: 2026-01-01 to 2026-04-30")
    plt.ylabel("Normalized NAV")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true", help="Download a fresh Yahoo Finance price cache.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    close, volume = load_prices(refresh=args.refresh_data)
    universe = [ticker for ticker in load_universe() if ticker in close.columns]
    fundamentals = pd.read_csv(FUNDAMENTAL_FILE, parse_dates=["reportdate", "date"])
    signals = build_monthly_signals(close, volume, universe, fundamentals)
    test_returns = close.pct_change().fillna(0.0).loc[TEST_START:TEST_END]

    all_returns: dict[str, pd.Series] = {}
    all_weights = {}
    rebalance_tables = []

    for spec in STRATEGIES:
        returns, weights, rebalance_table = run_strategy(close, universe, signals, spec)
        all_returns[spec.name] = returns
        all_weights[spec.name] = weights
        rebalance_tables.append(rebalance_table)

    all_returns["SPY"] = test_returns["SPY"]
    all_returns["QQQ"] = test_returns["QQQ"]

    returns_frame = pd.DataFrame(all_returns)
    nav = (1 + returns_frame).cumprod()
    metrics = pd.DataFrame([performance_metrics(label, returns_frame[label]) for label in returns_frame.columns])

    returns_frame.to_csv(OUTPUT_DIR / "daily_returns.csv", index_label="date")
    nav.to_csv(OUTPUT_DIR / "nav.csv", index_label="date")
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    all_weights["Integrated AI Strategy"].to_csv(OUTPUT_DIR / "daily_weights.csv", index_label="date")
    pd.concat(rebalance_tables, ignore_index=True).to_csv(OUTPUT_DIR / "rebalance_weights.csv", index=False)

    signal_exports = []
    for rebalance_date, (signal_date, signal_frame) in signals.items():
        export = signal_frame.copy()
        export["rebalance_date"] = rebalance_date.date().isoformat()
        export["signal_date"] = signal_date.date().isoformat()
        export["tic"] = export.index
        signal_exports.append(export.reset_index(drop=True))
    pd.concat(signal_exports, ignore_index=True).to_csv(OUTPUT_DIR / "signal_snapshot.csv", index=False)

    plot_nav(nav, OUTPUT_DIR / "equity_curve.png")

    display = metrics.copy()
    for column in ["cumulative_return", "annual_return", "annual_volatility", "max_drawdown", "win_rate"]:
        display[column] = display[column].map(lambda x: f"{x:.2%}")
    display["sharpe"] = display["sharpe"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
