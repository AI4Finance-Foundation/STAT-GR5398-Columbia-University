"""Build and backtest the FRG layered portfolio.

FRG is short for the FinRL / FinRobot / FinGPT layered selection pipeline.
The input file is still expected to contain one row per ticker/rebalance date
with a `composite_signal` column.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "frg_portfolio"
DEFAULT_STRATEGY_NAME = "FRG_Portfolio"
DEFAULT_SIGNAL_CSV = REPO_ROOT / "results" / "fingpt_score_only_signals.csv"
DEFAULT_ROBOT_CSV = REPO_ROOT / "results" / "finrobot2_rebalanced_panel(2).csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / f"{RESULT_PREFIX}_backtest"
DEFAULT_EXPECTED_ENVS = ("stock selection", "stock_selection", "stock_selection_env")
DEFAULT_MARKET_SLEEVE_CANDIDATES = ("SPY", "QQQ")
DEFAULT_MOMENTUM_HISTORY_DAYS = 174

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "requests",
    "yfinance",
    "bt",
    "matplotlib",
    "scipy",
)


def _normalise_env_name(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace("_", " ").replace("-", " ")


def _detect_conda_env() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name


def _preflight(expected_envs: Iterable[str], skip_env_check: bool) -> None:
    """Fail before doing any data work if the runtime is not ready."""
    detected = _detect_conda_env()
    expected = tuple(env for env in expected_envs if env)

    if expected and not skip_env_check:
        detected_norm = _normalise_env_name(detected)
        expected_norm = {_normalise_env_name(env) for env in expected}
        if detected_norm not in expected_norm:
            raise SystemExit(
                "Environment check failed.\n"
                f"  Expected conda env: {', '.join(expected)}\n"
                f"  Detected env/interpreter: {detected or sys.executable}\n"
                "Run with the stock selection conda env, for example:\n"
                '  conda run -n "stock selection" python scripts/run_fingpt_score_only_backtest.py\n'
                "Or pass --skip-env-check if you intentionally use another ready env."
            )

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing required Python package(s): "
            + ", ".join(missing)
            + "\nActivate/install the stock selection environment before rerunning."
        )


def _configure_runtime_paths() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.dont_write_bytecode = True

    python_env_root = Path(sys.executable).resolve().parent
    dll_dirs = [
        python_env_root / "Library" / "bin",
        python_env_root / "Scripts",
        python_env_root,
    ]
    if sys.platform == "win32":
        existing_path = os.environ.get("PATH", "")
        env_paths = [str(path) for path in dll_dirs if path.exists()]
        os.environ["PATH"] = ";".join(env_paths + [existing_path])
        for path in dll_dirs:
            if path.exists():
                os.add_dll_directory(str(path))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the weekly FRG layered portfolio."
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=DEFAULT_SIGNAL_CSV,
        help="Input signal CSV. Default: results/fingpt_score_only_signals.csv",
    )
    parser.add_argument(
        "--robot-panel",
        type=Path,
        default=DEFAULT_ROBOT_CSV,
        help="FinRobot/FinRL weekly panel CSV. Default: results/finrobot2_rebalanced_panel(2).csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated backtest outputs. Default: results/{RESULT_PREFIX}_backtest",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default=DEFAULT_STRATEGY_NAME,
        help=f"Display name used in reports and plots. Default: {DEFAULT_STRATEGY_NAME}.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of tickers to keep per rebalance date.",
    )
    parser.add_argument(
        "--selection-model",
        choices=("score", "momentum", "frg"),
        default="frg",
        help=(
            "Stock basket construction model. frg merges the FinRobot/FinRL panel "
            "with FinGPT signals; score and momentum use FinGPT signals only."
        ),
    )
    parser.add_argument(
        "--weighting",
        choices=("score_shifted", "equal", "positive_only", "softmax"),
        default="score_shifted",
        help=(
            "How selected scores become long-only weights. score_shifted ranks by "
            "composite_signal and shifts scores to non-negative values."
        ),
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.25,
        help="Temperature used only when --weighting softmax.",
    )
    parser.add_argument(
        "--price-source",
        choices=("yahoo_chart", "yfinance"),
        default="yahoo_chart",
        help="Historical price source. yahoo_chart uses Yahoo's direct chart API.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional price download start date YYYY-MM-DD. Defaults to first signal minus 7 days.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional price download end date YYYY-MM-DD. Defaults to last signal plus 7 days.",
    )
    parser.add_argument(
        "--trade-lag-days",
        type=int,
        default=1,
        help=(
            "Number of trading days to delay execution after each signal date. "
            "Default: 1 to avoid same-day signal lookahead."
        ),
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital passed to BacktestEngine.",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Flat transaction cost rate passed to BacktestEngine.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=("SPY", "QQQ"),
        help="Optional yfinance benchmarks backtested as buy-and-hold via BacktestEngine.",
    )
    parser.add_argument(
        "--market-sleeve",
        choices=("none", "spy", "qqq", "momentum"),
        default="qqq",
        help=(
            "Optional market ETF sleeve blended with the FinGPT stock basket. "
            "momentum picks the strongest ticker from --market-sleeve-candidates."
        ),
    )
    parser.add_argument(
        "--market-sleeve-weight",
        type=float,
        default=0.60,
        help="Portfolio weight assigned to the market sleeve. Default: 0.60.",
    )
    parser.add_argument(
        "--market-sleeve-candidates",
        nargs="*",
        default=DEFAULT_MARKET_SLEEVE_CANDIDATES,
        help="Ticker candidates used only when --market-sleeve momentum.",
    )
    parser.add_argument(
        "--market-momentum-window",
        type=int,
        default=20,
        help="Trading-day lookback used by --market-sleeve momentum.",
    )
    parser.add_argument(
        "--stock-momentum-window",
        type=int,
        default=80,
        help="Prior trading-day lookback used by --selection-model momentum.",
    )
    parser.add_argument(
        "--stock-momentum-top-n",
        type=int,
        default=3,
        help="Number of positive-momentum FinGPT candidates kept by --selection-model momentum.",
    )
    parser.add_argument(
        "--stock-momentum-weighting",
        choices=("equal", "momentum", "blend"),
        default="blend",
        help="Weighting inside the momentum-filtered stock basket.",
    )
    parser.add_argument(
        "--frg-robot-tilt",
        type=float,
        default=0.10,
        help="How strongly FinRobot/FinRL quality tilts final stock weights. Default: 0.10.",
    )
    parser.add_argument(
        "--frg-robot-gate-floor",
        type=float,
        default=0.25,
        help="Minimum multiplier applied to FinRobot rejects/reductions. Default: 0.25.",
    )
    parser.add_argument(
        "--expected-env",
        nargs="*",
        default=DEFAULT_EXPECTED_ENVS,
        help="Accepted conda environment names. Use empty with --skip-env-check to bypass.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip conda env-name validation, but still require Python packages.",
    )
    return parser.parse_args()


def _import_runtime_modules():
    import numpy as np
    import pandas as pd
    import requests
    import yfinance as yf

    from src.backtest.backtest_engine import BacktestConfig, BacktestEngine

    return np, pd, requests, yf, BacktestConfig, BacktestEngine


def _load_signals(signal_path: Path, pd) -> object:
    if not signal_path.exists():
        raise FileNotFoundError(f"Signal CSV not found: {signal_path}")

    df = pd.read_csv(signal_path)
    required = {"ticker", "rebalance_date", "composite_signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Signal CSV missing required column(s): {sorted(missing)}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], errors="coerce").dt.normalize()
    df["composite_signal"] = pd.to_numeric(df["composite_signal"], errors="coerce")

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

    df = df.dropna(subset=["ticker", "rebalance_date", "composite_signal"])
    if df.empty:
        raise ValueError("No valid signal rows after cleaning.")

    return df.sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)


def _load_frg_panel(robot_path: Path, signal_path: Path, pd) -> object:
    if not robot_path.exists():
        raise FileNotFoundError(f"FinRobot panel CSV not found: {robot_path}")

    robot = pd.read_csv(robot_path)
    required_robot = {
        "date",
        "ticker",
        "selected",
        "finrl_score",
        "finrobot_score",
        "finrobot_multiplier",
        "after_finrobot_weight",
        "technical_score",
        "ml_ensemble_score",
    }
    missing_robot = required_robot - set(robot.columns)
    if missing_robot:
        raise ValueError(f"FinRobot panel missing required column(s): {sorted(missing_robot)}")

    gpt = _load_signals(signal_path, pd)
    gpt_cols = [
        "ticker",
        "rebalance_date",
        "sentiment_score",
        "confidence_score",
        "materiality_score",
        "composite_signal",
    ]

    robot = robot.copy()
    robot["ticker"] = robot["ticker"].astype(str).str.strip().str.upper()
    robot["date"] = pd.to_datetime(robot["date"], errors="coerce").dt.normalize()
    robot = robot.dropna(subset=["date", "ticker"])

    panel = robot.merge(
        gpt[gpt_cols],
        left_on=["date", "ticker"],
        right_on=["rebalance_date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    panel["rebalance_date"] = panel["rebalance_date"].fillna(panel["date"])
    panel["composite_signal"] = pd.to_numeric(panel["composite_signal"], errors="coerce").fillna(0.0)
    for col in [
        "selected",
        "finrl_score",
        "finrobot_score",
        "finrobot_multiplier",
        "after_finrobot_weight",
        "technical_score",
        "ml_ensemble_score",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)

    return panel.sort_values(["rebalance_date", "ticker"]).reset_index(drop=True)


def _download_close_prices_yfinance(tickers: list[str], start: str, end: str, yf, pd) -> object:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise ValueError("No price data downloaded from yfinance.")

    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded yfinance data does not contain Close prices.")
        close = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Downloaded yfinance data does not contain Close prices.")
        close = data[["Close"]].copy()
        close.columns = tickers[:1]

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index().reindex(columns=tickers)
    close = close.ffill().dropna(axis=1, how="all").dropna(axis=0, how="all")

    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise ValueError(f"Price data missing ticker(s): {missing}")
    return close


def _to_epoch_seconds(date_value: str, pd) -> int:
    return int(pd.Timestamp(date_value, tz="UTC").timestamp())


def _download_one_yahoo_chart(ticker: str, start: str, end: str, requests, pd) -> object:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": _to_epoch_seconds(start, pd),
        "period2": _to_epoch_seconds(end, pd),
        "interval": "1d",
        "events": "history",
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code == 429:
        raise RuntimeError(f"Yahoo chart rate-limited ticker {ticker}.")
    response.raise_for_status()

    payload = response.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {ticker}: {error}")
    results = payload.get("chart", {}).get("result") or []
    if not results:
        raise ValueError(f"Yahoo chart returned no data for {ticker}.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quote = (indicators.get("quote") or [{}])[0]
    adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    close_values = adjclose or quote.get("close")
    if not timestamps or not close_values:
        raise ValueError(f"Yahoo chart returned no close prices for {ticker}.")

    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .tz_localize(None)
        .normalize()
    )
    series = pd.Series(close_values, index=dates, name=ticker, dtype="float64")
    return series.dropna()


def _download_close_prices_yahoo_chart(tickers: list[str], start: str, end: str, requests, pd) -> object:
    import time

    series_list = []
    failures = []
    for ticker in tickers:
        try:
            series_list.append(_download_one_yahoo_chart(ticker, start, end, requests, pd))
            time.sleep(0.2)
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")

    if failures:
        raise ValueError("Failed to download price data:\n" + "\n".join(failures))
    if not series_list:
        raise ValueError("No price data downloaded from Yahoo chart.")

    close = pd.concat(series_list, axis=1).sort_index()
    close = close.reindex(columns=tickers).ffill().dropna(axis=1, how="all").dropna(axis=0, how="all")
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise ValueError(f"Price data missing ticker(s): {missing}")
    return close


def _download_close_prices(tickers: list[str], start: str, end: str, args, requests, yf, pd) -> object:
    if args.price_source == "yfinance":
        return _download_close_prices_yfinance(tickers, start, end, yf, pd)
    return _download_close_prices_yahoo_chart(tickers, start, end, requests, pd)


def _default_price_window(signals, pd) -> tuple[str, str]:
    min_date = signals["rebalance_date"].min() - pd.Timedelta(days=7)
    max_date = signals["rebalance_date"].max() + pd.Timedelta(days=7)
    return min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")


def _resolve_rebalance_dates(signals, price_index, pd) -> object:
    available = pd.DatetimeIndex(price_index).sort_values().normalize()
    if available.empty:
        raise ValueError("Price data has no trading dates.")

    resolved = {}
    for target in signals["rebalance_date"].drop_duplicates().sort_values():
        eligible = available[available <= target]
        if eligible.empty:
            raise ValueError(
                f"No trading date on or before signal date {target.strftime('%Y-%m-%d')}."
            )
        resolved[target] = eligible[-1]

    out = signals.copy()
    out["date"] = out["rebalance_date"].map(resolved)
    return out


def _assign_weights_for_group(group, args, np, pd) -> object:
    ranked = group.sort_values(["composite_signal", "ticker"], ascending=[False, True]).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= args.top_n
    ranked["target_weight"] = 0.0

    selected = ranked[ranked["selected"]].copy()
    if args.weighting == "positive_only":
        selected = selected[selected["composite_signal"] > 0].copy()
        ranked["selected"] = ranked["ticker"].isin(selected["ticker"])

    if selected.empty:
        return ranked

    scores = selected["composite_signal"].astype(float)
    if args.weighting == "equal":
        weights = pd.Series(1.0 / len(selected), index=selected.index)
    elif args.weighting == "positive_only":
        total = scores.sum()
        weights = scores / total if total > 0 else pd.Series(1.0 / len(selected), index=selected.index)
    elif args.weighting == "softmax":
        temperature = max(float(args.softmax_temperature), 1e-9)
        shifted = (scores / temperature) - (scores / temperature).max()
        exp_scores = np.exp(shifted)
        weights = pd.Series(exp_scores / exp_scores.sum(), index=selected.index)
    else:
        spread = scores.max() - scores.min()
        if spread <= 1e-12:
            weights = pd.Series(1.0 / len(selected), index=selected.index)
        else:
            shifted = scores - scores.min() + 1e-6
            weights = shifted / shifted.sum()

    ranked.loc[weights.index, "target_weight"] = weights.astype(float)
    return ranked


def _build_portfolio(signals, price_index, args, np, pd) -> tuple[object, object]:
    resolved = _resolve_rebalance_dates(signals, price_index, pd)
    ranked_frames = [
        _assign_weights_for_group(group, args, np, pd)
        for _, group in resolved.groupby("date", sort=True)
    ]
    ranked = (
        pd.concat(ranked_frames, ignore_index=True)
        .sort_values(["date", "rank", "ticker"])
        .reset_index(drop=True)
    )

    weight_signals = (
        ranked.pivot_table(
            index="date",
            columns="ticker",
            values="target_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .astype(float)
    )

    return ranked, weight_signals


def _prior_momentum(ticker: str, rebalance_date, prices, lookback: int) -> float:
    if ticker not in prices.columns:
        return 0.0
    history = prices.loc[prices.index < rebalance_date, ticker].dropna()
    if len(history) < 2:
        return 0.0
    effective_lookback = min(max(int(lookback), 1), len(history) - 1)
    return float(history.iloc[-1] / history.iloc[-effective_lookback - 1] - 1.0)


def _assign_momentum_weights_for_group(group, rebalance_date, prices, args, np, pd) -> object:
    ranked = group.copy()
    ranked["prior_momentum"] = [
        _prior_momentum(str(ticker), rebalance_date, prices, args.stock_momentum_window)
        for ticker in ranked["ticker"]
    ]
    ranked = ranked.sort_values(
        ["prior_momentum", "composite_signal", "ticker"],
        ascending=[False, False, True],
    ).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["selected"] = False
    ranked["target_weight"] = 0.0

    candidates = ranked[ranked["prior_momentum"] > 0].copy()
    if candidates.empty:
        candidates = ranked.copy()
    selected = candidates.head(max(int(args.stock_momentum_top_n), 1)).copy()
    if selected.empty:
        return ranked

    ranked.loc[selected.index, "selected"] = True
    if args.stock_momentum_weighting == "equal":
        weights = pd.Series(1.0 / len(selected), index=selected.index)
    elif args.stock_momentum_weighting == "momentum":
        raw = selected["prior_momentum"].clip(lower=0).astype(float) + 1e-6
        weights = raw / raw.sum()
    else:
        scores = selected["composite_signal"].astype(float)
        score_raw = scores - scores.min() + 1e-6
        momentum_raw = selected["prior_momentum"].clip(lower=0).astype(float) + 1e-6
        raw = score_raw * momentum_raw
        weights = raw / raw.sum() if raw.sum() > 0 else pd.Series(1.0 / len(selected), index=selected.index)

    ranked.loc[weights.index, "target_weight"] = weights.astype(float)
    return ranked


def _build_momentum_portfolio(signals, prices, args, np, pd) -> tuple[object, object]:
    resolved = _resolve_rebalance_dates(signals, prices.index, pd)
    ranked_frames = [
        _assign_momentum_weights_for_group(group, date, prices, args, np, pd)
        for date, group in resolved.groupby("date", sort=True)
    ]
    ranked = (
        pd.concat(ranked_frames, ignore_index=True)
        .sort_values(["date", "rank", "ticker"])
        .reset_index(drop=True)
    )

    weight_signals = (
        ranked.pivot_table(
            index="date",
            columns="ticker",
            values="target_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .astype(float)
    )
    return ranked, weight_signals


def _percent_rank(series, pd) -> object:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True).fillna(0.0)


def _assign_frg_weights_for_group(group, rebalance_date, prices, args, np, pd) -> object:
    ranked = group.copy()
    ranked["prior_momentum"] = [
        _prior_momentum(str(ticker), rebalance_date, prices, args.stock_momentum_window)
        for ticker in ranked["ticker"]
    ]
    ranked["selected"] = pd.to_numeric(ranked["selected"], errors="coerce").fillna(0).astype(int).eq(1)
    ranked["robot_gate"] = (
        pd.to_numeric(ranked["finrobot_multiplier"], errors="coerce")
        .fillna(1.0)
        .clip(lower=max(float(args.frg_robot_gate_floor), 0.0))
    )
    ranked["r_finrl"] = _percent_rank(ranked["finrl_score"], pd)
    ranked["r_robot"] = _percent_rank(ranked["finrobot_score"], pd)
    ranked["r_gpt"] = _percent_rank(ranked["composite_signal"], pd)
    ranked["r_momentum"] = _percent_rank(ranked["prior_momentum"], pd)
    ranked["robot_quality"] = (
        0.5 * ranked["r_robot"]
        + 0.5 * ranked["r_finrl"]
    )
    ranked["target_weight"] = 0.0

    candidates = ranked[ranked["selected"]].copy()
    if candidates.empty:
        candidates = ranked.copy()
    positive_momentum = candidates[candidates["prior_momentum"] > 0].copy()
    if not positive_momentum.empty:
        candidates = positive_momentum

    candidates = candidates.sort_values(
        ["prior_momentum", "composite_signal", "ticker"],
        ascending=[False, False, True],
    ).head(max(int(args.stock_momentum_top_n), 1)).copy()

    ranked = ranked.sort_values(
        ["prior_momentum", "composite_signal", "ticker"],
        ascending=[False, False, True],
    ).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["frg_selected"] = ranked.index.isin(candidates.index)

    if candidates.empty:
        return ranked

    score_raw = pd.to_numeric(candidates["composite_signal"], errors="coerce").fillna(0.0)
    score_raw = score_raw - score_raw.min() + 1e-6
    momentum_raw = candidates["prior_momentum"].clip(lower=0.0) + 1e-6
    robot_tilt = max(0.0, min(float(args.frg_robot_tilt), 1.0))
    robot_quality = (1.0 - robot_tilt) + robot_tilt * candidates["robot_quality"]
    raw = score_raw * momentum_raw * robot_quality * candidates["robot_gate"]
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=candidates.index)
    weights = raw / raw.sum()

    ranked.loc[weights.index, "target_weight"] = weights.astype(float)
    return ranked


def _build_frg_portfolio(panel, prices, args, np, pd) -> tuple[object, object]:
    resolved = _resolve_rebalance_dates(panel, prices.index, pd)
    ranked_frames = [
        _assign_frg_weights_for_group(group, date, prices, args, np, pd)
        for date, group in resolved.groupby("date", sort=True)
    ]
    ranked = (
        pd.concat(ranked_frames, ignore_index=True)
        .sort_values(["date", "rank", "ticker"])
        .reset_index(drop=True)
    )
    weight_signals = (
        ranked.pivot_table(
            index="date",
            columns="ticker",
            values="target_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .astype(float)
    )
    return ranked, weight_signals


def _clean_ticker_list(values: Iterable[str]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _market_sleeve_tickers(args) -> list[str]:
    sleeve = str(args.market_sleeve).lower()
    sleeve_weight = max(0.0, min(float(args.market_sleeve_weight), 1.0))
    if sleeve == "none" or sleeve_weight <= 0:
        return []
    if sleeve in {"spy", "qqq"}:
        return [sleeve.upper()]
    return _clean_ticker_list(args.market_sleeve_candidates)


def _select_market_sleeve_ticker(rebalance_date, prices, args, pd) -> tuple[str | None, str]:
    sleeve = str(args.market_sleeve).lower()
    if sleeve in {"spy", "qqq"}:
        return sleeve.upper(), "fixed"
    if sleeve != "momentum":
        return None, "disabled"

    candidates = [ticker for ticker in _clean_ticker_list(args.market_sleeve_candidates) if ticker in prices.columns]
    if not candidates:
        return None, "no_candidates"

    window = max(int(args.market_momentum_window), 1)
    history = prices.loc[prices.index < pd.Timestamp(rebalance_date), candidates].dropna(how="all")
    if len(history) < 2:
        valid = history.dropna(axis=1, how="any").columns.tolist()
        return (valid[0], "fallback") if valid else (candidates[0], "fallback")

    lookback = min(window, len(history) - 1)
    recent = history.iloc[-1]
    prior = history.iloc[-lookback - 1]
    momentum = (recent / prior - 1.0).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if momentum.empty:
        return candidates[0], "fallback"
    return str(momentum.sort_values(ascending=False).index[0]), f"{lookback}d_momentum"


def _apply_market_sleeve(weekly_weight_signals, prices, args, pd) -> tuple[object, object]:
    """Blend the FinGPT stock basket with an ETF sleeve at each rebalance."""
    sleeve_weight = max(0.0, min(float(args.market_sleeve_weight), 1.0))
    if str(args.market_sleeve).lower() == "none" or sleeve_weight <= 0:
        details = pd.DataFrame(columns=["date", "market_sleeve", "market_sleeve_weight", "reason"])
        return weekly_weight_signals, details

    base = weekly_weight_signals.copy().sort_index().fillna(0.0)
    row_sums = base.sum(axis=1).replace(0.0, pd.NA)
    base = base.div(row_sums, axis=0).fillna(0.0).mul(1.0 - sleeve_weight)

    records = []
    for rebalance_date in base.index:
        ticker, reason = _select_market_sleeve_ticker(rebalance_date, prices, args, pd)
        if not ticker:
            records.append(
                {
                    "date": rebalance_date,
                    "market_sleeve": "",
                    "market_sleeve_weight": 0.0,
                    "reason": reason,
                }
            )
            continue
        if ticker not in prices.columns:
            raise ValueError(f"Market sleeve ticker {ticker} is missing from downloaded price data.")
        if ticker not in base.columns:
            base[ticker] = 0.0
        base.loc[rebalance_date, ticker] = base.loc[rebalance_date, ticker] + sleeve_weight
        records.append(
            {
                "date": rebalance_date,
                "market_sleeve": ticker,
                "market_sleeve_weight": sleeve_weight,
                "reason": reason,
            }
        )

    base = base.reindex(columns=sorted(base.columns)).fillna(0.0)
    details = pd.DataFrame(records)
    if not details.empty:
        details["date"] = pd.to_datetime(details["date"]).dt.strftime("%Y-%m-%d")
    return base, details


def _apply_trade_lag(weight_signals, price_index, lag_days: int, pd) -> object:
    lag_days = max(int(lag_days), 0)
    if lag_days == 0:
        return weight_signals

    trading_index = pd.DatetimeIndex(price_index).sort_values()
    shifted_dates = []
    keep_positions = []
    for pos, signal_date in enumerate(pd.DatetimeIndex(weight_signals.index)):
        next_pos = trading_index.searchsorted(signal_date, side="right") + lag_days - 1
        if next_pos < len(trading_index):
            shifted_dates.append(trading_index[next_pos])
            keep_positions.append(pos)

    if not keep_positions:
        raise ValueError("No rebalance dates remain after applying trade lag.")

    lagged = weight_signals.iloc[keep_positions].copy()
    lagged.index = pd.DatetimeIndex(shifted_dates)
    lagged = lagged[~lagged.index.duplicated(keep="last")].sort_index()
    return lagged


def _save_series(series, path: Path, value_name: str) -> None:
    frame = series.rename(value_name).reset_index()
    frame.columns = ["date", value_name]
    frame.to_csv(path, index=False)


def _expand_weights_to_drift_targets(rebalance_weights, prices, np, pd) -> object:
    """Create daily target weights that match passive drift between rebalances.

    BacktestEngine rebalances on every target-weight row after forward filling.
    Supplying drifted daily targets preserves weekly rebalance intent while
    keeping the existing engine unchanged.
    """
    rebalance_weights = (
        rebalance_weights.reindex(columns=prices.columns, fill_value=0.0)
        .sort_index()
        .astype(float)
    )
    daily = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal_dates = [date for date in rebalance_weights.index if date in prices.index]

    for idx, rebalance_date in enumerate(signal_dates):
        if idx + 1 < len(signal_dates):
            next_rebalance = signal_dates[idx + 1]
            period_index = prices.index[(prices.index >= rebalance_date) & (prices.index < next_rebalance)]
        else:
            period_index = prices.index[prices.index >= rebalance_date]
        if period_index.empty:
            continue

        target = rebalance_weights.loc[rebalance_date].fillna(0.0)
        if target.sum() <= 0:
            continue

        start_prices = prices.loc[rebalance_date].replace(0, np.nan)
        relative_prices = prices.loc[period_index].div(start_prices, axis=1)
        drift_values = relative_prices.mul(target, axis=1)
        denom = drift_values.sum(axis=1).replace(0, np.nan)
        daily.loc[period_index] = drift_values.div(denom, axis=0).fillna(0.0)

    return daily


def _save_result_outputs(
    result,
    ranked,
    weekly_weight_signals,
    bt_weight_signals,
    output_dir: Path,
    pd,
    sleeve_details=None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all_ranks": output_dir / f"{RESULT_PREFIX}_all_ranks.csv",
        "weekly_weights": output_dir / f"{RESULT_PREFIX}_weight_signals.csv",
        "bt_weights": output_dir / f"{RESULT_PREFIX}_bt_weight_signals.csv",
        "portfolio_values": output_dir / f"{RESULT_PREFIX}_portfolio_values.csv",
        "portfolio_returns": output_dir / f"{RESULT_PREFIX}_portfolio_returns.csv",
        "metrics": output_dir / f"{RESULT_PREFIX}_metrics.csv",
    }
    if sleeve_details is not None and not sleeve_details.empty:
        paths["market_sleeve"] = output_dir / f"{RESULT_PREFIX}_market_sleeve.csv"

    ranked_out = ranked.copy()
    ranked_out["rebalance_date"] = ranked_out["rebalance_date"].dt.strftime("%Y-%m-%d")
    ranked_out["date"] = ranked_out["date"].dt.strftime("%Y-%m-%d")
    ranked_out.to_csv(paths["all_ranks"], index=False)

    weights_out = weekly_weight_signals.copy()
    weights_out.index = weights_out.index.strftime("%Y-%m-%d")
    weights_out.to_csv(paths["weekly_weights"], index_label="date")

    bt_weights_out = bt_weight_signals.copy()
    bt_weights_out.index = bt_weights_out.index.strftime("%Y-%m-%d")
    bt_weights_out.to_csv(paths["bt_weights"], index_label="date")

    _save_series(result.portfolio_values, paths["portfolio_values"], "portfolio_value")
    _save_series(result.portfolio_returns, paths["portfolio_returns"], "portfolio_return")

    metrics = pd.DataFrame([result.metrics])
    metrics.insert(0, "strategy_name", result.strategy_name)
    metrics.to_csv(paths["metrics"], index=False)
    if "market_sleeve" in paths:
        sleeve_details.to_csv(paths["market_sleeve"], index=False)
    return paths


def _run_buy_hold_benchmarks(benchmarks, price_data, config, BacktestEngine, pd) -> object:
    rows = []
    for ticker in benchmarks:
        if ticker not in price_data.columns:
            continue
        weights = pd.DataFrame({ticker: 1.0}, index=[price_data.index.min()])
        engine = BacktestEngine(config)
        result = engine.run_backtest(f"{ticker}_BuyHold", price_data[[ticker]], weights)
        row = {"strategy_name": result.strategy_name}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _drawdown_from_returns(returns, pd) -> object:
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    return cumulative.div(running_max).sub(1.0).fillna(0.0)


def _format_metric(value, kind: str) -> str:
    if value is None:
        return ""
    if kind == "percent":
        return f"{float(value):.2%}"
    return f"{float(value):.2f}"


def _save_comparison_plot(
    result,
    backtest_prices,
    weight_signals,
    benchmark_df,
    benchmarks,
    save_path: Path,
    pd,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    colors = {
        result.strategy_name: "#0072B2",
        "SPY": "#D55E00",
        "QQQ": "#009E73",
    }
    line_styles = {
        result.strategy_name: "-",
        "SPY": "--",
        "QQQ": ":",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"height_ratios": [1.15, 1.0]})
    ax_curve, ax_drawdown, ax_table, ax_alloc = axes.flatten()

    curves = pd.DataFrame(index=result.portfolio_values.index)
    curves[result.strategy_name] = result.portfolio_values / result.portfolio_values.iloc[0] - 1.0
    for ticker in benchmarks:
        if ticker in backtest_prices.columns:
            series = backtest_prices[ticker].dropna()
            if not series.empty:
                curves[ticker] = series / series.iloc[0] - 1.0

    for column in curves.columns:
        color = colors.get(column, None)
        linewidth = 3.4 if column == result.strategy_name else 2.4
        alpha = 1.0 if column == result.strategy_name else 0.9
        curves[column].dropna().plot(
            ax=ax_curve,
            label=column,
            linewidth=linewidth,
            color=color,
            linestyle=line_styles.get(column, "-"),
            alpha=alpha,
        )
    ax_curve.set_title("Growth of $1", fontsize=13, fontweight="bold")
    ax_curve.set_ylabel("Total return")
    ax_curve.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_curve.legend(loc="upper left", frameon=True)

    drawdowns = pd.DataFrame(index=curves.index)
    drawdowns[result.strategy_name] = _drawdown_from_returns(result.portfolio_returns, pd)
    for ticker in benchmarks:
        if ticker in backtest_prices.columns:
            drawdowns[ticker] = _drawdown_from_returns(backtest_prices[ticker].pct_change(), pd)
    for column in drawdowns.columns:
        color = colors.get(column, None)
        linewidth = 2.8 if column == result.strategy_name else 2.1
        drawdowns[column].dropna().plot(
            ax=ax_drawdown,
            label=column,
            linewidth=linewidth,
            color=color,
            linestyle=line_styles.get(column, "-"),
            alpha=0.95,
        )
    ax_drawdown.fill_between(
        drawdowns.index,
        drawdowns[result.strategy_name].fillna(0.0).to_numpy(),
        0.0,
        color="#1f77b4",
        alpha=0.12,
    )
    ax_drawdown.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_drawdown.legend(loc="lower left", frameon=True)

    rows = [
        {
            "name": result.strategy_name,
            "total_return": result.metrics.get("total_return"),
            "annual_return": result.metrics.get("annual_return"),
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "max_drawdown": result.metrics.get("max_drawdown"),
        }
    ]
    if benchmark_df is not None and not benchmark_df.empty:
        for _, row in benchmark_df.iterrows():
            name = str(row.get("strategy_name", "")).replace("_BuyHold", "")
            rows.append(
                {
                    "name": name,
                    "total_return": row.get("total_return"),
                    "annual_return": row.get("annual_return"),
                    "sharpe_ratio": row.get("sharpe_ratio"),
                    "max_drawdown": row.get("max_drawdown"),
                }
            )

    ax_table.axis("off")
    table_data = [
        [
            row["name"],
            _format_metric(row["total_return"], "percent"),
            _format_metric(row["annual_return"], "percent"),
            _format_metric(row["sharpe_ratio"], "number"),
            _format_metric(row["max_drawdown"], "percent"),
        ]
        for row in rows
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Strategy", "Total", "Annual", "Sharpe", "Max DD"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee7")
        if row_idx == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row_idx == 1:
            cell.set_facecolor("#edf5ff")
        else:
            cell.set_facecolor("#ffffff")
    ax_table.set_title("Metric Snapshot", fontsize=13, fontweight="bold")

    alloc = weight_signals.copy().fillna(0.0)
    avg_weights = alloc.mean().sort_values(ascending=False)
    top_cols = avg_weights[avg_weights > 0.01].head(7).index.tolist()
    if top_cols:
        alloc_plot = alloc[top_cols].copy()
        other = alloc.drop(columns=top_cols, errors="ignore").sum(axis=1)
        if (other > 0.01).any():
            alloc_plot["Other"] = other
        alloc_plot.plot.area(ax=ax_alloc, linewidth=0.0, alpha=0.88, cmap="tab20")
        ax_alloc.set_ylim(0, 1.0)
        ax_alloc.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_alloc.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    ax_alloc.set_title("Target Allocation", fontsize=13, fontweight="bold")
    ax_alloc.set_ylabel("Weight")

    for ax in (ax_curve, ax_drawdown, ax_alloc):
        ax.set_xlabel("")
        ax.grid(True, color="#e7ebf0", linewidth=0.8)

    fig.suptitle("FRG Portfolio Backtest vs Market Benchmarks", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _print_report(
    result,
    ranked,
    weight_signals,
    paths: dict[str, Path],
    output_dir: Path,
    benchmark_df=None,
    calendar_benchmark_df=None,
    calendar_start=None,
) -> None:
    latest_date = ranked["date"].max()
    latest = ranked[(ranked["date"] == latest_date) & (ranked["target_weight"] > 0)].copy()
    latest_weights = weight_signals.loc[weight_signals.index.max()].sort_values(ascending=False)
    latest_weights = latest_weights[latest_weights > 1e-6]

    print("\nFRG portfolio backtest complete.")
    print(f"Strategy: {result.strategy_name}")
    calendar_start_text = calendar_start.date() if hasattr(calendar_start, "date") else calendar_start
    print(f"Signal calendar start: {calendar_start_text if calendar_start_text is not None else 'n/a'}")
    print(f"Investable backtest start: {weight_signals.index.min().date()}")
    print(f"Backtest end: {result.portfolio_values.index.max().date()}")
    print(f"Rebalance dates: {weight_signals.shape[0]}")
    print(f"Tickers: {', '.join(weight_signals.columns)}")
    print(f"Output directory: {output_dir.resolve()}")

    print("\nLatest layered stock picks:")
    cols = ["date", "ticker", "composite_signal", "rank", "target_weight"]
    print(latest[cols].to_string(index=False))

    print("\nLatest portfolio target weights:")
    print(latest_weights.rename("target_weight").to_string())

    print("\nStrategy metrics:")
    for key in ["total_return", "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]:
        value = result.metrics.get(key)
        if value is not None:
            print(f"{key}: {value:.6f}")

    if benchmark_df is not None and not benchmark_df.empty:
        print("\nAligned benchmark metrics (same investable start as strategy):")
        metric_cols = ["strategy_name", "total_return", "annual_return", "sharpe_ratio", "max_drawdown"]
        print(benchmark_df[metric_cols].to_string(index=False))

    if calendar_benchmark_df is not None and not calendar_benchmark_df.empty:
        print("\nCalendar benchmark metrics (from original signal calendar start):")
        metric_cols = ["strategy_name", "total_return", "annual_return", "sharpe_ratio", "max_drawdown"]
        print(calendar_benchmark_df[metric_cols].to_string(index=False))

    print("\nGenerated files:")
    for path in paths.values():
        print(path.resolve())


def main() -> None:
    args = _parse_args()
    _preflight(args.expected_env, args.skip_env_check)
    _configure_runtime_paths()

    np, pd, requests, yf, BacktestConfig, BacktestEngine = _import_runtime_modules()

    if args.selection_model == "frg":
        signals = _load_frg_panel(args.robot_panel, args.signals, pd)
    else:
        signals = _load_signals(args.signals, pd)
    default_start, default_end = _default_price_window(signals, pd)
    if args.start:
        start = args.start
    elif args.selection_model in {"momentum", "frg"}:
        history_days = max(
            DEFAULT_MOMENTUM_HISTORY_DAYS,
            int(args.stock_momentum_window * 2) + 14,
        )
        start = (signals["rebalance_date"].min() - pd.Timedelta(days=history_days)).strftime("%Y-%m-%d")
    else:
        start = default_start
    end = args.end or default_end

    signal_tickers = sorted(signals["ticker"].unique().tolist())
    benchmarks = sorted({ticker.upper() for ticker in (args.benchmarks or [])})
    market_sleeve_tickers = _market_sleeve_tickers(args)
    download_tickers = sorted(set(signal_tickers) | set(benchmarks) | set(market_sleeve_tickers))
    prices = _download_close_prices(download_tickers, start, end, args, requests, yf, pd)

    if args.selection_model == "frg":
        ranked, weekly_weight_signals = _build_frg_portfolio(
            signals,
            prices,
            args,
            np,
            pd,
        )
    elif args.selection_model == "momentum":
        ranked, weekly_weight_signals = _build_momentum_portfolio(
            signals,
            prices,
            args,
            np,
            pd,
        )
    else:
        ranked, weekly_weight_signals = _build_portfolio(
            signals,
            prices.index,
            args,
            np,
            pd,
        )
    weekly_weight_signals, sleeve_details = _apply_market_sleeve(
        weekly_weight_signals,
        prices,
        args,
        pd,
    )
    weekly_weight_signals = _apply_trade_lag(
        weekly_weight_signals,
        prices.index,
        args.trade_lag_days,
        pd,
    )
    if weekly_weight_signals.empty or (weekly_weight_signals.sum(axis=1) <= 0).all():
        raise ValueError("No positive target weights were generated.")

    first_rebalance = weekly_weight_signals.index.min()
    backtest_prices = prices.loc[prices.index >= first_rebalance].copy()
    strategy_prices = backtest_prices[weekly_weight_signals.columns]
    bt_weight_signals = _expand_weights_to_drift_targets(
        weekly_weight_signals,
        strategy_prices,
        np,
        pd,
    )
    config = BacktestConfig(
        start_date=start,
        end_date=end,
        rebalance_freq="W",
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        benchmark_tickers=[],
        integer_positions=False,
    )
    engine = BacktestEngine(config)
    strategy_name = args.strategy_name.strip() or DEFAULT_STRATEGY_NAME
    result = engine.run_backtest(
        strategy_name,
        strategy_prices,
        bt_weight_signals,
    )

    paths = _save_result_outputs(
        result,
        ranked,
        weekly_weight_signals,
        bt_weight_signals,
        args.output_dir,
        pd,
        sleeve_details,
    )

    benchmark_df = _run_buy_hold_benchmarks(
        benchmarks,
        backtest_prices,
        config,
        BacktestEngine,
        pd,
    )
    if not benchmark_df.empty:
        benchmark_path = args.output_dir / f"{RESULT_PREFIX}_benchmark_metrics.csv"
        benchmark_df.to_csv(benchmark_path, index=False)
        paths["benchmark_metrics"] = benchmark_path

    signal_calendar_start = prices.index[prices.index >= signals["rebalance_date"].min()].min()
    calendar_prices = prices.loc[prices.index >= signal_calendar_start].copy()
    calendar_benchmark_df = _run_buy_hold_benchmarks(
        benchmarks,
        calendar_prices,
        config,
        BacktestEngine,
        pd,
    )
    if not calendar_benchmark_df.empty:
        calendar_benchmark_path = args.output_dir / f"{RESULT_PREFIX}_calendar_benchmark_metrics.csv"
        calendar_benchmark_df.to_csv(calendar_benchmark_path, index=False)
        paths["calendar_benchmark_metrics"] = calendar_benchmark_path

    plot_path = args.output_dir / f"{RESULT_PREFIX}_backtest.png"
    _save_comparison_plot(
        result,
        backtest_prices,
        weekly_weight_signals,
        benchmark_df,
        benchmarks,
        plot_path,
        pd,
    )
    paths["plot"] = plot_path

    _print_report(
        result,
        ranked,
        weekly_weight_signals,
        paths,
        args.output_dir,
        benchmark_df,
        calendar_benchmark_df,
        signal_calendar_start,
    )


if __name__ == "__main__":
    main()
