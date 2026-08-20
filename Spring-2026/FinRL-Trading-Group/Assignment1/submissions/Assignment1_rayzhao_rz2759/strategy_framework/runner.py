from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from strategy_framework.backtest_engine import BacktestConfig, resolve_effective_lags, run_backtest
from strategy_framework.interfaces import AllocationStrategy, StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class DataPaths:
    stock_selected_csv: str
    final_ratios_csv: str
    stock_price_csv: str
    output_step2_dir: str


def load_selected_stocks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df


def load_ticker_sector_map(path: str) -> dict[str, float]:
    sector_df = pd.read_csv(path)
    required_cols = {"tic", "gsector"}
    if not required_cols.issubset(set(sector_df.columns)):
        raise ValueError(f"{path} must contain columns {required_cols}.")

    keep_cols = ["tic", "gsector"] + (["date"] if "date" in sector_df.columns else [])
    sector_df = sector_df[keep_cols].copy()
    sector_df["tic"] = sector_df["tic"].astype(str)
    sector_df["gsector"] = pd.to_numeric(sector_df["gsector"], errors="coerce")

    # Use earliest known sector per ticker to avoid future-information leakage.
    if "date" in sector_df.columns:
        sector_df["date"] = pd.to_datetime(sector_df["date"], errors="coerce").dt.normalize()
        sector_df = sector_df.sort_values(["tic", "date"], kind="mergesort")
    else:
        sector_df = sector_df.reset_index().rename(columns={"index": "_row_order"})
        sector_df = sector_df.sort_values(["tic", "_row_order"], kind="mergesort")

    earliest = sector_df.dropna(subset=["gsector"]).drop_duplicates(subset=["tic"], keep="first")
    return earliest.set_index("tic")["gsector"].to_dict()


def load_cc_return_matrix(stock_price_csv: str) -> pd.DataFrame:
    daily = pd.read_csv(stock_price_csv, usecols=["datadate", "tic", "ajexdi", "prccd"], parse_dates=["datadate"])
    daily["close_adj"] = daily["prccd"] / daily["ajexdi"]
    close_adj_df = daily.pivot(index="datadate", columns="tic", values="close_adj").sort_index()
    return close_adj_df.pct_change().replace([float("inf"), float("-inf")], float("nan"))


def run_strategy_pipeline(
    strategy: AllocationStrategy,
    data_paths: DataPaths,
    backtest_config: BacktestConfig | None = None,
    force_config_name: str | None = None,
) -> tuple[Path, StrategySignals, pd.DataFrame]:
    selected = load_selected_stocks(data_paths.stock_selected_csv)
    ticker_sector = load_ticker_sector_map(data_paths.final_ratios_csv)
    cc_ret_df = load_cc_return_matrix(data_paths.stock_price_csv)

    signals = strategy.generate_signals(selected_stocks_df=selected, ticker_sector=ticker_sector, cc_ret_df=cc_ret_df)

    force_name = normalize_config_name(force_config_name)
    signal_name = normalize_config_name(signals.config_name)
    config_name = force_name or signal_name
    if config_name is None:
        raise ValueError("Resolved config_name is empty. Provide a valid force_config_name or strategy config_name.")

    output_dir = Path(data_paths.output_step2_dir) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explicit overwrite behavior for repeated runs with the same config name.
    # This prevents stale artifacts from older runs being mistaken as fresh results.
    managed_outputs = [
        output_dir / "portfolio_weights.csv",
        output_dir / "portfolio_weights_exec.csv",
        output_dir / "regime_state.csv",
        output_dir / "strategy_result.csv",
    ]
    for fp in managed_outputs:
        if fp.exists() and fp.is_file():
            fp.unlink()

    signals.weights_signal_df.to_csv(output_dir / "portfolio_weights.csv")
    cfg = backtest_config or BacktestConfig()
    signals.regime_df.to_csv(output_dir / "regime_state.csv")

    daily_price_df = pd.read_csv(data_paths.stock_price_csv, parse_dates=["datadate"])
    result_obj = run_backtest(
        signals.weights_signal_df,
        signals.regime_df,
        daily_price_df,
        cfg,
        return_exec_weights=True,
    )
    if isinstance(result_obj, tuple):
        result_df, weights_exec_df = result_obj
    else:
        result_df = result_obj
        # Fallback path for compatibility if engine returns result only.
        weight_lag_days, _, _ = resolve_effective_lags(cfg)
        weights_exec_df = signals.weights_signal_df.sort_index().shift(weight_lag_days).ffill().fillna(0.0)
    weights_exec_df.to_csv(output_dir / "portfolio_weights_exec.csv")
    result_df.to_csv(output_dir / "strategy_result.csv")

    return output_dir, signals, result_df
