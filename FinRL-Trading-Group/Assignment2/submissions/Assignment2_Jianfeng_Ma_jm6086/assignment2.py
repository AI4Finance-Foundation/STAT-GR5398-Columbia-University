from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
DOWNLOAD_START = "2015-01-01"
WEEKLY_LOOKBACK = 12
SLOW_TREND_WEEKS = 26
SLOW_DRAWDOWN_WEEKS = 13
SLOW_DRAWDOWN_THRESHOLD = -0.10
FAST_PRICE_LOOKBACK_DAYS = 3
FAST_PRICE_THRESHOLD = -0.03
FAST_VIX_Z_THRESHOLD = 3.0
FAST_DELTA_VIX_Z_THRESHOLD = 3.5
FAST_DURATION_DAYS = 10
ABSOLUTE_STOP = -0.05
TRAILING_STOP = -0.10
COOLDOWN_DAYS = 20
MIN_VALID_FRACTION = 0.80
RISK_BUDGETS = {
    "risk_on": 1.00,
    "neutral": 0.90,
    "risk_off": 0.70,
}
FAST_RISK_BUDGET = 0.50

STATIC_GROUPS = {
    "growth_tech": ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"],
    "real_assets": ["XOM", "CVX", "FCX", "BHP", "GLD", "SLV"],
    "defensive": ["TLT", "IEF", "XLU", "XLV", "IAU", "SHY", "UUP"],
}
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IAU", "XLU", "XLV"]
MARKET_SYMBOLS = ["SPY", "QQQ", "^GSPC", "^VIX"]


@dataclass
class PositionState:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    peak_price: float


@dataclass
class StrategyDecision:
    date: pd.Timestamp
    regime_state: str
    invested_weight: float
    cash_weight: float
    selected_group: str
    selected_assets: List[str]
    selection_source: str
    notes: str


@dataclass
class BacktestResult:
    name: str
    nav: pd.Series
    weights: pd.DataFrame
    decisions: pd.DataFrame
    metrics: pd.Series


def ticker_to_filename(ticker: str) -> str:
    return ticker.replace("^", "caret_").replace("/", "_") + ".csv"


def compute_mad(series: pd.Series) -> float:
    median = series.median()
    return float((series - median).abs().median())


def latest_robust_zscore(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    center = clean.median()
    mad = compute_mad(clean)
    current = float(clean.iloc[-1])
    if mad < 1e-6:
        diff = current - center
        if abs(diff) < 1e-6:
            return 0.0
        return float(np.sign(diff) * 10.0)
    return float(np.clip((current - center) / mad, -20.0, 20.0))


def robust_information_ratio(excess_returns: pd.Series) -> float:
    clean = excess_returns.dropna()
    if len(clean) < max(3, int(np.ceil(WEEKLY_LOOKBACK * MIN_VALID_FRACTION))):
        return np.nan
    variability = compute_mad(clean)
    if variability < 1e-6 or not np.isfinite(variability):
        variability = float(clean.std())
    if variability < 1e-6 or not np.isfinite(variability):
        return np.nan
    return float(clean.mean() / variability)


def build_weekly_close_matrix(close_df: pd.DataFrame) -> pd.DataFrame:
    if close_df.empty:
        return close_df
    return close_df.groupby(close_df.index.to_period("W-FRI")).tail(1)


def compute_group_weekly_returns(
    weekly_close: pd.DataFrame,
    symbols: Iterable[str],
    as_of_date: pd.Timestamp,
) -> pd.Series:
    available = [symbol for symbol in symbols if symbol in weekly_close.columns]
    if not available:
        return pd.Series(dtype=float)
    window = weekly_close.loc[:as_of_date, available].tail(WEEKLY_LOOKBACK + 1)
    if len(window) < 2:
        return pd.Series(dtype=float)
    returns = window.pct_change().dropna(how="all")
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.mean(axis=1, skipna=True).dropna()


def compute_residual_rankings(
    weekly_close: pd.DataFrame,
    symbols: List[str],
    as_of_date: pd.Timestamp,
) -> List[Tuple[str, float, float]]:
    group_returns = compute_group_weekly_returns(weekly_close, symbols, as_of_date)
    if group_returns.empty:
        return []

    rankings: List[Tuple[str, float, float]] = []
    min_required = max(3, int(np.ceil(WEEKLY_LOOKBACK * MIN_VALID_FRACTION)))
    for symbol in symbols:
        if symbol not in weekly_close.columns:
            continue
        price_window = weekly_close.loc[:as_of_date, symbol].dropna().tail(WEEKLY_LOOKBACK + 1)
        if len(price_window) < min_required + 1:
            continue
        asset_returns = price_window.pct_change().dropna()
        common = asset_returns.index.intersection(group_returns.index)
        if len(common) < min_required:
            continue
        residual = asset_returns.loc[common] - group_returns.loc[common]
        zscore = latest_robust_zscore(residual.tail(WEEKLY_LOOKBACK))
        residual_momentum = float((1.0 + residual).prod() - 1.0)
        rankings.append((symbol, zscore, residual_momentum))

    rankings.sort(key=lambda row: (row[1], row[2]), reverse=True)
    return rankings


def compute_slow_regime(
    daily_close: pd.DataFrame,
    weekly_close: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    spx_weekly = weekly_close.loc[:as_of_date, "^GSPC"].dropna()
    vix_weekly = weekly_close.loc[:as_of_date, "^VIX"].dropna()

    trend_signal = False
    if len(spx_weekly) >= SLOW_TREND_WEEKS:
        trend_signal = bool(spx_weekly.iloc[-1] < spx_weekly.tail(SLOW_TREND_WEEKS).mean())

    drawdown_signal = False
    drawdown = 0.0
    if len(spx_weekly) >= SLOW_DRAWDOWN_WEEKS:
        drawdown_window = spx_weekly.tail(SLOW_DRAWDOWN_WEEKS)
        drawdown = float(drawdown_window.iloc[-1] / drawdown_window.max() - 1.0)
        drawdown_signal = drawdown <= SLOW_DRAWDOWN_THRESHOLD

    vix_signal = False
    vix_z = 0.0
    if len(vix_weekly) >= 52:
        vix_z = latest_robust_zscore(vix_weekly.tail(min(len(vix_weekly), 156)))
        vix_signal = vix_z >= FAST_VIX_Z_THRESHOLD

    risk_score = int(trend_signal) + int(drawdown_signal) + int(vix_signal)
    if risk_score == 0:
        state = "risk_on"
    elif risk_score == 1:
        state = "neutral"
    else:
        state = "risk_off"

    return {
        "state": state,
        "risk_budget": RISK_BUDGETS[state],
        "risk_score": risk_score,
        "drawdown_13w": drawdown,
        "vix_z": vix_z,
    }


def compute_fast_risk_off(
    daily_close: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    def three_day_return(symbol: str) -> float:
        prices = daily_close.loc[:as_of_date, symbol].dropna()
        if len(prices) < FAST_PRICE_LOOKBACK_DAYS + 1:
            return 0.0
        return float(prices.iloc[-1] / prices.iloc[-(FAST_PRICE_LOOKBACK_DAYS + 1)] - 1.0)

    spx_return = three_day_return("^GSPC")
    qqq_return = three_day_return("QQQ")

    vix_daily = daily_close.loc[:as_of_date, "^VIX"].dropna()
    vix_z = 0.0
    delta_vix_z = 0.0
    if len(vix_daily) >= 252:
        vix_window = vix_daily.tail(min(len(vix_daily), 252 * 3))
        vix_z = latest_robust_zscore(vix_window)
        delta_vix = vix_window.diff().dropna()
        if not delta_vix.empty:
            delta_vix_z = latest_robust_zscore(delta_vix)

    price_shock = (spx_return <= FAST_PRICE_THRESHOLD) or (qqq_return <= FAST_PRICE_THRESHOLD)
    volatility_shock = (vix_z >= FAST_VIX_Z_THRESHOLD) and (delta_vix_z >= FAST_DELTA_VIX_Z_THRESHOLD)
    return {
        "triggered": bool(price_shock or volatility_shock),
        "price_shock": bool(price_shock),
        "volatility_shock": bool(volatility_shock),
        "spx_return_3d": spx_return,
        "qqq_return_3d": qqq_return,
        "vix_z": vix_z,
        "delta_vix_z": delta_vix_z,
    }


class Assignment1Selector:
    def __init__(self, selector_path: Path):
        selector_df = pd.read_csv(selector_path, parse_dates=["trade_date"])
        selector_df = selector_df.sort_values(["trade_date", "predicted_return"], ascending=[True, False])
        self.grouped = {
            trade_date: group[["tic", "predicted_return"]].reset_index(drop=True)
            for trade_date, group in selector_df.groupby("trade_date")
        }
        self.dates = sorted(self.grouped)

    def candidates_as_of(self, as_of_date: pd.Timestamp) -> Tuple[List[str], Optional[pd.Timestamp]]:
        eligible_dates = [date for date in self.dates if date <= as_of_date]
        if not eligible_dates:
            return [], None
        latest_date = eligible_dates[-1]
        candidates = self.grouped[latest_date]["tic"].tolist()
        return candidates, latest_date


def select_group_candidates(
    group_name: str,
    as_of_date: pd.Timestamp,
    selector: Assignment1Selector,
    cooldowns: Dict[str, pd.Timestamp],
    use_assignment1_selector: bool,
    available_symbols: Iterable[str],
) -> Tuple[List[str], str]:
    available_set = set(available_symbols)

    def not_in_cooldown(symbol: str) -> bool:
        until = cooldowns.get(symbol)
        return until is None or as_of_date >= until

    static_candidates = [
        symbol
        for symbol in STATIC_GROUPS[group_name]
        if symbol in available_set and not_in_cooldown(symbol)
    ]

    if group_name != "growth_tech" or not use_assignment1_selector:
        return static_candidates, "static_group"

    selected, selector_date = selector.candidates_as_of(as_of_date)
    dynamic_candidates = [
        symbol for symbol in selected if symbol in available_set and not_in_cooldown(symbol)
    ]
    if len(dynamic_candidates) >= 2:
        return dynamic_candidates, f"assignment1_selector_asof_{selector_date.date()}"
    return static_candidates, "assignment1_selector_fallback_to_static"


def build_rebalance_decision(
    as_of_date: pd.Timestamp,
    daily_close: pd.DataFrame,
    weekly_close: pd.DataFrame,
    selector: Assignment1Selector,
    cooldowns: Dict[str, pd.Timestamp],
    use_assignment1_selector: bool,
    fast_active_until: Optional[pd.Timestamp],
) -> StrategyDecision:
    regime = compute_slow_regime(daily_close, weekly_close, as_of_date)
    risk_budget = float(regime["risk_budget"])
    if fast_active_until is not None and as_of_date <= fast_active_until:
        risk_budget = min(risk_budget, FAST_RISK_BUDGET)

    qqq_weekly = weekly_close.loc[:as_of_date, "QQQ"].dropna().tail(WEEKLY_LOOKBACK + 1)
    qqq_returns = qqq_weekly.pct_change().dropna()

    group_metrics = {}
    for group_name, symbols in STATIC_GROUPS.items():
        group_returns = compute_group_weekly_returns(weekly_close, symbols, as_of_date)
        common = group_returns.index.intersection(qqq_returns.index)
        if len(common) < max(3, int(np.ceil(WEEKLY_LOOKBACK * MIN_VALID_FRACTION))):
            continue
        excess = group_returns.loc[common] - qqq_returns.loc[common]
        group_metrics[group_name] = {
            "information_ratio": robust_information_ratio(excess),
            "excess_return": float((1.0 + group_returns.loc[common]).prod() - (1.0 + qqq_returns.loc[common]).prod()),
        }

    ranked_groups = sorted(
        group_metrics.items(),
        key=lambda item: item[1]["information_ratio"],
        reverse=True,
    )
    positive_groups = [
        group_name
        for group_name, metrics in ranked_groups
        if np.isfinite(metrics["information_ratio"]) and metrics["excess_return"] > 0
    ]

    if not positive_groups:
        invested = risk_budget
        cash = 1.0 - invested
        return StrategyDecision(
            date=as_of_date,
            regime_state=regime["state"],
            invested_weight=invested,
            cash_weight=cash,
            selected_group="default",
            selected_assets=list(DEFAULT_SYMBOLS),
            selection_source="fallback_default",
            notes="No positive-IR sector group qualified; used default basket.",
        )

    winning_group = positive_groups[0]
    candidates, source = select_group_candidates(
        group_name=winning_group,
        as_of_date=as_of_date,
        selector=selector,
        cooldowns=cooldowns,
        use_assignment1_selector=use_assignment1_selector,
        available_symbols=weekly_close.columns,
    )
    rankings = compute_residual_rankings(weekly_close, candidates, as_of_date)

    if not rankings:
        invested = risk_budget
        cash = 1.0 - invested
        return StrategyDecision(
            date=as_of_date,
            regime_state=regime["state"],
            invested_weight=invested,
            cash_weight=cash,
            selected_group="default",
            selected_assets=list(DEFAULT_SYMBOLS),
            selection_source=f"{source}_fallback_default",
            notes=f"Winning group {winning_group} had no valid residual-momentum ranks; used default basket.",
        )

    selected_assets = [symbol for symbol, _, _ in rankings[:2]]
    invested = risk_budget
    cash = 1.0 - invested
    return StrategyDecision(
        date=as_of_date,
        regime_state=regime["state"],
        invested_weight=invested,
        cash_weight=cash,
        selected_group=winning_group,
        selected_assets=selected_assets,
        selection_source=source,
        notes=f"Weekly rebalance into {winning_group}.",
    )


def equal_weight_series(
    symbols: List[str],
    invested_weight: float,
    weight_index: pd.Index,
) -> pd.Series:
    weights = pd.Series(0.0, index=weight_index, dtype=float)
    if not symbols or invested_weight <= 0:
        return weights
    weight = invested_weight / len(symbols)
    for symbol in symbols:
        if symbol in weights.index:
            weights.loc[symbol] = weight
    return weights


def compute_performance_metrics(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    daily_returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cumulative_return = float(nav.iloc[-1] - 1.0)
    annual_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0)
    annual_volatility = float(daily_returns.std() * np.sqrt(252))
    sharpe = float((daily_returns.mean() * 252) / annual_volatility) if annual_volatility > 0 else np.nan
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan
    win_rate = float((daily_returns > 0).mean()) if not daily_returns.empty else np.nan
    return pd.Series(
        {
            "cumulative_return": cumulative_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
        }
    )


def load_price_matrix(
    tickers: List[str],
    cache_dir: Path,
) -> pd.DataFrame:
    series_dict = {}
    for ticker in tickers:
        csv_path = cache_dir / ticker_to_filename(ticker)
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        series_dict[ticker] = df["Close"]
    close_df = pd.DataFrame(series_dict).sort_index()
    return close_df


def download_history(
    tickers: List[str],
    cache_dir: Path,
    start: str,
    end: str,
    refresh: bool = False,
) -> None:
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    end = pd.Timestamp(end).strftime("%Y-%m-%d")
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for ticker in tickers:
        if refresh or not (cache_dir / ticker_to_filename(ticker)).exists():
            missing.append(ticker)

    if not missing:
        print(f"Using cached data for {len(tickers)} tickers.")
        return

    print(f"Downloading {len(missing)} tickers from Yahoo Finance...")
    for idx, ticker in enumerate(missing, start=1):
        print(f"[{idx}/{len(missing)}] {ticker}")
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        if data.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        keep_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]
        data = data[keep_cols].reset_index()
        data.to_csv(cache_dir / ticker_to_filename(ticker), index=False)


def build_ticker_universe(selector: Assignment1Selector) -> List[str]:
    selector_tickers = sorted(
        {
            ticker
            for date in selector.dates
            for ticker in selector.grouped[date]["tic"].tolist()
        }
    )
    static_tickers = sorted(
        {
            ticker
            for symbols in list(STATIC_GROUPS.values()) + [DEFAULT_SYMBOLS, MARKET_SYMBOLS]
            for ticker in symbols
        }
    )
    return sorted(set(static_tickers + selector_tickers))


def run_backtest(
    name: str,
    close_df: pd.DataFrame,
    selector: Assignment1Selector,
    use_assignment1_selector: bool,
) -> BacktestResult:
    close_df = close_df.loc[(close_df.index >= pd.Timestamp(START_DATE)) & (close_df.index <= pd.Timestamp(END_DATE))].copy()
    calendar = close_df["SPY"].dropna().index
    close_df = close_df.reindex(calendar).ffill()
    tradable_symbols = [symbol for symbol in close_df.columns if symbol not in {"^GSPC", "^VIX"}]
    returns = close_df[tradable_symbols].pct_change().fillna(0.0)
    weekly_close = build_weekly_close_matrix(close_df)
    weekly_dates = set(weekly_close.index[(weekly_close.index >= pd.Timestamp(START_DATE)) & (weekly_close.index <= pd.Timestamp(END_DATE))])

    current_weights = pd.Series(0.0, index=tradable_symbols, dtype=float)
    current_positions: Dict[str, PositionState] = {}
    cooldowns: Dict[str, pd.Timestamp] = {}
    fast_active_until: Optional[pd.Timestamp] = None

    nav_value = 1.0
    nav_rows = []
    weight_rows = []
    decision_rows = []

    for idx, date in enumerate(calendar):
        if idx == 0:
            day_return = 0.0
        else:
            day_return = float((current_weights * returns.loc[date]).sum())
            nav_value *= 1.0 + day_return

        for symbol, position in list(current_positions.items()):
            price = close_df.at[date, symbol]
            if pd.notna(price) and price > position.peak_price:
                position.peak_price = float(price)

        action = "hold"
        action_notes = ""
        selected_group = ""
        selected_assets: List[str] = []
        selection_source = ""
        regime_state = ""

        if date in weekly_dates:
            decision = build_rebalance_decision(
                as_of_date=date,
                daily_close=close_df,
                weekly_close=weekly_close,
                selector=selector,
                cooldowns=cooldowns,
                use_assignment1_selector=use_assignment1_selector,
                fast_active_until=fast_active_until,
            )
            if decision.selected_group == "default":
                next_weights = equal_weight_series(decision.selected_assets, decision.invested_weight, current_weights.index)
            else:
                next_weights = equal_weight_series(decision.selected_assets, decision.invested_weight, current_weights.index)

            current_weights = next_weights
            current_positions = {}
            for symbol in decision.selected_assets:
                if symbol not in current_weights.index or current_weights.loc[symbol] <= 0:
                    continue
                price = close_df.at[date, symbol]
                if pd.isna(price):
                    continue
                current_positions[symbol] = PositionState(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=float(price),
                    peak_price=float(price),
                )

            action = "weekly_rebalance"
            action_notes = decision.notes
            selected_group = decision.selected_group
            selected_assets = decision.selected_assets
            selection_source = decision.selection_source
            regime_state = decision.regime_state
        else:
            fast_signal = compute_fast_risk_off(close_df, date)
            if fast_signal["triggered"] and (fast_active_until is None or date > fast_active_until):
                fast_active_until = date + pd.Timedelta(days=FAST_DURATION_DAYS)
                invested = float(current_weights.sum())
                if invested > FAST_RISK_BUDGET and invested > 0:
                    current_weights *= FAST_RISK_BUDGET / invested
                action = "fast_risk_off"
                action_notes = (
                    f"3-day shock detected. SPX={fast_signal['spx_return_3d']:.2%}, "
                    f"QQQ={fast_signal['qqq_return_3d']:.2%}, VIX_Z={fast_signal['vix_z']:.2f}."
                )
            else:
                stopped_symbols = []
                for symbol, position in list(current_positions.items()):
                    price = close_df.at[date, symbol]
                    if pd.isna(price):
                        continue
                    loss_from_entry = float(price / position.entry_price - 1.0)
                    loss_from_peak = float(price / position.peak_price - 1.0)
                    if loss_from_entry <= ABSOLUTE_STOP or loss_from_peak <= TRAILING_STOP:
                        stopped_symbols.append(symbol)
                if stopped_symbols:
                    for symbol in stopped_symbols:
                        cooldowns[symbol] = date + pd.Timedelta(days=COOLDOWN_DAYS)
                        current_positions.pop(symbol, None)
                        current_weights.loc[symbol] = 0.0
                    action = "stop_loss"
                    action_notes = "Stopped out: " + ", ".join(stopped_symbols)

        cash_weight = max(0.0, 1.0 - float(current_weights.sum()))
        nav_rows.append(
            {
                "date": date,
                "nav": nav_value,
                "strategy_return": day_return,
                "cash_weight": cash_weight,
                "action": action,
            }
        )
        weight_row = {"date": date, "cash": cash_weight}
        weight_row.update(current_weights.to_dict())
        weight_rows.append(weight_row)
        decision_rows.append(
            {
                "date": date,
                "action": action,
                "regime_state": regime_state,
                "selected_group": selected_group,
                "selected_assets": ", ".join(selected_assets),
                "selection_source": selection_source,
                "notes": action_notes,
                "fast_active_until": fast_active_until.strftime("%Y-%m-%d") if fast_active_until is not None else "",
            }
        )

    nav_df = pd.DataFrame(nav_rows).set_index("date")
    weights_df = pd.DataFrame(weight_rows)
    decisions_df = pd.DataFrame(decision_rows)
    metrics = compute_performance_metrics(nav_df["nav"])
    return BacktestResult(
        name=name,
        nav=nav_df["nav"],
        weights=weights_df,
        decisions=decisions_df,
        metrics=metrics,
    )


def format_metric(value: float, pct: bool = True) -> str:
    if pd.isna(value):
        return "N/A"
    if pct:
        return f"{value:.2%}"
    return f"{value:.2f}"


def save_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        comparison_df.index,
        comparison_df["Original Adaptive Rotation"],
        label="Original Adaptive Rotation",
        linewidth=2.4,
        color="#1d4ed8",
    )
    ax.plot(
        comparison_df.index,
        comparison_df["Modified Adaptive Rotation"],
        label="Modified + A1 Selector",
        linewidth=2.4,
        color="#059669",
    )
    ax.plot(
        comparison_df.index,
        comparison_df["QQQ"],
        label="QQQ",
        linewidth=1.8,
        color="#d97706",
        alpha=0.9,
    )
    ax.plot(
        comparison_df.index,
        comparison_df["SPY"],
        label="SPY",
        linewidth=1.8,
        color="#6b7280",
        alpha=0.9,
    )
    ax.set_title("Assignment 2: Adaptive Rotation vs Modified Strategy vs Benchmarks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    chart_path: Path,
) -> None:
    report_path = output_dir.parent / "Assignment2_Report.md"
    markdown_table = metrics_df.copy()
    markdown_table["Cumulative Return"] = markdown_table["cumulative_return"].map(lambda x: format_metric(x))
    markdown_table["Annual Return"] = markdown_table["annual_return"].map(lambda x: format_metric(x))
    markdown_table["Annual Volatility"] = markdown_table["annual_volatility"].map(lambda x: format_metric(x))
    markdown_table["Sharpe Ratio"] = markdown_table["sharpe_ratio"].map(lambda x: format_metric(x, pct=False))
    markdown_table["Max Drawdown"] = markdown_table["max_drawdown"].map(lambda x: format_metric(x))
    markdown_table["Calmar Ratio"] = markdown_table["calmar_ratio"].map(lambda x: format_metric(x, pct=False))
    markdown_table["Win Rate"] = markdown_table["win_rate"].map(lambda x: format_metric(x))
    markdown_table = markdown_table[
        [
            "Cumulative Return",
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
        ]
    ]
    headers = ["Portfolio"] + markdown_table.columns.tolist()
    separator = ["---"] * len(headers)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for index, row in markdown_table.iterrows():
        table_lines.append("| " + " | ".join([str(index)] + [str(value) for value in row.tolist()]) + " |")
    markdown_table_text = "\n".join(table_lines)

    report = f"""# Assignment 2 Report

Course: GR5398 26 Spring - FinRL-Trading Quantitative Trading Strategy Track  
Author: `Jianfeng Ma` (`jm6086`)  
Backtest Window: `{START_DATE}` to `{END_DATE}`

## 1. Objective

This submission implements the Adaptive Rotation Strategy described in the Assignment 2 prompt and compares:

- Original Adaptive Rotation
- Modified Adaptive Rotation with Assignment 1 QQQ rolling-selection logic
- `QQQ` benchmark

## 2. Strategy Setup

### 2.1 Original Adaptive Rotation

- Weekly rebalance using trailing 12-week Information Ratio versus `QQQ` to choose the strongest asset group
- Intra-group ranking by residual momentum
- Final allocation to the top 2 assets in the chosen group
- Slow regime overlay from `^GSPC` trend / drawdown / `^VIX`
- Fast risk-off and stop-loss overlays for daily risk control

### 2.2 Modified Version

The modified strategy keeps the same regime and risk framework, but changes the growth-tech stock-selection pool:

- When the Assignment 1 rolling selector is available, the growth-tech candidate pool is replaced by the latest available Assignment 1 picks
- Because the Assignment 1 fundamentals start in early 2018 and the original rolling model needs four years of history, the Assignment 1 selector only becomes available from `2023-02-15`
- Before `2023-02-15`, the modified strategy falls back to the original static growth-tech basket to preserve the required `2018-01-01` to `2025-12-31` evaluation window

## 3. Results

{markdown_table_text}

Chart:

![Assignment 2 comparison]({chart_path.name})

## 4. Discussion

- The original strategy provides the baseline adaptive rotation behavior from the assignment prompt
- The modified strategy tests whether Assignment 1 stock-selection signals improve the growth-tech sleeve without changing the broader regime logic
- The warm-up limitation from Assignment 1 is handled explicitly rather than backfilling nonexistent pre-2023 selector signals

## 5. Output Files

- `outputs/comparison_metrics.csv`
- `outputs/comparison_nav.csv`
- `outputs/comparison_chart.png`
- `outputs/original_weights.csv`
- `outputs/modified_weights.csv`
- `outputs/original_decisions.csv`
- `outputs/modified_decisions.csv`
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GR5398 Assignment 2 adaptive rotation backtests.")
    parser.add_argument("--refresh-data", action="store_true", help="Redownload market data even if cached CSVs exist.")
    args = parser.parse_args()

    submission_dir = Path(__file__).resolve().parent
    output_dir = submission_dir / "outputs"
    cache_dir = submission_dir / "data_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    selector_path = (
        submission_dir.parent.parent.parent
        / "Assignment1"
        / "submissions"
        / "Assignment1_Jianfeng_Ma_jm6086"
        / "outputs"
        / "step2"
        / "stock_selected.csv"
    )
    selector = Assignment1Selector(selector_path)

    tickers = build_ticker_universe(selector)
    download_history(
        tickers=tickers,
        cache_dir=cache_dir,
        start=DOWNLOAD_START,
        end=(pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        refresh=args.refresh_data,
    )
    close_df = load_price_matrix(tickers, cache_dir)

    original = run_backtest(
        name="Original Adaptive Rotation",
        close_df=close_df,
        selector=selector,
        use_assignment1_selector=False,
    )
    modified = run_backtest(
        name="Modified Adaptive Rotation",
        close_df=close_df,
        selector=selector,
        use_assignment1_selector=True,
    )

    benchmark_df = close_df.loc[(close_df.index >= pd.Timestamp(START_DATE)) & (close_df.index <= pd.Timestamp(END_DATE)), ["QQQ", "SPY"]].dropna()
    benchmark_nav = benchmark_df / benchmark_df.iloc[0]

    comparison_df = pd.concat(
        [
            original.nav.rename("Original Adaptive Rotation"),
            modified.nav.rename("Modified Adaptive Rotation"),
            benchmark_nav["QQQ"].rename("QQQ"),
            benchmark_nav["SPY"].rename("SPY"),
        ],
        axis=1,
    ).dropna()

    metrics_df = pd.DataFrame(
        {
            "Original Adaptive Rotation": compute_performance_metrics(comparison_df["Original Adaptive Rotation"]),
            "Modified Adaptive Rotation": compute_performance_metrics(comparison_df["Modified Adaptive Rotation"]),
            "QQQ": compute_performance_metrics(comparison_df["QQQ"]),
            "SPY": compute_performance_metrics(comparison_df["SPY"]),
        }
    ).T

    comparison_df.to_csv(output_dir / "comparison_nav.csv")
    metrics_df.to_csv(output_dir / "comparison_metrics.csv")
    original.weights.to_csv(output_dir / "original_weights.csv", index=False)
    modified.weights.to_csv(output_dir / "modified_weights.csv", index=False)
    original.decisions.to_csv(output_dir / "original_decisions.csv", index=False)
    modified.decisions.to_csv(output_dir / "modified_decisions.csv", index=False)

    chart_path = output_dir / "comparison_chart.png"
    save_plot(comparison_df, chart_path)
    write_report(output_dir, metrics_df, chart_path)

    print("\nBacktest metrics:")
    print(
        metrics_df[
            [
                "cumulative_return",
                "annual_return",
                "annual_volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ].round(4)
    )
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
