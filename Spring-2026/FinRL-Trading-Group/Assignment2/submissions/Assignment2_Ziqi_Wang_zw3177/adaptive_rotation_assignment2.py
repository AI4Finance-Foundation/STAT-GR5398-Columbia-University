# Assignment 2 - Adaptive Rotation Strategy with QQQ Rolling Selection Logic
# Author: Ziqi Wang
# Backtest Window: 2018-01-01 to 2025-12-31

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
DOWNLOAD_END = "2026-01-01"
OUTPUT_DIR = Path("assignment2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

growth_tech = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]
real_assets = ["XOM", "CVX", "FCX", "BHP", "GLD", "SLV"]
defensive_assets = ["TLT", "IEF", "XLU", "XLV", "IAU", "SHY", "UUP"]
default_assets = ["SPY", "QQQ", "IAU", "XLU", "XLV"]
market_symbols = ["SPY", "QQQ", "^GSPC", "^VIX"]

asset_groups = {
    "growth_tech": growth_tech,
    "real_assets": real_assets,
    "defensive": defensive_assets,
    "default": default_assets,
}

all_tickers = sorted(set(growth_tech + real_assets + defensive_assets + default_assets + market_symbols))

raw = yf.download(
    all_tickers,
    start=START_DATE,
    end=DOWNLOAD_END,
    auto_adjust=True,
    progress=False,
    group_by="ticker",
    threads=True
)

close_dict = {}
for ticker in all_tickers:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                close_dict[ticker] = raw[ticker]["Close"]
        else:
            close_dict[ticker] = raw["Close"]
    except Exception as e:
        print(f"Skip {ticker}: {e}")

close = pd.DataFrame(close_dict).sort_index()
close = close.dropna(how="all").ffill()
close = close.loc[(close.index >= START_DATE) & (close.index <= END_DATE)]

daily_ret = close.pct_change().fillna(0)
weekly_close = close.resample("W-FRI").last().dropna(how="all").ffill()
weekly_ret = weekly_close.pct_change().dropna(how="all")

print("Number of tickers:", len(all_tickers))
print("Daily close shape:", close.shape)
print("Weekly close shape:", weekly_close.shape)

LOOKBACK_WEEKS = 12
TOP_N = 2
REBALANCE_FREQ = "W-FRI"

def compute_group_return(weekly_close, symbols):
    available = [s for s in symbols if s in weekly_close.columns]
    group_prices = weekly_close[available].dropna(how="all").ffill()
    group_returns = group_prices.pct_change().mean(axis=1)
    return group_returns

def compute_information_ratio(group_ret, benchmark_ret, as_of_date, lookback=12):
    group_window = group_ret.loc[:as_of_date].tail(lookback)
    bench_window = benchmark_ret.loc[:as_of_date].tail(lookback)
    common_idx = group_window.index.intersection(bench_window.index)
    if len(common_idx) < max(4, int(lookback * 0.7)):
        return np.nan
    excess = group_window.loc[common_idx] - bench_window.loc[common_idx]
    if excess.std() == 0 or pd.isna(excess.std()):
        return np.nan
    return excess.mean() / excess.std()

def residual_momentum_rank(weekly_close, symbols, as_of_date, lookback=12):
    available = [s for s in symbols if s in weekly_close.columns]
    if len(available) == 0:
        return []
    price_window = weekly_close.loc[:as_of_date, available].tail(lookback + 1)
    asset_returns = price_window.pct_change().dropna(how="all")
    if asset_returns.empty:
        return []
    group_return = asset_returns.mean(axis=1)
    scores = {}
    for symbol in available:
        r = asset_returns[symbol].dropna()
        common_idx = r.index.intersection(group_return.index)
        if len(common_idx) < max(4, int(lookback * 0.7)):
            continue
        residual = r.loc[common_idx] - group_return.loc[common_idx]
        residual_momentum = (1 + residual).prod() - 1
        if pd.notna(residual_momentum):
            scores[symbol] = residual_momentum
    if len(scores) == 0:
        return []
    ranked = pd.Series(scores).sort_values(ascending=False)
    return ranked.index.tolist()

def run_original_adaptive_rotation(close, weekly_close, asset_groups):
    qqq_weekly_ret = weekly_close["QQQ"].pct_change()
    rebalance_dates = weekly_close.index
    tradable_assets = sorted(set(growth_tech + real_assets + defensive_assets + default_assets))
    weights = pd.DataFrame(np.nan, index=close.index, columns=tradable_assets)
    decisions = []
    for date in rebalance_dates:
        if date < weekly_close.index[LOOKBACK_WEEKS]:
            continue
        group_irs = {}
        for group_name, symbols in asset_groups.items():
            group_ret = compute_group_return(weekly_close, symbols)
            ir = compute_information_ratio(group_ret, qqq_weekly_ret, date, LOOKBACK_WEEKS)
            group_irs[group_name] = ir
        group_irs_series = pd.Series(group_irs).dropna()
        selected_group = "default" if group_irs_series.empty else group_irs_series.sort_values(ascending=False).index[0]
        candidate_symbols = asset_groups[selected_group]
        ranked_assets = residual_momentum_rank(weekly_close, candidate_symbols, date, LOOKBACK_WEEKS)
        if len(ranked_assets) == 0:
            selected_assets = default_assets[:TOP_N]
            selected_group = "default"
        else:
            selected_assets = ranked_assets[:TOP_N]
        daily_date = close.index[close.index >= date]
        if len(daily_date) == 0:
            continue
        rebalance_day = daily_date[0]
        weights.loc[rebalance_day, :] = 0.0
        weight = 1.0 / len(selected_assets)
        weights.loc[rebalance_day, selected_assets] = weight
        decisions.append({
            "date": rebalance_day,
            "selected_group": selected_group,
            "selected_assets": ", ".join(selected_assets),
            "group_ir": group_irs,
        })
    weights = weights.ffill().fillna(0.0)
    weight_sum = weights.sum(axis=1)
    weights = weights.div(weight_sum.replace(0, np.nan), axis=0).fillna(0.0)
    decisions_df = pd.DataFrame(decisions)
    return weights, decisions_df

original_weights, original_decisions = run_original_adaptive_rotation(close, weekly_close, asset_groups)

FEE_RATE = 0.001

def run_backtest_from_weights(close, weights, name="Strategy", fee_rate=0.001):
    tradable_assets = [c for c in weights.columns if c in close.columns]
    price = close[tradable_assets].copy()
    ret = price.pct_change().fillna(0.0)
    common_dates = ret.index.intersection(weights.index)
    ret = ret.loc[common_dates]
    w = weights.loc[common_dates, tradable_assets]
    w_lag = w.shift(1).fillna(0.0)
    gross_ret = (w_lag * ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = fee_rate * turnover
    net_ret = gross_ret - cost
    nav = (1.0 + net_ret).cumprod()
    result = pd.DataFrame({
        "gross_return": gross_ret,
        "turnover": turnover,
        "cost": cost,
        "net_return": net_ret,
        "nav": nav
    })
    result["strategy"] = name
    return result

def compute_performance_metrics(nav, periods_per_year=252):
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    cumulative_return = nav.iloc[-1] / nav.iloc[0] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    annual_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    return pd.Series({
        "Cumulative Return": cumulative_return,
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    })

original_result = run_backtest_from_weights(close, original_weights, "Original Adaptive Rotation", FEE_RATE)

SKIP_WEEKS = 4
LOOKBACK_6M_WEEKS = 26
LOOKBACK_12M_WEEKS = 52
VOL_LOOKBACK_WEEKS = 13

def qqq_rolling_prefilter(
    weekly_close,
    symbols,
    as_of_date,
    top_k=4,
    skip_weeks=4,
    lookback_6m=26,
    lookback_12m=52,
    vol_lookback=13
):
    available = [s for s in symbols if s in weekly_close.columns]
    if len(available) <= top_k:
        return available
    price_window = weekly_close.loc[:as_of_date, available].ffill()
    min_len = lookback_12m + skip_weeks + 2
    if len(price_window) < min_len:
        return available
    past = price_window.iloc[:-skip_weeks] if skip_weeks > 0 else price_window
    scores = {}
    for symbol in available:
        s = past[symbol].dropna()
        if len(s) < lookback_12m + 1:
            continue
        mom6 = s.iloc[-1] / s.iloc[-lookback_6m] - 1
        mom12 = s.iloc[-1] / s.iloc[-lookback_12m] - 1
        raw_momentum = 0.5 * mom6 + 0.5 * mom12
        returns = s.pct_change().dropna()
        recent_vol = returns.tail(vol_lookback).std()
        if pd.isna(recent_vol) or recent_vol <= 0:
            continue
        score = raw_momentum / recent_vol
        scores[symbol] = score
    if len(scores) == 0:
        return available
    ranked = pd.Series(scores).sort_values(ascending=False)
    selected = ranked.head(min(top_k, len(ranked))).index.tolist()
    return selected

def run_modified_adaptive_rotation(close, weekly_close, asset_groups):
    qqq_weekly_ret = weekly_close["QQQ"].pct_change()
    rebalance_dates = weekly_close.index
    tradable_assets = sorted(set(growth_tech + real_assets + defensive_assets + default_assets))
    weights = pd.DataFrame(np.nan, index=close.index, columns=tradable_assets)
    decisions = []
    for date in rebalance_dates:
        if date < weekly_close.index[LOOKBACK_WEEKS]:
            continue
        group_irs = {}
        for group_name, symbols in asset_groups.items():
            group_ret = compute_group_return(weekly_close, symbols)
            ir = compute_information_ratio(group_ret, qqq_weekly_ret, date, LOOKBACK_WEEKS)
            group_irs[group_name] = ir
        group_irs_series = pd.Series(group_irs).dropna()
        selected_group = "default" if group_irs_series.empty else group_irs_series.sort_values(ascending=False).index[0]
        candidate_symbols = asset_groups[selected_group]
        selection_source = "original_group"
        if selected_group == "growth_tech":
            candidate_symbols = qqq_rolling_prefilter(
                weekly_close=weekly_close,
                symbols=candidate_symbols,
                as_of_date=date,
                top_k=4,
                skip_weeks=SKIP_WEEKS,
                lookback_6m=LOOKBACK_6M_WEEKS,
                lookback_12m=LOOKBACK_12M_WEEKS,
                vol_lookback=VOL_LOOKBACK_WEEKS
            )
            selection_source = "qqq_rolling_prefilter"
        ranked_assets = residual_momentum_rank(weekly_close, candidate_symbols, date, LOOKBACK_WEEKS)
        if len(ranked_assets) == 0:
            selected_assets = default_assets[:TOP_N]
            selected_group = "default"
            selection_source = "fallback_default"
        else:
            selected_assets = ranked_assets[:TOP_N]
        daily_date = close.index[close.index >= date]
        if len(daily_date) == 0:
            continue
        rebalance_day = daily_date[0]
        weights.loc[rebalance_day, :] = 0.0
        weight = 1.0 / len(selected_assets)
        weights.loc[rebalance_day, selected_assets] = weight
        decisions.append({
            "date": rebalance_day,
            "selected_group": selected_group,
            "selected_assets": ", ".join(selected_assets),
            "candidate_assets": ", ".join(candidate_symbols),
            "selection_source": selection_source,
            "group_ir": group_irs,
        })
    weights = weights.ffill().fillna(0.0)
    weight_sum = weights.sum(axis=1)
    weights = weights.div(weight_sum.replace(0, np.nan), axis=0).fillna(0.0)
    decisions_df = pd.DataFrame(decisions)
    return weights, decisions_df

modified_weights, modified_decisions = run_modified_adaptive_rotation(close, weekly_close, asset_groups)
modified_result = run_backtest_from_weights(close, modified_weights, "Modified Adaptive Rotation", FEE_RATE)

comparison_nav = pd.DataFrame({
    "Original Adaptive Rotation": original_result["nav"],
    "Modified Adaptive Rotation": modified_result["nav"],
    "QQQ": close["QQQ"].loc[original_result.index] / close["QQQ"].loc[original_result.index].iloc[0],
    "SPY": close["SPY"].loc[original_result.index] / close["SPY"].loc[original_result.index].iloc[0],
}).dropna()

metrics_df = pd.DataFrame({
    "Original Adaptive Rotation": compute_performance_metrics(comparison_nav["Original Adaptive Rotation"]),
    "Modified Adaptive Rotation": compute_performance_metrics(comparison_nav["Modified Adaptive Rotation"]),
    "QQQ": compute_performance_metrics(comparison_nav["QQQ"]),
    "SPY": compute_performance_metrics(comparison_nav["SPY"]),
}).T

print(metrics_df)

plt.figure(figsize=(14, 7))
plt.plot(comparison_nav.index, comparison_nav["Original Adaptive Rotation"], label="Original Adaptive Rotation", linewidth=2.5)
plt.plot(comparison_nav.index, comparison_nav["Modified Adaptive Rotation"], label="Modified Adaptive Rotation + QQQ Rolling Logic", linewidth=2.5)
plt.plot(comparison_nav.index, comparison_nav["QQQ"], label="QQQ", linewidth=2)
plt.plot(comparison_nav.index, comparison_nav["SPY"], label="SPY", linewidth=2)
plt.title("Assignment 2: Original vs Modified Adaptive Rotation vs QQQ/SPY")
plt.xlabel("Date")
plt.ylabel("Normalized NAV")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
chart_path = OUTPUT_DIR / "comparison_chart.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

comparison_nav.to_csv(OUTPUT_DIR / "comparison_nav.csv")
metrics_df.to_csv(OUTPUT_DIR / "comparison_metrics.csv")
original_weights.to_csv(OUTPUT_DIR / "original_weights.csv")
modified_weights.to_csv(OUTPUT_DIR / "modified_weights.csv")
original_decisions.to_csv(OUTPUT_DIR / "original_decisions.csv", index=False)
modified_decisions.to_csv(OUTPUT_DIR / "modified_decisions.csv", index=False)
original_result.to_csv(OUTPUT_DIR / "original_backtest.csv")
modified_result.to_csv(OUTPUT_DIR / "modified_backtest.csv")

print("Saved files:")
for file in OUTPUT_DIR.iterdir():
    print(file)
