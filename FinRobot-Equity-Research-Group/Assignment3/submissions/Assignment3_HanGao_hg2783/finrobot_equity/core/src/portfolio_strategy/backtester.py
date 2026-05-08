# portfolio_strategy/backtester.py

import numpy as np
import pandas as pd


def get_rebalance_dates(monthly_returns, freq="Q"):
    """
    Get quarterly rebalance dates from monthly return index.
    Also force the first available month to be a rebalance date,
    so the portfolio is invested from the start of the backtest window.
    """
    dates = monthly_returns.index.to_series()
    rebalance_dates = list(dates.groupby(dates.dt.to_period(freq)).max().values)

    first_date = monthly_returns.index[0]
    if first_date not in rebalance_dates:
        rebalance_dates = [first_date] + rebalance_dates

    return rebalance_dates


def build_equal_weight_portfolio(scores, monthly_returns, universe, top_n=15, rebalance_freq="Q"):
    """
    Build equal-weight portfolio using top N scores at each rebalance date.
    """
    weights = pd.DataFrame(0.0, index=monthly_returns.index, columns=universe)
    rebalance_dates = get_rebalance_dates(monthly_returns, rebalance_freq)

    for date in rebalance_dates:
        if date not in scores.index:
            continue

        available_scores = scores.loc[date, universe].dropna()
        if len(available_scores) < top_n:
            continue

        selected = available_scores.nlargest(top_n).index
        weights.loc[date, selected] = 1.0 / top_n

    weights = weights.replace(0, np.nan).ffill().fillna(0.0)
    return weights


def build_score_weighted_portfolio(scores, monthly_returns, universe, top_n=15, rebalance_freq="Q", max_weight=0.10):
    """
    Build score-weighted portfolio using top N scores at each rebalance date.
    Weight_i = score_i / sum(scores)
    with max single-name weight cap.
    """
    weights = pd.DataFrame(0.0, index=monthly_returns.index, columns=universe)
    rebalance_dates = get_rebalance_dates(monthly_returns, rebalance_freq)

    for date in rebalance_dates:
        if date not in scores.index:
            continue

        available_scores = scores.loc[date, universe].dropna()
        if len(available_scores) < top_n:
            continue

        selected_scores = available_scores.nlargest(top_n)
        raw_weights = selected_scores / selected_scores.sum()

        capped_weights = raw_weights.clip(upper=max_weight)
        capped_weights = capped_weights / capped_weights.sum()

        weights.loc[date, capped_weights.index] = capped_weights

    weights = weights.replace(0, np.nan).ffill().fillna(0.0)
    return weights


def compute_portfolio_returns(weights, monthly_returns, transaction_cost=0.001, use_current_weights=True):
    """
    Compute portfolio returns.

    If use_current_weights=True:
        Use weights at date t to earn return at date t.
        This is useful when weights are interpreted as beginning-of-period weights.

    If use_current_weights=False:
        Use weights at date t-1 to earn return at date t.
        This is stricter when weights are formed at the end of date t-1.
    """
    aligned_returns = monthly_returns[weights.columns].fillna(0.0)

    if use_current_weights:
        effective_weights = weights.copy()
    else:
        effective_weights = weights.shift(1).fillna(0.0)

    gross_returns = (effective_weights * aligned_returns).sum(axis=1)

    turnover = 0.5 * weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * transaction_cost

    net_returns = gross_returns - costs

    result = pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": costs,
        "net_return": net_returns
    })

    return result


def compute_core_satellite_returns(qqq_returns, active_returns, qqq_weight=0.5, active_weight=0.5):
    """
    50% QQQ + 50% active stock strategy.
    """
    return qqq_weight * qqq_returns + active_weight * active_returns


def performance_summary(returns, risk_free_monthly=0.0):
    """
    Compute annualized return, volatility, Sharpe, max drawdown, and cumulative return.
    """
    returns = returns.dropna()
    excess_returns = returns - risk_free_monthly

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    n_months = len(returns)
    ann_return = (1 + total_return) ** (12 / n_months) - 1
    ann_vol = returns.std() * np.sqrt(12)

    sharpe = np.nan
    if ann_vol != 0:
        sharpe = (excess_returns.mean() * 12) / ann_vol

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "cumulative_return": total_return,
        "max_drawdown": max_drawdown
    }


def summarize_all_strategies(strategy_returns):
    """
    strategy_returns: dict of strategy_name -> monthly return series
    """
    rows = []
    for name, ret in strategy_returns.items():
        stats = performance_summary(ret)
        stats["strategy"] = name
        rows.append(stats)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "strategy",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "cumulative_return",
            "max_drawdown"
        ]
    ]

    return summary