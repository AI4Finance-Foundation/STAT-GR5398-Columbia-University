# portfolio_strategy/baselines.py

import numpy as np
import pandas as pd


def compute_all_tech_equal_weight_returns(monthly_returns, universe):
    """
    Compute an equal-weight return series for the full technology universe.

    This baseline answers:
    Did the Top15 active strategy outperform because of stock selection,
    or because the full technology universe performed well?

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Monthly returns for all available tickers.
    universe : list[str]
        Technology/growth stock universe used by the strategy.

    Returns
    -------
    pd.Series
        Monthly equal-weight returns across all available universe stocks.
    """
    available_universe = [ticker for ticker in universe if ticker in monthly_returns.columns]

    if not available_universe:
        raise ValueError("No universe tickers found in monthly_returns.")

    return monthly_returns[available_universe].fillna(0.0).mean(axis=1)


def compute_random_topn_equal_weight_baselines(
    monthly_returns,
    universe,
    top_n=15,
    n_simulations=500,
    seed=42,
):
    """
    Simulate random Top-N equal-weight portfolios from the same technology universe.

    This baseline answers:
    Did the score-based Top15 strategy beat portfolios that randomly select 15 names
    from the same universe?

    Returns
    -------
    random_returns : pd.DataFrame
        Monthly returns for each random simulation.
    random_summary : pd.DataFrame
        Cumulative return, annualized return, volatility, Sharpe, and max drawdown
        for each random simulation.
    percentile_summary : pd.DataFrame
        Cross-simulation percentile summary for key metrics.
    """
    rng = np.random.default_rng(seed)
    available_universe = [ticker for ticker in universe if ticker in monthly_returns.columns]

    if len(available_universe) < top_n:
        raise ValueError(
            f"Not enough tickers for random Top-{top_n}. "
            f"Available: {len(available_universe)}"
        )

    returns = monthly_returns[available_universe].fillna(0.0)
    random_return_series = {}
    summary_rows = []

    for sim_id in range(n_simulations):
        selected = rng.choice(available_universe, size=top_n, replace=False)
        sim_returns = returns[list(selected)].mean(axis=1)
        random_return_series[f"random_top{top_n}_{sim_id:04d}"] = sim_returns

        cumulative = (1 + sim_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_months = len(sim_returns)
        ann_return = (1 + total_return) ** (12 / n_months) - 1
        ann_vol = sim_returns.std() * np.sqrt(12)
        sharpe = (sim_returns.mean() * 12) / ann_vol if ann_vol != 0 else np.nan
        drawdown = cumulative / cumulative.cummax() - 1
        max_drawdown = drawdown.min()

        summary_rows.append({
            "simulation": f"random_top{top_n}_{sim_id:04d}",
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "cumulative_return": total_return,
            "max_drawdown": max_drawdown,
            "selected_tickers": ",".join(selected),
        })

    random_returns = pd.DataFrame(random_return_series)
    random_summary = pd.DataFrame(summary_rows)

    metrics = [
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "cumulative_return",
        "max_drawdown",
    ]

    percentile_rows = []
    for metric in metrics:
        values = random_summary[metric].dropna()
        percentile_rows.append({
            "metric": metric,
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "median": values.quantile(0.50),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "mean": values.mean(),
        })

    percentile_summary = pd.DataFrame(percentile_rows)

    return random_returns, random_summary, percentile_summary
