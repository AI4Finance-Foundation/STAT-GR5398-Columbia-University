# portfolio_strategy/analytics.py

import numpy as np
import pandas as pd


def active_return_summary(strategy_returns, benchmark_name="SPY"):
    benchmark = strategy_returns[benchmark_name]
    rows = []

    for col in strategy_returns.columns:
        if col == benchmark_name:
            continue

        active = strategy_returns[col] - benchmark

        rows.append({
            "strategy": col,
            "benchmark": benchmark_name,
            "average_monthly_active_return": active.mean(),
            "cumulative_active_return": strategy_returns[col].add(1).prod() - benchmark.add(1).prod(),
            "monthly_win_rate": (active > 0).mean(),
            "tracking_error_annualized": active.std() * np.sqrt(12),
            "information_ratio": (active.mean() * 12) / (active.std() * np.sqrt(12)) if active.std() != 0 else np.nan
        })

    return pd.DataFrame(rows)


def simple_beta_alpha(strategy_returns, strategy_name, benchmark_name):
    y = strategy_returns[strategy_name].dropna()
    x = strategy_returns[benchmark_name].dropna()

    common_index = y.index.intersection(x.index)
    y = y.loc[common_index]
    x = x.loc[common_index]

    X = np.column_stack([np.ones(len(x)), x.values])
    alpha_monthly, beta = np.linalg.lstsq(X, y.values, rcond=None)[0]

    residuals = y.values - X @ np.array([alpha_monthly, beta])

    return {
        "strategy": strategy_name,
        "benchmark": benchmark_name,
        "monthly_alpha": alpha_monthly,
        "annualized_alpha": (1 + alpha_monthly) ** 12 - 1,
        "beta": beta,
        "residual_vol_annualized": residuals.std() * np.sqrt(12)
    }


def regression_summary(strategy_returns, benchmarks=("SPY", "QQQ")):
    rows = []

    for strategy in strategy_returns.columns:
        if strategy in benchmarks:
            continue

        for benchmark in benchmarks:
            rows.append(simple_beta_alpha(strategy_returns, strategy, benchmark))

    return pd.DataFrame(rows)