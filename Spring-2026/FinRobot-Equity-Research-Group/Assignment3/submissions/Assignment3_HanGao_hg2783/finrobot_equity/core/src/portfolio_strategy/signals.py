# portfolio_strategy/signals.py

import pandas as pd


def compute_momentum(monthly_prices, lookback_months=12, skip_recent_months=1):
    """
    12-month momentum excluding the most recent month.
    Momentum_t = Price_{t-1} / Price_{t-12} - 1
    """
    past_price = monthly_prices.shift(skip_recent_months + lookback_months)
    recent_price = monthly_prices.shift(skip_recent_months)

    momentum = recent_price / past_price - 1
    return momentum


def compute_volatility(monthly_returns, lookback_months=12):
    """
    Rolling 12-month volatility.
    """
    vol = monthly_returns.rolling(lookback_months).std()
    return vol


def cross_sectional_rank_score(df, higher_is_better=True):
    """
    Convert raw signal values into cross-sectional rank scores between 0 and 1.
    """
    if higher_is_better:
        return df.rank(axis=1, pct=True)
    else:
        return df.rank(axis=1, pct=True, ascending=False)


def compute_quant_score(monthly_prices, monthly_returns):
    """
    Final Quant Score =
    70% momentum score
    + 30% low-volatility score
    """
    momentum = compute_momentum(monthly_prices)
    vol = compute_volatility(monthly_returns)

    momentum_score = cross_sectional_rank_score(momentum, higher_is_better=True)
    low_vol_score = cross_sectional_rank_score(vol, higher_is_better=False)

    quant_score = 0.7 * momentum_score + 0.3 * low_vol_score
    return quant_score