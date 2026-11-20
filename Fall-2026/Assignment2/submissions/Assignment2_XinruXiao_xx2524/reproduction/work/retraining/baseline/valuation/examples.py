"""Worked examples of the DataView and FinRobot DCF calls. Copy and improve them.

Both are deliberately simple; their assumptions (one-year revenue growth fading
linearly, a 5% equity risk premium, zero working-capital change) are the first
things to question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity


def _ttm(fund: pd.DataFrame) -> pd.DataFrame:
    """Last four quarters per ticker; tickers with fewer than four are dropped."""
    f = fund.sort_values(["ticker", "datadate"])
    last4 = f.groupby("ticker").tail(4)
    return last4[last4.groupby("ticker")["ticker"].transform("size") == 4]


def trailing_pe_value(as_of, tickers, data) -> pd.DataFrame:
    """Fair value = sector-median trailing P/E x own TTM EPS; range = sector P25..P75."""
    members = data.members()
    universe = sorted(members | set(tickers))
    last4 = _ttm(data.fundamentals(universe))
    g = last4.groupby("ticker").agg(eps_ttm=("EPS", "sum"), sector=("gsector", "last"))
    g["price"] = pd.Series(data.last_close(g.index))
    g["pe"] = g["price"] / g["eps_ttm"]
    peers = g[(g["eps_ttm"] > 0) & g.index.isin(members) & g["pe"].between(0, 200)]
    bands = peers.groupby("sector")["pe"].quantile([0.25, 0.5, 0.75]).unstack()
    rows = []
    for t in tickers:
        if t not in g.index or g.at[t, "eps_ttm"] <= 0 or g.at[t, "sector"] not in bands.index:
            continue
        eps, price = g.at[t, "eps_ttm"], g.at[t, "price"]
        lo, mid, hi = (bands.loc[g.at[t, "sector"], q] * eps for q in (0.25, 0.5, 0.75))
        if not np.isfinite(price):
            continue
        rows.append({"ticker": t, "fair_value": mid, "low": lo, "high": hi,
                     "mispricing": mid / price - 1, "method_spread": 1.0})
    return pd.DataFrame(rows, columns=["ticker", "fair_value", "low", "high", "mispricing", "method_spread"])


def weekly_beta(ticker, data, years=5) -> float | None:
    """Regression beta of weekly returns against SPY over the last ``years``."""
    px = data.close([ticker, "SPY"], lookback=252 * years).dropna()
    wk = px.resample("W-FRI").last().pct_change().dropna()
    if len(wk) < 52:
        return None
    return float(np.cov(wk[ticker], wk["SPY"])[0, 1] / wk["SPY"].var())


def per_share_dcf(as_of, ticker, data, *, erp=0.05, terminal_growth=0.025,
                  credit_spread=0.02, years=10) -> dict | None:
    """FinRobot's calculate_dcf on per-share inputs (shares_outstanding = 1).

    fundamentals.db has per-share and ratio columns but no totals, so every
    input is per share. D&A is not stored; it equals EBITDA margin minus
    operating margin.
    """
    f = data.fundamentals([ticker]).sort_values("datadate")
    if len(f) < 8:
        return None
    q, prev = f.tail(4), f.iloc[-8:-4]
    revenue = q["revenue_per_share"].sum()
    if not revenue > 0:
        return None
    first_growth = float(np.clip(revenue / prev["revenue_per_share"].sum() - 1, -0.10, 0.30))
    growth = list(np.linspace(first_growth, terminal_growth, years))
    beta = weekly_beta(ticker, data)
    rf = data.macro("DGS10")
    if beta is None or rf.empty:
        return None
    rf = float(rf.iloc[-1]) / 100
    clip = lambda x, lo, hi: float(np.clip(x, lo, hi)) if np.isfinite(x) else lo
    inputs = DCFInputs(
        revenue_base=revenue,
        revenue_growth_rates=growth,
        ebitda_margin=clip(q["ebitda_margin"].median(), 0, 1),
        da_pct_revenue=clip((q["ebitda_margin"] - q["operating_margin"]).median(), 0, 0.5),
        capex_pct_revenue=clip(q["capex_per_share"].abs().sum() / revenue, 0, 1),
        nwc_pct_revenue=0.0,
        tax_rate=clip(q["effective_tax_rate"].median(), 0, 0.35),
        risk_free_rate=clip(rf, 0, 0.15),
        beta=clip(adjust_beta_blume(beta), 0, 5),
        equity_risk_premium=erp,
        cost_of_debt=clip(rf + credit_spread, 0, 0.20),
        debt_ratio=clip(q["debt_to_capital"].iloc[-1], 0, 0.95),
        terminal_growth_rate=terminal_growth,
        shares_outstanding=1.0,
        net_debt=float(q["interest_debt_per_share"].iloc[-1] - q["cash_per_share"].iloc[-1]),
    )
    try:
        res = calculate_dcf(inputs)
    except (ValueError, ZeroDivisionError):
        return None
    w = res.wacc
    grid = calculate_sensitivity(inputs, wacc_range=[w - 0.01, w, w + 0.01],
                                 tg_range=[terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005])
    cells = [p for row in grid["implied_prices"] for p in row if p is not None]
    return {"ticker": ticker, "fair_value": res.implied_price, "low": min(cells) if cells else None,
            "high": max(cells) if cells else None, "wacc": w}
