"""Performance metrics shared by every evaluation script."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Drawdowns the timing module is judged on: (peak, trough, end of rebound window).
CRISES = {
    "2018Q4": ("2018-09-20", "2018-12-24", "2019-04-30"),
    "2020-03": ("2020-02-19", "2020-03-23", "2020-08-31"),
    "2022": ("2022-01-03", "2022-10-12", "2023-03-31"),
}


def performance(equity: pd.Series) -> dict:
    """Total return, CAGR, volatility, Sharpe (rf = 0), max drawdown, Calmar."""
    s = equity.dropna()
    r = s.pct_change().dropna()
    if len(r) < 2:
        return {}
    years = len(r) / 252
    vol = r.std() * np.sqrt(252)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    mdd = (s / s.cummax() - 1).min()
    return {
        "total_return_pct": round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(r.mean() * 252 / vol, 3) if vol > 0 else None,
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else None,
    }


def crisis_table(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Drawdown avoided and rebound captured in each crisis inside the series."""
    rows = []
    for name, (peak, trough, end) in CRISES.items():
        peak, trough, end = map(pd.Timestamp, (peak, trough, end))
        if strategy.index.min() > peak or strategy.index.max() < end:
            continue

        def ret(s, a, b):
            s = s.loc[a:b].dropna()
            return (s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else np.nan
        fall_s, fall_b = ret(strategy, peak, trough), ret(benchmark, peak, trough)
        up_s, up_b = ret(strategy, trough, end), ret(benchmark, trough, end)
        rows.append({"crisis": name, "fall_strategy_pct": round(fall_s, 2), "fall_benchmark_pct": round(fall_b, 2),
                     "fall_avoided_pct": round(fall_s - fall_b, 2),
                     "rebound_strategy_pct": round(up_s, 2), "rebound_benchmark_pct": round(up_b, 2),
                     "rebound_missed_pct": round(up_b - up_s, 2)})
    return pd.DataFrame(rows)
