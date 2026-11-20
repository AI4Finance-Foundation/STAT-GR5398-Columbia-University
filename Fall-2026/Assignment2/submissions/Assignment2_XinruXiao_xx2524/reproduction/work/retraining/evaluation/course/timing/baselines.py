"""Timing baselines every submission is compared with. Same signature as
timing.student_timing.exposure(as_of, data)."""
from __future__ import annotations


def full(as_of, data) -> float:
    """Always fully invested."""
    return 1.0


def sma200(as_of, data) -> float:
    """Invested while SPY closes above its 200-session average, otherwise cash."""
    spy = data.close(["SPY"], lookback=200)["SPY"].dropna()
    if len(spy) < 200:
        return 1.0
    return 1.0 if spy.iloc[-1] > spy.mean() else 0.0


def engine(as_of, data) -> float:
    """The risk engine in timing/engine.py.

    exposure = clamp(16% / SPY 20-day realized volatility x (0.3 + 0.7 x risk score), 0.10, 1.0)
    risk score = 22% VIX percentile + 17% yield curve + 17% high-yield spread
                 + 17% SPY momentum + 15% sector breadth + 12% news sentiment
    Historically the news factor is fixed at 0.5 and the high-yield factor is
    neutral before 2023-09 (FRED publishes only ~3 years of that series), so
    four factors carry the signal.
    """
    from timing.engine import exposure_from_history
    res = exposure_from_history(data.market_frame(), data.macro_dict(), as_of)
    return float(res.exposureMultiplier) if res is not None else 1.0


BASELINES = {"full": full, "sma200": sma200, "engine": engine}
