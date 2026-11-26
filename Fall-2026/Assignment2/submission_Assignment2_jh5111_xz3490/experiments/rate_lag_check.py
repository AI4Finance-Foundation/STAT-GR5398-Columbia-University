"""Look-ahead audit: DGS10 lagged one more observation, in the filter's rate gate and/or the DCF risk-free rate.

DGS10 for day t comes from quotes taken about 3:30 pm on t; the backtest decides after the close of t and
trades at the open of t+1. If the publication time were later than the open, using the previous observation
is the safe version. This compares the two.
    PYTHONHASHSEED=0 .venv/bin/python experiments/rate_lag_check.py --period dev
    (val runs only same_day and lagged_1, the gate check; the DCF variants are dev only)
Writes output/experiments/rate_lag_<period>.csv. Does not change the chosen configuration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

KEEP = ["total_return_pct", "sharpe", "max_drawdown_pct", "calmar", "turnover_per_year", "avg_removed_per_week"]


def rates_rising_lagged(as_of, data) -> bool:
    r = data.macro("DGS10").dropna().iloc[:-1]
    past = r.loc[: pd.Timestamp(as_of) - pd.Timedelta(days=sv.RATE_LOOKBACK_DAYS)]
    return bool(len(r) and len(past) and r.iloc[-1] > past.iloc[-1])


class _LaggedMacro:
    """A DataView whose DGS10 series stops one observation earlier; everything else passes through."""

    def __init__(self, view):
        self._v = view

    def macro(self, series_id):
        s = self._v.macro(series_id)
        return s.dropna().iloc[:-1] if series_id == "DGS10" else s

    def __getattr__(self, name):
        return getattr(self._v, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test
    period = ap.parse_args().period
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    orig, orig_dcf, rows = sv.rates_rising, sv.dcf_inputs_table, []

    def dcf_lagged(as_of, tickers, data):
        return orig_dcf(as_of, tickers, _LaggedMacro(data))

    variants = [("same_day", orig, orig_dcf), ("lagged_1", rates_rising_lagged, orig_dcf)]
    if period == "dev":
        variants += [("dcf_lagged", orig, dcf_lagged), ("both_lagged", rates_rising_lagged, dcf_lagged)]
    try:
        for name, fn, dcf_fn in variants:
            sv.rates_rising, sv.dcf_inputs_table = fn, dcf_fn
            sv._VALUE_CACHE.clear()
            s = rp.run_one(f"ratelag_{period}_{name}", start, end, valuation=True, timing="none", stops=False, ctx=ctx)
            rows.append({"dgs10": name, **{c: s.get(c) for c in KEEP}})
    finally:
        sv.rates_rising, sv.dcf_inputs_table = orig, orig_dcf
        sv._VALUE_CACHE.clear()
    t = pd.DataFrame(rows)
    out = Path("output") / "experiments"
    t.to_csv(out / f"rate_lag_{period}.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
