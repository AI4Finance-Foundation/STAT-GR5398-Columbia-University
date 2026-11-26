"""Transaction-cost sensitivity of A (selection) and B (+ valuation filter) on dev.

The course backtester charges commission 1 bp + slippage 5 bp = 6 bp of traded notional. B trades more
(turnover 12.5 vs 10.5 a year), so its edge shrinks as costs rise. dev only.
    PYTHONHASHSEED=0 .venv/bin/python experiments/cost_sensitivity.py
Writes output/experiments/cost_sensitivity_dev.csv.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import run_portfolio as rp
from portfolio.backtester import BacktestConfig
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

TOTAL_BPS = (6, 12, 20)          # commission stays 1 bp; slippage = total - 1
KEEP = ["total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "calmar", "turnover_per_year"]


def main():
    start, end = resolve("dev", None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    rows = []
    try:
        for bps in TOTAL_BPS:
            rp.BacktestConfig = functools.partial(BacktestConfig, commission=0.0001, slippage=(bps - 1) / 10_000)
            for run, val in (("A", False), ("B", True)):
                sv._VALUE_CACHE.clear()
                s = rp.run_one(f"cost_dev_{bps}bp_{run}", start, end, valuation=val, timing="full", stops=False, ctx=ctx)
                rows.append({"cost_bps": bps, "run": run, **{k: s.get(k) for k in KEEP}})
                print(rows[-1], flush=True)
    finally:
        rp.BacktestConfig = BacktestConfig
    t = pd.DataFrame(rows)
    t.to_csv(Path("output") / "experiments" / "cost_sensitivity_dev.csv", index=False)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
