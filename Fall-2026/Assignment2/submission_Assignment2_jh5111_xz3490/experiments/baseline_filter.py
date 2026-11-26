"""The chosen filter rule (student_valuation.FILTER_RULE) fed by the course's trailing-P/E example instead of value().

Answers: does our three-method valuation add anything over a simple P/E inside the same rule?
    PYTHONHASHSEED=0 .venv/bin/python experiments/baseline_filter.py --period dev
    PYTHONHASHSEED=0 .venv/bin/python experiments/baseline_filter.py --period val    comparison only
Writes output/experiments/baseline_filter_<period>.csv. Does not change the chosen configuration.
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
from valuation.examples import trailing_pe_value

KEEP = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
        "turnover_per_year", "avg_removed_per_week"]


def pe_table(as_of, tickers, data):
    df = trailing_pe_value(as_of, tickers, data).copy()
    df["n_methods"] = 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test
    period = ap.parse_args().period
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    saved = sv.combined_table
    rows = []
    try:
        for name, table in (("value", saved), ("trailing_pe", pe_table)):
            sv.combined_table = table
            sv._VALUE_CACHE.clear()
            print(f"running {name}", flush=True)
            s = rp.run_one(f"basefilter_{period}_{name}", start, end, valuation=True, timing="none",
                           stops=False, ctx=ctx)
            rows.append({"rule": sv.FILTER_RULE, "valuation": name, **{c: s.get(c) for c in KEEP}})
    finally:
        sv.combined_table = saved
        sv._VALUE_CACHE.clear()
    t = pd.DataFrame(rows)
    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"baseline_filter_{period}.csv", index=False)
    print("\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
