"""Supplementary ablation with the course's engine timing (dev / val only).

    PYTHONHASHSEED=0 .venv/bin/python experiments/engine_ablation.py --period dev
    PYTHONHASHSEED=0 .venv/bin/python experiments/engine_ablation.py --period val

The required valuation-group table uses --timing full, so C = A and D = B. This runs the same four cells with
timing = engine, to show whether valuation adds the same on top of a timing module (D - C vs B - A).
Writes output/experiments/ablation_engine_<period>.csv; output/ablation_<period>.csv is not touched.
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

KEEP = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
        "turnover_per_year", "avg_exposure", "avg_removed_per_week"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["dev", "val"])  # never test
    period = ap.parse_args().period
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    rows = []
    for name, val, tim in (("A_selection", False, "none"), ("B_valuation", True, "none"),
                           ("C_timing_engine", False, "engine"), ("D_both_engine", True, "engine")):
        print("running", name, flush=True)
        s = rp.run_one(f"engine_{period}_{name}", start, end, valuation=val, timing=tim, stops=False, ctx=ctx)
        rows.append({"run": name, **{c: s.get(c) for c in KEEP}})
    table = pd.DataFrame(rows).set_index("run")
    table.to_csv(Path("output") / "experiments" / f"ablation_engine_{period}.csv")
    with pd.option_context("display.width", 200):
        print(table.to_string())


if __name__ == "__main__":
    main()
