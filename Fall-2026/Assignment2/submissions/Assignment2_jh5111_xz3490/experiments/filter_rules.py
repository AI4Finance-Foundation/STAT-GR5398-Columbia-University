"""Run every pre-registered filter rule (student_valuation.FILTER_RULES) through the portfolio backtest.

    .venv/bin/python experiments/filter_rules.py --period dev
    .venv/bin/python experiments/filter_rules.py --period val     once, to choose the final rule

Timing is "none", which gives the same numbers as the ablation's "full" (checked on dev and val). dev / val only.
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

KEEP = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
        "turnover_per_year", "avg_removed_per_week"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test
    period = ap.parse_args().period
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    rows = []
    default = sv.FILTER_RULE
    try:
        for rule in sv.FILTER_RULES:
            sv.FILTER_RULE = rule
            print(f"running {rule}", flush=True)
            rows.append(rp.run_one(f"rules_{period}_{rule}", start, end, valuation=rule != "none",
                                   timing="none", stops=False, ctx=ctx))
    finally:
        sv.FILTER_RULE = default
    table = pd.DataFrame(rows).set_index("run").reindex(columns=KEEP)
    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"filter_rules_{period}.csv")
    with pd.option_context("display.width", 200):
        print("\n" + table.to_string())


if __name__ == "__main__":
    main()
