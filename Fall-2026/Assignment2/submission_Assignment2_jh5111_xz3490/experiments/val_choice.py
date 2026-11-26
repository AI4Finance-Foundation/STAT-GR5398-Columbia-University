"""Val check of the 11/25 candidate rule cheap_k against the current worst_k_rates. Run once.

    PYTHONHASHSEED=0 .venv/bin/python experiments/val_choice.py

Registered before running (team doc, 11/25): k = 2 for both; adopt cheap_k only if its val Sharpe AND
val Calmar are both higher than worst_k_rates'. Otherwise keep worst_k_rates.
Writes output/experiments/val_choice.csv.
"""
from __future__ import annotations

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
    start, end = resolve("val", None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    saved, rows = sv.FILTER_RULE, []
    try:
        for rule in ("worst_k_rates", "cheap_k"):
            sv.FILTER_RULE = rule
            print("running", rule, flush=True)
            s = rp.run_one(f"valchoice_{rule}", start, end, valuation=True, timing="none", stops=False, ctx=ctx)
            rows.append({"rule": rule, **{c: s.get(c) for c in KEEP}})
    finally:
        sv.FILTER_RULE = saved
    table = pd.DataFrame(rows).set_index("rule")
    table.to_csv(Path("output") / "experiments" / "val_choice.csv")
    cur, new = table.loc["worst_k_rates"], table.loc["cheap_k"]
    adopt = new["sharpe"] > cur["sharpe"] and new["calmar"] > cur["calmar"]
    print(table.to_string())
    print("\nregistered rule -> adopt cheap_k:", adopt)


if __name__ == "__main__":
    main()
