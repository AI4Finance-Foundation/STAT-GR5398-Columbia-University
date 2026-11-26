"""Year-by-year A (selection) vs B (+ valuation filter) from saved equity curves. No new backtests.

    .venv/bin/python experiments/yearly_ab.py
Writes output/experiments/yearly_ab.csv.
"""
from pathlib import Path

import pandas as pd

RUNS = {"dev": ("dev_A_selection", "dev_B_valuation"), "val": ("val_A_selection", "val_B_valuation"),
        "test": ("test_A_selection", "test_B_valuation")}


def yearly(eq):
    out = {}
    for y, s in eq.groupby(eq.index.year):
        prev = eq[eq.index.year < y]
        base = prev.iloc[-1] if len(prev) else s.iloc[0]
        dd = (s / s.cummax() - 1).min()
        out[y] = {"ret_pct": (s.iloc[-1] / base - 1) * 100, "maxdd_pct": dd * 100}
    return pd.DataFrame(out).T


rows = []
for period, (a, b) in RUNS.items():
    ea = pd.read_csv(Path("output") / a / "equity.csv", index_col=0, parse_dates=True)["equity"]
    eb = pd.read_csv(Path("output") / b / "equity.csv", index_col=0, parse_dates=True)["equity"]
    ya, yb = yearly(ea), yearly(eb)
    for y in ya.index:
        rows.append({"period": period, "year": y, "A_ret": ya.loc[y, "ret_pct"], "B_ret": yb.loc[y, "ret_pct"],
                     "B_minus_A": yb.loc[y, "ret_pct"] - ya.loc[y, "ret_pct"],
                     "A_maxdd": ya.loc[y, "maxdd_pct"], "B_maxdd": yb.loc[y, "maxdd_pct"]})
t = pd.DataFrame(rows).round(2)
Path("output/experiments").mkdir(parents=True, exist_ok=True)
t.to_csv("output/experiments/yearly_ab.csv", index=False)
print(t.to_string(index=False))
