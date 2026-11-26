"""Robustness of the chosen filter (student_valuation.FILTER_RULE) on dev. The chosen rule is not changed by this.

    PYTHONHASHSEED=0 .venv/bin/python experiments/robustness_dev.py
    PYTHONHASHSEED=0 .venv/bin/python experiments/robustness_dev.py --period val
        val: diagnostic only (part 2, chosen setting, no grid). It does not change the rule.

1. Sensitivity: DROP_WORST_K in 1/2/3 x FILTER_DEPTH in 10/15/20, and RATE_LOOKBACK_DAYS in 91/182/365
   (k=2, depth 15), timing "none" (same numbers as "full").
2. Removed names vs their backfills: forward returns 5 / 21 / 63 sessions after each weekly decision
   (chosen setting k=2, depth=15, lookback 182). A window that would run past the period end is left empty.
Writes output/experiments/sensitivity_<period>.csv, removed_vs_backfill_<period>.csv, removed_log_<period>.csv.
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
KS, DEPTHS, LOOKBACKS = (1, 2, 3), (10, 15, 20), (91, 182, 365)
HORIZONS = (5, 21, 63)
CHOSEN = (2, 15, 182)


def top10(cands, members):
    g = sorted((c for c in cands if c.bucket == sv.BUCKET and c.symbol in members),
               key=lambda c: c.rank_score, reverse=True)
    return [c.symbol for c in g[:10]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test
    period = ap.parse_args().period
    grid = ([(k, d, 182) for k in KS for d in DEPTHS] + [(2, 15, lb) for lb in LOOKBACKS if lb != 182]
            if period == "dev" else [CHOSEN])
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    orig, saved = sv.filter_candidates, (sv.DROP_WORST_K, sv.FILTER_DEPTH, sv.RATE_LOOKBACK_DAYS)
    log = []

    def logged(as_of, candidates, data):
        kept = orig(as_of, candidates, data)
        if (sv.DROP_WORST_K, sv.FILTER_DEPTH, sv.RATE_LOOKBACK_DAYS) == CHOSEN:
            members = data.members()
            before, after = top10(candidates, members), top10(kept, members)
            log.append({"as_of": pd.Timestamp(as_of),
                        "removed": [t for t in before if t not in after],
                        "backfill": [t for t in after if t not in before]})
        return kept

    rows = []
    sv.filter_candidates = logged
    try:
        for k, d, lb in grid:
            sv.DROP_WORST_K, sv.FILTER_DEPTH, sv.RATE_LOOKBACK_DAYS = k, d, lb
            sv._VALUE_CACHE.clear()  # each run as if standalone
            print(f"running k={k} depth={d} lookback={lb}", flush=True)
            s = rp.run_one(f"sens_{period}_k{k}_d{d}_lb{lb}", start, end, valuation=True, timing="none",
                           stops=False, ctx=ctx)
            rows.append({"rule": sv.FILTER_RULE, "k": k, "depth": d, "rate_lookback": lb,
                         **{c: s.get(c) for c in KEEP}})
    finally:
        sv.filter_candidates = orig
        sv.DROP_WORST_K, sv.FILTER_DEPTH, sv.RATE_LOOKBACK_DAYS = saved

    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    sens = pd.DataFrame(rows)
    sens.to_csv(out / f"sensitivity_{period}.csv", index=False)

    close = prices.close.loc[:str(end)]  # forward windows stop at the period end (val never reads test prices)
    idx = close.index
    recs = []
    for e in log:
        i = idx.searchsorted(e["as_of"], side="right") - 1
        for side in ("removed", "backfill"):
            for t in e[side]:
                r = {"as_of": e["as_of"].date(), "side": side, "ticker": t}
                for h in HORIZONS:
                    j = i + h
                    r[f"fwd{h}"] = (close[t].iloc[j] / close[t].iloc[i] - 1) if j < len(idx) and t in close else None
                recs.append(r)
    det = pd.DataFrame(recs)
    det.to_csv(out / f"removed_log_{period}.csv", index=False)
    fw = [f"fwd{h}" for h in HORIZONS]
    summ = det.groupby("side")[fw].agg(["mean", "median", "count"])
    summ.to_csv(out / f"removed_vs_backfill_{period}.csv")

    with pd.option_context("display.width", 200):
        print("\n" + sens.to_string(index=False))
        print("\n" + summ.to_string())
        print("\nmost removed:\n" + det[det.side == "removed"]["ticker"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
