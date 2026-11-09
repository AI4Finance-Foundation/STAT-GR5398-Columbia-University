"""Evaluate a timing module on SPY alone, without stock selection.

    python evaluate_timing.py --period dev                          student vs all baselines
    python evaluate_timing.py --period dev --timing sma200 engine

Each week the exposure is decided at the close of the last session before
the week's first session and held until the next decision. Daily return =
exposure x SPY return, minus 6 bp per unit of exposure traded.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.metrics import crisis_table, performance
from shared.periods import record_test_run, resolve
from timing.baselines import BASELINES

COST = 0.0006


def exposure_path(fn, bundle, sessions) -> pd.Series:
    """Exposure held on each session: decided at the close before each week's first session."""
    held, current, last_week = [], None, None
    for day in sessions:
        week = day.isocalendar()[:2]
        if week != last_week:
            last_week = week
            as_of = bundle.prices.last_session_on_or_before(day.date() - timedelta(days=1))
            with student_call():
                current = float(fn(as_of, DataView(bundle, as_of)))
            if not 0.0 <= current <= 1.0:
                raise ValueError(f"exposure({as_of}) returned {current}; it must be between 0.0 and 1.0")
        held.append(current)
    return pd.Series(held, index=sessions)


def simulate(expo: pd.Series, spy: pd.Series) -> pd.Series:
    ret = spy.pct_change().reindex(expo.index).fillna(0.0)
    cost = expo.diff().abs().fillna(expo.iloc[0]) * COST
    return (1 + expo * ret - cost).cumprod()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["dev", "val", "test", "forward"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--timing", nargs="+", default=["student", *BASELINES])
    a = ap.parse_args(argv)
    start, end = resolve(a.period, a.start, a.end)
    bundle = DataBundle.load()
    prices = bundle.prices
    end = min(end, prices.sessions[-1].date())
    record_test_run("evaluate_timing.py", sys.argv[1:], start, end)
    sessions = prices.sessions[(prices.sessions >= pd.Timestamp(start)) & (prices.sessions <= pd.Timestamp(end))]
    first = prices.sessions.searchsorted(sessions[0])
    spy = prices.close["SPY"].loc[prices.sessions[max(first - 5, 0)]: sessions[-1]]
    buy_hold = (spy / spy.loc[sessions[0]]).loc[sessions]
    rows, crises = [], []
    for name in a.timing:
        if name == "student":
            from timing import student_timing
            fn = student_timing.exposure
        else:
            fn = BASELINES[name]
        expo = exposure_path(fn, bundle, sessions)
        eq = simulate(expo, spy)
        rows.append({"timing": name, **performance(eq), "avg_exposure": round(expo.mean(), 3),
                     "time_in_market_pct": round((expo > 0.5).mean() * 100, 1),
                     "exposure_changes": int((expo.diff().abs() > 1e-9).sum())})
        ct = crisis_table(eq, buy_hold)
        if len(ct):
            crises.append(ct.assign(timing=name))
    rows.append({"timing": "SPY buy & hold", **performance(buy_hold), "avg_exposure": 1.0})
    table = pd.DataFrame(rows).set_index("timing")
    label = a.period or f"{start}_{end}"
    out = Path("output"); out.mkdir(exist_ok=True)
    table.to_csv(out / f"timing_eval_{label}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"\nSPY timing, {sessions[0].date()} -> {sessions[-1].date()}\n")
        print(table.to_string())
        if crises:
            ct = pd.concat(crises).set_index(["crisis", "timing"]).sort_index()
            ct.to_csv(out / f"timing_crises_{label}.csv")
            print("\nCrises, percent (fall_avoided > 0 is good, rebound_missed > 0 is bad)\n")
            print(ct.to_string())


if __name__ == "__main__":
    main()
