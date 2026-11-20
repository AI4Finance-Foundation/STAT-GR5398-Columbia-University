"""Backtest the portfolio: selection, plus your valuation filter and/or timing.

    python run_portfolio.py --period dev                                   A  selection only
    python run_portfolio.py --period dev --valuation                       B  + valuation filter
    python run_portfolio.py --period dev --timing student                  C  + timing
    python run_portfolio.py --period dev --valuation --timing student      D  both
    python run_portfolio.py --period dev --ablation --timing student       A, B, C and D in one run

--timing takes student (timing/student_timing.py) or a baseline: full, sma200, engine.
--scores replays selection scores you rebuilt with selection/build_scores.py --out
(bonus; run names then end in _<file name>).
Results go to output/<run>/ (equity.csv, trades.csv, stats.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from portfolio.backtester import BacktestConfig, run_backtest
from portfolio.hooks import FilteredPredictions, exposure_fn
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.metrics import performance
from shared.periods import record_test_run, resolve
from timing.baselines import BASELINES

CONFIG = "selection/configs/growth.json"


def timing_function(name):
    if name == "student":
        from timing import student_timing
        return student_timing.exposure
    return BASELINES[name]


def run_one(name, start, end, *, valuation, timing, stops, ctx):
    preds = ctx["scores"]
    filtered = None
    if valuation:
        from valuation import student_valuation
        filtered = preds = FilteredPredictions(preds, student_valuation.filter_candidates, ctx["bundle"])
    expo = exposure_fn(timing_function(timing), ctx["bundle"]) if timing != "none" else None
    cfg = BacktestConfig(start=start, end=end, stops=stops, use_exposure_engine=expo is not None)
    res = run_backtest(ctx["prices"], preds, ctx["params"], ctx["universe"], cfg,
                       global_unified=ctx["options"].inference_mode == "global-unified", exposure_fn=expo)
    out = Path("output") / name
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "equity.csv")
    res.trades.to_csv(out / "trades.csv", index=False)
    traded = res.trades["notional"].abs().sum() if len(res.trades) else 0.0
    stats = {"run": name, "start": str(start), "end": str(end), **performance(res.equity),
             "trades": int(len(res.trades)),
             "turnover_per_year": round(traded / res.equity.mean() / (len(res.equity) / 252), 2)}
    if expo is not None and expo.log:
        stats["avg_exposure"] = round(sum(expo.log) / len(expo.log), 3)
    if filtered is not None and filtered.calls:
        stats["avg_removed_per_week"] = round(filtered.removed / filtered.calls, 1)
    for b in ("SPY", "QQQ"):
        s = ctx["prices"].close[b].loc[str(start):str(end)].dropna()
        stats[f"{b}_total_return_pct"] = round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
        stats[f"{b}_sharpe"] = performance(s).get("sharpe")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["dev", "val", "test", "forward"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--valuation", action="store_true", help="apply valuation/student_valuation.filter_candidates")
    ap.add_argument("--timing", default="none", choices=["none", "student", *BASELINES])
    ap.add_argument("--ablation", action="store_true", help="run A/B/C/D; --timing sets C and D (default student)")
    ap.add_argument("--stops", action="store_true", help="two-layer 15%% stop (portfolio/backtester.py)")
    ap.add_argument("--scores", type=Path, help="selection scores from build_scores.py --out (default: the course scores)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    start, end = resolve(a.period, a.start, a.end)
    _, options, params, prices, _, universe = load(CONFIG)
    end = min(end, prices.sessions[-1].date())
    record_test_run("run_portfolio.py", sys.argv[1:], start, end)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(a.scores) if a.scores else MonthlyScores(),
           "bundle": DataBundle.load(prices=prices)}
    label = (a.period or f"{start}_{end}") + (f"_{a.scores.stem}" if a.scores else "")
    if a.ablation:
        t = a.timing if a.timing != "none" else "student"
        runs = [("A_selection", False, "none"), ("B_valuation", True, "none"),
                (f"C_timing_{t}", False, t), (f"D_both_{t}", True, t)]
    else:
        tag = "_".join(["valuation"] * a.valuation + [f"timing_{a.timing}"] * (a.timing != "none")) or "selection"
        runs = [(tag, a.valuation, a.timing)]
    rows = []
    for name, val, tim in runs:
        print(f"running {name}  {start} -> {end}", flush=True)
        rows.append(run_one(f"{label}_{name}{'_stops' if a.stops else ''}", start, end,
                            valuation=val, timing=tim, stops=a.stops, ctx=ctx))
    table = pd.DataFrame(rows).set_index("run")
    if a.ablation:
        table.to_csv(Path("output") / f"ablation_{label}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.T.to_string())


if __name__ == "__main__":
    main()
