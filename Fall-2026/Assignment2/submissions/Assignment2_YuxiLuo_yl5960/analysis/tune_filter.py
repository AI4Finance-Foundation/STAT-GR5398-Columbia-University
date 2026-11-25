"""Run portfolio B (selection + valuation filter) with module constants overridden.

Run from source_code/ (dev or val only; never test):
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/tune_filter.py --period dev --tag p050 --set FILTER_PREMIUM=0.5

Writes output/tuning/<period>_B_<tag>/ and appends one JSON line to
../submissions/Assignment2_YuxiLuo_yl5960/analysis/tuning_log.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

LOG = Path(__file__).resolve().parent / "tuning_log.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--log", default=str(LOG), help="JSON-lines file to append the stats to")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(sv, k, eval(v))
        overrides[k] = getattr(sv, k)
    start, end = resolve(a.period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    end = min(end, prices.sessions[-1].date())
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    t0 = time.time()
    stats = rp.run_one(f"tuning/{a.period}_B_{a.tag}", start, end, valuation=True, timing="none", stops=False, ctx=ctx)
    stats.update({"tag": a.tag, "period": a.period, "overrides": overrides, "seconds": round(time.time() - t0)})
    with Path(a.log).open("a") as fh:
        fh.write(json.dumps(stats, default=str) + "\n")
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
