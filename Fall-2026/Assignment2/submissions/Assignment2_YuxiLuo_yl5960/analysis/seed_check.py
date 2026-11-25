"""Portfolio A (selection only) under one PYTHONHASHSEED, for the hash-seed note. Dev and val only.

Run from source_code/, once per seed (the seed must be set before Python starts):
    for s in 0 1 2 3 4; do PYTHONHASHSEED=$s python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/seed_check.py --period val; done

Writes output/tuning/<period>_A_seed<s>/ (never output/<period>_selection/) and appends one JSON line to
analysis/study/seeds.jsonl. B under seeds 1-4 is already in tuning_log.jsonl (final_seed*).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve

OUT = Path(__file__).resolve().parent / "study" / "seeds.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], required=True)
    a = ap.parse_args()
    seed = os.environ.get("PYTHONHASHSEED")
    if seed is None:
        sys.exit("set PYTHONHASHSEED")
    start, end = resolve(a.period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    end = min(end, prices.sessions[-1].date())
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    stats = rp.run_one(f"tuning/{a.period}_A_seed{seed}", start, end, valuation=False, timing="none", stops=False, ctx=ctx)
    stats.update({"period": a.period, "seed": int(seed)})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(json.dumps(stats) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
