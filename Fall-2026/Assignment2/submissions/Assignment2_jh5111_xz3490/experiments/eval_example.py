"""The course's trailing-P/E example (evaluate_valuation.py --example), saved under its own name.

    .venv/bin/python experiments/eval_example.py --period dev
    .venv/bin/python experiments/eval_example.py --period val
    .venv/bin/python experiments/eval_example.py --period test     once, with the final test run

evaluate_valuation.py writes both value() and --example to output/valuation_eval_<period>.csv.
This keeps value()'s file and writes the example's to output/valuation_eval_example_<period>.csv.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_valuation


def main():
    args = sys.argv[1:]
    if "--period" not in args:
        raise SystemExit("pass --period dev, val or test")
    period = args[args.index("--period") + 1]
    ours = Path("output") / f"valuation_eval_{period}.csv"
    keep = ours.with_suffix(".keep") if ours.exists() else None
    if keep:
        shutil.copy(ours, keep)
    try:
        sys.argv = [sys.argv[0]] + args + ["--example"]
        evaluate_valuation.main(args + ["--example"])
        shutil.move(ours, Path("output") / f"valuation_eval_example_{period}.csv")
    finally:
        if keep:
            shutil.move(keep, ours)


if __name__ == "__main__":
    main()
