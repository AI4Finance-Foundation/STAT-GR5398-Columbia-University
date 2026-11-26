"""Evaluate one valuation method on its own with the course's evaluate_valuation.py.

    .venv/bin/python experiments/eval_method.py dcf --period dev
    .venv/bin/python experiments/eval_method.py ev  --period dev
    .venv/bin/python experiments/eval_method.py peer_pe --period dev
    .venv/bin/python experiments/eval_method.py value --period dev     the combined value()

Each method returns METHOD_COLUMNS; this wraps it into value()'s COLUMNS (method_spread = 1)
and runs the course evaluator unchanged. dev / val only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import evaluate_valuation
from valuation import student_valuation as sv

METHODS = {"dcf": sv.dcf_value, "ev": sv.ev_ebitda_value, "peer_pe": sv.peer_pe_value}


def as_value(method):
    def value(as_of, tickers, data):
        r = method(as_of, tickers, data)
        r = r[~r["missing"]].copy()
        px = pd.Series(data.last_close(r["ticker"]))
        r["mispricing"] = r["fair_value"].to_numpy() / px.reindex(r["ticker"]).to_numpy() - 1
        r["method_spread"] = 1.0
        return r[sv.COLUMNS]
    return value


def main():
    name, rest = sys.argv[1], sys.argv[2:]
    if "test" in rest:
        raise SystemExit("dev / val only: the test period is run once, at the end")
    if name == "value":
        evaluate_valuation.main(rest)
        return
    sv.value = as_value(METHODS[name])
    # the course evaluator writes output/valuation_eval_<period>.csv; keep that file for value()
    official = Path("output") / f"valuation_eval_{rest[rest.index('--period') + 1]}.csv"
    saved = official.read_bytes() if official.exists() else None
    evaluate_valuation.main(rest)
    official.replace(Path("output/experiments") / f"method_eval_{name}_{official.stem.split('_')[-1]}.csv")
    if saved is not None:
        official.write_bytes(saved)


if __name__ == "__main__":
    main()
