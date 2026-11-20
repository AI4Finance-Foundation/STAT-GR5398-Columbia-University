"""Automatic checks for your module. Run before you submit.

    python check_submission.py              both modules
    python check_submission.py --timing     timing/student_timing.py
    python check_submission.py --valuation  valuation/student_valuation.py

A FAIL means the submission is not graded until it is fixed.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

from evaluate_valuation import check_frame, top_candidates
from selection import MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call

SAMPLE_DATES = [date(2017, 6, 30), date(2018, 12, 21), date(2020, 3, 20),
                date(2021, 9, 30), date(2022, 10, 12), date(2023, 6, 30)]
SLOW = {"exposure": 2.0, "value": 30.0, "filter_candidates": 30.0}
CALLS = {"exposure": "about 520 weekly calls per backtest", "value": "about 60 monthly calls per evaluation period",
         "filter_candidates": "about 520 weekly calls per backtest"}

FAIL_PATTERNS = [
    (r"\b(requests|urllib|httpx|aiohttp|socket)\b", "network access"),
    (r"\b(openai|anthropic|google\.generativeai|ollama)\b", "an LLM call (it has seen the future)"),
    (r"\b(PriceStore|FundamentalStore|TradeableUniverse|DataBundle|load_exposure_inputs)\b|shared\.(store|prices|universe)",
     "loading course data directly instead of through the DataView"),
    (r"daily_adjusted|fundamentals\.db|market_series|monthly_scores|data/(prices|macro|reference|selection)",
     "reading course data files directly"),
    (r"\b(trade_price|adj_close_q|y_return)\b", "a column dated after as_of"),
    (r"['\"](pe|ev_multiple)['\"]", "the database pe / ev_multiple columns (basis not verified)"),
]
WARN_PATTERNS = [
    (r"\bread_(parquet|csv|excel|json|pickle|feather)\b|\bopen\(|joblib\.load|np\.load",
     "reads a file at import time: it must hold nothing dated after the as_of it is used on"),
    (r"\brandom\b", "randomness: fix the seed so results reproduce"),
]

results = []


def report(status, check, detail=""):
    results.append(status)
    print(f"[{status}] {check}" + (f": {detail}" if detail else ""), flush=True)


def scan(path: Path):
    code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", path.read_text())
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    clean = True
    for pat, why in FAIL_PATTERNS:
        for m in re.finditer(pat, code):
            clean = False
            report("FAIL", f"{path} static scan", f"{why}: {m.group(0)!r}")
    for pat, why in WARN_PATTERNS:
        if re.search(pat, code):
            report("WARN", f"{path} static scan", why)
    if clean:
        report("PASS", f"{path} static scan", "no direct data, network or LLM access")


def call(fn, *args):
    t0 = time.perf_counter()
    with student_call():
        out = fn(*args)
    return out, time.perf_counter() - t0


def speed(name, worst):
    report("WARN" if worst > SLOW[name] else "PASS", f"{name}() speed",
           f"slowest call {worst:.2f}s; {CALLS[name]}")


def check_timing(bundle):
    scan(Path("timing/student_timing.py"))
    from timing import student_timing
    worst = 0.0
    for d in SAMPLE_DATES:
        as_of = bundle.prices.last_session_on_or_before(d)
        try:
            x1, dt = call(student_timing.exposure, as_of, DataView(bundle, as_of))
            x2, _ = call(student_timing.exposure, as_of, DataView(bundle, as_of))
            x1, x2 = float(x1), float(x2)
        except Exception as exc:
            return report("FAIL", f"exposure({as_of})", f"{type(exc).__name__}: {exc}")
        if not 0.0 <= x1 <= 1.0:
            return report("FAIL", f"exposure({as_of})", f"returned {x1}; must be in [0, 1]")
        if x1 != x2:
            return report("FAIL", f"exposure({as_of}) repeatability", f"{x1} then {x2}")
        worst = max(worst, dt)
    report("PASS", f"exposure() on {len(SAMPLE_DATES)} sample dates", "a number in [0, 1], same answer twice")
    speed("exposure", worst)


def check_valuation(bundle):
    scan(Path("valuation/student_valuation.py"))
    from valuation import student_valuation
    scores = MonthlyScores()
    worst_v = worst_f = 0.0
    valued = asked = 0
    for d in SAMPLE_DATES:
        as_of = bundle.prices.last_session_on_or_before(d)
        rebalance = bundle.prices.sessions[bundle.prices.sessions > pd.Timestamp(as_of)][0].date()
        view = DataView(bundle, as_of)
        tickers = top_candidates(scores, rebalance, view.members(), 20)
        try:
            df1, dt = call(student_valuation.value, as_of, tickers, view)
            df1 = check_frame(df1, as_of, tickers, view)
            df2, _ = call(student_valuation.value, as_of, tickers, DataView(bundle, as_of))
            df2 = check_frame(df2, as_of, tickers, view)
        except Exception as exc:
            return report("FAIL", f"value({as_of})", f"{type(exc).__name__}: {exc}")
        if not df1.reset_index(drop=True).equals(df2.reset_index(drop=True)):
            return report("FAIL", f"value({as_of}) repeatability", "two calls returned different frames")
        worst_v, valued, asked = max(worst_v, dt), valued + len(df1), asked + len(tickers)
        cands = scores.get(rebalance)
        try:
            kept, dt = call(student_valuation.filter_candidates, as_of, cands, DataView(bundle, as_of))
            kept = tuple(kept)
        except Exception as exc:
            return report("FAIL", f"filter_candidates({as_of})", f"{type(exc).__name__}: {exc}")
        if not set(kept) <= set(cands):
            return report("FAIL", f"filter_candidates({as_of})", "returned rows that were not in its input")
        worst_f = max(worst_f, dt)
    report("PASS", f"value() on {len(SAMPLE_DATES)} sample dates", "columns, ranges and mispricing consistent; repeatable")
    report("PASS" if valued else "WARN", "value() coverage", f"valued {valued} of {asked} candidates")
    report("PASS", f"filter_candidates() on {len(SAMPLE_DATES)} sample dates", "only removes rows")
    speed("value", worst_v)
    speed("filter_candidates", worst_f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--valuation", action="store_true")
    a = ap.parse_args(argv)
    both = not (a.timing or a.valuation)
    bundle = DataBundle.load()
    for wanted, fn in ((a.timing or both, check_timing), (a.valuation or both, check_valuation)):
        if wanted:
            try:
                fn(bundle)
            except Exception:
                report("FAIL", fn.__name__, traceback.format_exc(limit=3))
    log = Path("output") / "test_runs.jsonl"
    n = sum(1 for _ in log.open()) if log.exists() else 0
    report("PASS" if n <= 1 else "WARN", "test-period runs", f"{n} logged in {log}; report this number")
    print(f"\n{results.count('FAIL')} FAIL, {results.count('WARN')} WARN, {results.count('PASS')} PASS")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    main()
