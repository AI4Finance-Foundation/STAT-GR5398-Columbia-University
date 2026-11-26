"""Shared setup for the Opus strategy research.

Runs against the frozen team tree (valuation_candidate/source_code) without
modifying it. All outputs go to Opus_handoff/strategy_research/.
Only dev (2017-2021) and val (2022-2023) are ever touched here; the test
period is refused.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "valuation_candidate" / "source_code"
CACHE = HERE / "cache"
OUT = HERE / "results"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

os.chdir(SRC)                      # course code resolves data/ relative to cwd
sys.path.insert(0, str(SRC))

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

from shared.periods import PERIODS                          # noqa: E402
from portfolio.backtester import BacktestConfig, run_backtest  # noqa: E402

ALLOWED = {"dev": PERIODS["dev"], "val": PERIODS["val"]}


def period(name: str):
    if name not in ALLOWED:
        raise SystemExit("This research only uses dev and val; test is closed.")
    return ALLOWED[name]


def load_course():
    from selection.config import load
    from shared.data_view import DataBundle
    _, _, params, prices, _, univ = load("selection/configs/growth.json")
    bundle = DataBundle.load(prices=prices)
    return params, prices, univ, bundle


def rebalance_days(prices, start: date, end: date):
    cal = prices.sessions.to_series().loc[str(start):str(end)]
    iso = cal.dt.isocalendar()
    return [d.date() for d in cal.groupby([iso.year, iso.week]).first()]


class RemovalScores:
    """Wrap MonthlyScores and remove a precomputed set of symbols each week."""

    def __init__(self, base, removed_by_day):
        self.base, self.removed = base, removed_by_day
        self.cutoff = None

    def get(self, day):
        items = self.base.get(day)
        self.cutoff = self.base.cutoff
        drop = self.removed.get(day, set())
        return tuple(p for p in items if p.symbol not in drop)


def backtest(prices, params, univ, scores, start, end):
    res = run_backtest(prices, scores, params, univ, BacktestConfig(start=start, end=end))
    return res


def turnover(res):
    t = res.trades
    if not len(t):
        return 0.0
    return float(t.notional.abs().sum() / res.equity.mean() / (len(res.equity) / 252))
