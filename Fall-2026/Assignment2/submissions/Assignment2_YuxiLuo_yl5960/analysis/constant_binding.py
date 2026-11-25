"""How often do the post-2017 constants actually bind? Dev and val only.

Run from source_code/ with PYTHONHASHSEED=0:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/constant_binding.py

On the evaluate_valuation sample (top 20 growth_tech on the first session of each month):
  * MIN_GORDON_SPREAD (FinRobot, set in 2026): DCF base cases rejected because WACC - g < 1.5pp,
    and sensitivity-grid cells dropped for the same reason
  * STATUTORY_TAX = 21% (law from 2018): rows where effective_tax_rate is missing in the last
    four quarters, so the fallback is used
Also checks that every month's selection-score cutoff is before the decision date, and which
name changes the filter on 2022-06-30 when DGS10/BAA10Y are lagged one observation.
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

import valuation.finrobot.dcf as fdcf
from evaluate_valuation import top_candidates
from selection import MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.periods import resolve
from valuation import student_valuation as sv

bundle = DataBundle.load()
scores = MonthlyScores()
sessions = bundle.prices.sessions
count = {"dcf_calls": 0, "gordon_reject": 0, "grid_cells": 0, "grid_dropped_spread": 0, "tax_fallback": 0}
real_dcf, real_sens = sv.calculate_dcf, sv.calculate_sensitivity


def dcf(inputs, *a, **k):
    count["dcf_calls"] += 1
    if inputs.tax_rate == sv.STATUTORY_TAX:
        count["tax_fallback"] += 1
    try:
        return real_dcf(inputs, *a, **k)
    except ValueError as e:
        if "minimum" in str(e):
            count["gordon_reject"] += 1
        raise


def sens(inputs, wacc_range, tg_range, *a, **k):
    for w in wacc_range:
        for g in tg_range:
            count["grid_cells"] += 1
            if g < w and (w - g) < fdcf.MIN_GORDON_SPREAD:
                count["grid_dropped_spread"] += 1
    return real_sens(inputs, wacc_range, tg_range, *a, **k)


sv.calculate_dcf, sv.calculate_sensitivity = dcf, sens
late_cutoffs = []
for period in ("dev", "val"):
    start, end = resolve(period, None, None)
    win = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    before = dict(count)
    for day in win.to_series().groupby([win.year, win.month]).first():
        d = day.date()
        as_of = bundle.prices.last_session_on_or_before(d - timedelta(days=1))
        view = DataView(bundle, as_of)
        tickers = top_candidates(scores, d, view.members(), 20)
        if pd.Timestamp(scores.cutoff) > pd.Timestamp(as_of):
            late_cutoffs.append((d, scores.cutoff))
        with student_call():
            sv.value_detail(as_of, tickers, view)
    print(period, {k: count[k] - before[k] for k in count})
print("months whose score cutoff is after as_of:", late_cutoffs or "none")

sv.calculate_dcf, sv.calculate_sensitivity = real_dcf, real_sens
d = sessions[sessions.searchsorted(pd.Timestamp(2022, 7, 1))].date()
as_of = bundle.prices.last_session_on_or_before(d - timedelta(days=1))
preds = scores.get(d)
lag = __import__("copy").copy(bundle)
lag.macro_series = dict(bundle.macro_series)
for k in ("DGS10", "BAA10Y"):
    lag.macro_series[k] = bundle.macro_series[k].shift(1).dropna()
with student_call():
    k0 = {c.symbol for c in sv.filter_candidates(as_of, preds, DataView(bundle, as_of))}
    k1 = {c.symbol for c in sv.filter_candidates(as_of, preds, DataView(lag, as_of))}
    tick = sorted({c.symbol for c in preds} - k0 ^ k1 if False else (k0 ^ k1))
    det0 = sv.value_detail(as_of, tick, DataView(bundle, as_of))
    det1 = sv.value_detail(as_of, tick, DataView(lag, as_of))
print(f"2022-06-30 lagged macro: removed only with lag {sorted(k0 - k1)}, only without {sorted(k1 - k0)}")
print("DGS10 on as_of vs previous:", bundle.macro_series["DGS10"].loc[:str(as_of)].tail(2).to_dict())
cols = ["ticker", "price"] + [f"fair_{m}" for m in sv.METHODS]
print(det0[cols].round(2).to_string(index=False)); print(det1[cols].round(2).to_string(index=False))
