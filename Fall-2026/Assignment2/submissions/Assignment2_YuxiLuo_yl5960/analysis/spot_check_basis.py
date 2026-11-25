"""Spot-check the share-basis alignment on the names it was built for. Dev/val dates only.

Run from source_code/ with PYTHONHASHSEED=0:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/spot_check_basis.py [--out ../submissions/Assignment2_YuxiLuo_yl5960/analysis/study/spot_check_basis.csv]

For each ticker and as_of date (DataView limited to as_of, as in the backtest):
  price           last adjusted close on as_of (the price file's 2026 basis)
  basis           the factor _align_basis applied to the latest visible quarter
  pe_aligned      price / aligned TTM EPS          (what the module uses)
  pe_raw          price / unaligned TTM EPS        (what it would use without the fix)
  pe_own          the filing's own trailing P/E at its period end:
                  (ps x revenue_per_share) / TTM EPS, both on the filing's own basis,
                  so a split or currency mismatch cannot distort it
  fair_* / mis    each method's value and the combined mispricing, with and without alignment
A correct alignment gives pe_aligned close to pe_own (they differ only by the price move
between the period end and as_of); pe_raw is off by the basis factor.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from valuation import student_valuation as sv

TICKERS = ("KLAC", "APH", "NTES", "MSFT")
DATES = ("2018-06-29", "2019-06-28", "2020-06-30", "2021-06-30", "2022-06-30", "2023-06-29", "2023-12-28")


def ttm_eps(view, t, align):
    sv.ALIGN_BASIS = align
    try:
        q = sv._prepare(view.fundamentals([t]), view.close([t]))
    finally:
        sv.ALIGN_BASIS = True
    return q.iloc[-1] if len(q) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "study" / "spot_check_basis.csv"))
    a = ap.parse_args()
    bundle = DataBundle.load()
    rows = []
    for d in DATES:
        as_of = bundle.prices.last_session_on_or_before(pd.Timestamp(d).date())
        view = DataView(bundle, as_of)
        with student_call():
            det = {al: None for al in (True, False)}
            for al in det:
                sv.ALIGN_BASIS = al
                det[al] = sv.value_detail(as_of, list(TICKERS), view).set_index("ticker")
            sv.ALIGN_BASIS = True
        for t in TICKERS:
            raw = view.fundamentals([t]).sort_values("datadate")
            if len(raw) < 4:
                continue
            l4 = raw.tail(4)
            own_price = float(l4["ps"].iloc[-1] * l4["revenue_per_share"].iloc[-1])
            pe_own = own_price / float(l4["EPS"].sum())
            al, un = ttm_eps(view, t, True), ttm_eps(view, t, False)
            price = view.last_close([t]).get(t)
            rec = {"as_of": as_of, "ticker": t, "datadate": str(pd.Timestamp(al["datadate"]).date()),
                   "price": round(price, 2), "basis": round(float(al["basis"]), 3),
                   "eps_ttm_aligned": round(float(al["eps_ttm"]), 3), "eps_ttm_raw": round(float(un["eps_ttm"]), 3),
                   "pe_aligned": round(price / al["eps_ttm"], 1), "pe_raw": round(price / un["eps_ttm"], 1),
                   "pe_own": round(pe_own, 1)}
            for tag, frame in (("al", det[True]), ("raw", det[False])):
                r = frame.loc[t] if t in frame.index else None
                for m in sv.METHODS:
                    rec[f"{m}_{tag}"] = round(float(r[f"fair_{m}"]), 2) if r is not None and np.isfinite(r[f"fair_{m}"]) else np.nan
                rec[f"mis_{tag}"] = round(float(r["mispricing"]), 3) if r is not None else np.nan
                rec[f"flag_{tag}"] = bool(sv.overvalued(frame.loc[[t]].reset_index()).iloc[0]) if r is not None else None
            rows.append(rec)
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(exist_ok=True)
    out.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
