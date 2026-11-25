"""Empirical look-ahead tests for valuation/student_valuation.py. Dev/val dates only.

Run from source_code/ with PYTHONHASHSEED=0:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/lookahead_test.py [--tmp DIR]

On 14 as_of dates (the session before the first session of January and July, 2017-2023),
for the 25 best-ranked growth_tech candidates (FILTER_DEPTH, as filter_candidates sees them):

1. Poisoned future. A copy of fundamentals.db where every filing row not visible on as_of
   (datadate after as_of, or accepted/filed after as_of; this includes later restatements
   of old quarters) has each numeric column multiplied by a random factor in [0.2, 5.2],
   and every membership snapshot after as_of is replaced by a single ticker. Prices
   (close and open), the market series and the macro series after as_of are multiplied by
   random factors too. value_detail() and filter_candidates() must return exactly the same
   output as on the real data.
2. Price basis. KLAC's and NVDA's whole price history is multiplied by 10, as if the price
   file had been adjusted for a different future split. With share-basis alignment every
   mispricing must be unchanged; the fair values of the two names scale by 10.
3. Macro one session later. DGS10 and BAA10Y are lagged by one observation (the value of the
   previous day), in case the as_of value was not yet published before the next open.
   Reports how much mispricing and the filter's decisions move.
"""
import argparse
import copy
import shutil
import sqlite3
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from evaluate_valuation import top_candidates
from selection import MonthlyScores
from selection.config import DATA
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.store import FundamentalStore
from shared.universe import TradeableUniverse
from valuation import student_valuation as sv

TEXT = {"ticker", "datadate", "gsector", "filing_date", "accepted_date"}
SCALED = ("KLAC", "NVDA")


def run(bundle, as_of, tickers, preds):
    view = DataView(bundle, as_of)
    with student_call():
        det = sv.value_detail(as_of, tickers, view)
        kept = tuple(c.symbol for c in sv.filter_candidates(as_of, preds, view))
    if det.empty:   # early 2017: fewer than four quarters of filings are visible yet
        det = pd.DataFrame(columns=["ticker", "fair_value", "mispricing", "fair_dcf"])
    return det.sort_values("ticker").reset_index(drop=True), kept


def poisoned_bundle(bundle, as_of, tmp, rng):
    ts, iso = pd.Timestamp(as_of), str(as_of)
    db = Path(tmp) / f"poison_{iso}.db"
    shutil.copy(DATA / "fundamentals.db", db)
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("pragma table_info(fundamental_data)") if r[1] not in TEXT]
    # Not visible on as_of under the course's rule (shared/store.py): rows with neither date
    # become visible about two months after quarter end.
    future = """NOT (datadate <= :d AND (
        (COALESCE(accepted_date, '') != '' AND accepted_date <= :d)
        OR (COALESCE(accepted_date, '') = '' AND COALESCE(filing_date, '') != '' AND filing_date <= :d)
        OR (COALESCE(accepted_date, '') = '' AND COALESCE(filing_date, '') = '' AND COALESCE(CASE
                WHEN substr(datadate, 6, 5) = '03-31' THEN date(substr(datadate, 1, 4) || '-06-01')
                WHEN substr(datadate, 6, 5) = '06-30' THEN date(substr(datadate, 1, 4) || '-09-01')
                WHEN substr(datadate, 6, 5) = '09-30' THEN date(substr(datadate, 1, 4) || '-12-01')
                WHEN substr(datadate, 6, 5) = '12-31' THEN date((CAST(substr(datadate, 1, 4) AS INTEGER) + 1) || '-03-01')
            END, '9999') <= date(:d))))"""
    n_rows = con.execute(f"select count(*) from fundamental_data where {future}", {"d": iso}).fetchone()[0]
    sets = ", ".join(f'"{c}" = "{c}" * (0.2 + (abs(random()) % 5000) / 1000.0)' for c in cols)
    con.execute(f"update fundamental_data set {sets} where {future}", {"d": iso})
    n_snap = con.execute("update universe_membership set tickers = 'ZZZZ' where snapshot_date > ?", (iso,)).rowcount
    con.commit(); con.close()

    b = copy.copy(bundle)
    b.prices = copy.copy(bundle.prices)
    for side in ("close", "open"):
        f = getattr(bundle.prices, side).copy()
        later = f.index > ts
        f.loc[later] = f.loc[later].to_numpy() * rng.uniform(0.2, 5.2, size=f.loc[later].shape)
        setattr(b.prices, side, f)
    m = bundle.market_frame.copy()
    later = m["date"] > ts
    m.loc[later, "close"] = m.loc[later, "close"] * rng.uniform(0.2, 5.2, size=later.sum())
    b.market_frame = m
    b.macro_series = {k: s.where(s.index <= ts, s * rng.uniform(0.2, 5.2, size=len(s)))
                      for k, s in bundle.macro_series.items()}
    b.store = FundamentalStore(str(db))
    b.universe = TradeableUniverse.from_db_path(str(db), ("sp500", "nasdaq100"))
    return b, n_rows, n_snap, db


def scaled_bundle(bundle):
    b = copy.copy(bundle)
    b.prices = copy.copy(bundle.prices)
    for side in ("close", "open"):
        f = getattr(bundle.prices, side).copy()
        f[list(SCALED)] = f[list(SCALED)] * 10.0
        setattr(b.prices, side, f)
    return b


def lagged_macro_bundle(bundle):
    b = copy.copy(bundle)
    b.macro_series = dict(bundle.macro_series)
    for k in ("DGS10", "BAA10Y"):
        s = bundle.macro_series[k]
        b.macro_series[k] = s.shift(1).dropna()
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", help="directory for the poisoned database copies (default: a temp dir)")
    ap.add_argument("--version", default="v1", help="v1 = committed module; v2_0/v2a/v2b/v2c = valuation_v2.py")
    a = ap.parse_args()
    if a.version != "v1":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import valuation_v2
        valuation_v2.install(a.version)
    tmp = a.tmp or tempfile.mkdtemp(prefix="lookahead_")
    rng = np.random.default_rng(0)
    bundle = DataBundle.load()
    scores = MonthlyScores()
    sessions = bundle.prices.sessions
    scaled, lagged = scaled_bundle(bundle), lagged_macro_bundle(bundle)

    rows, lag_frames = [], []
    for y in range(2017, 2024):
        for mth in (1, 7):
            day = sessions[sessions.searchsorted(pd.Timestamp(y, mth, 1))].date()
            as_of = bundle.prices.last_session_on_or_before(day - timedelta(days=1))
            preds = scores.get(day)
            members = DataView(bundle, as_of).members()
            tickers = top_candidates(scores, day, members, sv.FILTER_DEPTH)
            base, kept = run(bundle, as_of, tickers, preds)

            pb, n_rows, n_snap, db = poisoned_bundle(bundle, as_of, tmp, rng)
            pdet, pkept = run(pb, as_of, tickers, preds)
            universe = sorted(members | set(tickers))
            f0 = DataView(bundle, as_of).fundamentals(universe)
            f1 = DataView(pb, as_of).fundamentals(universe)
            visible_same = f0.reset_index(drop=True).equals(f1.reset_index(drop=True))
            members_same = DataView(pb, as_of).members() == members
            db.unlink()
            try:
                pd.testing.assert_frame_equal(base, pdet, check_exact=True)
                same = True
            except AssertionError:
                same = False

            sdet, skept = run(scaled, as_of, tickers, preds)
            mis_diff = float((base["mispricing"] - sdet["mispricing"]).abs().max())
            ratio = (sdet.set_index("ticker")["fair_value"] / base.set_index("ticker")["fair_value"]).reindex(SCALED).dropna()

            ldet, lkept = run(lagged, as_of, tickers, preds)
            j = base.merge(ldet, on="ticker", suffixes=("", "_lag"))
            j["as_of"] = as_of
            lag_frames.append(j)
            rows.append({
                "as_of": as_of, "candidates": len(preds), "valued": len(base),
                "future_rows_poisoned": n_rows, "future_snapshots_poisoned": n_snap,
                "poison_visible_rows_identical": visible_same, "poison_members_identical": members_same,
                "poison_identical_values": same, "poison_identical_filter": kept == pkept,
                "basis_max_mispricing_diff": mis_diff,
                "basis_fair_ratio": " ".join(f"{t}={v:.6f}" for t, v in ratio.items()),
                "basis_identical_filter": kept == skept,
                "lag_identical_filter": kept == lkept,
                "removed": len(preds) - len(kept),
            })
            print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    lag = pd.concat(lag_frames)
    d = (lag["mispricing_lag"] - lag["mispricing"]).abs()
    dd = (lag["fair_dcf_lag"] - lag["fair_dcf"]).abs() / lag["fair_dcf"]
    here = Path(__file__).resolve().parent / "study"
    here.mkdir(exist_ok=True)
    out.to_csv(here / ("lookahead_test.csv" if a.version == "v1" else f"lookahead_test_{a.version}.csv"), index=False)
    print("\n" + out.drop(columns=["basis_fair_ratio"]).to_string(index=False))
    print(f"\npoisoned future: values identical on {out['poison_identical_values'].sum()} of {len(out)} dates, "
          f"filter identical on {out['poison_identical_filter'].sum()} of {len(out)}")
    print(f"price basis x10: max |mispricing change| {out['basis_max_mispricing_diff'].max():.2e}; "
          f"filter identical on {out['basis_identical_filter'].sum()} of {len(out)}")
    print(f"macro lagged one observation: {len(lag)} ticker-dates; |mispricing change| median {d.median():.4f}, "
          f"p95 {d.quantile(0.95):.4f}, max {d.max():.4f}; DCF value change median {dd.median():.2%}, max {dd.max():.2%}; "
          f"filter identical on {out['lag_identical_filter'].sum()} of {len(out)} dates")


if __name__ == "__main__":
    main()
