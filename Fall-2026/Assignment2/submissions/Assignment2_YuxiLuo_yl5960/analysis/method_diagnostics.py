"""Per-method diagnostics for valuation/student_valuation.py, on the same sample as
evaluate_valuation.py (top 20 growth_tech candidates on the first session of each month).

Run from source_code/:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/method_diagnostics.py --period dev [--set NAME=VALUE ...]

Reports, for the combined value and for each method alone: coverage, rank IC of
mispricing against the next 63/126/252 session returns (mean, t), and the share of
prices inside [low, high] 252 sessions later (combined only). Never use --period test.
"""
import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from evaluate_valuation import HORIZONS, top_candidates
from selection import MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.periods import resolve
from valuation import student_valuation as sv


def ic_table(res, col):
    out = {}
    for h in HORIZONS:
        r = res.dropna(subset=[col, f"ret_{h}"])
        ic = r.groupby("as_of").apply(
            lambda g: g[col].corr(g[f"ret_{h}"], method="spearman") if len(g) >= 5 else np.nan,
            include_groups=False).dropna()
        if len(ic) > 1 and ic.std() > 0:
            out[f"IC{h}"] = round(ic.mean(), 3)
            out[f"t{h}"] = round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], default="dev")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--set", action="append", default=[], help="override a module constant, e.g. ALIGN_BASIS=False")
    ap.add_argument("--save")
    ap.add_argument("--json", help="write the summary as JSON here")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(sv, k, eval(v))
        print(f"set {k} = {getattr(sv, k)!r}")
    start, end = resolve(a.period, a.start, a.end)
    bundle = DataBundle.load()
    prices, sessions = bundle.prices, bundle.prices.sessions
    scores = MonthlyScores()
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    month_starts = window.to_series().groupby([window.year, window.month]).first()
    recs, asked, worst = [], 0, 0.0
    for day in month_starts:
        as_of = prices.last_session_on_or_before(day.date() - timedelta(days=1))
        view = DataView(bundle, as_of)
        tickers = top_candidates(scores, day.date(), view.members(), a.top)
        asked += len(tickers)
        t0 = time.perf_counter()
        with student_call():
            det = sv.value_detail(as_of, tickers, view)
        worst = max(worst, time.perf_counter() - t0)
        pos = sessions.searchsorted(pd.Timestamp(as_of))
        for r in det.to_dict("records"):
            p0 = prices.close.at[sessions[pos], r["ticker"]]
            r["as_of"] = as_of
            for h in HORIZONS:
                if pos + h < len(sessions):
                    later = prices.close[r["ticker"]].iloc[pos + 1: pos + h + 1].dropna()
                    r[f"ret_{h}"] = later.iloc[-1] / p0 - 1 if len(later) else np.nan
                    if h == 252:
                        r["price_252"] = later.iloc[-1] if len(later) else np.nan
            recs.append(r)
    res = pd.DataFrame(recs)
    for m in sv.METHODS:
        res[f"mis_{m}"] = res[f"fair_{m}"] / res["price"] - 1
    rows = {"combined": {"coverage": round(len(res) / asked, 3), **ic_table(res, "mispricing")}}
    for m in sv.METHODS:
        rows[m] = {"coverage": round(res[f"fair_{m}"].notna().sum() / asked, 3), **ic_table(res, f"mis_{m}")}
    print(f"\n{start} -> {end}: {len(month_starts)} months, {asked} candidate-months, slowest value_detail {worst:.2f}s")
    print(pd.DataFrame(rows).T.to_string())
    r = res.dropna(subset=["price_252"])
    inside = ((r["price_252"] >= r["low"]) & (r["price_252"] <= r["high"])).mean()
    print(f"price 252 sessions later inside [low, high]: {inside:.0%} of {len(r)};  "
          f"median (high/low - 1) {((res['high'] / res['low']) - 1).median():.0%}")
    print(f"median mispricing {res['mispricing'].median():+.1%}, median method_spread {res['method_spread'].median():.2f}, "
          f"share n_methods>=2 {(res['n_methods'] >= 2).mean():.0%}")
    flag = sv.overvalued(res).fillna(False)
    f126 = res.dropna(subset=["ret_126"])
    fl = flag.loc[f126.index]
    summary = {
        "period": a.period, "overrides": {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in a.set},
        "months": len(month_starts), "asked": asked, "rows": rows,
        "range_inside": round(float(inside), 3), "range_n": len(r),
        "median_width": round(float(((res["high"] / res["low"]) - 1).median()), 3),
        "median_mispricing": round(float(res["mispricing"].median()), 3),
        "median_spread": round(float(res["method_spread"].median()), 3),
        "share_2plus": round(float((res["n_methods"] >= 2).mean()), 3),
        "flag_share": round(float(flag.sum() / asked), 3),
        "flagged_ret126": round(float(f126.loc[fl, "ret_126"].mean()), 4),
        "rest_ret126": round(float(f126.loc[~fl, "ret_126"].mean()), 4),
        "slowest_s": round(worst, 2),
    }
    print(f"flagged (price above every method) share {summary['flag_share']:.1%}; mean next-126 return "
          f"flagged {summary['flagged_ret126']:+.1%} vs rest {summary['rest_ret126']:+.1%}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))
    if a.save:
        res.to_csv(a.save, index=False)


if __name__ == "__main__":
    main()
