"""Evaluate a valuation module on the stocks the selection model actually considers.

    python evaluate_valuation.py --period dev
    python evaluate_valuation.py --period dev --example       the trailing-P/E example in valuation/examples.py

On the first session of each month it values the top --top growth_tech
candidates (as of the previous close) and reports
  * rank IC: Spearman correlation of mispricing with the next 63 / 126 / 252
    session returns, averaged over months, with a t-statistic
  * range coverage: share of valued stocks whose price 252 sessions later is
    inside [low, high]
  * valuation coverage: share of candidates you returned a value for
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from selection import BUCKET, MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.periods import record_test_run, resolve

HORIZONS = (63, 126, 252)
REQUIRED = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]


def top_candidates(scores, day, members, n):
    cands = [c for c in scores.get(day) if c.bucket == BUCKET and c.symbol in members]
    return [c.symbol for c in sorted(cands, key=lambda c: c.rank_score, reverse=True)[:n]]


def check_frame(df, as_of, tickers, data):
    df = pd.DataFrame(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"value({as_of}) is missing columns {missing}")
    df = df[REQUIRED].dropna(subset=["fair_value", "mispricing"]).copy()
    extra = set(df["ticker"]) - set(tickers)
    if extra:
        raise ValueError(f"value({as_of}) returned tickers it was not asked for: {sorted(extra)[:5]}")
    bad = df[(df["low"] > df["fair_value"] + 1e-9) | (df["high"] < df["fair_value"] - 1e-9)]
    if len(bad):
        raise ValueError(f"value({as_of}): low <= fair_value <= high fails for {bad['ticker'].tolist()[:5]}")
    px = pd.Series(data.last_close(df["ticker"]))
    idx = df.set_index("ticker")
    off = ((idx["fair_value"] / px - 1) - idx["mispricing"]).abs() > 0.01
    if off.any():
        raise ValueError(f"value({as_of}): mispricing is not fair_value / price - 1 for {off[off].index.tolist()[:5]}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["dev", "val", "test", "forward"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--example", action="store_true")
    a = ap.parse_args(argv)
    start, end = resolve(a.period, a.start, a.end)
    bundle = DataBundle.load()
    prices = bundle.prices
    end = min(end, prices.sessions[-1].date())
    record_test_run("evaluate_valuation.py", sys.argv[1:], start, end)
    if a.example:
        from valuation.examples import trailing_pe_value as value
    else:
        from valuation.student_valuation import value
    scores = MonthlyScores()
    window = prices.sessions[(prices.sessions >= pd.Timestamp(start)) & (prices.sessions <= pd.Timestamp(end))]
    month_starts = window.to_series().groupby([window.year, window.month]).first()
    sessions = prices.sessions
    records, asked = [], 0
    for day in month_starts:
        as_of = prices.last_session_on_or_before(day.date() - timedelta(days=1))
        view = DataView(bundle, as_of)
        tickers = top_candidates(scores, day.date(), view.members(), a.top)
        asked += len(tickers)
        with student_call():
            df = value(as_of, tickers, view)
        df = check_frame(df, as_of, tickers, view)
        pos = sessions.searchsorted(pd.Timestamp(as_of))
        for r in df.itertuples(index=False):
            p0 = prices.close.at[sessions[pos], r.ticker] if r.ticker in prices.close.columns else np.nan
            rec = r._asdict() | {"as_of": as_of, "price": p0}
            for h in HORIZONS:
                if pos + h < len(sessions):
                    later = prices.close[r.ticker].iloc[pos + 1: pos + h + 1].dropna()
                    rec[f"ret_{h}"] = later.iloc[-1] / p0 - 1 if len(later) else np.nan
                    if h == 252:
                        rec["price_252"] = later.iloc[-1] if len(later) else np.nan
            records.append(rec)
    res = pd.DataFrame(records)
    label = a.period or f"{start}_{end}"
    out = Path("output"); out.mkdir(exist_ok=True)
    res.to_csv(out / f"valuation_eval_{label}.csv", index=False)
    print(f"\nValuation, {start} -> {end}, top {a.top} {BUCKET} candidates each month")
    print(f"valued {len(res)} of {asked} candidate-months ({len(res) / max(asked, 1):.0%})")
    if res.empty:
        return
    for h in HORIZONS:
        col = f"ret_{h}"
        if col not in res:
            continue
        ic = (res.dropna(subset=[col]).groupby("as_of")
              .apply(lambda g: g["mispricing"].corr(g[col], method="spearman") if len(g) >= 5 else np.nan)
              .dropna())
        if len(ic) > 1:
            t = ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan
            print(f"rank IC {h:>3} sessions: mean {ic.mean():+.3f}   t {t:+.2f}   months {len(ic)}")
    if "price_252" in res:
        r = res.dropna(subset=["price_252", "low", "high"])
        if len(r):
            inside = ((r["price_252"] >= r["low"]) & (r["price_252"] <= r["high"])).mean()
            print(f"price 252 sessions later inside [low, high]: {inside:.0%} of {len(r)}")
    print(f"median mispricing {res['mispricing'].median():+.1%}   median method_spread {res['method_spread'].median():.2f}")


if __name__ == "__main__":
    main()
