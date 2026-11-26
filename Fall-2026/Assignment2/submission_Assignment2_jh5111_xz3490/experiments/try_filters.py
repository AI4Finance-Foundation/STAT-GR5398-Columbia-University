"""First-pass experiment: which filter rules are worth building properly?

NOT the submission. Uses rough stand-ins for the three valuation methods:
  dcf   valuation.examples.per_share_dcf (course example, unchanged)
  pe    valuation.examples.trailing_pe_value (course example, sector-median trailing P/E)
  ev    own-history EV/EBITDA band: median of the stock's own past (price + net debt) / TTM EBITDA,
        over up to 20 past filings, times current TTM EBITDA, minus net debt (all per share)
Fair values are cached per (ticker, month) at the first weekly call of the month, so later weeks of
a month use slightly older filings (never newer ones). Mispricing always uses the current price.

Rules (missing valuation -> keep the stock; only growth_tech names are ever removed):
  abs_X   combined = median of available methods; drop if combined mispricing < -X
  rel_k   among the current top 10, drop the k most overvalued (combined mispricing)
  agree_X drop only if >= 2 methods are available and every one says overvalued by more than X

Run from source_code/:  .venv/bin/python experiments/try_filters.py --period val
--period dev (default) or val. Never test: it is run once, at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import run_portfolio as rp
from selection import BUCKET, MonthlyScores
from selection.config import load
from shared.data_view import DataBundle, DataView
from shared.periods import resolve
from valuation import student_valuation
from valuation.examples import _ttm, per_share_dcf, trailing_pe_value

TOP = 20  # how deep into the growth_tech ranking we value each week

# ---------------------------------------------------------------- stand-in valuations

_FV: dict = {}  # (ticker, ym) -> {"dcf": x, "pe": x, "ev": x}


def _ym(as_of):
    ts = pd.Timestamp(as_of)
    return ts.year, ts.month


def ev_band_value(ticker, data, min_quarters=8, max_quarters=20):
    f = data.fundamentals([ticker]).sort_values("datadate").reset_index(drop=True)
    if len(f) < 4 + min_quarters - 1:
        return None
    f["ebitda_ps"] = f["ebitda_margin"] * f["revenue_per_share"]
    f["ebitda_ttm"] = f["ebitda_ps"].rolling(4).sum()
    f["net_debt"] = f["interest_debt_per_share"] - f["cash_per_share"]
    px = data.close([ticker], lookback=252 * 6)[ticker].dropna()
    if px.empty:
        return None
    f["px"] = [px.asof(pd.Timestamp(d)) if pd.Timestamp(d) >= px.index[0] else np.nan
               for d in pd.to_datetime(f["accepted_date"])]
    f["mult"] = (f["px"] + f["net_debt"]) / f["ebitda_ttm"]
    hist = f.iloc[:-1].dropna(subset=["mult"])  # past filings only, not the latest
    hist = hist[(hist["ebitda_ttm"] > 0) & hist["mult"].between(0, 200)].tail(max_quarters)
    last = f.iloc[-1]
    if len(hist) < min_quarters or not (last["ebitda_ttm"] > 0):
        return None
    fv = hist["mult"].median() * last["ebitda_ttm"] - last["net_debt"]
    return fv if fv > 0 else None


def fair_values(as_of, tickers, data):
    ym = _ym(as_of)
    need = [t for t in tickers if (t, ym) not in _FV]
    if need:
        pe =trailing_pe_value(as_of, need, data).set_index("ticker")["fair_value"]
        for t in need:
            try:
                d = per_share_dcf(as_of, t, data)
            except ValueError:  # e.g. NaN net debt fails DCFInputs validation
                d = None
            dcf = d["fair_value"] if d and d["fair_value"] is not None and d["fair_value"] > 0 else None
            p = pe.get(t)
            _FV[(t, ym)] = {"dcf": dcf, "pe": p if p is not None and p > 0 else None,
                            "ev": ev_band_value(t, data)}
    return {t: _FV[(t, ym)] for t in tickers}


def mispricing_table(as_of, tickers, data):
    fv = fair_values(as_of, tickers, data)
    px = data.last_close(tickers)
    rows = []
    for t in tickers:
        p = px.get(t)
        if p is None or not np.isfinite(p):
            continue
        m = {k: (v / p - 1 if v else np.nan) for k, v in fv[t].items()}
        avail = [v for v in fv[t].values() if v]
        m["combined"] = np.median(avail) / p - 1 if avail else np.nan
        m["n"] = len(avail)
        rows.append({"ticker": t, **m})
    return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------- filter rules

def ranked_growth(candidates, data):
    members = data.members()
    g = [c for c in candidates if c.bucket == BUCKET and c.symbol in members]
    return [c.symbol for c in sorted(g, key=lambda c: c.rank_score, reverse=True)][:TOP]


def make_filter(rule, x):
    def filt(as_of, candidates, data):
        ranked = ranked_growth(candidates, data)
        if not ranked:
            return candidates
        m = mispricing_table(as_of, ranked, data)
        drop = set()
        if rule == "abs":
            drop = set(m.index[m["combined"] < -x])
        elif rule == "rel":
            top10 = m.reindex(ranked[:10])["combined"].dropna().sort_values()
            drop = set(top10.index[: int(x)])
        elif rule == "agree":
            cols = m[["dcf", "pe", "ev"]]
            ok = (m["n"] >= 2) & cols.apply(lambda r: r.dropna().lt(-x).all(), axis=1)
            drop = set(m.index[ok])
        return tuple(c for c in candidates if c.symbol not in drop)
    return filt


RULES = [("abs", 0.3), ("abs", 0.5), ("abs", 0.7), ("rel", 1), ("rel", 2), ("agree", 0.3), ("agree", 0.5)]


# ---------------------------------------------------------------- method diagnostics (IC, how expensive)

def method_diagnostics(bundle, scores, start, end):
    prices = bundle.prices
    sessions = prices.sessions
    window = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    month_starts = window.to_series().groupby([window.year, window.month]).first()
    recs = []
    for day in month_starts:
        as_of = prices.last_session_on_or_before(day.date() - pd.Timedelta(days=1))
        view = DataView(bundle, as_of)
        ranked = ranked_growth(scores.get(day.date()), view)
        m = mispricing_table(as_of, ranked, view)
        pos = sessions.searchsorted(pd.Timestamp(as_of))
        for t, r in m.iterrows():
            p0 = prices.close.at[sessions[pos], t]
            rec = {"as_of": as_of, "ticker": t, "top10": t in ranked[:10], **r.to_dict()}
            for h in (63, 126):
                if pos + h < len(sessions):
                    later = prices.close[t].iloc[pos + 1: pos + h + 1].dropna()
                    rec[f"ret_{h}"] = later.iloc[-1] / p0 - 1 if len(later) else np.nan
            recs.append(rec)
    res = pd.DataFrame(recs)
    print("\nStand-in methods on the top-20 growth_tech candidates, first session of each month")
    rows = []
    for k in ("dcf", "pe", "ev", "combined"):
        r = {"method": k, "coverage": res[k].notna().mean(),
             "median_mispricing_top10": res.loc[res["top10"], k].median(),
             "share_top10_over_50pct_overvalued": (res.loc[res["top10"], k] < -0.5).mean()}
        for h in (63, 126):
            ic = (res.dropna(subset=[k, f"ret_{h}"]).groupby("as_of")
                  .apply(lambda g: g[k].corr(g[f"ret_{h}"], method="spearman") if len(g) >= 5 else np.nan)
                  .dropna())
            r[f"IC_{h}"] = ic.mean()
            r[f"t_{h}"] = ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 1 and ic.std() > 0 else np.nan
        rows.append(r)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).set_index("method").to_string())
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test: it is run once, at the end
    period = ap.parse_args().period
    start, end = resolve(period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    bundle = DataBundle.load(prices=prices)
    scores = MonthlyScores()
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": scores, "bundle": bundle}
    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)

    diag = method_diagnostics(bundle, scores, start, end)
    diag.to_csv(out / f"method_diagnostics_{period}.csv", index=False)

    rows = [rp.run_one(f"exp_{period}_A_selection", start, end, valuation=False, timing="none", stops=False, ctx=ctx)]
    original = student_valuation.filter_candidates
    try:
        for rule, x in RULES:
            name = f"exp_{period}_{rule}_{x}"
            print(f"running {name}", flush=True)
            student_valuation.filter_candidates = make_filter(rule, x)
            rows.append(rp.run_one(name, start, end, valuation=True, timing="none", stops=False, ctx=ctx))
    finally:
        student_valuation.filter_candidates = original
    keep = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
            "turnover_per_year", "avg_removed_per_week"]
    table = pd.DataFrame(rows).set_index("run")[keep]
    table.to_csv(out / f"filter_rules_{period}.csv")
    with pd.option_context("display.width", 200):
        print("\n" + table.to_string())


if __name__ == "__main__":
    main()
