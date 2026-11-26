"""Literature-based variants of the chosen configuration (pre-registered in the team doc on 11/21).

Each variant changes one thing; everything else is the chosen worst_k (k=2, depth 15).
    PYTHONHASHSEED=0 .venv/bin/python experiments/variants.py --period dev
    PYTHONHASHSEED=0 .venv/bin/python experiments/variants.py --period val --only V4 F1   dev survivors, once
    PYTHONHASHSEED=0 .venv/bin/python experiments/variants.py --period dev --eval        + valuation metrics (V*)

Valuation variants (value() changes, filter rule does not):
  V4  peer P/E replaced by a cross-sectional regression of log(EV/Sales) on growth, margin, sector
      (Bhojraj & Lee 2002; Bartram & Grinblatt 2018)
  V5  peer P/E centred on the harmonic mean instead of the median (Liu, Nissim & Thomas 2002)
  V6  DCF growth fades from year 2 (HOLD_YEARS = 1) (Chan, Karceski & Lakonishok 2003)
Filter variants (value() unchanged; only which top-10 names may be dropped):
  F1  only names whose 63-session return is below the top 10's median (Asness, Moskowitz & Pedersen 2013)
  F2  only names whose TTM EBITDA margin is below a year earlier (Novy-Marx 2013)
  F3  filter only when DGS10 is above its level 182 days earlier (Dechow, Sloan & Soliman 2004)
Writes output/experiments/variants_<period>.csv (and variants_eval_dev.csv with --eval).
Saved results were run before the 11/23 merge of the teammate's EV/EBITDA and peer P/E.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import evaluate_valuation
import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

KEEP = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
        "turnover_per_year", "avg_removed_per_week"]
REG_MIN_OBS = 30
REG_GROWTH_CLIP, REG_MARGIN_CLIP = (-0.5, 1.0), (-0.5, 0.8)
MOM_SESSIONS = 63
RATE_LOOKBACK_DAYS = 182
PE_RANGE = (0.0, 200.0)


def _row(t, fair=np.nan, low=np.nan, high=np.nan, reason=""):
    return {"ticker": t, "fair_value": fair, "low": low, "high": high, "missing": bool(reason), "reason": reason}


def _usable_latest(f, as_of, price):
    """Reason the latest filing cannot be used, or ""."""
    if len(f) < 4 or price is None or not np.isfinite(price):
        return "insufficient_data"
    if not sv.basis_ok(f.iloc[-1], price):
        return "basis_mismatch"
    if (as_of - pd.Timestamp(f["datadate"].iloc[-1])).days > sv.MAX_FILING_AGE_DAYS:
        return "insufficient_data"
    if not sv.contiguous_quarters(pd.to_datetime(f["datadate"]), 4):
        return "insufficient_data"
    return ""


# ---------------------------------------------------------------- V4: regression peer value
def _reg_features(f, as_of, price):
    """(features dict, "") for one ticker's sorted filings, or (None, reason)."""
    reason = _usable_latest(f, as_of, price)
    if reason:
        return None, reason
    if len(f) < 8 or not sv.contiguous_quarters(pd.to_datetime(f["datadate"]), 8):
        return None, "insufficient_data"
    rev = f["revenue_per_share"].to_numpy(dtype=float)
    ebitda = (f["ebitda_margin"] * f["revenue_per_share"]).to_numpy(dtype=float)
    sales, prev = rev[-4:].sum(), rev[-8:-4].sum()
    net_debt = float(f["interest_debt_per_share"].iloc[-1] - f["cash_per_share"].iloc[-1])
    if not (sales > 0 and prev > 0 and np.isfinite(net_debt)):
        return None, "insufficient_data"
    ev = price + net_debt
    if not ev > 0:
        return None, "negative_value"
    x = {"y": np.log(ev / sales), "sales": sales, "net_debt": net_debt, "sector": f["gsector"].iloc[-1],
         "growth": float(np.clip(sales / prev - 1, *REG_GROWTH_CLIP)),
         "margin": float(np.clip(ebitda[-4:].sum() / sales, *REG_MARGIN_CLIP))}
    if not all(np.isfinite(x[k]) for k in ("y", "growth", "margin")) or pd.isna(x["sector"]):
        return None, "insufficient_data"
    return x, ""


def peer_reg_value(as_of, tickers, data) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return sv._empty_methods()
    as_of = pd.Timestamp(as_of)
    members = data.members()
    universe = sorted(members | set(tickers))
    prices = data.last_close(universe)
    fund = data.fundamentals(universe).sort_values(["ticker", "datadate"])
    feats, reasons = {}, {}
    for t, f in fund.groupby("ticker"):
        x, why = _reg_features(f, as_of, prices.get(t))
        if x is None:
            reasons[t] = why
        else:
            feats[t] = x
    table = pd.DataFrame.from_dict(feats, orient="index")
    rows = []
    if table.empty:
        return pd.DataFrame([_row(t, reason=reasons.get(t, "insufficient_data")) for t in tickers],
                            columns=sv.METHOD_COLUMNS)
    dummies = pd.get_dummies(table["sector"].astype(str), prefix="s", drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=table.index, name="const"), table[["growth", "margin"]], dummies], axis=1)
    in_fit = table.index.isin(members)
    for t in tickers:
        if t not in table.index:
            rows.append(_row(t, reason=reasons.get(t, "insufficient_data")))
            continue
        fit = in_fit & (table.index != t)
        if fit.sum() < REG_MIN_OBS:
            rows.append(_row(t, reason="insufficient_data"))
            continue
        beta, *_ = np.linalg.lstsq(X[fit].to_numpy(), table.loc[fit, "y"].to_numpy(), rcond=None)
        resid = table.loc[fit, "y"].to_numpy() - X[fit].to_numpy() @ beta
        pred = float(X.loc[t].to_numpy() @ beta)
        me = table.loc[t]
        lo_q, hi_q = np.quantile(resid, [sv.PE_BAND_Q[0], sv.PE_BAND_Q[1]])
        fair, lo, hi = (np.exp(pred + q) * me["sales"] - me["net_debt"] for q in (0.0, lo_q, hi_q))
        if not fair > 0:
            rows.append(_row(t, reason="negative_value"))
            continue
        rows.append(_row(t, fair, max(lo, 0.0), hi))
    return pd.DataFrame(rows, columns=sv.METHOD_COLUMNS)


# ---------------------------------------------------------------- V5: harmonic-mean peer P/E
def peer_pe_harmonic(as_of, tickers, data) -> pd.DataFrame:
    """sv.peer_pe_value with the centre moved from the peer median to the peer harmonic mean."""
    base = sv.peer_pe_value(as_of, tickers, data)
    ok = ~base["missing"].astype(bool)
    if not ok.any():
        return base
    members = data.members()
    universe = sorted(members | set(tickers))
    prices = data.last_close(universe)
    fund = data.fundamentals(universe).sort_values(["ticker", "datadate"])
    info = {}
    for t, f in fund.groupby("ticker"):
        if _usable_latest(f, pd.Timestamp(as_of), prices.get(t)):
            continue
        eps = float(f["EPS"].tail(4).sum())
        if eps > 0:
            info[t] = {"eps": eps, "sector": f["gsector"].iloc[-1], "pe": prices[t] / eps}
    table = pd.DataFrame.from_dict(info, orient="index")
    peers = table[table.index.isin(members) & table["pe"].between(*PE_RANGE, inclusive="neither")]
    for i in base.index[ok]:
        t = base.at[i, "ticker"]
        group = peers[(peers["sector"] == table.at[t, "sector"]) & (peers.index != t)]["pe"]
        fair = len(group) / (1.0 / group).sum() * table.at[t, "eps"]
        base.at[i, "fair_value"] = fair
        base.at[i, "low"] = min(base.at[i, "low"], fair)
        base.at[i, "high"] = max(base.at[i, "high"], fair)
    return base


# ---------------------------------------------------------------- F1-F3: who may be dropped
def _eligible_momentum(as_of, top, data):
    px = data.close(top, lookback=MOM_SESSIONS + 1)
    ret = px.iloc[-1] / px.iloc[0] - 1
    return set(ret.index[ret < ret.median()])


def _eligible_quality(as_of, top, data):
    fund = data.fundamentals(top).sort_values(["ticker", "datadate"])
    out = set()
    for t, f in fund.groupby("ticker"):
        if len(f) < 8 or not sv.contiguous_quarters(pd.to_datetime(f["datadate"]), 8):
            continue
        rev = f["revenue_per_share"].to_numpy(dtype=float)
        ebitda = (f["ebitda_margin"] * f["revenue_per_share"]).to_numpy(dtype=float)
        now, before = ebitda[-4:].sum() / rev[-4:].sum(), ebitda[-8:-4].sum() / rev[-8:-4].sum()
        if np.isfinite(now) and np.isfinite(before) and now < before:
            out.add(t)
    return out


def _eligible_rates(as_of, top, data):
    r = data.macro("DGS10").dropna()
    past = r.loc[: pd.Timestamp(as_of) - pd.Timedelta(days=RATE_LOOKBACK_DAYS)]
    rising = len(past) and r.iloc[-1] > past.iloc[-1]
    return set(top) if rising else set()


def make_filter(eligible):
    def filter_candidates(as_of, candidates, data):
        as_of = pd.Timestamp(as_of)
        members = data.members()
        growth = sorted((c for c in candidates if c.bucket == sv.BUCKET and c.symbol in members),
                        key=lambda c: c.rank_score, reverse=True)
        ranked = [c.symbol for c in growth[:sv.FILTER_DEPTH]]
        if not ranked:
            return candidates
        m = sv._monthly_values(as_of, ranked, data)
        top = m["mispricing"].reindex(ranked[:10]).dropna()
        ok = eligible(as_of, ranked[:10], data)
        top = top[(top < 0) & top.index.isin(ok)]
        drop = set(top.nsmallest(sv.DROP_WORST_K).index)
        return tuple(c for c in candidates if c.symbol not in drop)
    return filter_candidates


# ---------------------------------------------------------------- runner
def variant_patches(name):
    """dict of student_valuation attributes to set for this variant."""
    methods = dict(sv.VALUE_METHODS)
    if name == "V4":
        methods.pop("peer_pe")
        methods["peer_reg"] = peer_reg_value
        return {"VALUE_METHODS": tuple(methods.items())}
    if name == "V5":
        methods["peer_pe"] = peer_pe_harmonic
        return {"VALUE_METHODS": tuple(methods.items())}
    if name == "V6":
        return {"HOLD_YEARS": 1}
    if name in ("F1", "F2", "F3"):
        fn = {"F1": _eligible_momentum, "F2": _eligible_quality, "F3": _eligible_rates}[name]
        return {"filter_candidates": make_filter(fn)}
    if name == "current":
        return {}
    raise ValueError(name)


@contextlib.contextmanager
def patched(name):
    patches = variant_patches(name)
    saved = {k: getattr(sv, k) for k in patches} | {"FILTER_RULE": sv.FILTER_RULE}
    try:
        sv.FILTER_RULE = "worst_k"
        for k, v in patches.items():
            setattr(sv, k, v)
        sv._VALUE_CACHE.clear()
        yield
    finally:
        for k, v in saved.items():
            setattr(sv, k, v)
        sv._VALUE_CACHE.clear()


def eval_metrics(name, period):
    """evaluate_valuation.py on the variant's value(); keeps value()'s own output file."""
    ours = Path("output") / f"valuation_eval_{period}.csv"
    keep = ours.with_suffix(".keep")
    shutil.copy(ours, keep)
    try:
        buf = io.StringIO()
        with patched(name), contextlib.redirect_stdout(buf):
            evaluate_valuation.main(["--period", period])
        txt = buf.getvalue()
    finally:
        shutil.move(keep, ours)
    row = {"variant": name}
    m = re.search(r"\((\d+)%\)", txt)
    row["coverage"] = int(m.group(1)) / 100 if m else None
    for h in (63, 126, 252):
        m = re.search(rf"rank IC\s+{h} sessions: mean ([-\d.]+)\s+t ([-\d.]+)", txt)
        row[f"ic{h}"], row[f"t{h}"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"inside \[low, high\]: (\d+)%", txt)
    row["range_cover"] = int(m.group(1)) / 100 if m else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="dev", choices=["dev", "val"])  # never test
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    names = a.only or ["current", "V4", "V5", "V6", "F1", "F2", "F3"]
    start, end = resolve(a.period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}
    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    rows, evals = [], []
    for name in names:
        print(f"running {name}", flush=True)
        with patched(name):
            s = rp.run_one(f"variant_{a.period}_{name}", start, end, valuation=True, timing="none",
                           stops=False, ctx=ctx)
        rows.append({"variant": name, **{c: s.get(c) for c in KEEP}})
        print(rows[-1], flush=True)
        if a.eval and name in ("current", "V4", "V5", "V6"):
            evals.append(eval_metrics(name, a.period))
            print(evals[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / f"variants_{a.period}.csv", index=False)
    print("\n" + t.to_string(index=False))
    if evals:
        e = pd.DataFrame(evals)
        e.to_csv(out / f"variants_eval_{a.period}.csv", index=False)
        print("\n" + e.to_string(index=False))


if __name__ == "__main__":
    main()
