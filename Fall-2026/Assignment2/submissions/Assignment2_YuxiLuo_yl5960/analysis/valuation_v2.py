"""Post-test exploration (dev and val only): finer peer groups and a growth-adjusted peer P/E.

NOT the submitted module. The submission is valuation/student_valuation.py at commit 8288c74, which was the
only version run on test. This file changes one of its three methods, the peer multiple, in two ways that can
be switched on separately:

  corr_peers   peers = the N_PEERS index members (on as_of) whose weekly returns over the last CORR_WEEKS
               weeks correlate most with the stock (at least MIN_OVERLAP_WEEKS weeks of overlap), instead of
               the coarse FMP sector (one "Technology" sector holds 127 tickers). Falls back to the sector
               when fewer than MIN_PEERS qualify. Prices up to as_of and members on as_of only.
  growth_adj   PEG-style: each peer's P/E / (100 x g), where g = the module's year-1 revenue growth estimate
               (0.5 x last-year growth + 0.5 x 3-year CAGR) clipped to [G_FLOOR, G_CAP]; the stock's fair P/E =
               median peer PEG x 100 x its own clipped g. Range = peer PEG P25..P75 on the same scale.

The DCF, the EV/EBITDA history, the combination and the filter rule are the committed module's, unchanged.
All constants below were fixed before any v2 run and are not tuned.

    import valuation_v2 as v2; v2.install("v2c")   # patches student_valuation.value_detail in this process
"""
import numpy as np
import pandas as pd

from valuation import student_valuation as sv

N_PEERS = 15
CORR_WEEKS = 52
MIN_OVERLAP_WEEKS = 40
G_FLOOR, G_CAP = 0.05, 0.50

VARIANTS = {  # name: (corr_peers, growth_adj)
    "v2_0": (False, False),   # both off: must reproduce the committed module exactly
    "v2a": (True, False),
    "v2b": (False, True),
    "v2c": (True, True),
}
CONFIG = {"corr_peers": True, "growth_adj": True}
STATS = {"corr_used": 0, "corr_fallback": 0}


def install(variant):
    CONFIG["corr_peers"], CONFIG["growth_adj"] = VARIANTS[variant]
    sv.value_detail = value_detail


def revenue_growth(q):
    """The committed DCF's year-1 revenue growth, before its clip; None if not computable."""
    if q is None or q.empty:
        return None
    now = q["rev_ttm"].iloc[-1]
    if not (np.isfinite(now) and now > 0):
        return None
    one, three = sv._ttm_years_ago(q, 1), sv._ttm_years_ago(q, 3)
    if one is None:
        return None
    yoy = now / one - 1
    if three is None:
        return float(yoy)
    return float(sv.GROWTH_WEIGHT_YOY * yoy + (1 - sv.GROWTH_WEIGHT_YOY) * ((now / three) ** (1 / 3) - 1))


def peer_table(latest, members, prices, by_ticker):
    peers = sv._peer_table(latest, members, prices)            # ticker, gsector, multiple (trailing P/E)
    if CONFIG["growth_adj"]:
        g = peers["ticker"].map(lambda t: revenue_growth(by_ticker.get(t)))
        peers = peers.assign(g=g.astype(float)).dropna(subset=["g"])
        peers["peg"] = peers["multiple"] / (100 * peers["g"].clip(G_FLOOR, G_CAP))
    return peers


def weekly_returns(closes):
    c = closes.ffill(limit=5)
    wk = c.groupby(c.index.to_period("W-FRI")).tail(1)
    return wk.pct_change(fill_method=None).iloc[1:].tail(CORR_WEEKS)


def corr_peer_names(t, wr, eligible):
    if t not in wr.columns:
        return []
    x = wr[t]
    ok = x.notna()
    if ok.sum() < MIN_OVERLAP_WEEKS:
        return []
    cols = [c for c in eligible if c != t and c in wr.columns]
    sub = wr.loc[ok, cols]
    n = sub.notna().sum()
    c = sub.corrwith(x[ok])
    c = c[n >= MIN_OVERLAP_WEEKS].dropna().sort_values(ascending=False, kind="stable")
    return list(c.index[:N_PEERS])


def peer_value(row, t, peers, wr, g_own):
    eps = row["eps_ttm"]
    if not (np.isfinite(eps) and eps > 0):
        return None
    group = None
    if CONFIG["corr_peers"] and wr is not None:
        names = corr_peer_names(t, wr, list(peers["ticker"]))
        if len(names) >= sv.MIN_PEERS:
            group = peers.set_index("ticker").loc[names]
            STATS["corr_used"] += 1
        else:
            STATS["corr_fallback"] += 1
    if group is None:
        group = peers[(peers["gsector"] == row["gsector"]) & (peers["ticker"] != row["ticker"])]
    if len(group) < sv.MIN_PEERS:
        return None
    qs = (sv.PEER_BAND[0], 0.5, sv.PEER_BAND[1])
    if CONFIG["growth_adj"]:
        if g_own is None or not np.isfinite(g_own):
            return None
        scale = 100 * float(np.clip(g_own, G_FLOOR, G_CAP)) * eps
        lo, mid, hi = (float(group["peg"].quantile(p)) * scale for p in qs)
    else:
        lo, mid, hi = (float(group["multiple"].quantile(p)) * eps for p in qs)
    return mid, lo, hi


def value_detail(as_of, tickers, data) -> pd.DataFrame:
    """The committed value_detail with the peer step replaced (see the module docstring)."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()
    members = data.members()
    universe = sorted(members | set(tickers))
    closes = data.close(universe)
    fund = sv._prepare(data.fundamentals(universe), closes)
    latest = fund.groupby("ticker", sort=False).tail(1)
    prices = {k: v for k, v in data.last_close(sorted(set(latest["ticker"]) | set(tickers))).items()
              if v is not None and np.isfinite(v) and v > 0}
    by_ticker = dict(tuple(fund.groupby("ticker", sort=False)))
    peers = peer_table(latest, members, prices, by_ticker)
    px = data.close(tickers + ["SPY"], lookback=sv.HISTORY_SESSIONS)
    dgs10, baa = data.macro("DGS10").dropna(), data.macro("BAA10Y").dropna()
    rf = float(dgs10.iloc[-1]) / 100 if len(dgs10) else None
    spread = float(baa.iloc[-1]) / 100 if len(baa) else None
    wr = weekly_returns(closes.tail(5 * CORR_WEEKS + 15)) if CONFIG["corr_peers"] else None

    rows = []
    for t in tickers:
        price, q = prices.get(t), by_ticker.get(t)
        if price is None or q is None or q.empty:
            continue
        found = {
            "dcf": sv._dcf(q, price, sv._beta(px, t), rf, spread) if rf is not None and spread is not None else None,
            "ev_ebitda": sv._ev_ebitda(q, px[t]),
            "peer_multiple": peer_value(q.iloc[-1], t, peers, wr,
                                        revenue_growth(q) if CONFIG["growth_adj"] else None),
        }
        found = {k: v for k, v in found.items() if v is not None and all(np.isfinite(v)) and v[0] > 0}
        if not found:
            continue
        fairs = [v[0] for v in found.values()]
        fair = float(np.median(fairs))
        low = min(fair, float(np.median([v[1] for v in found.values()])))
        high = max(fair, float(np.median([v[2] for v in found.values()])))
        rec = {"ticker": t, "fair_value": fair, "low": low, "high": high,
               "mispricing": fair / price - 1, "method_spread": max(fairs) / min(fairs),
               "price": price, "n_methods": len(found)}
        for m in sv.METHODS:
            rec[f"fair_{m}"] = found[m][0] if m in found else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
