"""Point-in-time valuation: peer P/E, own EV/EBITDA history and FinRobot DCF.

Preselected assumptions: ERP 5%, terminal growth 2.5%, debt spread 2%,
zero working-capital change (the course FinRobot example).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.examples import per_share_dcf

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]
MIN_PEERS = 3
MAX_SPREAD = 3.0
SELL_FRACTION = 0.70


def _history(f, prices, ticker, ebitda, net_debt):
    """Historical EV/EBITDA using only prices as of each filing's availability."""
    f = f.sort_values("datadate").drop_duplicates("datadate", keep="last").copy()
    if len(f) < 8 or ticker not in prices:
        return None
    known = pd.to_datetime(f["accepted_date"], errors="coerce")
    known = known.fillna(pd.to_datetime(f["filing_date"], errors="coerce"))
    known = known.fillna(pd.to_datetime(f["datadate"]) + pd.Timedelta(days=60))
    quarterly = pd.to_numeric(f["revenue_per_share"], errors="coerce") * pd.to_numeric(f["ebitda_margin"], errors="coerce")
    trailing = quarterly.rolling(4, min_periods=4).sum()
    past = prices[ticker].dropna()
    multiples = []
    for i in range(3, len(f)):
        if not np.isfinite(trailing.iloc[i]) or trailing.iloc[i] <= 0:
            continue
        px = past.loc[:known.iloc[i]]
        debt = pd.to_numeric(f["interest_debt_per_share"].iloc[i], errors="coerce")
        cash = pd.to_numeric(f["cash_per_share"].iloc[i], errors="coerce")
        if px.empty or not np.isfinite(debt) or not np.isfinite(cash):
            continue
        m = (float(px.iloc[-1]) + debt - cash) / trailing.iloc[i]
        if np.isfinite(m) and 1 < m < 100:
            multiples.append(m)
    if len(multiples) < 5 or not np.isfinite(net_debt):
        return None
    band = np.quantile(multiples, [0.25, 0.5, 0.75]) * ebitda - net_debt
    return band if np.isfinite(band).all() and min(band) > 0 else None


def _calculate(as_of, tickers, data):
    names = tuple(dict.fromkeys(tickers))
    if not names:
        return {}
    members = data.members()
    universe = sorted(members | set(names))
    filings = data.fundamentals(universe)
    if filings.empty:
        return {}
    rows = []
    for symbol, f in filings.groupby("ticker", sort=True):
        q = f.sort_values("datadate").drop_duplicates("datadate", keep="last").tail(4)
        if len(q) != 4:
            continue
        eps = pd.to_numeric(q["EPS"], errors="coerce")
        rev = pd.to_numeric(q["revenue_per_share"], errors="coerce")
        margin = pd.to_numeric(q["ebitda_margin"], errors="coerce")
        debt = pd.to_numeric(q["interest_debt_per_share"].iloc[-1], errors="coerce")
        cash = pd.to_numeric(q["cash_per_share"].iloc[-1], errors="coerce")
        rows.append(dict(ticker=symbol, sector=q["gsector"].iloc[-1],
                         eps=eps.sum() if eps.notna().all() else np.nan,
                         ebitda=(rev * margin).sum() if rev.notna().all() and margin.notna().all() else np.nan,
                         net_debt=debt - cash))
    if not rows:
        return {}
    frame = pd.DataFrame(rows).set_index("ticker")
    frame["price"] = pd.Series(data.last_close(universe))
    frame["trailing_multiple"] = frame["price"] / frame["eps"]
    peers = frame.loc[frame.index.isin(members) & frame["eps"].gt(0) & frame["trailing_multiple"].between(1, 120)]
    bands = {s: g["trailing_multiple"].quantile([0.25, 0.5, 0.75]).to_numpy()
             for s, g in peers.groupby("sector") if len(g) >= MIN_PEERS}
    history_prices = data.close(names, lookback=1260)
    result = {}
    for name in names:
        if name not in frame.index:
            continue
        r = frame.loc[name]
        price = r["price"]
        if not np.isfinite(price) or price <= 0:
            continue
        methods = {}
        if r["eps"] > 0 and r["sector"] in bands:
            methods["peer_earnings"] = bands[r["sector"]] * r["eps"]
        if np.isfinite(r["ebitda"]) and r["ebitda"] > 0:
            own = _history(filings.loc[filings["ticker"] == name], history_prices,
                           name, r["ebitda"], r["net_debt"])
            if own is not None:
                methods["ev"] = own
        # Missing debt/cash makes FinRobot's net_debt invalid. Other missing
        # accounting inputs may also be rejected by its validated input model.
        dcf = None
        if np.isfinite(r["net_debt"]):
            try:
                dcf = per_share_dcf(as_of, name, data)
            except (ValueError, TypeError, ArithmeticError):
                pass
        if dcf is not None:
            methods["dcf"] = [dcf["low"], dcf["fair_value"], dcf["high"]]
        methods = {k: np.asarray(v, dtype=float) for k, v in methods.items()
                   if np.asarray(v).shape == (3,) and np.isfinite(v).all() and min(v) > 0}
        if len(methods) < 2:
            continue
        mids = np.array([v[1] for v in methods.values()])
        fair = float(np.median(mids))
        result[name] = (
            dict(ticker=name, fair_value=fair,
                 low=min(float(min(v[0] for v in methods.values())), fair),
                 high=max(float(max(v[2] for v in methods.values())), fair),
                 mispricing=fair / price - 1,
                 method_spread=float(max(mids) / min(mids))),
            methods, float(price))
    return result


def value(as_of, tickers, data) -> pd.DataFrame:
    computed = _calculate(as_of, tickers, data)
    return pd.DataFrame([computed[t][0] for t in dict.fromkeys(tickers) if t in computed],
                        columns=COLUMNS)


def filter_candidates(as_of, candidates, data):
    ranked = sorted((c for c in candidates if c.bucket == "growth_tech"),
                    key=lambda c: c.rank_score, reverse=True)
    computed = _calculate(as_of, [c.symbol for c in ranked[:13]], data)
    removed = set()
    for c in ranked[:10]:
        if c.symbol not in computed:
            continue
        summary, methods, price = computed[c.symbol]
        if (summary["method_spread"] <= MAX_SPREAD
                and sum(v[1] < SELL_FRACTION * price for v in methods.values()) >= 2):
            removed.add(c.symbol)
            if len(removed) >= 3:
                break
    return tuple(c for c in candidates if c.symbol not in removed)
