"""Valuation module: is a stock overpriced or underpriced?

Three point-in-time methods, all per share (fundamentals.db has per-share
columns but no totals), combined into one fair value per ticker:

  1. DCF        FinRobot's calculate_dcf / calculate_sensitivity on inputs built
                from the last four quarters. Risk-free rate = DGS10 on as_of,
                cost of debt = DGS10 + BAA10Y on as_of, revenue growth from the
                ticker's own revenue history (no analyst consensus), range from
                the WACC x terminal-growth sensitivity grid.
  2. EV/EBITDA  the ticker's own history: weekly EV / TTM EBITDA over the last
                five years, each week paired with the filings that were public
                that week. Fair value = median multiple x current TTM EBITDA,
                minus current net debt; range = the P25..P75 multiples.
  3. Peer P/E   trailing (not forward) P/E: median P/E of same-sector index
                members on as_of x own TTM EPS; range = sector P25..P75.

fair_value     median of the available method values
low, high      median of the method lows / highs, widened to contain fair_value
mispricing     fair_value / last close - 1   (positive = undervalued)
method_spread  highest / lowest method value (1.0 for one method)

filter_candidates() values the FILTER_DEPTH best-ranked growth_tech members
each week and removes a name only when at least FILTER_MIN_METHODS methods
value it AND the price exceeds (1 + FILTER_PREMIUM) x every one of their values.
Every other candidate is returned unchanged.

Share-basis alignment (_align_basis): some per-share rows are on a different
share count or currency than the adjusted price file (KLAC and APH before their
2024 splits, RMB-reporting ADRs). Each row is rescaled by adjusted close /
(ps x revenue_per_share) at its fiscal period end, smoothed over quarters.

Rules followed
  * Data only through ``data`` (DataView): nothing dated after as_of.
  * No LLM numbers. The database pe / ev_multiple columns are not used;
    multiples are computed from EPS, revenue_per_share, ebitda_margin and prices.
  * Per-share and ratio columns are single-quarter values: TTM = sum of four
    consecutive quarters (a gap in the quarters voids the TTM figure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]
GROWTH_BUCKET = "growth_tech"
METHODS = ("dcf", "ev_ebitda", "peer_multiple")

# ---------------------------------------------------------------------------
# Constants. Each one is a stated assumption; the report gives the source.
# ---------------------------------------------------------------------------
# DCF
EQUITY_RISK_PREMIUM = 0.05      # long-run US equity risk premium (Damodaran implied ERP, 2017-2023 range 4.2-6.0%)
TERMINAL_GROWTH = 0.025         # long-run nominal growth, below the 2017-2026 average 10y yield + inflation
PROJECTION_YEARS = 10           # explicit forecast window, as in FinRobot's DCF
GROWTH_FLOOR, GROWTH_CAP = -0.05, 0.35   # first-year revenue growth bounds
GROWTH_WEIGHT_YOY = 0.5         # first-year growth = 0.5 x last-year growth + 0.5 x 3-year CAGR
NWC_TO_REVENUE = 0.10           # operating NWC / revenue; dNWC = 10% of the revenue increase
BETA_YEARS = 5                  # weekly returns against SPY over five years
MIN_BETA_WEEKS = 52             # prices start 2016-06, so 2017 needs a one-year beta
BETA_BOUNDS = (0.3, 3.0)
MAX_DEBT_WEIGHT = 0.8           # market-value D / (D + E), capped
STATUTORY_TAX = 0.21            # used only when effective_tax_rate is missing
WACC_STEP, TG_STEP = 0.01, 0.005  # sensitivity grid: WACC +-1pp, terminal growth +-0.5pp
# Share-basis alignment (see _align_basis)
ALIGN_BASIS = True
BASIS_WINDOW = 9                # quarters in the centred rolling median of the basis factor
PER_SHARE_COLUMNS = ("EPS", "revenue_per_share", "capex_per_share", "cash_per_share",
                     "interest_debt_per_share", "BPS", "fcf_per_share", "ocf_per_share")
# EV/EBITDA own history
HISTORY_SESSIONS = 1260         # five years of sessions
MIN_HISTORY_WEEKS = 52          # at least one year of weekly multiples (prices start 2016-06)
HIST_BAND = (0.25, 0.75)
# Peer P/E
PEER_BAND = (0.25, 0.75)
MAX_PEER_MULTIPLE = 200.0       # peers with P/E above this are noise, not a valuation
MIN_PEERS = 5
# filter_candidates
FILTER_DEPTH = 25               # growth_tech names valued per week (the portfolio holds 10)
FILTER_MIN_METHODS = 2
FILTER_PREMIUM = 0.0            # drop when price > (1 + premium) x every method's fair value

_TTM_COLUMNS = {"rev_ttm": "revenue_per_share", "eps_ttm": "EPS",
                "ebitda_ttm": "ebitda_q", "capex_ttm": "capex_per_share"}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _clip(x, lo, hi, default=None):
    x = float(x) if x is not None else np.nan
    if not np.isfinite(x):
        return default
    return float(min(max(x, lo), hi))


def _align_basis(f: pd.DataFrame, closes: pd.DataFrame, accepted: pd.Series) -> pd.DataFrame:
    """Put per-share columns on the same share basis (and currency) as the price file.

    The price file is split- and dividend-adjusted to its 2026 basis, but some
    per-share rows are not: KLAC and APH rows before mid-2024 are on the
    pre-split share count (about 10x and 2x off), and NTES / BIDU / PDD report
    RMB per ordinary share against a USD ADS price (about 7x off). ps x
    revenue_per_share is the filing's own price at fiscal period end on the
    filing's basis, so adjusted close / (ps x revenue_per_share) is the factor
    between the two bases. A centred rolling median over the ticker's quarters
    removes the noise from the calendar datadate not matching the fiscal
    quarter end, while keeping a genuine step (a split) sharp.
    """
    # Fiscal period end: the calendar datadate, or about 25 days before
    # acceptance for fiscal calendars whose datadate label falls after the filing.
    ref = pd.concat([f["datadate"], accepted - pd.Timedelta(days=25)], axis=1).min(axis=1)
    filled = closes.ffill()
    pos = filled.index.searchsorted(ref.to_numpy(), side="right") - 1
    col = filled.columns.get_indexer(f["ticker"])
    ok = (pos >= 0) & (col >= 0)
    adj = np.full(len(f), np.nan)
    adj[ok] = filled.to_numpy()[pos[ok], col[ok]]
    own = (f["ps"] * f["revenue_per_share"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where((own > 0) & (adj > 0), adj / own, np.nan)
    k = pd.Series(k, index=f.index)
    k = k.groupby(f["ticker"], sort=False).transform(
        lambda s: s.rolling(BASIS_WINDOW, center=True, min_periods=1).median().ffill().bfill())
    f["basis"] = k.fillna(1.0)
    for c in PER_SHARE_COLUMNS:
        f[c] = f[c] * f["basis"]
    return f


def _prepare(fund: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, quarter) with TTM sums and the date each row became public."""
    f = fund.copy()
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = f.sort_values(["ticker", "datadate"]).drop_duplicates(["ticker", "datadate"], keep="last")
    accepted = pd.to_datetime(f["accepted_date"], errors="coerce").dt.normalize()
    filed = pd.to_datetime(f["filing_date"], errors="coerce").dt.normalize()
    known = accepted.fillna(filed).fillna(f["datadate"] + pd.Timedelta(days=90))
    if ALIGN_BASIS:
        f = _align_basis(f, closes, known)
    # The DataView shows a quarter only once both its filing is accepted and its
    # datadate has passed, so a historical row was public from the later of the two.
    f["known"] = known.where(known > f["datadate"], f["datadate"])
    f["ebitda_q"] = f["ebitda_margin"] * f["revenue_per_share"]
    g = f.groupby("ticker", sort=False)
    for out, src in _TTM_COLUMNS.items():
        f[out] = g[src].transform(lambda s: s.rolling(4).sum())
    span = (f["datadate"] - g["datadate"].shift(3)).dt.days
    f.loc[~span.between(250, 300), list(_TTM_COLUMNS)] = np.nan   # missing quarter -> no TTM
    f["net_debt"] = f["interest_debt_per_share"] - f["cash_per_share"]
    return f.reset_index(drop=True)


def _ttm_years_ago(q: pd.DataFrame, years: int):
    """TTM revenue per share from the row whose datadate is ``years`` before the latest."""
    target = q["datadate"].iloc[-1] - pd.Timedelta(days=round(365.25 * years))
    near = q[(q["datadate"] - target).abs() <= pd.Timedelta(days=45)]
    if near.empty:
        return None
    v = near["rev_ttm"].iloc[-1]
    return float(v) if np.isfinite(v) and v > 0 else None


def _weekly(series: pd.Series) -> pd.Series:
    """Last close of each week, indexed by the real session date."""
    s = series.dropna()
    return s.groupby(s.index.to_period("W-FRI")).tail(1)


def _beta(px: pd.DataFrame, ticker: str):
    both = px[[ticker, "SPY"]].dropna().tail(252 * BETA_YEARS)
    wk = both.groupby(both.index.to_period("W-FRI")).tail(1).pct_change().dropna()
    if len(wk) < MIN_BETA_WEEKS or wk["SPY"].var() <= 0:
        return None
    raw = float(np.cov(wk[ticker], wk["SPY"])[0, 1] / wk["SPY"].var())
    return _clip(adjust_beta_blume(raw), *BETA_BOUNDS)


# ---------------------------------------------------------------------------
# Method 1: DCF
# ---------------------------------------------------------------------------
def _growth_path(q: pd.DataFrame):
    now = float(q["rev_ttm"].iloc[-1])
    one, three = _ttm_years_ago(q, 1), _ttm_years_ago(q, 3)
    if one is None:
        return None
    yoy = now / one - 1
    first = yoy if three is None else GROWTH_WEIGHT_YOY * yoy + (1 - GROWTH_WEIGHT_YOY) * ((now / three) ** (1 / 3) - 1)
    first = _clip(first, GROWTH_FLOOR, GROWTH_CAP)
    return list(np.linspace(first, TERMINAL_GROWTH, PROJECTION_YEARS))


def _dcf(q: pd.DataFrame, price: float, beta, rf: float, credit_spread: float):
    if beta is None or len(q) < 8:
        return None
    last, l4 = q.iloc[-1], q.tail(4)
    revenue = last["rev_ttm"]
    if not (np.isfinite(revenue) and revenue > 0):
        return None
    growth = _growth_path(q)
    margin = _clip(l4["ebitda_margin"].median(), 0.0, 0.9)
    if growth is None or margin is None or margin <= 0:
        return None
    mean_g = float(np.mean(growth))
    debt = max(float(last["interest_debt_per_share"]), 0.0) if np.isfinite(last["interest_debt_per_share"]) else 0.0
    net_debt = float(last["net_debt"]) if np.isfinite(last["net_debt"]) else None
    if net_debt is None:
        return None
    try:
        inputs = DCFInputs(
            revenue_base=float(revenue),
            revenue_growth_rates=growth,
            ebitda_margin=margin,
            da_pct_revenue=_clip((l4["ebitda_margin"] - l4["operating_margin"]).median(), 0.0, 0.3, 0.0),
            capex_pct_revenue=_clip(l4["capex_per_share"].abs().sum() / revenue, 0.0, 0.5, 0.0),
            nwc_pct_revenue=NWC_TO_REVENUE * mean_g / (1 + mean_g),
            terminal_nwc_pct_revenue=NWC_TO_REVENUE * TERMINAL_GROWTH / (1 + TERMINAL_GROWTH),
            tax_rate=_clip(l4["effective_tax_rate"].median(), 0.0, 0.35, STATUTORY_TAX),
            risk_free_rate=_clip(rf, 0.0, 0.15),
            beta=beta,
            equity_risk_premium=EQUITY_RISK_PREMIUM,
            cost_of_debt=_clip(rf + credit_spread, 0.0, 0.20),
            debt_ratio=min(debt / (debt + price), MAX_DEBT_WEIGHT),
            terminal_growth_rate=TERMINAL_GROWTH,
            shares_outstanding=1.0,
            net_debt=net_debt,
        )
        res = calculate_dcf(inputs)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None
    w = res.wacc
    grid = calculate_sensitivity(inputs, wacc_range=[w - WACC_STEP, w, w + WACC_STEP],
                                 tg_range=[TERMINAL_GROWTH - TG_STEP, TERMINAL_GROWTH, TERMINAL_GROWTH + TG_STEP])
    cells = [p for row in grid["implied_prices"] for p in row if p is not None and np.isfinite(p)]
    fair = float(res.implied_price)
    return fair, min(cells + [fair]), max(cells + [fair])


# ---------------------------------------------------------------------------
# Method 2: EV / EBITDA against the ticker's own history
# ---------------------------------------------------------------------------
def _ev_ebitda(q: pd.DataFrame, closes: pd.Series):
    last = q.iloc[-1]
    ebitda_now, nd_now = last["ebitda_ttm"], last["net_debt"]
    if not (np.isfinite(ebitda_now) and ebitda_now > 0 and np.isfinite(nd_now)):
        return None
    filings = q.dropna(subset=["ebitda_ttm", "net_debt"]).sort_values("known")[["known", "ebitda_ttm", "net_debt"]]
    wk = _weekly(closes)
    if filings.empty or wk.empty:
        return None
    weeks = pd.DataFrame({"date": wk.index, "price": wk.values})
    hist = pd.merge_asof(weeks, filings, left_on="date", right_on="known").dropna()
    hist = hist[hist["ebitda_ttm"] > 0]
    mult = (hist["price"] + hist["net_debt"]) / hist["ebitda_ttm"]
    mult = mult[mult > 0]
    if len(mult) < MIN_HISTORY_WEEKS:
        return None
    lo, mid, hi = (float(mult.quantile(p)) * ebitda_now - nd_now for p in (HIST_BAND[0], 0.5, HIST_BAND[1]))
    if not mid > 0:
        return None
    return mid, max(lo, 0.0), hi


# ---------------------------------------------------------------------------
# Method 3: trailing P/E against same-sector index members
# ---------------------------------------------------------------------------
def _peer_table(latest: pd.DataFrame, members: set, prices: dict) -> pd.DataFrame:
    peers = latest[latest["ticker"].isin(members)].copy()
    peers["price"] = peers["ticker"].map(prices)
    peers["multiple"] = peers["price"] / peers["eps_ttm"]
    ok = (peers["eps_ttm"] > 0) & peers["multiple"].between(0, MAX_PEER_MULTIPLE)
    return peers.loc[ok, ["ticker", "gsector", "multiple"]]


def _peer_multiple(row: pd.Series, peers: pd.DataFrame):
    eps = row["eps_ttm"]
    if not (np.isfinite(eps) and eps > 0):
        return None
    same = peers[(peers["gsector"] == row["gsector"]) & (peers["ticker"] != row["ticker"])]["multiple"]
    if len(same) < MIN_PEERS:
        return None
    lo, mid, hi = (float(same.quantile(p)) * eps for p in (PEER_BAND[0], 0.5, PEER_BAND[1]))
    return mid, lo, hi


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------
def value_detail(as_of, tickers, data) -> pd.DataFrame:
    """Per-method values for each ticker; value() and filter_candidates() read this."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()
    members = data.members()
    universe = sorted(members | set(tickers))
    fund = _prepare(data.fundamentals(universe), data.close(universe))
    latest = fund.groupby("ticker", sort=False).tail(1)
    prices = {k: v for k, v in data.last_close(sorted(set(latest["ticker"]) | set(tickers))).items()
              if v is not None and np.isfinite(v) and v > 0}
    peers = _peer_table(latest, members, prices)
    px = data.close(tickers + ["SPY"], lookback=HISTORY_SESSIONS)
    dgs10, baa = data.macro("DGS10").dropna(), data.macro("BAA10Y").dropna()
    rf = float(dgs10.iloc[-1]) / 100 if len(dgs10) else None
    spread = float(baa.iloc[-1]) / 100 if len(baa) else None
    by_ticker = dict(tuple(fund.groupby("ticker", sort=False)))

    rows = []
    for t in tickers:
        price, q = prices.get(t), by_ticker.get(t)
        if price is None or q is None or q.empty:
            continue
        found = {
            "dcf": _dcf(q, price, _beta(px, t), rf, spread) if rf is not None and spread is not None else None,
            "ev_ebitda": _ev_ebitda(q, px[t]),
            "peer_multiple": _peer_multiple(q.iloc[-1], peers),
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
        for m in METHODS:
            rec[f"fair_{m}"] = found[m][0] if m in found else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def value(as_of, tickers, data) -> pd.DataFrame:
    detail = value_detail(as_of, tickers, data)
    if detail.empty:
        return pd.DataFrame(columns=COLUMNS)
    return detail[COLUMNS].reset_index(drop=True)


def overvalued(detail: pd.DataFrame) -> pd.Series:
    """True where enough methods agree the price is above every one of their values."""
    if detail.empty:
        return pd.Series(dtype=bool)
    fairs = detail[[f"fair_{m}" for m in METHODS]]
    return (detail["n_methods"] >= FILTER_MIN_METHODS) & \
        (detail["price"] > (1 + FILTER_PREMIUM) * fairs.max(axis=1))


def filter_candidates(as_of, candidates, data):
    members = data.members()
    growth = sorted((c for c in candidates if c.bucket == GROWTH_BUCKET and c.symbol in members),
                    key=lambda c: c.rank_score, reverse=True)[:FILTER_DEPTH]
    if not growth:
        return candidates
    detail = value_detail(as_of, [c.symbol for c in growth], data)
    if detail.empty:
        return candidates
    drop = set(detail.loc[overvalued(detail), "ticker"])
    return tuple(c for c in candidates if c.symbol not in drop)
