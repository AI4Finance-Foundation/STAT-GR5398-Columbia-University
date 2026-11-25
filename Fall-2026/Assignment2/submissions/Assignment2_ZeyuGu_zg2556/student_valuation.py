"""Point-in-time per-share DCF, own-history EV/EBITDA and sector trailing P/E."""
from __future__ import annotations

import numpy as np
import pandas as pd

from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity


FILTER_THRESHOLD = 0.15
MAX_METHOD_SPREAD = 2.5
OUTPUT = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread",
          "n_methods", "dcf_value", "ev_value", "peer_value",
          "dcf_low", "dcf_high", "ev_low", "ev_high", "peer_low", "peer_high"]
FIELDS = ["ticker", "datadate", "accepted_date", "filing_date", "gsector", "EPS",
          "revenue_per_share", "ebitda_margin", "operating_margin", "capex_per_share",
          "effective_tax_rate", "debt_to_capital", "interest_debt_per_share", "cash_per_share"]


def _peer_pool(f, members, prices, date):
    last = f.groupby("ticker", sort=False).tail(4)
    groups = last.groupby("ticker", sort=False)
    base = groups.agg(sector=("gsector", "last"), newest=("quarter", "last"), oldest=("quarter", "first"),
                      latest_period=("datadate", "last"))
    base["eps"] = groups["EPS"].sum(min_count=4)
    base["price"] = prices.reindex(base.index)
    fresh = (date - base["latest_period"]).dt.days.le(180)
    base = base.loc[(base.newest - base.oldest == 3) & fresh & base.eps.gt(0) & base.price.gt(0)]
    multiples = base.price / base.eps
    peers = base.assign(multiple=multiples).loc[base.index.isin(members) & multiples.le(200)]
    return base, {sector: rows["multiple"] for sector, rows in peers.groupby("sector")}


def _peer_band(ticker, base, sectors):
    if ticker not in base.index:
        return None
    sector = sectors.get(base.at[ticker, "sector"])
    if sector is None:
        return None
    peers = sector.drop(ticker, errors="ignore")
    if len(peers) < 5:
        return None
    return np.quantile(peers, [0.25, 0.5, 0.75]) * base.at[ticker, "eps"]


def _dcf_band(q, beta, rate):
    if len(q) < 8 or not np.isfinite(beta) or not np.isfinite(rate):
        return None
    if q.quarter.iloc[-1] - q.quarter.iloc[-8] != 7:
        return None
    last = q.tail(4)
    revenue = last.revenue_per_share.sum(min_count=4)
    previous = q.revenue_per_share.iloc[-8:-4].sum(min_count=4)
    if not (revenue > 0 and previous > 0):
        return None
    # Use a slower three-year CAGR as a second estimate when all periods exist.
    growth = revenue / previous - 1
    if len(q) >= 16 and q.quarter.iloc[-1] - q.quarter.iloc[-16] == 15:
        older = q.revenue_per_share.iloc[-16:-12].sum(min_count=4)
        if older > 0:
            growth = (growth + (revenue / older) ** (1 / 3) - 1) / 2
    growth = float(np.clip(growth, -0.10, 0.30))
    terminal = 0.025
    ebitda = last.ebitda_margin.median()
    depreciation = (last.ebitda_margin - last.operating_margin).median()
    capex = last.capex_per_share.abs().sum(min_count=4) / revenue
    tax = last.effective_tax_rate.median()
    debt = last.debt_to_capital.iloc[-1]
    net_debt = last.interest_debt_per_share.iloc[-1] - last.cash_per_share.iloc[-1]
    if not np.isfinite([ebitda, depreciation, capex, tax, debt, net_debt]).all() or ebitda <= 0:
        return None

    inputs = DCFInputs(
        revenue_base=revenue, revenue_growth_rates=np.linspace(growth, terminal, 10).tolist(),
        ebitda_margin=float(np.clip(ebitda, 0, 1)),
        da_pct_revenue=float(np.clip(depreciation, 0, 0.5)),
        capex_pct_revenue=float(np.clip(capex, 0, 1)), nwc_pct_revenue=0.0,
        tax_rate=float(np.clip(tax, 0, 0.35)), risk_free_rate=float(np.clip(rate, 0, 0.15)),
        beta=float(np.clip(adjust_beta_blume(beta), 0, 5)), equity_risk_premium=0.05,
        cost_of_debt=float(np.clip(rate + 0.02, 0, 0.20)), debt_ratio=float(np.clip(debt, 0, 0.95)),
        terminal_growth_rate=terminal, shares_outstanding=1.0, net_debt=net_debt,
    )
    try:
        result = calculate_dcf(inputs)
    except ValueError:
        return None
    wacc = result.wacc
    grid = calculate_sensitivity(inputs, [wacc - 0.01, wacc, wacc + 0.01],
                                [terminal - 0.005, terminal, terminal + 0.005])
    outcomes = [x for row in grid["implied_prices"] for x in row
                if x is not None and np.isfinite(x) and x > 0]
    if not outcomes:
        return None
    lo, hi = np.quantile(outcomes, [0.25, 0.75])
    return np.array([min(lo, result.implied_price), result.implied_price, max(hi, result.implied_price)])


def _ev_band(q, history, date):
    """Sample at the first session strictly after all four quarterly inputs became known."""
    qebitda = q.revenue_per_share * q.ebitda_margin
    ttm = qebitda.rolling(4, min_periods=4).sum().to_numpy()
    debts = (q.interest_debt_per_share - q.cash_per_share).to_numpy()
    if not (ttm[-1] > 0 and np.isfinite(debts[-1])):
        return None
    # Historical reconstructions need real filing metadata for each of the four
    # rows. If the visible version was revised, its newer timestamp governs.
    known = q["known"].to_numpy(dtype="datetime64[ns]")
    quarters = q.quarter.to_numpy()
    dates = history.index
    prices = history.to_numpy()
    cutoff = date - pd.DateOffset(years=5)
    samples = {}
    for i in range(3, len(q)):
        window = known[i-3:i+1]
        if quarters[i] - quarters[i-3] != 3 or np.isnat(window).any() or not ttm[i] > 0:
            continue
        accepted = pd.Timestamp(window.max())
        if accepted < cutoff or accepted < dates[0] or accepted >= date:
            continue
        index = dates.searchsorted(accepted, side="right")
        if index >= len(dates):
            continue
        if i + 1 < len(q):
            # Once a newer fiscal period is public, this older TTM is obsolete.
            newer = known[i+1:]
            if np.any(newer[~np.isnat(newer)] <= dates[index].to_datetime64()):
                continue
        multiple = (prices[index] + debts[i]) / ttm[i]
        if np.isfinite(multiple) and 0 < multiple <= 100:
            samples[index] = multiple
    if len(samples) < 4:
        return None
    estimates = np.quantile(list(samples.values()), [0.25, 0.5, 0.75]) * ttm[-1] - debts[-1]
    if not estimates[1] > 0:
        return None
    estimates[0] = max(0.0, estimates[0])
    return estimates


def _weekly_betas(prices, date):
    weekly = prices.resample("W-FRI").last().loc[:date].pct_change(fill_method=None)
    benchmark = weekly["SPY"]
    betas = {}
    for ticker in weekly.columns.drop("SPY"):
        aligned = weekly[ticker].notna() & benchmark.notna()
        if aligned.sum() >= 52:
            market = benchmark[aligned]
            variance = market.var()
            if variance > 0:
                betas[ticker] = weekly.loc[aligned, ticker].cov(market) / variance
    return betas


def value(as_of, tickers, data) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=OUTPUT)
    date = pd.Timestamp(as_of).normalize()
    members = set(data.members())
    universe = sorted(members | set(tickers))
    f = data.fundamentals(universe).loc[:, FIELDS].copy()
    if f.empty:
        return pd.DataFrame(columns=OUTPUT)
    f["datadate"] = pd.to_datetime(f["datadate"])
    accepted = pd.to_datetime(f.accepted_date, errors="coerce")
    filed = accepted.fillna(pd.to_datetime(f.filing_date, errors="coerce"))
    # Decisions occur at the close. Intraday release times are unavailable;
    # do not use a dated filing on its own publication day. Some vendor fiscal
    # labels postdate the filing; both dates must pass before a historical mark.
    f["known"] = filed.where(filed.isna() | (filed >= f.datadate), f.datadate)
    f = f.loc[(f.datadate < date) & f.known.notna() & (f.known < date)].copy()
    if f.empty:
        return pd.DataFrame(columns=OUTPUT)
    # Calendar-quarter keys allow missing fiscal periods to be detected.
    f["quarter"] = f.datadate.dt.year * 4 + f.datadate.dt.quarter
    f = f.sort_values(["ticker", "datadate"]).drop_duplicates(["ticker", "quarter"], keep="last")
    # Peer observations must have an actual close at the decision date. Targets
    # use the same current close as the reported fair-value-to-price ratio.
    current = data.close(universe, lookback=1)
    if current.empty:
        return pd.DataFrame(columns=OUTPUT)
    prices = current.iloc[-1].replace([np.inf, -np.inf], np.nan)
    base, sectors = _peer_pool(f, members, prices, date)
    selected = f[f.ticker.isin(tickers)].copy()
    history = data.close(sorted(set(tickers) | {"SPY"}), lookback=1260)
    betas = _weekly_betas(history, date)
    treasury = data.macro("DGS10").dropna()
    rate = treasury.iloc[-1] / 100 if len(treasury) else np.nan
    rows = []
    for ticker, q in selected.groupby("ticker", sort=False):
        price = prices.get(ticker, np.nan)
        if not (price > 0 and len(q) >= 4 and q.quarter.iloc[-1] - q.quarter.iloc[-4] == 3):
            continue
        if (date - q.datadate.iloc[-1]).days > 180:
            continue
        bands = [
            _dcf_band(q, betas.get(ticker, np.nan), rate),
            _ev_band(q, history[ticker], date),
            _peer_band(ticker, base, sectors),
        ]
        valid = [b for b in bands if b is not None]
        if not valid:
            continue
        central = np.asarray(valid)[:, 1]
        fair = float(np.median(central))
        low, high = np.quantile(np.asarray(valid).ravel(), [0.25, 0.75])
        points = [b[1] if b is not None else np.nan for b in bands]
        ends = [b[end] if b is not None else np.nan for b in bands for end in (0, 2)]
        rows.append([ticker, fair, min(low, fair), max(high, fair), fair / price - 1,
                     central.max() / central.min(), len(valid), *points, *ends])
    return pd.DataFrame(rows, columns=OUTPUT)


def screen_values(as_of, candidates, data):
    members = set(data.members())
    growth = (c for c in candidates if c.bucket == "growth_tech" and c.symbol in members)
    ranked = sorted(growth, key=lambda c: c.rank_score, reverse=True)
    return value(as_of, [c.symbol for c in ranked[:20]], data)


def removed_symbols(rows, threshold):
    exclude = (rows.n_methods >= 2) & (rows.method_spread <= MAX_METHOD_SPREAD) & (rows.mispricing < -threshold)
    return set(rows.loc[exclude, "ticker"])


def filter_candidates(as_of, candidates, data):
    dropped = removed_symbols(screen_values(as_of, candidates, data), FILTER_THRESHOLD)
    return tuple(c for c in candidates if c.symbol not in dropped or c.bucket != "growth_tech")
