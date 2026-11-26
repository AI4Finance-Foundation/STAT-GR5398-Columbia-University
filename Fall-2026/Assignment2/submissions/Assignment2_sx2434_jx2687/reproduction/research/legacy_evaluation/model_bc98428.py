"""Three-method candidate using the team workbook convention and an independent history implementation."""
import numpy as np
from scipy.optimize import brentq
import pandas as pd
from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]
TERMINAL = .025


def growth_path(start):
    return list(np.linspace(start, TERMINAL, 10))


def dcf_estimates(row, beta, rf, reverse=False):
    if not np.isfinite([beta, rf, row.growth, row.margin, row.da, row.capex, row.tax,
                        row.net_debt, row.debt_share, row.revenue]).all():
        return {}
    if row.revenue <= 0 or row.margin <= 0:
        return {}
    g = float(np.clip(row.growth, -.10, .30))
    try:
        inputs = DCFInputs(revenue_base=row.revenue, revenue_growth_rates=growth_path(g),
            ebitda_margin=np.clip(row.margin, 0, 1), da_pct_revenue=np.clip(row.da, 0, .5),
            capex_pct_revenue=np.clip(row.capex, 0, 1), nwc_pct_revenue=0, terminal_nwc_pct_revenue=0,
            tax_rate=np.clip(row.tax, 0, .35), risk_free_rate=rf,
            beta=np.clip(adjust_beta_blume(beta), 0, 5), equity_risk_premium=.05,
            cost_of_debt=rf+.02, debt_ratio=np.clip(row.debt_share, 0, .95),
            shares_outstanding=1, net_debt=row.net_debt, terminal_growth_rate=TERMINAL)
        result = calculate_dcf(inputs)
        wacc = result.wacc
        grid = calculate_sensitivity(inputs, [wacc-.01, wacc, wacc+.01], [.02, TERMINAL, .03])
        lowcase = inputs.model_copy(update={"revenue_growth_rates": growth_path(g-.05)})
        highcase = inputs.model_copy(update={"revenue_growth_rates": growth_path(g+.05)})
        down = calculate_dcf(lowcase, wacc_override=wacc+.01, tg_override=.02).implied_price
        up = calculate_dcf(highcase, wacc_override=wacc-.01, tg_override=.03).implied_price
        cells = [v for line in grid["implied_prices"] for v in line if v is not None and np.isfinite(v) and v > 0]
        points = cells + [result.implied_price, down, up]
        audit = {}
        if reverse:
            # Invert the same ten-year fading growth path while keeping other inputs fixed.
            def gap(start_growth):
                variant = inputs.model_copy(update={"revenue_growth_rates": growth_path(start_growth)})
                try:
                    return calculate_dcf(variant).implied_price-row.price
                except ValueError:
                    return np.nan
            grid_g = np.linspace(-.30, 1., 66)
            gaps = [gap(x) for x in grid_g]
            roots = [brentq(gap, a, b) for a,b,fa,fb in zip(grid_g[:-1],grid_g[1:],gaps[:-1],gaps[1:]) if fa*fb<=0]
            audit = dict(implied_start_growth=roots[0] if len(roots)==1 else np.nan,
                         beta=inputs.beta, ttm_revenue=row.revenue, ebitda_margin=inputs.ebitda_margin,
                         net_debt=row.net_debt, lower_growth_value=calculate_dcf(lowcase).implied_price)
        return dict(**audit, dcf=result.implied_price, dcf_low=min(points), dcf_high=max(points),
                    growth=g, wacc=wacc)
    except (ValueError, ArithmeticError):
        return {}


def valuations(as_of, tickers, data, *, reverse=False):
    members = sorted(data.members())
    symbols = sorted(set(tickers).intersection(members))
    if not symbols:
        return pd.DataFrame()
    day = pd.Timestamp(as_of)
    prices = data.close(sorted(set(members) | {"SPY"}), lookback=1260)
    current = prices.iloc[-1]
    rates = data.macro("DGS10").dropna()
    rf = float(rates.iloc[-1])/100 if len(rates) else np.nan
    weekly = prices[list(dict.fromkeys(symbols+["SPY"]))].resample("W-FRI").last().pct_change(fill_method=None)
    market = weekly["SPY"]
    paired_market = weekly.notna().mul(market, axis=0).where(weekly.notna())
    betas = weekly.corrwith(market)*weekly.where(market.notna(),axis=0).std()/paired_market.std()
    betas = betas.where(paired_market.count() >= 52)
    f = data.fundamentals(members)
    if f.empty:
        return pd.DataFrame()
    # Require a visible release date. Same-day releases may be after the close.
    released = pd.to_datetime(f.accepted_date, errors="coerce").fillna(pd.to_datetime(f.filing_date, errors="coerce"))
    f = f.sort_values(["ticker","datadate"]).copy()
    f["released"] = released
    f["quarter"] = pd.PeriodIndex(f.datadate, freq="Q").asi8
    grouped = f.groupby("ticker")
    last = grouped.tail(1).set_index("ticker")
    fresh = (day-pd.to_datetime(last.datadate)).dt.days.between(0,185)
    q = grouped.tail(4).copy()
    gq = q.groupby("ticker")
    counts = gq[["EPS","ebitda_margin","operating_margin","capex_per_share","effective_tax_rate","released"]].count()
    valid4 = (gq.size()==4) & (gq.quarter.max()-gq.quarter.min()==3) & fresh & (counts.released==4) & (gq.released.max()<day)
    info = pd.DataFrame(index=last.index)
    info["eps"] = gq.EPS.sum(min_count=4).where(valid4)
    info["sector"] = last.gsector
    info["price"] = current
    info["trailing_multiple"] = (info.price/info.eps).where((info.eps>0)&(info.price>0))
    peer_pools = {s:g.trailing_multiple.dropna() for s,g in info.groupby("sector")}
    q["da"] = q.ebitda_margin-q.operating_margin
    q["abs_capex"] = q.capex_per_share.abs()
    info["revenue"] = gq.revenue_per_share.sum(min_count=4)
    info["margin"] = gq.ebitda_margin.median().where(counts.ebitda_margin==4)
    info["da"] = gq.da.median().where((counts.operating_margin==4)&(counts.ebitda_margin==4))
    info["capex"] = gq.abs_capex.sum(min_count=4)/info.revenue
    info["tax"] = gq.effective_tax_rate.median().where(counts.effective_tax_rate==4)
    info["debt_share"] = last.debt_to_capital
    info["net_debt"] = last.interest_debt_per_share-last.cash_per_share
    eight = grouped.tail(8).copy()
    eightg = eight.groupby("ticker")
    prev = eightg.revenue_per_share.shift(4)
    eight["yoy"] = (eight.revenue_per_share/prev-1).where((prev>0)&(eight.revenue_per_share>0))
    valid8 = valid4 & (eightg.size()==8) & (eightg.quarter.max()-eightg.quarter.min()==7) & (eightg.released.count()==8) & (eightg.released.max()<day)
    info["growth"] = eightg.yoy.median().where(valid8 & (eightg.yoy.count()==4))
    # Snapshot each historical month using the filing version visible at that month end.
    hist = data.extra("history_monthly")
    hist = hist.loc[(hist.date < day) & (hist.date >= day-pd.DateOffset(years=5)) & hist.ticker.isin(symbols)]
    hg = hist.groupby("ticker").multiple
    quant = hg.quantile([.25,.5,.75]).unstack().reindex(columns=[.25,.5,.75])
    ttm = (q.revenue_per_share*q.ebitda_margin).groupby(q.ticker).sum(min_count=4)
    complete = (gq.size()==4) & (gq.quarter.max()-gq.quarter.min()==3)
    own = quant.mul(ttm.where(complete & (ttm>0)),axis=0).sub(info.net_debt,axis=0)
    own = own.where((hg.count().reindex(own.index,fill_value=0)>=12) & (own[.25]>0), axis=0)
    info["history_n"] = hg.count()
    info[["hist_low","hist_mid","hist_high"]] = own.reindex(info.index).to_numpy()
    rows = []
    for t in symbols:
        if t not in info.index or not np.isfinite(current.get(t,np.nan)) or current[t]<=0:
            continue
        row = info.loc[t]
        rec = dict(ticker=t, price=row.price, hist_low=row.hist_low, hist_mid=row.hist_mid, hist_high=row.hist_high, history_n=row.history_n, eps=row.eps, sector=row.sector,
                   quarter_end=last.at[t,"datadate"], latest_release=last.at[t,"released"], rf=rf)
        peers = peer_pools.get(row.sector)
        if peers is not None and row.eps>0:
            peers = peers.drop(t, errors="ignore")
            if len(peers)>=5:
                lo, mid, hi = np.quantile(peers, [.25,.5,.75])
                rec.update(peer_low=lo*row.eps, peer=mid*row.eps, peer_high=hi*row.eps,
                           peer_n=len(peers), peer_multiple=mid)
        rec.update(dcf_estimates(row,betas.get(t,np.nan),rf,reverse))
        rows.append(rec)
    return pd.DataFrame(rows).reindex(columns=["ticker","price","hist_low","hist_mid","hist_high","history_n","dcf","dcf_low","dcf_high","peer","peer_low","peer_high",
            "growth","wacc","rf","eps","sector","peer_n","peer_multiple","quarter_end","latest_release"] + (["implied_start_growth","beta","ttm_revenue","ebitda_margin","net_debt","lower_growth_value"] if reverse else []))


def value(as_of, tickers, data):
    d = valuations(as_of,tickers,data)
    if d.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = d.dropna(subset=["dcf","peer","hist_mid"],thresh=2).copy()
    centers = d[["dcf","peer","hist_mid"]]
    d["fair_value"] = centers.median(axis=1)
    d["low"] = d[["dcf_low","peer_low","hist_low"]].min(axis=1)
    d["high"] = d[["dcf_high","peer_high","hist_high"]].max(axis=1)
    d["mispricing"] = d.fair_value/d.price-1
    d["method_spread"] = centers.max(axis=1)/centers.min(axis=1)
    return d[COLUMNS]


DEFAULT_RULE = "two_of_three"


def rejected(d, rule):
    if d.empty:
        return set()
    cashflow = d.price.gt(d.dcf_high)
    peer = d.price.gt(d.peer_high)
    history = d.price.gt(d.hist_high)
    if rule == "dcf_confirmation":
        flag = cashflow & (peer | history)
    elif rule == "two_of_three":
        flag = (cashflow & peer) | (cashflow & history) | (history & peer)
    elif rule == "dcf_peer":
        flag = cashflow & peer
    else:
        raise ValueError(f"unknown rule {rule}")
    return set(d.loc[flag, "ticker"])


def filter_candidates(as_of, candidates, data):
    d = valuations(as_of, [c.symbol for c in candidates if c.bucket=="growth_tech"], data)
    removed = rejected(d, DEFAULT_RULE)
    return tuple(c for c in candidates if c.symbol not in removed)
