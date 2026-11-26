"""Own-history EV/EBITDA, using only the supplied course DataView.

Call historical_ev_ebitda_value(as_of, tickers, data). Set return_history=True
to receive (results, monthly_details). All money fields are USD per share on
the course data basis. No file reads, network calls, or portfolio filters.
"""
from __future__ import annotations

from copy import copy
import numpy as np
import pandas as pd

FIELDS = ['revenue_per_share', 'ebitda_margin',
          'interest_debt_per_share', 'cash_per_share']
RESULT_COLUMNS = ['ticker', 'as_of', 'price', 'price_date', 'valid_months',
                  'multiple_p25', 'multiple_p50', 'multiple_p75',
                  'low', 'fair_value', 'high', 'mispricing',
                  'ttm_ebitda_per_share', 'net_debt_per_share',
                  'latest_quarter', 'latest_accepted_date',
                  'history_first', 'history_last', 'status', 'reason']
HISTORY_COLUMNS = ['ticker', 'month_end', 'price', 'ttm_start', 'ttm_end',
                   'latest_accepted_date', 'ttm_ebitda_per_share',
                   'debt_per_share', 'cash_per_share', 'net_debt_per_share',
                   'multiple', 'valid', 'reason']


def _view_at(data, day):
    """Narrow a DataView's public cutoff without changing the caller's view.

    The course DataView exposes a writable as_of attribute. A shallow copy
    shares its read-only data provider but queries filings at the earlier
    cutoff, preserving filing-version selection. We never read its private
    bundle, and never widen the original information horizon.
    """
    day = pd.Timestamp(day).normalize()
    if day > pd.Timestamp(data.as_of).normalize():
        raise ValueError('Historical cutoff is after the supplied DataView')
    view = copy(data)
    view.as_of = day
    return view


def _snapshot(frame, day):
    """Four consecutive visible quarters; flow sum, latest balance stocks."""
    if frame.empty:
        return {}, '没有当时可用的财报'
    q = frame.sort_values('datadate').tail(4).copy()
    if len(q) < 4:
        return {}, '不足四个季度财报'
    dates = pd.to_datetime(q['datadate'], errors='coerce')
    if dates.isna().any() or (dates > pd.Timestamp(day)).any():
        return {}, '季度日期缺失或晚于观察日'
    periods = dates.dt.to_period('Q').astype('int64').to_numpy()
    if not np.all(np.diff(periods) == 1):
        return {}, '最近四季度不连续'
    nums = q[FIELDS].apply(pd.to_numeric, errors='coerce')
    # Debt and cash are stocks: only the latest quarter is required.
    if not np.isfinite(nums[['revenue_per_share', 'ebitda_margin']].to_numpy()).all():
        return {}, '四季度收入或EBITDA利润率缺失'
    if not np.isfinite(nums[['interest_debt_per_share', 'cash_per_share']].iloc[-1].to_numpy()).all():
        return {}, '最新季度债务或现金缺失'
    accepted = pd.to_datetime(q['accepted_date'], errors='coerce')
    filed = pd.to_datetime(q['filing_date'], errors='coerce')
    known = accepted.fillna(filed)
    if known.isna().any():
        return {}, '财报公开日期缺失（不采用推定滞后日期）'
    if (known > pd.Timestamp(day)).any():
        return {}, '财报在观察日之后才公开'
    e = float((nums['revenue_per_share'] * nums['ebitda_margin']).sum())
    debt = float(nums['interest_debt_per_share'].iloc[-1])
    cash = float(nums['cash_per_share'].iloc[-1])
    snap = dict(ttm_start=dates.iloc[0].strftime('%Y-%m-%d'),
                ttm_end=dates.iloc[-1].strftime('%Y-%m-%d'),
                latest_accepted_date=known.max().strftime('%Y-%m-%d'),
                ttm_ebitda_per_share=e, debt_per_share=debt,
                cash_per_share=cash, net_debt_per_share=debt-cash)
    return snap, '' if e > 0 else 'TTM EBITDA非正'


def historical_ev_ebitda_value(as_of, tickers, data, *,
                             lookback_years=5, min_months=12,
                             return_history=False):
    """Return one row per requested ticker, including every failed ticker.

    Historical dates are completed calendar months' last SPY trading session,
    within [as_of - 5 years, as_of). Never replace a missing stock price with
    an earlier stock-specific session. Require >=12 valid positive multiples.
    EBITDA/share = sum(quarter revenue/share * quarter EBITDA margin).
    EV/share = price + latest debt/share - latest cash/share.
    Fair/share = historical median multiple * current TTM EBITDA/share
                 - current net debt/share. Quantiles use linear interpolation.

    The course dividend-adjusted prices and supplied per-share fundamentals
    are retained for reconciliation. This is a course-basis valuation proxy,
    not a verified unadjusted historical share price.
    """
    if not isinstance(lookback_years, int) or not 1 <= lookback_years <= 5:
        raise ValueError('lookback_years must be an integer between 1 and 5')
    if not isinstance(min_months, int) or min_months < 12:
        raise ValueError('min_months must be at least 12')
    cutoff = pd.Timestamp(as_of).normalize()
    view = _view_at(data, cutoff)
    tickers = list(dict.fromkeys(str(t).strip().upper() for t in tickers))
    if not tickers:
        result = pd.DataFrame(columns=RESULT_COLUMNS)
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
        return (result, history) if return_history else result
    prices = view.close(list(dict.fromkeys(['SPY', *tickers]))).sort_index()
    calendar = prices['SPY'].dropna()
    lower = cutoff - pd.DateOffset(years=lookback_years)
    month_ends = []
    for month, group in calendar.groupby(calendar.index.to_period('M')):
        last = group.index[-1]
        if month.end_time.normalize() <= cutoff and lower <= last < cutoff:
            month_ends.append(last)
    records = []
    for day in month_ends:
        filings = _view_at(view, day).fundamentals(tickers)
        for ticker in tickers:
            rec = dict(ticker=ticker, month_end=day.strftime('%Y-%m-%d'),
                       price=float(prices.at[day, ticker]), multiple=np.nan,
                       valid=False, reason='')
            snap, reason = _snapshot(filings[filings['ticker'] == ticker], day)
            rec.update(snap)
            if not reason and (not np.isfinite(rec['price']) or rec['price'] <= 0):
                reason = '月末当天股价缺失或非正'
            if not reason:
                multiple = (rec['price'] + snap['net_debt_per_share']) / snap['ttm_ebitda_per_share']
                if not np.isfinite(multiple) or multiple <= 0:
                    reason = 'EV/EBITDA非正或无效'
                else:
                    rec.update(multiple=float(multiple), valid=True)
            rec['reason'] = reason
            records.append(rec)
    history = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    result = _current_valuations(cutoff, tickers, view, prices, history, min_months)
    return (result, history) if return_history else result


def cached_history_value(as_of, tickers, data):
    """Team adapter: reuse dated monthly multiples, then apply Jiao Xu's current valuation."""
    cutoff = pd.Timestamp(as_of).normalize()
    view = _view_at(data, cutoff)
    history = view.extra("history_monthly")
    eligible = (
        history.ticker.isin(tickers)
        & (history.date >= cutoff - pd.DateOffset(years=5))
        & (history.date < cutoff)
        & (history.date + pd.offsets.MonthEnd(0) <= cutoff)
    )
    history = history.loc[eligible].rename(columns={"date": "month_end"}).assign(valid=True)
    history["month_end"] = history.month_end.dt.strftime("%Y-%m-%d")
    prices = view.close(tickers)
    return _current_valuations(cutoff, tickers, view, prices, history, 12)


def _current_valuations(cutoff, tickers, view, prices, history, min_months):
    current = view.fundamentals(tickers)
    current_prices = view.last_close(tickers)
    results = []
    for ticker in tickers:
        hist = history[(history['ticker'] == ticker) & history['valid'].eq(True)]
        price = float(current_prices.get(ticker, np.nan))
        series = prices[ticker].dropna()
        row = dict(ticker=ticker, as_of=cutoff.strftime('%Y-%m-%d'), price=price,
                   price_date=series.index[-1].strftime('%Y-%m-%d') if len(series) else None,
                   valid_months=len(hist), status='ok', reason='')
        if len(hist):
            row.update(history_first=hist.iloc[0]['month_end'], history_last=hist.iloc[-1]['month_end'])
        snap, reason = _snapshot(current[current['ticker'] == ticker], cutoff)
        if snap:
            row.update(ttm_ebitda_per_share=snap['ttm_ebitda_per_share'],
                       net_debt_per_share=snap['net_debt_per_share'],
                       latest_quarter=snap['ttm_end'],
                       latest_accepted_date=snap['latest_accepted_date'])
        problems = []
        if len(hist) < min_months:
            problems.append(f'历史数据不足：{len(hist)}个有效月末，至少需要{min_months}个')
        if reason:
            problems.append(reason)
        if not np.isfinite(price) or price <= 0:
            problems.append('估值日最新可得股价缺失或非正')
        if problems:
            row.update(status='unavailable', reason='；'.join(problems))
        else:
            qs = hist['multiple'].quantile([.25, .5, .75], interpolation='linear').to_numpy()
            vals = qs * snap['ttm_ebitda_per_share'] - snap['net_debt_per_share']
            row.update(multiple_p25=float(qs[0]), multiple_p50=float(qs[1]), multiple_p75=float(qs[2]),
                       low=float(vals[0]), fair_value=float(vals[1]), high=float(vals[2]),
                       mispricing=float(vals[1]/price-1))
            if vals[1] <= 0:
                row.update(status='nonpositive_value', reason='中位倍数隐含股权价值非正，不宜直接用于过滤')
        results.append(row)
    result = pd.DataFrame(results, columns=RESULT_COLUMNS)
    return result


# Team portfolio valuation and filtering.
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
    history = cached_history_value(day, symbols, data).set_index("ticker")
    info["history_n"] = history.valid_months
    # All three portfolio methods require positive, usable valuation bands.
    usable = history.status.eq("ok") & history.low.gt(0)
    for target, source in (("hist_low", "low"), ("hist_mid", "fair_value"), ("hist_high", "high")):
        info[target] = history[source].where(usable)
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
