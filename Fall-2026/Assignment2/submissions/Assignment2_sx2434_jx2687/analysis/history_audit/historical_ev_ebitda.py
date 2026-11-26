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
    return (result, history) if return_history else result
