"""Point-in-time three-method valuation. All parameters are research assumptions.

Course DCF engine is unchanged. No data access except supplied DataView.
Missing valuations are retained. Filtering requires two methods, modest
method disagreement and price above the union of their upper estimates.
"""
import numpy as np
import pandas as pd
from valuation.finrobot import DCFInputs, calculate_dcf, calculate_sensitivity, adjust_beta_blume

COLUMNS = ['ticker', 'fair_value', 'low', 'high', 'mispricing', 'method_spread']
CONFIG = dict(erp=.05, credit_spread=.02, terminal_growth=.025, years=10,
              history_months=36, min_history=12, min_peers=5,
              min_beta_weeks=52, growth_floor=-.10, growth_cap=.30,
              max_method_spread=3.0, minimum_methods=2, candidate_limit=None)
_CACHE = {}


def _four(f):
    q = f.sort_values('datadate').tail(4)
    if len(q) != 4:
        return None
    labels = [str(x)[:10] for x in q.datadate]
    quarters = [int(x[:4])*4+(int(x[5:7])-1)//3 for x in labels]
    return q if all(b-a == 1 for a,b in zip(quarters,quarters[1:])) else None


def _finite(x):
    return bool(np.isfinite(x))


def _bounded(x, lo, hi):
    if not _finite(x):
        raise ValueError('missing input')
    return float(np.clip(x, lo, hi))


def _available_dates(f):
    # Revisions returned by the current view are NOT backfilled into earlier
    # history. Missing earlier vintages reduce coverage instead.
    accepted = pd.to_datetime(f.accepted_date, errors='coerce')
    filed = pd.to_datetime(f.filing_date, errors='coerce')
    lag = pd.to_datetime(f.datadate) + pd.offsets.QuarterEnd(0) + pd.DateOffset(months=2, days=1)
    return accepted.fillna(filed).fillna(lag)


def _context(as_of, data):
    key = (data, str(as_of))
    # Cache only exact decision dates; never reuse a later-date result.
    if key in _CACHE:
        return _CACHE[key]
    members = data.members()
    f = data.fundamentals(sorted(members)).sort_values(['ticker', 'datadate'])
    groups = {t: g.copy() for t, g in f.groupby('ticker')}
    prices = data.last_close(sorted(members))
    peer_rows = []
    last = f.groupby('ticker', sort=False).tail(4).copy()
    last['_quarter'] = pd.PeriodIndex(pd.to_datetime(last.datadate), freq='Q').asi8
    agg = last.groupby('ticker').agg(n=('EPS','size'), valid=('EPS','count'),
                                    eps=('EPS','sum'), sector=('gsector','last'),
                                    first=('_quarter','min'), final=('_quarter','max'))
    for t, r in agg.iterrows():
        if r.n != 4 or r.valid != 4 or r.final-r['first'] != 3:
            continue
        price = prices.get(t, np.nan)
        multiple = price / r.eps if r.eps > 0 else np.nan
        if _finite(multiple) and 0 < multiple < 200:
            peer_rows.append((t, str(r.sector), multiple))
    ctx = (groups, prices, peer_rows)
    if len(_CACHE) >= 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = ctx
    return ctx


def _dcf(t, f, q, data):
    prev = _four(f.sort_values('datadate').iloc[:-4])
    if prev is None:
        raise ValueError('need eight consecutive quarters')
    if q.revenue_per_share.isna().any() or prev.revenue_per_share.isna().any():
        raise ValueError('missing revenue')
    rev, old = float(q.revenue_per_share.sum()), float(prev.revenue_per_share.sum())
    if rev <= 0 or old <= 0:
        raise ValueError('nonpositive revenue')
    if (pd.Period(q.datadate.iloc[0], freq='Q').ordinal -
            pd.Period(prev.datadate.iloc[-1], freq='Q').ordinal) != 1:
        raise ValueError('gap in eight-quarter growth input')
    growth = _bounded(rev/old-1, CONFIG['growth_floor'], CONFIG['growth_cap'])
    px = data.close([t, 'SPY'], lookback=1260).dropna()
    wk = px.resample('W-FRI').last().pct_change().dropna()
    if len(wk) < CONFIG['min_beta_weeks'] or wk.SPY.var() <= 0:
        raise ValueError('insufficient beta history')
    raw_beta = float(wk[t].cov(wk.SPY)/wk.SPY.var())
    rf_series = data.macro('DGS10').dropna()
    if rf_series.empty:
        raise ValueError('missing risk free rate')
    rf = _bounded(float(rf_series.iloc[-1])/100, 0, .15)
    required = ['ebitda_margin', 'operating_margin', 'capex_per_share',
                'effective_tax_rate', 'interest_debt_per_share', 'cash_per_share', 'debt_to_capital']
    if q[required].isna().any().any():
        raise ValueError('missing DCF financial input')
    margin = float(q.ebitda_margin.median())
    if margin <= 0:
        raise ValueError('nonpositive median EBITDA margin')
    inputs = DCFInputs(
        revenue_base=rev, revenue_growth_rates=list(np.linspace(growth, CONFIG['terminal_growth'], CONFIG['years'])),
        ebitda_margin=_bounded(margin, 0, 1),
        da_pct_revenue=_bounded((q.ebitda_margin-q.operating_margin).median(), 0, .5),
        capex_pct_revenue=_bounded(q.capex_per_share.abs().sum()/rev, 0, 1),
        nwc_pct_revenue=0., terminal_nwc_pct_revenue=0.,
        tax_rate=_bounded(q.effective_tax_rate.median(), 0, .35),
        risk_free_rate=rf, beta=_bounded(adjust_beta_blume(raw_beta), 0, 5),
        equity_risk_premium=CONFIG['erp'], cost_of_debt=_bounded(rf+CONFIG['credit_spread'], 0, .2),
        debt_ratio=_bounded(q.debt_to_capital.iloc[-1], 0, .95),
        terminal_growth_rate=CONFIG['terminal_growth'], shares_outstanding=1.,
        net_debt=float(q.interest_debt_per_share.iloc[-1]-q.cash_per_share.iloc[-1]))
    result = calculate_dcf(inputs)
    grid = calculate_sensitivity(inputs, [result.wacc-.01, result.wacc, result.wacc+.01], [.02,.025,.03])
    cells = [x for row in grid['implied_prices'] for x in row if x is not None and _finite(x) and x > 0]
    return dict(mid=result.implied_price, low=min(cells), high=max(cells),
                beta_weeks=len(wk), growth=growth, wacc=result.wacc,
                terminal_share=result.pv_terminal/result.enterprise_value,
                actual_ebitda_margin=float((q.revenue_per_share*q.ebitda_margin).sum()/rev),
                forecast_ebitda_margin=margin)


def _history(t, f, q, data, as_of):
    px = data.close([t], lookback=1260)[t].dropna()
    known = _available_dates(f).to_numpy(dtype='datetime64[ns]')
    dated = pd.to_datetime(f.datadate).to_numpy(dtype='datetime64[ns]')
    ordinals = pd.PeriodIndex(pd.to_datetime(f.datadate), freq='Q').asi8
    earnings = (f.revenue_per_share*f.ebitda_margin).to_numpy(dtype=float)
    debts = (f.interest_debt_per_share-f.cash_per_share).to_numpy(dtype=float)
    multiples = []
    dates = pd.date_range(end=pd.Timestamp(as_of), periods=CONFIG['history_months'], freq='ME')
    # Array arithmetic is equivalent to filtering each snapshot separately;
    # current filing versions are only admitted after their known date.
    for d in dates:
        stamp = d.to_datetime64()
        ids = np.flatnonzero((dated <= stamp) & (known <= stamp))[-4:]
        pos = px.index.searchsorted(d, side='right')-1
        if len(ids) != 4 or pos < 0 or not np.all(np.diff(ordinals[ids]) == 1):
            continue
        ebitda = float(earnings[ids].sum())
        debt = float(debts[ids[-1]])
        v = (float(px.iloc[pos])+debt)/ebitda if ebitda > 0 else np.nan
        if _finite(v) and v > 0:
            multiples.append(v)
    if len(multiples) < CONFIG['min_history']:
        raise ValueError('insufficient historical multiple snapshots')
    if q[['revenue_per_share','ebitda_margin']].isna().any().any():
        raise ValueError('missing EBITDA')
    current = float((q.revenue_per_share*q.ebitda_margin).sum())
    debt = float(q.interest_debt_per_share.iloc[-1]-q.cash_per_share.iloc[-1])
    if current <= 0:
        raise ValueError('nonpositive TTM EBITDA')
    lo, mid, hi = np.quantile(multiples, [.25,.5,.75])*current-debt
    if not all(_finite(x) and x > 0 for x in [lo,mid,hi]):
        raise ValueError('nonpositive historical-band equity')
    return dict(low=float(lo),mid=float(mid),high=float(hi),observations=len(multiples))


def audit_value(as_of, tickers, data):
    groups, prices, peers = _context(as_of, data)
    records, audit = [], []
    for t in sorted(set(tickers)):
        f = groups.get(t)
        price = prices.get(t, np.nan)
        methods, reasons = {}, {}
        q = None if f is None else _four(f)
        if q is not None and _finite(price) and price > 0:
            for name, fn in [('dcf', lambda:_dcf(t,f,q,data)),
                             ('history', lambda:_history(t,f,q,data,as_of))]:
                try:
                    methods[name] = fn()
                except (ValueError, ArithmeticError, IndexError, KeyError) as exc:
                    reasons[name] = str(exc)
            sector = str(q.gsector.iloc[-1])
            # Exclude target itself; same-sector members known on decision date.
            pm = [m for sym,s,m in peers if sym != t and s == sector]
            if len(pm) >= CONFIG['min_peers'] and q.EPS.notna().all() and q.EPS.sum() > 0:
                lo, mid, hi = np.quantile(pm,[.25,.5,.75])*float(q.EPS.sum())
                methods['peer'] = dict(low=float(lo),mid=float(mid),high=float(hi),peers=len(pm))
            else:
                reasons['peer'] = 'nonpositive/missing EPS or too few peers'
        else:
            reasons['input'] = 'incomplete four quarters, missing price or nonmember'
        valid = list(methods.values())
        record = None
        if len(valid) >= CONFIG['minimum_methods']:
            mids = [m['mid'] for m in valid]
            mid = float(np.median(mids))
            record = dict(ticker=t, fair_value=mid, low=min(m['low'] for m in valid),
                          high=max(m['high'] for m in valid), mispricing=mid/price-1,
                          method_spread=max(mids)/min(mids))
            records.append(record)
        audit.append(dict(ticker=t,price=price,methods=methods,reasons=reasons,aggregate=record))
    return pd.DataFrame(records, columns=COLUMNS), audit


def value(as_of, tickers, data):
    return audit_value(as_of,tickers,data)[0]


def filter_candidates(as_of, candidates, data):
    # Extended study: assess every growth-tech candidate supplied this week.
    ranked = sorted((p for p in candidates if p.bucket == 'growth_tech'),
                    key=lambda p: p.rank_score, reverse=True)
    tickers = [p.symbol for p in ranked[:CONFIG['candidate_limit']]]
    values = value(as_of,tickers,data).set_index('ticker')
    # Conservative agreement filter: abstain under missingness/disagreement.
    removed = {t for t,r in values.iterrows()
               if r.method_spread <= CONFIG['max_method_spread']
               and r.fair_value/(1+r.mispricing) > r.high}
    return tuple(p for p in candidates if p.symbol not in removed)
