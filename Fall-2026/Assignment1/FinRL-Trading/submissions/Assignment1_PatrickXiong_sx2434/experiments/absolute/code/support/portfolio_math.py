import numpy as np
import pandas as pd

def metrics(r):
    nav=(1+r).cumprod();days=(r.index[-1]-r.index[0]).days
    sd=r.std(ddof=1)
    return dict(cumulative=float(nav.iloc[-1]-1),cagr=float(nav.iloc[-1]**(365.25/days)-1),
        volatility=float(sd*np.sqrt(252)),sharpe_rf0=float(r.mean()/sd*np.sqrt(252)) if sd else 0.,
        max_drawdown=float((nav/nav.cummax().clip(lower=1)-1).min()))


def finance_trade(drift,target,missing,fee):
    """Dollar holdings in units of pre-trade NAV, with costs funded by sales.

Frozen unquoted positions retain their dollar value, not just their weight.
The residual is cash if no quoted targets remain.
"""
    frozen=np.where(missing,drift,0.)
    tradable=np.where(missing,0.,target)
    total=tradable.sum()
    if total>0: tradable=tradable/total
    def plan(cost):
        return frozen+tradable*max(0.,1-cost-frozen.sum())
    lo=0.;hi=2*fee
    for _ in range(45):
        mid=(lo+hi)/2
        needed=fee*np.abs(plan(mid)-drift).sum()
        if needed>mid: lo=mid
        else: hi=mid
    cost=(lo+hi)/2;dollars=plan(cost);turnover=np.abs(dollars-drift).sum()
    assert abs(cost-fee*turnover)<1e-10
    weights=dollars/(1-cost)
    assert weights.sum()<=1+1e-10 and weights.min()>=-1e-12
    return weights,cost,turnover


