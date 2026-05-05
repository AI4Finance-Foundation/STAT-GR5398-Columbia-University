from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

def weekly_returns(ret,freq): return (1+ret.fillna(0)).resample(freq).prod()-1

def regime_state(prices,dates,window):
    spy=prices["SPY"].ffill(); ma=spy.rolling(window,min_periods=max(20,window//2)).mean()
    return (spy>ma).map({True:"risk_on",False:"risk_off"}).reindex(dates,method="ffill").fillna("risk_off")

def group_information_ratios(wret,groups,asof,lookback,benchmark):
    hist=wret.loc[:asof].iloc[:-1].tail(lookback); out={}
    for g,ts in groups.items():
        valid=[t for t in ts if t in hist.columns]
        if hist.empty or not valid: out[g]=np.nan; continue
        gr=hist[valid].mean(axis=1); ex=gr if benchmark=="zero" else gr-hist.get(benchmark,pd.Series(0,index=hist.index)).fillna(0)
        sd=ex.std(ddof=1); out[g]=float(ex.mean()/sd) if sd and sd>1e-12 else np.nan
    return out

def residual_momentum_scores(ret,assets,asof,lookback):
    h=ret.loc[:asof].iloc[:-1].tail(lookback); spy=pd.to_numeric(h.get("SPY"),errors="coerce"); out={}
    for a in assets:
        if a not in h: continue
        xy=pd.concat([pd.to_numeric(h[a],errors="coerce").rename("a"),spy.rename("s")],axis=1).dropna()
        if len(xy)<max(20,lookback//3): continue
        x=xy.s.to_numpy(float); y=xy.a.to_numpy(float); var=float(((x-x.mean())**2).sum()/max(len(x)-1,1)); cov=float(((y-y.mean())*(x-x.mean())).sum()/max(len(x)-1,1)); beta=cov/var if var>1e-12 else 0
        out[a]=float(np.prod(1+(y-beta*x))-1)
    return out

def robust_z(s):
    x=pd.to_numeric(s,errors="coerce")
    if x.notna().sum()<2: return pd.Series(0,index=s.index)
    x=x.clip(x.quantile(.05),x.quantile(.95)); sd=x.std(ddof=1)
    return ((x-x.mean())/sd).fillna(0) if sd>1e-12 else pd.Series(0,index=s.index)

def qqq_rolling_candidates(prices,asof,candidate_assets,windows,top_k,assignment1_selected_path:Path|None=None):
    h=prices.loc[:asof].iloc[:-1]; scores={}
    for t in candidate_assets:
        if t not in h: continue
        s=h[t].dropna(); vals=[float(s.iloc[-1]/s.iloc[-int(w)]-1) for w in windows if len(s)>int(w)]
        if vals: scores[t]=float(np.mean(vals))
    if assignment1_selected_path and assignment1_selected_path.exists():
        sel=pd.read_csv(assignment1_selected_path,parse_dates=["trade_date"])
        sel=sel[(sel.trade_date<asof)&(sel.tic.isin(candidate_assets))]
        if not sel.empty:
            z=robust_z(sel.sort_values("trade_date").groupby("tic").tail(1).set_index("tic")["predicted_return"])
            for t,v in z.items(): scores[t]=scores.get(t,0)+0.25*float(v)
    ranked=sorted(scores,key=lambda t:scores[t],reverse=True)
    return ranked[:top_k],scores
