import numpy as np
def equity_curve(r): return (1+r.fillna(0)).cumprod()
def drawdown(eq): return eq/eq.cummax()-1
def performance_metrics(r,rebalances=0,avg_turnover=0):
    r=r.dropna(); eq=equity_curve(r); yrs=max((r.index[-1]-r.index[0]).days/365.25,1e-9)
    cum=float(eq.iloc[-1]-1); cagr=float(eq.iloc[-1]**(1/yrs)-1); vol=float(r.std(ddof=1)*np.sqrt(252)); mdd=float(drawdown(eq).min())
    return {"cumulative_return":cum,"annualized_return":cagr,"annualized_volatility":vol,"sharpe_ratio":cagr/vol if vol>1e-12 else np.nan,"max_drawdown":mdd,"cagr":cagr,"calmar_ratio":cagr/abs(mdd) if mdd<0 else np.nan,"number_of_rebalances":float(rebalances),"average_turnover":float(avg_turnover),"hit_rate":float((r>0).mean())}
