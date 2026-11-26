"""Create a teammate reconciliation table and validation diagnostics with fully matured labels."""
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_valuation import top_candidates,check_frame
from selection import MonthlyScores
from shared.data_view import DataBundle,DataView
from shared.guard import student_call
from valuation.student_valuation import value,valuations
out=Path("../results")
bundle=DataBundle.load()
scores=MonthlyScores()
for dt,prior,names in [("2019-08-05","2019-08-02",None),("2021-06-30","2021-06-30",["MSFT"])]:
    view=DataView(bundle,prior)
    names=names or top_candidates(scores,pd.Timestamp(dt).date(),view.members(),20)
    with student_call():
        result=valuations(prior,names,view).set_index("ticker").reindex(names)
        hist=view.extra("history_monthly")
        hist=hist[(hist.date<pd.Timestamp(prior))&(hist.date>=pd.Timestamp(prior)-pd.DateOffset(years=5))]
        f=view.fundamentals(names).sort_values(["ticker","datadate"])
    quartiles=hist.groupby("ticker").multiple.quantile([.25,.5,.75]).unstack()
    diag=result[["price","hist_low","hist_mid","hist_high"]].join(quartiles.rename(columns={.25:"p25",.5:"p50",.75:"p75"}))
    diag["valid_months"]=hist.groupby("ticker").size().reindex(names,fill_value=0)
    for t in names:
        q=f[f.ticker==t].tail(4)
        if pd.notna(diag.at[t,"hist_mid"]): reason="valued"
        elif len(q)<4 or not np.all(np.diff(pd.PeriodIndex(q.datadate,freq="Q").asi8)==1): reason="insufficient consecutive quarters"
        elif diag.at[t,"valid_months"]<12: reason="fewer than 12 valid historical month ends"
        elif q[["ebitda_margin","revenue_per_share"]].isna().any().any(): reason="missing quarterly inputs"
        elif (q.ebitda_margin*q.revenue_per_share).sum()<=0: reason="nonpositive current TTM EBITDA"
        else: reason="missing price/debt/cash or invalid equity band"
        diag.loc[t,"status"]=reason
    diag.insert(0,"as_of",prior)
    diag.to_csv(out/f"reference_{dt}.csv")
    print(dt,len(diag),diag.status.value_counts().to_dict())
cal=bundle.prices.sessions
close=bundle.prices.close.loc[:"2023-12-29"]
months=cal[(cal>="2022-01-03")&(cal<="2023-12-29")].to_series()
rows=[]
asked=0
for day in months.groupby(months.dt.to_period("M")).first():
    prior=bundle.prices.last_session_on_or_before(day.date()-timedelta(days=1))
    view=DataView(bundle,prior);tickers=top_candidates(scores,day.date(),view.members(),20)
    asked+=len(tickers)
    with student_call():
        d=value(prior,tickers,view)
    d=check_frame(d,prior,tickers,view).set_index("ticker")
    d["as_of"]=prior
    idx=close.index.get_loc(pd.Timestamp(prior))
    for h in (63,126,252):
        d[f"ret_{h}"]=close.iloc[idx+h].reindex(d.index)/close.iloc[idx].reindex(d.index)-1 if idx+h<len(close) else np.nan
    rows.append(d.reset_index())
records=pd.concat(rows,ignore_index=True)
records.to_csv(out/"val_matured_signals.csv",index=False)
print("valued",len(records),"of",asked)
for h in (63,126,252):
    samp=records.dropna(subset=[f"ret_{h}"])
    ic=[g.mispricing.corr(g[f"ret_{h}"],method="spearman") for _,g in samp.groupby("as_of") if len(g)>=5 and g.mispricing.nunique()>1 and g[f"ret_{h}"].nunique()>1]
    print(h,"mature",len(samp),"months",len(ic),"mean monthly IC",np.mean(ic))
