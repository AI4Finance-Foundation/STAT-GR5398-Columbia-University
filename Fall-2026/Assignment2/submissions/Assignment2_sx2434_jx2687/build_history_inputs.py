"""Export one point-in-time EV/EBITDA observation at each completed calendar month's last session."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from shared.data_view import DataBundle, DataView

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--through-month",required=True,help="Last complete calendar month (YYYY-MM)")
    end=pd.Period(parser.parse_args().through_month,freq="M")
    b=DataBundle.load()
    sessions=b.prices.sessions.to_series()
    # Require a later session to establish that the last requested month is complete.
    if sessions.iloc[-1].to_period("M")<=end:
        raise ValueError("Requested month has no subsequent trading month in the bundle")
    spy = b.prices.close["SPY"].dropna().index.to_series()
    days=spy[spy.dt.to_period("M")<=end]
    records=[]
    for day in days.groupby(days.dt.to_period("M")).last():
        view=DataView(b,day)
        f=view.fundamentals().sort_values(["ticker","datadate"]).groupby("ticker").tail(4).copy()
        if f.empty:
            continue
        f["q"]=pd.PeriodIndex(f.datadate,freq="Q").asi8
        f["ebitda"]=f.revenue_per_share*f.ebitda_margin
        released = pd.to_datetime(f.accepted_date, errors="coerce").fillna(pd.to_datetime(f.filing_date, errors="coerce"))
        f["eligible"] = (
            released.le(day) & pd.to_datetime(f.datadate).le(day)
            & np.isfinite(f.revenue_per_share) & np.isfinite(f.ebitda_margin)
        )
        g=f.groupby("ticker")
        last=g.tail(1).set_index("ticker")
        ebitda=g.ebitda.sum(min_count=4)
        price=view.close(last.index,lookback=1).iloc[-1]
        ratio=(price+last.interest_debt_per_share-last.cash_per_share)/ebitda
        use=g.eligible.all()&(g.size()==4)&((g.q.max()-g.q.min())==3)&(ebitda>0)&(price>0)&(ratio>0)&np.isfinite(ratio)
        records.append(ratio[use].rename_axis("ticker").rename("multiple").reset_index().assign(date=day))
    df=pd.concat(records,ignore_index=True)
    assert not df.duplicated(["date","ticker"]).any()
    df.to_parquet(Path("student_data")/"history_monthly.parquet",index=False)
    print(len(df),df.date.min(),df.date.max())
if __name__=="__main__":
    main()
