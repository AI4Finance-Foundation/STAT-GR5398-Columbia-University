"""Attempt public recovery of missing symbols; never silently add them to returns.

Symbol identity, corporate actions, and terminal delisting proceeds still require
verification. The raw downloads are diagnostic and excluded from submission ZIPs.
"""
import argparse,json,time
from pathlib import Path
import pandas as pd
import yfinance as yf

ap=argparse.ArgumentParser();ap.add_argument('--exclusions',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
tickers=sorted(pd.read_csv(a.exclusions).ticker.unique())
mapping={t: {'BRK.B':'BRK-B','BF.B':'BF-B'}.get(t,t) for t in tickers}
records=[]
for j in range(0,len(tickers),8):
    batch=tickers[j:j+8]
    try:
        data=yf.download([mapping[t] for t in batch],start='2015-06-15',end='2025-10-01',
            auto_adjust=True,group_by='ticker',threads=4,progress=False,timeout=12)
    except Exception as exc:
        print('batch failure',str(exc),flush=True);data=pd.DataFrame()
    for t in batch:
        symbol=mapping[t]
        try:
            frame=data[symbol].dropna(subset=['Close'])
        except (KeyError,TypeError): frame=pd.DataFrame()
        rec=dict(original_ticker=t,provider_symbol=symbol,rows=len(frame),status='quotes_returned' if len(frame) else 'unavailable',
                 first_date=str(frame.index.min().date()) if len(frame) else '',last_date=str(frame.index.max().date()) if len(frame) else '',
                 issuer_identity_verified=False,delisting_proceeds_verified=False)
        if len(frame): frame.to_csv(a.output/(symbol+'.csv'))
        records.append(rec)
    pd.DataFrame(records).to_csv(a.output/'recovery_inventory.csv',index=False)
    print(json.dumps(dict(completed=len(records),total=len(tickers),quoted=sum(x['rows']>0 for x in records))),flush=True)
(a.output/'provenance.json').write_text(json.dumps(dict(provider='Yahoo Finance via yfinance',downloaded_at=pd.Timestamp.now(tz='America/New_York').isoformat(),start='2015-06-15',end_exclusive='2025-10-01',auto_adjust=True,used_in_primary_backtest=False),indent=2))
