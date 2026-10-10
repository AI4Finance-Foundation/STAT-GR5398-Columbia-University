"""Trace sampled model inputs back to supplied financial payloads, read-only."""
import argparse,json,sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(a.cache/'fundamental_data_full.csv',parse_dates=['datadate','filing_date','accepted_date','actual_tradedate'])
con=sqlite3.connect(a.cache/'finrl_trading.db');data=pd.read_sql_query("select ticker,payload,row_json from raw_payloads where payload!='profile'",con);rows=[]
for rec in data.itertuples():
 x=json.loads(rec.row_json);period=str(x.get('period','')).upper();year=x.get('calendarYear');period='Q4' if period=='FY' else period
 try:key=pd.Period(str(year)+period).end_time.normalize() if year and period in ['Q1','Q2','Q3','Q4'] else pd.to_datetime(x['date']).to_period('Q').end_time.normalize()
 except Exception:continue
 rows.append({'ticker':rec.ticker,'datadate':key,'payload':rec.payload,'fiscal_end':pd.to_datetime(x.get('date'),errors='coerce'),'filed':pd.to_datetime(x.get('fillingDate') or x.get('filingDate'),errors='coerce'),'accepted':pd.to_datetime(x.get('acceptedDate'),errors='coerce'),'x':x})
r=pd.DataFrame(rows)
g=r.groupby(['ticker','datadate']).agg(fiscal_min=('fiscal_end','min'),fiscal_max=('fiscal_end','max'),payload_count=('payload','nunique')).reset_index();g['fiscal_span_days']=(g.fiscal_max-g.fiscal_min).dt.days
g.to_csv(a.output/'payload_period_alignment.csv',index=False)
# Deterministic sample fixed before new returns: tech leaders plus known fiscal anomalies and restored symbols.
tickers=['AAPL','MSFT','NVDA','AMZN','META','AVGO','TSLA','A','COST','WMT','BRK.B','BF.B','ABC','ANTM','AAL','AAP']
checks=[]
for t in tickers:
 sub=f.loc[(f.ticker==t)&f.datadate.between('2020-01-01','2025-06-30')]
 for z in sub.iloc[::max(1,len(sub)//3)].head(3).itertuples():
  anchor=r.loc[(r.ticker==t)&(r.payload=='income')&(r.filed.dt.normalize()==z.filing_date.normalize())];anchor_end=anchor.fiscal_end.max() if len(anchor) else pd.NaT
  rr=r.loc[(r.ticker==t)&(r.fiscal_end==anchor_end)] if pd.notna(anchor_end) else r.iloc[0:0];inc=rr.loc[rr.payload=='income'];bal=rr.loc[rr.payload=='balance'];ratio=rr.loc[rr.payload=='ratios']
  dic={w.payload:w.x for w in rr.itertuples()};i=dic.get('income',{});b=dic.get('balance',{});v=dic.get('ratios',{})
  available=pd.Series([z.filing_date,z.accepted_date]+rr.filed.tolist()+rr.accepted.tolist()+rr.fiscal_end.tolist()).max()
  record={'ticker':t,'datadate':z.datadate,'input_filing_date':z.filing_date,'input_accepted_date':z.accepted_date,'trade_date':z.actual_tradedate,'matched_payloads':len(rr),'fiscal_dates':';'.join(sorted(set(rr.fiscal_end.dropna().astype(str)))),'max_recorded_date':available,'known_before_trade':bool(available<z.actual_tradedate),'EPS_input':z.EPS,'EPS_raw':i.get('eps'),'pe_input':z.pe,'pe_raw':v.get('priceEarningsRatio'),'cur_ratio_input':z.cur_ratio,'cur_ratio_raw':v.get('currentRatio')}
  for name in ['EPS','pe','cur_ratio']:
   left=record[name+'_input'];right=record[name+'_raw'];record[name+'_match']= bool(np.isclose(left,float(right),rtol=1e-7,atol=1e-8)) if right is not None and pd.notna(left) else None
  checks.append(record)
s=pd.DataFrame(checks)
px=pd.read_csv(a.cache/'prepared/daily.csv',usecols=['tic','datadate','prccd'],parse_dates=['datadate'])
s['quarter_price']=np.nan
for ix,record in s.iterrows():
 q=px.loc[(px.tic==record.ticker)&(px.datadate>=pd.Timestamp(record.datadate))].sort_values('datadate')
 if len(q):s.at[ix,'quarter_price']=q.iloc[0].prccd
s['pe_times_eps']=s.pe_input*s.EPS_input
s['pe_price_relative_difference']=s.pe_times_eps/s.quarter_price-1
s.to_csv(a.output/'financial_spotchecks.csv',index=False)
summary={'sample_tickers':tickers,'sample_rows':len(s),'matched_raw_payload_rows':int((s.matched_payloads>0).sum()),'sample_with_date_after_trade':int((~s.known_before_trade).sum()),'all_matched_buckets':len(g),'buckets_with_fiscal_span_over_7_days':int((g.fiscal_span_days>7).sum()),'feature_matches':{k:dict(compared=int(s[k+'_match'].notna().sum()),match=int(s[k+'_match'].fillna(False).sum())) for k in ['EPS','pe','cur_ratio']},'PE_definition':'supplied PE is quarter-date price divided by quarterly EPS; not the raw API PE definition', 'pe_quarter_price_checks':int(s.pe_price_relative_difference.notna().sum()), 'pe_quarter_price_max_relative_error':float(s.pe_price_relative_difference.abs().max()), 'raw_join_correction':'match filing_date to income, then common fiscal_end; fiscal quarter is not calendar quarter', 'vintages_verified':False,'interpretation':'Checks concern supplied snapshot identity/mapping and timestamps, not original as-published values. Missing raw payload coverage is retained as an explicit risk.'}
(a.output/'financial_spotcheck_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
