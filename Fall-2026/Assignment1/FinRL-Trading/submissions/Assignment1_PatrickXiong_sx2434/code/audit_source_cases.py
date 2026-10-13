"""Exhaustive frozen-source event shape, universe, and future-prefix checks."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
from build_features import calculate,FEATURES
ROOT=Path(__file__).resolve().parents[1]
INP=Path(os.environ.get('FINRL_INPUT_DIR',str(ROOT/'inputs')))
DATA=Path(os.environ.get('FINRL_DATA_DIR',str(ROOT/'data-source')))
RUN=Path(os.environ.get('FINRL_RUN_DIR',str(ROOT/'run')))
d=pd.read_parquet(INP/'crsp_daily_raw.parquet').drop_duplicates(['permno','dlycaldt'])
panel=pd.read_csv(DATA/'model_panel.csv.gz',parse_dates=['decision_date','label_end'])
features=pd.read_csv(RUN/'price_features_candidate.csv.gz',parse_dates=['decision_date'],float_precision='round_trip')
pred=pd.read_csv(RUN/'predictions.csv',parse_dates=['decision_date'])
events=pd.read_csv(DATA/'event_recognition.csv',parse_dates=['crsp_return_date','recognition_date','execution_date'])
m=pd.read_csv(INP/'crsp_sp500_members.csv.gz',usecols=['PERMNO','DlyCalDt','MbrStartDt','MbrEndDt']);m.columns=m.columns.str.lower()
for c in ['dlycaldt','mbrstartdt','mbrenddt']:m[c]=pd.to_datetime(m[c])
spells=m[['permno','mbrstartdt','mbrenddt']].drop_duplicates()
calendar=pd.DatetimeIndex(d[d.permno.eq(84398)].dlycaldt.sort_values().unique())
shapes=[]
for ev in events.itertuples(index=False):
 q=d[d.permno.eq(ev.permno)].set_index('dlycaldt').sort_index()
 after=q[q.index>=ev.crsp_return_date]
 exact=q.loc[ev.crsp_return_date] if ev.crsp_return_date in q.index else None
 row=dict(permno=ev.permno,source_date=ev.crsp_return_date,recognition_date=ev.recognition_date,execution_date=ev.execution_date,
   source_flag=exact.dlyretmissflg if exact is not None else None,
   finite_return_rows_at_or_after_source=int(after.dlyret.notna().sum()),
   finite_return_rows_after_source=int(after.loc[after.index>ev.crsp_return_date,'dlyret'].notna().sum()),
   reconstruction=bool(ev.reconstructed_from_dated_quote),unknown_proceeds=bool(ev.unknown_complete_proceeds))
 assert row['finite_return_rows_after_source']==0,row
 assert row['source_flag'] not in ['NT','MP'],row
 assert pd.isna(ev.execution_date) or ev.execution_date>=ev.recognition_date
 shapes.append(row)
pd.DataFrame(shapes).to_csv(RUN/'source_event_shape_audit.csv',index=False)
counts=[];prefix_differences=[]
by={p:g.set_index('dlycaldt').sort_index() for p,g in d.groupby('permno')}
for day,g in panel.groupby('decision_date'):
 expected=set(spells[spells.mbrstartdt.le(day)&(spells.mbrenddt.ge(day)|spells.mbrenddt.isna())].permno)
 assert set(g.permno)==expected
 inf=pred[pred.decision_date.eq(day)]
 if day>=pd.Timestamp('2020-06-01'):assert set(inf.security_id)==set(g.security_id)
 counts.append(dict(date=day,official_effective_members=len(expected),panel_rows=len(g),prediction_rows=len(inf),retention_complete=True))
 fday=features[features.decision_date.eq(day)].set_index('security_id')
 for x in g.itertuples(index=False):
  q=by[x.permno].loc[lambda z:z.index<day].copy() # raw future rows deleted BEFORE wealth
  r=q.dlyret.reindex(calendar[calendar<day]).copy()
  r.loc[q.index[q.dlyretmissflg.eq('NT')]]=0.
  # MP remains NaN: ratios require complete windows. No future resumption
  # quote, terminal payout date, or last-record flag enters this feature path.
  if len(q) and q.iloc[0].dlyretmissflg=='NS':r.at[q.index[0]]=0.
  ev=events[events.permno.eq(x.permno)]
  if len(ev) and pd.notna(ev.iloc[0].execution_date) and ev.iloc[0].execution_date<day:
   event=ev.iloc[0]
   r.loc[(r.index>=event.crsp_return_date)&(r.index<event.execution_date)]=0.
   r.at[event.execution_date]=event.terminal_return;r.loc[r.index>event.execution_date]=0.
  wealth=(1+r).cumprod(skipna=True)*100
  wealth.loc[q.index[q.dlyretmissflg.eq('MP')]]=np.nan
  if len(q):wealth.loc[wealth.index<q.index[0]]=np.nan
  v,_=calculate(pd.DataFrame({'adj_close':wealth,'source':'CRSP_CIZ'}),calendar,day)
  actual=fday.loc[x.security_id,FEATURES].to_numpy(float);check=np.array([v[k] for k in FEATURES])
  assert np.array_equal(np.isnan(actual),np.isnan(check)),(x.security_id,str(day),'missingness')
  delta=float(np.nanmax(np.abs(actual-check))) if np.isfinite(actual).any() else 0.
  assert delta<1e-12,(x.security_id,str(day),delta)
  prefix_differences.append(delta)
pd.DataFrame(counts).to_csv(RUN/'universe_retention_checks.csv',index=False)
panel.assign(year=panel.decision_date.dt.year).groupby(['year','label_status']).size().rename('cases').to_csv(RUN/'label_status_by_year.csv')
sel=pd.read_csv(RUN/'backtest/SPY80_Best20_quarterly_selected.csv',parse_dates=['decision_date'])
sel.merge(panel[['permno','decision_date','label_status']],on=['permno','decision_date'],validate='one_to_one').groupby('label_status').size().rename('selected_cases').to_csv(RUN/'selected_label_status_counts.csv')
post=panel.merge(events[['permno','crsp_return_date']],on='permno')
post=post[post.decision_date.ge(post.crsp_return_date)]
assert post.target.isna().all()
status_rows=d[d.dlyclose.gt(0)&d.tradingstatusflg.ne('A')]
z=dict(positive_close_rows_excluded_by_nonactive_status=int(len(status_rows)),
 post_source_event_decision_cases=len(post),post_source_event_finite_labels=int(post.target.notna().sum()),
 official_membership_cases=len(panel),inference_cases=len(pred),inference_retention_complete=True,
 independent_raw_future_deletion_cases=len(prefix_differences),raw_future_deletion_max_difference=max(prefix_differences),
 raw_total_loss_records=int(d.dlyret.eq(-1).sum()),observed_total_loss_label_floor_cases=int(((panel.target<=np.log(1e-8)) & panel.label_status.eq('observed_CRSP_total_return')).sum()),
 source_events=len(shapes),source_events_with_return_after_source=0,source_events_masked_by_NT_MP=0,
 reconstructed_PRCF_no_cash_component_events=int(events.reconstructed_from_dated_quote.sum()),
 reconstruction_scope='Only PRCF price-of-subsequent-trade, zero/missing distribution, last pre-event price equals delisting mark.',
 non_session_recognition_in_sample=int((~events.recognition_date.isin(calendar)&events.recognition_date.le(calendar[-1])).sum()),
 events_recognized_after_sample=int(events.recognition_date.gt(calendar[-1]).sum()),
 universe_receipt_scope='All available official CRSP effective spells; source itself is retrospective and not certified against historical release vintages.')
(RUN/'source_case_checks.json').write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
