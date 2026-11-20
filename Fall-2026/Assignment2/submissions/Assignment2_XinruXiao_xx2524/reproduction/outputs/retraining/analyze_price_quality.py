"""Audit raw quote freshness and strict horizon endpoints; never changes a decision."""
import argparse,sys
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();sys.path.insert(0,str(r/'work/retraining/baseline'))
from shared.prices import PriceStore
store=PriceStore(r/'work/course/Fall-2026/Assignment2/source_code/data/prices/daily_adjusted.parquet');sessions=store.sessions;px=store.close.reindex(sessions)
valid_positions={t:np.flatnonzero(px[t].notna().to_numpy()) for t in px};records=[];summary=[]
for variant in ['course','retrained_baseline','retrained_improved']:
 out=r/'work/retraining/evaluation'/variant/'output'
 for period in ['dev','val','test','forward']:
  for scope in ['top20','all_growth']:
   f=out/f'valuation_eval_{period}_{scope}.csv'
   if not f.exists():continue
   values=pd.read_csv(f);aud=pd.read_csv(out/f'monthly_input_audit_{period}_{scope}.csv')[['as_of','ticker','price']].rename(columns={'price':'decision_price'})
   d=values.merge(aud,on=['as_of','ticker'],validate='one_to_one');current=[]
   for row in d.to_dict('records'):
    pos=int(sessions.searchsorted(pd.Timestamp(row['as_of'])));t=row['ticker'];valid=valid_positions.get(t,np.array([],dtype=int));past=valid[valid<=pos];last=int(past[-1]) if len(past) else None
    rec={'variant':variant,'period':period,'scope':scope,'as_of':row['as_of'],'ticker':t,'mispricing':row['mispricing'],'decision_price':row['decision_price'],'raw_current_price':row['price'],'quote_age_sessions':pos-last if last is not None else None,'quote_age_calendar_days':int((sessions[pos]-sessions[last]).days) if last is not None else None,'decision_price_consistent':bool(np.isclose(row['mispricing'],row['fair_value']/row['decision_price']-1,rtol=1e-8,atol=1e-10))}
    for h in [63,126,252]:
     mature=pos+h<len(sessions);current_raw=px[t].iloc[pos] if t in px else np.nan;endpoint=px[t].iloc[pos+h] if mature and t in px else np.nan
     strict=bool(mature and pd.notna(current_raw) and pd.notna(endpoint) and current_raw>0)
     rec[f'global_mature_{h}']=mature;rec[f'strict_endpoint_{h}']=strict;rec[f'strict_return_{h}']=endpoint/current_raw-1 if strict else np.nan
     rec[f'official_return_{h}']=row.get(f'ret_{h}',np.nan)
     rec[f'missing_endpoint_with_official_return_{h}']=bool(mature and pd.isna(endpoint) and pd.notna(rec[f'official_return_{h}']))
    current.append(rec);records.append(rec)
   q=pd.DataFrame(current);assert q.decision_price_consistent.all()
   s={'variant':variant,'period':period,'scope':scope,'valued_records':len(q),'stale_asof_records':int(q.quote_age_sessions.gt(0).sum()),'max_quote_age_sessions':q.quote_age_sessions.max(),'max_quote_age_calendar_days':q.quote_age_calendar_days.max()}
   for h in [63,126,252]:
    s[f'global_mature_n_{h}']=int(q[f'global_mature_{h}'].sum());s[f'strict_endpoint_n_{h}']=int(q[f'strict_endpoint_{h}'].sum());s[f'missing_endpoint_with_official_return_n_{h}']=int(q[f'missing_endpoint_with_official_return_{h}'].sum())
    sub=q[q[f'strict_endpoint_{h}']];ics=[g.mispricing.corr(g[f'strict_return_{h}'],method='spearman') for _,g in sub.groupby('as_of') if len(g)>=5 and g.mispricing.nunique()>1 and g[f'strict_return_{h}'].nunique()>1]
    s[f'strict_mean_monthly_ic_{h}']=float(np.nanmean(ics)) if ics else None
   summary.append(s)
dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True);pd.DataFrame(records).to_csv(dest/'price_endpoint_audit.csv',index=False);pd.DataFrame(summary).to_csv(dest/'price_endpoint_summary.csv',index=False)
print(pd.DataFrame(summary).query('scope=="all_growth"')[['variant','period','valued_records','stale_asof_records','missing_endpoint_with_official_return_n_252']].to_string(index=False))
