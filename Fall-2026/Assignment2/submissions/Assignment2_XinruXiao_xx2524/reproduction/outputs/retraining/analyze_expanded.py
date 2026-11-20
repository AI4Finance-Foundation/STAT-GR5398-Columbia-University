"""Deterministic summary of completed official evaluations; never invent missing runs."""
import argparse,json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True)
metrics=[];coverage=[];signals=[];reasons=[]
for variant in ['course','retrained_baseline','retrained_improved']:
 out=r/'work/retraining/evaluation'/variant/'output'
 for period in ['dev','val','test','forward']:
  files=list(out.glob('ablation_'+period+'_*.csv'))
  if files:
   d=pd.read_csv(files[0]);d.insert(0,'period',period);d.insert(0,'variant',variant);metrics.append(d)
  for scope in ['top20','all_growth']:
   f=out/f'monthly_input_audit_{period}_{scope}.csv'
   if f.exists():
    d=pd.read_csv(f);v=d.aggregate_valid.astype(str).str.lower().eq('true')
    row={'variant':variant,'period':period,'scope':scope,'candidate_months':len(d),'valued':int(v.sum()),'coverage':float(v.mean()),'distinct_tickers':d.ticker.nunique(),'as_of_count':d.as_of.nunique()}
    for method in ['dcf','history','peer']:row[method+'_valid']=int(d[method+'_valid'].astype(str).str.lower().eq('true').sum())
    coverage.append(row)
    for s in d.loc[~v,'reasons'].dropna():reasons.append({'variant':variant,'period':period,'scope':scope,'reasons':s})
   f=out/f'valuation_eval_{period}_{scope}.csv'
   if f.exists():
    d=pd.read_csv(f);row={'variant':variant,'period':period,'scope':scope,'valued_rows':len(d)}
    row['median_relative_interval_width']=float(((d.high-d.low)/d.fair_value).median()) if len(d) else None
    for n in [63,126,252]:
     col='ret_'+str(n)
     if col not in d:continue
     subset=d.dropna(subset=['mispricing',col]);row[f'mature_n_{n}']=len(subset)
     row[f'pooled_spearman_{n}']=subset.mispricing.corr(subset[col],method='spearman') if len(subset)>2 else None
     ics=[x.mispricing.corr(x[col],method='spearman') for _,x in subset.groupby('as_of') if len(x)>=5 and x.mispricing.nunique()>1 and x[col].nunique()>1]
     row[f'mean_monthly_spearman_{n}']=float(np.nanmean(ics)) if ics else None
     row[f'ic_months_{n}']=len(ics)
     row[f'ic_naive_t_{n}']=float(np.mean(ics)/np.std(ics,ddof=1)*np.sqrt(len(ics))) if len(ics)>1 and np.std(ics,ddof=1)>0 else None
    if 'price_252' in d:
     m=d.dropna(subset=['price_252','low','high']);row['interval_hit_n_252']=len(m);row['interval_hit_rate_252']=float(m.price_252.between(m.low,m.high).mean()) if len(m) else None
    signals.append(row)
if metrics:
 m=pd.concat(metrics,ignore_index=True);m.to_csv(dest/'portfolio_metrics.csv',index=False)
 paired=[]
 for (v,p),g in m.groupby(['variant','period']):
  base=g[g.run.str.contains('_A_selection')];filtered=g[g.run.str.contains('_B_valuation')]
  if len(base)==1 and len(filtered)==1:
   row={'variant':v,'period':p}
   for col in ['total_return_pct','cagr_pct','sharpe','max_drawdown_pct','turnover_per_year']:row['delta_'+col]=float(filtered.iloc[0][col]-base.iloc[0][col])
   paired.append(row)
 pd.DataFrame(paired).to_csv(dest/'paired_valuation_effect.csv',index=False)
for name,data in [('valuation_coverage',coverage),('signal_diagnostics',signals),('missing_valuation_reasons',reasons)]:pd.DataFrame(data).to_csv(dest/(name+'.csv'),index=False)
status={'available_variants':sorted(set(x['variant'] for x in coverage)),'note':'Partial until all three variants complete. Rank correlations are descriptive, not causal alpha. Forward horizons may have zero mature observations.'}
(dest/'analysis_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
