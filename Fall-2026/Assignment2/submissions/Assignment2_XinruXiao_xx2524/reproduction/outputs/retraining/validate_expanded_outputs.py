"""Fail closed on incomplete scores or missing prescribed evaluations."""
from pathlib import Path
import json,argparse,hashlib
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();o=r/'outputs/retraining';report={'score_integrity':{},'evaluation_integrity':{}}
expected=pd.period_range('2017-01','2026-11',freq='M').astype(str).tolist()
for v in ['baseline','improved']:
 path=o/(v+'_scores.parquet');d=pd.read_parquet(path);months=pd.to_datetime(d.month).dt.to_period('M').astype(str)
 assert sorted(months.unique())==expected
 assert not d.duplicated(['month','bucket','symbol']).any()
 assert np.isfinite(d[['rank_score','weight_score']].to_numpy(dtype=float)).all()
 assert (pd.to_datetime(d.cutoff)<pd.to_datetime(d.month)).all()
 assert d.rank_score.between(0,1).all()
 counts=d.groupby([months,'bucket']).size();assert set(d.bucket)=={'growth_tech','cyclical','real_assets','defensive'}
 report['score_integrity'][v]={'months':len(expected),'rows':len(d),'bucket_months':len(counts),'min_names_per_bucket_month':int(counts.min()),'max_names_per_bucket_month':int(counts.max()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'duplicate_keys':0,'nonfinite_scores':0,'cutoff_before_month':True}
for variant in ['course','retrained_baseline','retrained_improved']:
 out=r/'work/retraining/evaluation'/variant/'output';assert (out/'evaluation_complete.json').exists();summary=[]
 for period in ['dev','val','test','forward']:
  f=list(out.glob('ablation_'+period+'_*.csv'));assert len(f)==1
  d=pd.read_csv(f[0]);assert len(d)==4
  for left,right in [('A_selection','C_timing_full'),('B_valuation','D_both_full')]:
   x=d[d.run.str.endswith(left)].iloc[0];y=d[d.run.str.endswith(right)].iloc[0]
   for col in ['total_return_pct','cagr_pct','volatility_pct','sharpe','max_drawdown_pct','turnover_per_year']:assert x[col]==y[col],(variant,period,col)
   pd.testing.assert_frame_equal(pd.read_csv(out/x.run/'equity.csv'),pd.read_csv(out/y.run/'equity.csv'),check_exact=True)
  for scope in ['top20','all_growth']:
   for prefix in ['valuation_eval','monthly_input_audit']:assert (out/f'{prefix}_{period}_{scope}.csv').exists()
   values=pd.read_csv(out/f'valuation_eval_{period}_{scope}.csv')
   fields=['fair_value','low','high','mispricing','method_spread']
   assert np.isfinite(values[fields].to_numpy(dtype=float)).all()
   assert (values.low>0).all() and (values.low<=values.fair_value).all() and (values.fair_value<=values.high).all()
   assert (values.method_spread>=1).all()
   decision=pd.read_csv(out/f'monthly_input_audit_{period}_{scope}.csv')[['as_of','ticker','price']].rename(columns={'price':'decision_price'})
   check=values.merge(decision,on=['as_of','ticker'],validate='one_to_one')
   assert np.isclose(check.mispricing,check.fair_value/check.decision_price-1,rtol=1e-8,atol=1e-10).all()
  assert (out/f'weekly_all_candidates_{period}.csv').exists();summary.append(period)
 report['evaluation_integrity'][variant]={'periods':summary,'four_groups_each':True,'A_equals_C_and_B_equals_D':True}
report['status']='PASS';(o/'expanded_integrity_checks.json').write_text(json.dumps(report,indent=2));print('PASS: complete 119-month score tables, finite unique dated scores, all prescribed evaluation groups and files.')
