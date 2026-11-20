"""Development-only rule sensitivity on saved all-candidate values, no new backtest."""
from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();out=r/'work/retraining/evaluation/course/output';dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True)
audit=pd.read_csv(out/'monthly_input_audit_dev_all_growth.csv');values=pd.read_csv(out/'valuation_eval_dev_all_growth.csv')
cols=['as_of','ticker'];assert not audit.duplicated(cols).any();assert not values.duplicated(cols).any()
merged=audit.merge(values[cols+['method_spread']],on=cols,how='left',validate='one_to_one')
method_cols=['dcf_valid','history_valid','peer_valid'];merged['method_count']=sum(merged[c].astype(str).str.lower().eq('true').astype(int) for c in method_cols)
merged['year']=pd.to_datetime(merged.as_of).dt.year;rows=[]
for methods in [2,3]:
 for threshold in [2.,3.,4.]:
  for year,g in [('all_dev',merged)]+list(merged.groupby('year')):
   covered=g.method_count.ge(methods)&g.fair_value.notna();removed=covered&g.method_spread.le(threshold)&g.price.gt(g.high)
   rows.append({'scope':'saved_monthly_all_growth_dev_only','year':str(year),'minimum_methods':methods,'max_spread':threshold,'candidate_months':len(g),'valid_aggregate':int(covered.sum()),'flagged_for_removal':int(removed.sum()),'flagged_share':float(removed.mean()),'monthly_snapshots':g.as_of.nunique(),'note':'Rule flags only; no alternative portfolio backtests, no test-period tuning.'})
pd.DataFrame(rows).to_csv(dest/'development_all_candidate_rule_sensitivity.csv',index=False);print(pd.DataFrame(rows).query('year=="all_dev"').to_string(index=False))
