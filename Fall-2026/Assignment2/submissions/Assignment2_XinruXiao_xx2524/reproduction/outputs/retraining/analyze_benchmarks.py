"""Full SPY/QQQ reference metrics with unchanged course performance conventions."""
from pathlib import Path
import argparse,sys
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();sys.path.insert(0,str(r/'work/retraining/baseline'))
from shared.prices import PriceStore
from shared.metrics import performance
prices=PriceStore(r/'work/course/Fall-2026/Assignment2/source_code/data/prices/daily_adjusted.parquet').close
out=r/'work/retraining/evaluation/course/output';rows=[];curves=[]
for period in ['dev','val','test','forward']:
 f=list(out.glob('ablation_'+period+'_*.csv'))[0];tab=pd.read_csv(f);first=tab.iloc[0]
 for benchmark in ['SPY','QQQ']:
  s=prices[benchmark].loc[first.start:first.end].dropna();stats=performance(s)
  assert stats['total_return_pct']==first[benchmark+'_total_return_pct']
  assert stats['sharpe']==first[benchmark+'_sharpe']
  rows.append({'period':period,'benchmark':benchmark,'start':str(s.index.min().date()),'end':str(s.index.max().date()),**stats,'turnover_per_year':None,'cost_basis':'passive adjusted-price reference, no simulated transaction costs; turnover not defined by course'})
  curves.extend({'period':period,'date':str(day.date()),'benchmark':benchmark,'normalized_value':float(value/s.iloc[0]*100)} for day,value in s.items())
dest=r/'outputs/retraining/analysis';dest.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(dest/'benchmark_metrics.csv',index=False);print('PASS: full benchmark metrics agree with official return and Sharpe columns.')
pd.DataFrame(curves).to_csv(dest/'benchmark_curves.csv',index=False)
