"""Compare scores within common month/bucket; no returns used for selection."""
import argparse
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('--task-root',default='.');a=p.parse_args();r=Path(a.task_root).resolve();o=r/'outputs/retraining';dest=o/'analysis';dest.mkdir(exist_ok=True)
paths={'course':r/'work/course/Fall-2026/Assignment2/source_code/data/selection/monthly_scores.parquet','retrained_baseline':o/'baseline_scores.parquet','retrained_improved':o/'improved_scores.parquet'}
scores={name:pd.read_parquet(path) for name,path in paths.items() if path.exists()};rows=[];coverage=[]
expected=set(pd.period_range('2017-01','2026-11',freq='M').astype(str))
for name,frame in scores.items():
 for bucket,g in frame.groupby('bucket'):
  months=pd.to_datetime(g.month).dt.to_period('M').astype(str);counts=months.value_counts()
  coverage.append({'variant':name,'bucket':bucket,'months':months.nunique(),'rows':len(g),'min_names_per_available_month':int(counts.min()),'max_names_per_available_month':int(counts.max()),'missing_months':','.join(sorted(expected-set(months)))})
pd.DataFrame(coverage).to_csv(dest/'score_coverage_by_bucket.csv',index=False)
for left,right in [('course','retrained_baseline'),('retrained_baseline','retrained_improved')]:
 if left not in scores or right not in scores:continue
 a,b=scores[left],scores[right]
 a['month']=a.month.astype(str);b['month']=b.month.astype(str)
 merged=a.merge(b,on=['month','bucket','symbol'],suffixes=('_left','_right'),validate='one_to_one')
 for (month,bucket),g in merged.groupby(['month','bucket']):
  la=a[(a.month==month)&(a.bucket==bucket)].nlargest(10,'rank_score');rb=b[(b.month==month)&(b.bucket==bucket)].nlargest(10,'rank_score')
  rows.append({'left':left,'right':right,'month':month,'bucket':bucket,'common_names':len(g),'spearman':g.rank_score_left.corr(g.rank_score_right,method='spearman'),'top10_overlap':len(set(la.symbol)&set(rb.symbol)),'left_rows':len(a[(a.month==month)&(a.bucket==bucket)]),'right_rows':len(b[(b.month==month)&(b.bucket==bucket)])})
pd.DataFrame(rows).to_csv(dest/'monthly_score_comparison.csv',index=False)
print('Score comparisons:',len(rows))
