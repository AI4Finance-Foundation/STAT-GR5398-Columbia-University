"""Official evaluation entry points with score injection and exact-repeat memoization.

All financial calculations still occur inside DataView guards. Memoization is
local to one immutable DataBundle/score experiment and is reset each period.
"""
import argparse,sys,os,json,shutil,time,hashlib,fcntl
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--scores',required=True);p.add_argument('--variant',required=True);p.add_argument('--periods',nargs='+',choices=['dev','val','test','forward'],default=['dev','val','test','forward']);a=p.parse_args()
source=Path(a.source).resolve();scores_path=Path(a.scores).resolve();os.chdir(source);sys.path.insert(0,str(source))
lock=(source/'.evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
scores_frame=pd.read_parquet(scores_path)
period_months={'dev':('2017-01','2021-12'),'val':('2022-01','2023-12'),'test':('2024-01','2026-10'),'forward':('2026-10','2026-11')}
def fingerprint(period):
 first,last=period_months[period];months=pd.to_datetime(scores_frame.month).dt.to_period('M').astype(str)
 frame=scores_frame.loc[months.between(first,last)].sort_values(['month','bucket','symbol']).reset_index(drop=True)
 expected=pd.period_range(first,last,freq='M').astype(str).tolist()
 assert sorted(pd.to_datetime(frame.month).dt.to_period('M').astype(str).unique())==expected,'Incomplete score months for '+period
 return {'score_months':[first,last],'score_scope_sha256':hashlib.sha256(frame.to_csv(index=False,float_format='%.17g',date_format='%Y-%m-%d').encode()).hexdigest(),'valuation_sha256':hashlib.sha256((source/'valuation/student_valuation.py').read_bytes()).hexdigest()}
import run_portfolio as portfolio
import evaluate_valuation as evaluation
from selection.scores import MonthlyScores
from valuation import student_valuation as student
from shared.data_view import DataBundle
base_audit=student.audit_value;base_filter=student.filter_candidates
all_records=[]

def capture_value(as_of,tickers,data):
 df,details=base_audit(as_of,tickers,data)
 for d in details:
  methods=d['methods']
  all_records.append(dict(as_of=str(as_of),ticker=d['ticker'],price=d['price'],
       dcf_valid='dcf' in methods,history_valid='history' in methods,peer_valid='peer' in methods,
       aggregate_valid=d['aggregate'] is not None,
       reasons=json.dumps(d['reasons'],ensure_ascii=False),
       fair_value=d['aggregate']['fair_value'] if d['aggregate'] else None,
       low=d['aggregate']['low'] if d['aggregate'] else None,
       high=d['aggregate']['high'] if d['aggregate'] else None))
 return df
student.value=capture_value
# The course scorer class uses precomputed tables; replace only its path.
evaluation.MonthlyScores=lambda:MonthlyScores(scores_path)
output=source/'output';output.mkdir(exist_ok=True)
for period in a.periods:
 mark=output/f'evaluation_period_complete_{period}.json';identity=fingerprint(period)
 if mark.exists() and json.loads(mark.read_text()).get('identity')==identity:
  print('PERIOD RESUMED',a.variant,period,flush=True);continue
 print('PERIOD',a.variant,period,flush=True);memo={};all_records.clear()
 def cached_filter(as_of,candidates,data):
  key=(str(as_of),tuple(candidates))
  if key not in memo:memo[key]=base_filter(as_of,candidates,data)
  return memo[key]
 student.filter_candidates=cached_filter
 # Same DataBundle and deterministic filter for B/D. Reusing its exact result
 # does not reuse another period's later data or modify any trade calculation.
 sys.argv=['run_evaluation.py','--variant',a.variant,'--scores',str(scores_path),'--period',period,'--ablation','--timing','full']
 portfolio.main(['--period',period,'--ablation','--timing','full','--scores',str(scores_path)])
 pd.DataFrame(all_records).to_csv(output/f'weekly_all_candidates_{period}.csv',index=False)
 for top,label in [(20,'top20'),(10000,'all_growth')]:
  all_records.clear()
  sys.argv=['run_evaluation.py','--variant',a.variant,'--scores',str(scores_path),'--period',period,'--top',str(top)]
  evaluation.main(['--period',period,'--top',str(top)])
  generated=output/f'valuation_eval_{period}.csv'
  shutil.copy2(generated,output/f'valuation_eval_{period}_{label}.csv')
  pd.DataFrame(all_records).to_csv(output/f'monthly_input_audit_{period}_{label}.csv',index=False)
 print('PERIOD COMPLETE',a.variant,period,flush=True)
 mark.write_text(json.dumps({'variant':a.variant,'period':period,'identity':identity,'status':'complete'},indent=2))
complete=True
for period in ['dev','val','test','forward']:
 mark=output/f'evaluation_period_complete_{period}.json'
 try:complete=complete and mark.exists() and json.loads(mark.read_text()).get('identity')==fingerprint(period)
 except AssertionError:complete=False
if complete:
 (output/'evaluation_complete.json').write_text(json.dumps({'variant':a.variant,'scores':str(scores_path),'periods':['dev','val','test','forward'],'candidate_limit':None,'status':'complete'},indent=2))
 print('EVALUATION COMPLETE',a.variant,flush=True)
else:print('REQUESTED PERIODS COMPLETE; remaining periods pending',a.variant,flush=True)
