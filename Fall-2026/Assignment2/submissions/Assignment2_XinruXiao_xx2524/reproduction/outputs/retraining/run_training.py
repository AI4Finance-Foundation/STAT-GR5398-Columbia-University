"""Resumable two-version official monthly retraining, never overwrites course scores."""
import argparse,concurrent.futures,json,os,subprocess,sys,time,hashlib
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--python',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
root=Path(a.task_root).resolve();out=root/'outputs/retraining';start=time.monotonic()
versions={'baseline':'selection/configs/growth.json','improved':'selection/configs/improved.json'}
months=[str(m) for m in pd.period_range('2017-01','2026-11',freq='M')]
env=os.environ.copy()
env.update(DYLD_LIBRARY_PATH=str(root/'work/openmp/lib'),PYTHONPATH=str(root/'work/a2-retrain-deps')+os.pathsep+str(root/'work/a2-deps'),LOKY_MAX_CPU_COUNT='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
logs=out/'training_logs';logs.mkdir(exist_ok=True)
state={'total_months_per_variant':len(months),'variants':versions,'status':'running','completed':{},'failed':[]}
def update():
 for version in versions:state['completed'][version]=len(list((out/(version+'_parts')).glob('????-??.parquet')))
 state['elapsed_seconds']=round(time.monotonic()-start,1)
 temp=out/'training_status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'training_status.json')
def train(job):
 version,month=job;parts=out/(version+'_parts');parts.mkdir(exist_ok=True);target=parts/(month+'.parquet')
 if target.exists():return {'version':version,'month':month,'status':'resumed'}
 log=logs/(version+'_'+month+'.log')
 cmd=[a.python,str(out/'train_one_month.py'),'--source',str(root/'work/retraining'/version),'--month',month,'--parts',str(parts),'--config',versions[version]]
 with log.open('w') as f:
  r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 return {'version':version,'month':month,'status':'ok' if r.returncode==0 and target.exists() else 'failed','returncode':r.returncode}
jobs=[(v,m) for m in months for v in versions]
update()
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 futures={pool.submit(train,j):j for j in jobs}
 for future in concurrent.futures.as_completed(futures):
  job=futures[future]
  try:r=future.result()
  except Exception as e:r={'version':job[0],'month':job[1],'status':'failed','error':str(e)}
  if r['status']=='failed':state['failed'].append(r)
  update();print(json.dumps({'task':r,'completed':state['completed'],'elapsed':state['elapsed_seconds']}),flush=True)
if state['failed']:
 state['status']='failed';update();raise SystemExit('Some months failed. Inspect logs; rerun to resume.')
for version in versions:
 frames=[]
 for month in months:
  f=pd.read_parquet(out/(version+'_parts')/(month+'.parquet'))
  assert len(f)>0 and set(['month','cutoff','symbol','bucket','rank_score','weight_score'])<=set(f)
  assert pd.to_datetime(f.cutoff).max()<pd.Timestamp(month+'-01')
  assert not f.duplicated(['month','symbol','bucket']).any()
  frames.append(f)
 df=pd.concat(frames,ignore_index=True);dest=out/(version+'_scores.parquet');df.to_parquet(dest,index=False)
 meta={'version':version,'months':[months[0],months[-1]],'rows':len(df),'monthly_parts':len(frames),'model_config':versions[version]}
 for key,path in [('selector_sha256',root/'work/retraining'/version/'selection/selector.py'),('prices_sha256',root/'work/course/Fall-2026/Assignment2/source_code/data/prices/daily_adjusted.parquet'),('fundamentals_sha256',root/'work/course/Fall-2026/Assignment2/source_code/data/fundamentals.db')]:meta[key]=hashlib.sha256(path.read_bytes()).hexdigest()
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2))
state['status']='complete';update();print('TRAINING COMPLETE',flush=True)
