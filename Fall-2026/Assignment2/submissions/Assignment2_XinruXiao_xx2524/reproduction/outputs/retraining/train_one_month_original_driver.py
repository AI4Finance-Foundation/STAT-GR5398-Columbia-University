"""One official monthly training task with bounded numerical parallelism."""
import argparse,sys,os,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--month',required=True);p.add_argument('--parts',required=True);p.add_argument('--config',required=True);a=p.parse_args()
source=Path(a.source).resolve();parts=Path(a.parts).resolve();parts.mkdir(parents=True,exist_ok=True)
os.chdir(source);sys.path.insert(0,str(source))
from threadpoolctl import threadpool_limits
import selection.selector as selector
original=selector._build_models

def bounded_models():
 models=original()
 for m in models.values():
  if 'n_jobs' in m.get_params():m.set_params(n_jobs=1)
 return models
selector._build_models=bounded_models
from selection.build_scores import _init,_run_month
start=time.monotonic()
with threadpool_limits(limits=1):
 _init(a.config,str(parts))
 result=_run_month(a.month)
print(json.dumps({'month':a.month,'seconds':round(time.monotonic()-start,2),'result':result}),flush=True)
