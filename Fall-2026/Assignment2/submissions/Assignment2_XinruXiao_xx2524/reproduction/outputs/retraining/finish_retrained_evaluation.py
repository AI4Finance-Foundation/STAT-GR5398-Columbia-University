"""Continue evaluations automatically once both complete score tables exist."""
import argparse,os,sys,json,time,shutil,subprocess,concurrent.futures
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--python',required=True);a=p.parse_args()
r=Path(a.task_root).resolve();out=r/'outputs/retraining'
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(r/'work/openmp/lib'),PYTHONPATH=str(r/'work/a2-retrain-deps')+os.pathsep+str(r/'work/a2-deps'),LOKY_MAX_CPU_COUNT='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
while True:
 try:s=json.loads((out/'training_status.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):s={}
 if s.get('status')=='complete':break
 if s.get('status')=='failed':raise SystemExit('Training failed: evaluation not started.')
 time.sleep(20)

def evaluate(version):
 source=r/'work/retraining'/version;dest=r/'work/retraining/evaluation'/('retrained_'+version)
 shutil.copytree(source,dest,ignore=shutil.ignore_patterns('data','output','__pycache__','.DS_Store'),dirs_exist_ok=True)
 if not (dest/'data').exists():(dest/'data').symlink_to(r/'work/course/Fall-2026/Assignment2/source_code/data',target_is_directory=True)
 with (r/'work'/('evaluation_retrained_'+version+'.log')).open('w') as f:
  proc=subprocess.run([a.python,str(out/'run_evaluation.py'),'--source',str(dest),'--scores',str(out/(version+'_scores.parquet')),'--variant','retrained_'+version],env=env,stdout=f,stderr=subprocess.STDOUT)
 return {'variant':version,'returncode':proc.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 results=list(pool.map(evaluate,['baseline','improved']))
(out/'retrained_evaluation_status.json').write_text(json.dumps(results,indent=2));print(results,flush=True)
if any(x['returncode'] for x in results):raise SystemExit(1)
