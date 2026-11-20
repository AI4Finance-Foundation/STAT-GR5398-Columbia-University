"""Evaluate complete early periods while later independent months train."""
import argparse,os,json,shutil,subprocess,concurrent.futures
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task-root',required=True);p.add_argument('--python',required=True);p.add_argument('--periods',nargs='+',default=['dev']);a=p.parse_args();r=Path(a.task_root).resolve();o=r/'outputs/retraining'
env=os.environ.copy();env.update(PYTHONPATH=str(r/'work/a2-deps'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
def run(version):
 dest=r/'work/retraining/evaluation'/('retrained_'+version)
 shutil.copytree(r/'work/retraining'/version,dest,ignore=shutil.ignore_patterns('data','output','__pycache__','.DS_Store'),dirs_exist_ok=True)
 if not (dest/'data').exists():(dest/'data').symlink_to(r/'work/course/Fall-2026/Assignment2/source_code/data',target_is_directory=True)
 with (r/'work'/('evaluation_retrained_'+version+'_'+'_'.join(a.periods)+'_early.log')).open('w') as f:
  result=subprocess.run([a.python,str(o/'run_evaluation.py'),'--source',str(dest),'--scores',str(o/(version+'_scores.parquet')),'--variant','retrained_'+version,'--periods',*a.periods],env=env,stdout=f,stderr=subprocess.STDOUT)
 return {'variant':version,'periods':a.periods,'returncode':result.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['baseline','improved']))
(o/('early_evaluation_'+'_'.join(a.periods)+'_status.json')).write_text(json.dumps(results,indent=2));print(results,flush=True)
if any(x['returncode'] for x in results):raise SystemExit(1)
