"""Finish the authorized local pipeline after an already-running training job."""
import argparse,json,subprocess,sys,time,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True,type=Path);ap.add_argument('--report-python',required=True);ap.add_argument('--wait-minutes',type=int,default=45);a=ap.parse_args();r=Path(__file__).resolve().parents[1]
def update(stage,**kwargs):
 x=dict(stage=stage,updated_at=datetime.datetime.now().astimezone().isoformat(),**kwargs);(r/'progress.json').write_text(json.dumps(x,indent=2));(r/'progress.log').open('a').write(json.dumps(x)+'\n');print(json.dumps(x),flush=True)
def call(script,*args,python=sys.executable):subprocess.run([python,str(r/'code'/script),*map(str,args)],cwd=r,check=True)
try:
 end=time.monotonic()+60*a.wait_minutes
 while True:
  status=json.loads((r/'audited_run/status.json').read_text())
  if status['stage']=='completed':break
  if status['stage']=='failed':raise RuntimeError('Training failed: '+str(status))
  if time.monotonic()>end:raise TimeoutError('Training did not complete within bounded finish window')
  time.sleep(20)
 for window in ['full','common']:
  dest='comparison_'+window;update('comparing_'+window)
  call('compare_backtests.py','--cache',a.cache,'--baseline',r/'reference/original','--prior',r/'reference/overnight','--audited',r/'audited_run','--output',r/dest,'--window',window)
  call('verify_outputs.py','--root',r,'--run','audited_run','--comparison',dest)
 update('building_report')
 call('build_report.py','--root',r,'--run','audited_run','--comparison','comparison_full','--common','comparison_common',python=a.report_python)
 update('numerical_outputs_and_report_ready',visual_qa_passed=False,publication=False,course_submission=False)
except BaseException as e:
 update('finish_failed',error=str(e));raise
