"""Serial development follow-up after the original V7 plan has fully exited."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
up=Path.home()/'.cache/finrobot-zg2556/upstream'
def run(*argv):subprocess.run([sys.executable,*map(str,argv)],cwd=ROOT,check=True)
run('scripts/prepare_verifier_v7_1.py','--activate')
run('scripts/run_experiment_plan.py','--plan','config/verifier_v7_1_plan.json',
    '--config',args.config,'--upstream',up,'--log','evidence/progress/verifier_v7_1_plan.jsonl')
run('scripts/prepare_v7_adjudication.py','NVDA','AMD','INTC','--version','v7_1')
for ticker in ['NVDA','AMD','INTC']:
    run('scripts/review_changed_v7.py',ticker,'--version','v7_1','--config',args.config)
print('Development follow-up generated changed-unit proposals; explicit review and release decision remain required.')
