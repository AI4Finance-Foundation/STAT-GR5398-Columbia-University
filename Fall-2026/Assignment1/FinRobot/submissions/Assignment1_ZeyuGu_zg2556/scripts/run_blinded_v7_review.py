"""Run the three independent changed-unit reviews serially with warning blinding."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
for ticker in ['NVDA','AMD','INTC']:
    subprocess.run([sys.executable,'scripts/review_changed_v7.py',ticker,'--version','v7_1',
        '--blind-warnings','--config',args.config],cwd=root,check=True)
print('Blinded machine proposals ready for explicit review; no automatic canonical promotion.')
