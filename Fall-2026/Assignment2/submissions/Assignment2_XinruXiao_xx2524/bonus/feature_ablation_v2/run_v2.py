"""Isolated, resumable feature-ablation retrain; never writes the course score file."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[2]
out = args.out.resolve()
if out != here / 'scores_v2.parquet':
    parser.error('--out must be the isolated scores_v2.parquet next to this driver')
parts = here / 'scores_v2_parts'
logs = here / 'logs'
parts.mkdir(exist_ok=True)
logs.mkdir(exist_ok=True)
source = root / 'work/retraining_v2/model'
driver = here.parent / 'train_one_month.py'
course = source / 'data/selection/monthly_scores.parquet'
reference = here.parent / 'baseline_scores.parquet'
config = source / 'selection/configs/feature_ablation.json'
selector = source / 'selection/selector.py'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

identity = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha(p)
            for p in (course.resolve(), reference, driver, config, selector,
                      (source/'data/prices/daily_adjusted.parquet').resolve(),
                      (source/'data/fundamentals.db').resolve(), Path(__file__).resolve())}
identity.update(runtime=sys.version, pandas=pd.__version__, numpy=np.__version__)
assert sha(course) == (here/'course_scores.sha256').read_text().split()[0]
assert sha(config) == sha(here/'feature_ablation.json')
assert sha(selector) == sha(here/'selector.py')
manifest = here / 'run_identity.json'
if manifest.exists():
    assert json.loads(manifest.read_text()) == identity, 'Inputs changed: use a fresh experiment'
else:
    assert not list(parts.glob('*.parquet')), 'Unidentified checkpoints'
    manifest.write_text(json.dumps(identity, indent=2))

env = os.environ.copy()
env.update({key:'1' for key in ('LOKY_MAX_CPU_COUNT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                                'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
baseline = pd.read_parquet(reference)
keys = ['month','cutoff','bucket','symbol']
baseline['_month'] = pd.to_datetime(baseline.month).dt.strftime('%Y-%m')
expected = {month: part[keys].sort_values(keys).reset_index(drop=True)
            for month, part in baseline.groupby('_month')}

def validate(month):
    frame = pd.read_parquet(parts / (month+'.parquet'))
    assert not frame.duplicated(['month','symbol','bucket']).any(), month
    assert np.isfinite(frame[['rank_score','weight_score']]).all().all(), month
    pd.testing.assert_frame_equal(frame[keys].sort_values(keys).reset_index(drop=True),expected[month])
    return frame

def train(month):
    if not (parts/(month+'.parquet')).is_file():
        with (logs/(month+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(driver),'--source',str(source),'--month',month,
                            '--parts',str(parts),'--config','selection/configs/feature_ablation.json'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    return month,len(validate(month))

def stage(months):
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(train, m) for m in months]):
            print(*future.result(), flush=True)

import fcntl
lock = (here/'.run.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
pilot = ['2019-05','2019-06']
stage(pilot)
print('PILOT PASSED; continuing to full 119 months',flush=True)
stage(sorted(set(expected)-set(pilot)))
all_scores = pd.concat([validate(m) for m in sorted(expected)],ignore_index=True)
tmp = out.with_suffix('.tmp')
all_scores.to_parquet(tmp,index=False)
tmp.replace(out)
assert sha(course) == (here/'course_scores.sha256').read_text().split()[0]
print(f'TRAINING COMPLETE: {len(expected)} months, {len(all_scores)} rows',flush=True)
for period in ('dev','val'):
    for variant, extra in [('A',[]),('B',['--valuation'])]:
        path = logs / f'backtest_{period}_{variant}.log'
        with path.open('w') as log:
            subprocess.run([sys.executable,'run_portfolio.py','--period',period,
                            '--scores',str(out),*extra],cwd=source,env=env,stdout=log,
                           stderr=subprocess.STDOUT,check=True)
        print(f'BACKTEST COMPLETE: {period} {variant}',flush=True)
assert sha(course) == (here/'course_scores.sha256').read_text().split()[0]
subprocess.run([sys.executable,str(here/'write_comparison.py')], check=True)
