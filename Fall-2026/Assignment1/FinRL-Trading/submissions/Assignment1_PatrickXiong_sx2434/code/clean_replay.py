"""Replay saved predictions in a new directory with frozen inputs; no refit."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--destination', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--report-python', default=sys.executable)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dest = args.destination.resolve()
if dest.exists():
    raise SystemExit('Choose a new, nonexistent destination; existing files are preserved.')
dest.mkdir(parents=True)
for name in ['code', 'reference', 'audited_run']:
    shutil.copytree(root/name, dest/name, ignore=shutil.ignore_patterns('__pycache__'))
for name in ['data_sources.json', 'data_preparation_verification.json',
             'candidate_plan.json', 'requirements.txt', 'environment_training.txt', 'README.md']:
    shutil.copy2(root/name, dest/name)
shutil.copytree(root/'evidence', dest/'evidence', ignore=shutil.ignore_patterns('raw'))
with (dest/'replay.log').open('w') as log:
    subprocess.run([sys.executable, str(dest/'code/reproduce.py'), '--cache',
                    str(args.cache.resolve()), '--prefix', 'clean', '--reuse-saved-training',
                    '--report-python', args.report_python], cwd=dest,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
checked = []
for window in ['full', 'common']:
    for source in sorted((root/('comparison_'+window)).glob('*.csv')):
        target = dest/('clean_'+window)/source.name
        if not target.exists() or source.read_bytes() != target.read_bytes():
            raise RuntimeError(f'Replay differs: {window}/{source.name}')
        checked.append(dict(file=f'{window}/{source.name}',
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
pdf = dest/'output/pdf/FinRL_Assignment1_Research_Draft.pdf'
if not pdf.exists():
    raise RuntimeError('Report was not regenerated.')
result = dict(passed=True, mode='frozen inputs and saved predictions; downstream replay',
              fresh_training=False, fresh_network_download=False,
              destination=str(dest), numeric_files=checked, report_regenerated=True,
              note='Pinned course files reused from a verified cache; PDF metadata may differ.')
(root/'evidence/clean_replay_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=True, numeric_files=len(checked), destination=str(dest)), indent=2))
