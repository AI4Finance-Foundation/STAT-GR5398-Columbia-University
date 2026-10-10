"""One command from source verification through full/common reports.

A clean numeric reproduction uses a fresh prefix and refits the models. Raw
public recovery prices remain mutable: retain their manifests with each run.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True,type=Path);ap.add_argument('--prefix',default='reproduced');ap.add_argument('--report-python',default=sys.executable);ap.add_argument('--reuse-saved-training',action='store_true',help='Rebuild downstream outputs only; not a fresh model refit');a=ap.parse_args();root=Path(__file__).resolve().parents[1]
def call(script,*args,python=sys.executable):subprocess.run([python,str(root/'code'/script),*map(str,args)],cwd=root,check=True)
run='audited_run' if a.reuse_saved_training else a.prefix+'_run';full=a.prefix+'_full';common=a.prefix+'_common'
if a.reuse_saved_training:
 # Replaying fitted predictions requires the exact market inputs used to fit.
 audit=json.loads((root/run/'audit.json').read_text())
 for ticker,expected in audit['recovery_price_sha256'].items():
  path=root/'evidence/price_recovery/accepted'/f'{ticker}.csv'
  if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
   raise RuntimeError(f'Frozen recovery input missing or changed: {ticker}; restore the private input snapshot.')
 from market_data import aliases
 if aliases()!=audit['security_aliases']:
  raise RuntimeError('Security alias mapping differs from the fitted run.')
 for rec in json.loads((root/'data_sources.json').read_text()):
  path=a.cache/rec['file']
  if not path.exists():raise RuntimeError(f'Pinned source missing: {path}; run bootstrap_data.py first.')
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):h.update(block)
  if h.hexdigest()!=rec['sha256']:raise RuntimeError(f'Pinned source changed: {path}')
if not a.reuse_saved_training:
 call('bootstrap_data.py','--cache',a.cache)
 call('fetch_benchmarks.py','--output',root/'reference/overnight')
 call('probe_missing_prices.py','--exclusions',root/'reference/overnight/entry_exclusions.csv','--output',root/'public_price_probe')
 call('recover_prices.py','--cache',a.cache,'--probe',root/'public_price_probe','--output',root/'evidence/price_recovery')
 call('financial_spotcheck.py','--cache',a.cache,'--output',root/'evidence/financial')
 call('audit_and_train.py','--cache',a.cache,'--prior',root/'reference/overnight','--output',root/run)
for window,dest in [('full',full),('common',common)]:
 call('compare_backtests.py','--cache',a.cache,'--baseline',root/'reference/original','--prior',root/'reference/overnight','--audited',root/run,'--output',root/dest,'--window',window)
 call('verify_outputs.py','--root',root,'--run',run,'--comparison',dest)
call('build_report.py','--root',root,'--run',run,'--comparison',full,'--common',common,python=a.report_python)
(root/(a.prefix+'_reproduction.json')).write_text(json.dumps({'run':run,'full':full,'common':common,'fresh_training':not a.reuse_saved_training,'completed':True},indent=2))
