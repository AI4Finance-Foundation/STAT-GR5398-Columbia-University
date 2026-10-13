"""Rebuild common CRSP inputs, then reproduce every research target."""
from pathlib import Path
import argparse,subprocess,sys,json
a=argparse.ArgumentParser();a.add_argument('--input-dir',required=True);a.add_argument('--output-dir',required=True);a.add_argument('--skip-absolute',action='store_true');x=a.parse_args()
P=Path(__file__).resolve().parents[1];out=Path(x.output_dir).resolve()
if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new empty output directory')
out.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(P/'code/reproduce.py'),'--input-dir',x.input_dir,'--output-dir',str(out/'spy')],check=True)
for tag in ['qqq']+([] if x.skip_absolute else ['absolute']):
 subprocess.run([sys.executable,str(P/'experiments'/tag/'code/reproduce.py'),'--input-dir',x.input_dir,'--data-dir',str(out/'spy/data-source'),'--reference-run',str(out/'spy/run'),'--run-dir',str(out/tag/'run')],check=True)
(out/'reproduction_complete.json').write_text(json.dumps({'targets':['spy','qqq']+([] if x.skip_absolute else ['absolute']),'licensed_inputs_external':True},indent=2))
print('All requested source/model/portfolio audits passed')
