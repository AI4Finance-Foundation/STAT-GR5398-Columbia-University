"""Reproduce this WRDS experiment in a new output directory."""
from pathlib import Path
import argparse,os,subprocess,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
p.add_argument('--conservative-label-floor',action='store_true')
a=p.parse_args();code=Path(__file__).resolve().parent;inp=Path(a.input_dir).resolve();out=Path(a.output_dir).resolve()
if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory; completed results are preserved.')
out.mkdir(parents=True,exist_ok=True);(out/'data-source').mkdir();(out/'run').mkdir()
required=['crsp_daily_raw.parquet','crsp_names_all.csv','crsp_sp500_members.csv.gz','crsp_delisting.csv']
manifest=json.loads((inp/'frozen_manifest.json').read_text())
for name in required:
 actual=hashlib.sha256((inp/name).read_bytes()).hexdigest()
 assert actual==manifest[name]['sha256'],('Input hash mismatch',name)
env=os.environ.copy();env.update(FINRL_INPUT_DIR=str(inp),FINRL_DATA_DIR=str(out/'data-source'),FINRL_RUN_DIR=str(out/'run'))
def run(script,*args):
 with (out/(script+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(code/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(script+' completed',flush=True)
run('prepare_crsp.py')
if a.conservative_label_floor:
 env['FINRL_LABEL_FLOOR_STRESS']='1'
 import pandas as pd,numpy as np
 panel=out/'data-source/model_panel.csv.gz';d=pd.read_csv(panel)
 mask=d.label_status.eq('unknown_complete_delisting_proceeds')
 d.loc[mask,'target']=np.log(1e-8);d.loc[mask,'training_label_eligible']=True
 d.to_csv(panel,index=False)
 (out/'floor_sensitivity.json').write_text(json.dumps({'cases':int(mask.sum()),'gross_return_floor':1e-8,'scope':'Training sensitivity only; unknown cash-plus-rights terminal payouts are not measured as total losses.'},indent=2))
run('run_models.py');run('backtest_crsp.py');run('audit_model_inputs.py',str(out/'run'))
run('independent_ledger_audit.py');run('audit_source_cases.py')
run('summarize_results.py');run('audit_failure_validation.py')

if not a.conservative_label_floor:run('prove_bias_controls.py')
