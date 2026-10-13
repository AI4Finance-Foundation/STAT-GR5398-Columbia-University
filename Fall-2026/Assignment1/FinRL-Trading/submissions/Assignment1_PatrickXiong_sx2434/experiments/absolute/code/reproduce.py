from pathlib import Path
import argparse,os,sys,subprocess,shutil
a=argparse.ArgumentParser();a.add_argument('--input-dir',required=True);a.add_argument('--data-dir',required=True);a.add_argument('--reference-run',required=True);a.add_argument('--run-dir',required=True);x=a.parse_args()
C=Path(__file__).resolve().parent;r=Path(x.run_dir).resolve();r.mkdir(parents=True,exist_ok=True)
if any(r.iterdir()):raise RuntimeError('Use a new empty experiment run directory')
ref=Path(x.reference_run).resolve()
for f in ['price_features_candidate.csv.gz','feature_checks.json']:shutil.copy2(ref/f,r/f)
env=os.environ.copy();env.update(FINRL_INPUT_DIR=str(Path(x.input_dir).resolve()),FINRL_DATA_DIR=str(Path(x.data_dir).resolve()),FINRL_REFERENCE_RUN=str(ref),FINRL_RUN_DIR=str(r))
for name,args in [('run_models.py',[]),('backtest_crsp.py',[]),('audit_model_inputs.py',[str(r)]),('prove_bias_controls.py',[]),('independent_ledger_audit.py',[])]:
 with (r/(name+'.log')).open('w') as log:subprocess.run([sys.executable,str(C/name),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name+' completed',flush=True)
