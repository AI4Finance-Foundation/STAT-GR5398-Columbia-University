from pathlib import Path
import subprocess,configparser,sys,os,json,time
ROOT=Path(__file__).resolve().parents[1];workspace=ROOT.parents[1]
config_path=workspace/'work/private/config.ini';c=configparser.ConfigParser();c.read(config_path)
keys=[c['API_KEYS'][k] for k in ['fmp_api_key','openai_api_key']]
companies={'NVDA':('NVIDIA Corporation',['AMD','INTC']),'AMD':('Advanced Micro Devices',['NVDA','INTC']),
'INTC':('Intel Corporation',['AMD','NVDA']),'AAPL':('Apple Inc.',['MSFT','NVDA']),
'MSFT':('Microsoft Corporation',['AAPL','NVDA']),'GOOGL':('Alphabet Inc.',['MSFT','AAPL'])}
env=os.environ.copy();env['FMP_CACHE_DIR']=str(ROOT/'actual_runs/fmp_cache');env['MPLCONFIGDIR']=str(workspace/'work/mpl')
for ticker,(name,peers) in companies.items():
 out=ROOT/'actual_runs/baseline'/ticker;analysis=out/'analysis';analysis.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(ROOT/'upstream/finrobot_equity/core/src/generate_financial_analysis.py'),
 '--company-ticker',ticker,'--company-name',name,'--config-file',str(config_path),
 '--peer-tickers',*peers,'--output-dir',str(analysis),'--enable-sensitivity-analysis']
 start=time.perf_counter();r=subprocess.run(cmd,env=env,cwd=ROOT/'upstream',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 log=r.stdout
 for key in keys:log=log.replace(key,'[REDACTED]')
 (out/'data_only.log').write_text(log)
 status={'ticker':ticker,'seconds':time.perf_counter()-start,'returncode':r.returncode,
 'metrics_generated':(analysis/'financial_metrics_and_forecasts.csv').exists(),'llm_called':False}
 (out/'data_only_manifest.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
