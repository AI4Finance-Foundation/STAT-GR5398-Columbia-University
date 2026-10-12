"""Run the actual course pipeline; redact API keys before writing any logs.
Requires PRIVATE config path. Does not push, publish, or submit anything.
"""
from pathlib import Path
import argparse,configparser,subprocess,sys,json,time,os
ROOT=Path(__file__).resolve().parents[1]
UNIVERSE={'NVDA':('NVIDIA Corporation',['AMD','INTC']), 'AMD':('Advanced Micro Devices',['NVDA','INTC']),
 'INTC':('Intel Corporation',['AMD','NVDA']), 'AAPL':('Apple Inc.',['MSFT','NVDA']),
 'MSFT':('Microsoft Corporation',['AAPL','NVDA']), 'GOOGL':('Alphabet Inc.',['META','MSFT'])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ticker',choices=UNIVERSE)
 p.add_argument('--improved',action='store_true');p.add_argument('--preflight',action='store_true');args=p.parse_args()
 config_path=Path(args.config).resolve();config=configparser.ConfigParser();config.read(config_path)
 keys=[config.get('API_KEYS',x,fallback='').strip() for x in ['fmp_api_key','openai_api_key']]
 if not all(keys) or any(k.startswith('YOUR_') for k in keys):sys.exit('Missing FMP/OpenAI credentials. No network requests made.')
 def clean(text):
  for k in keys:text=text.replace(k,'[REDACTED]')
  return text
 if args.preflight:
  import requests
  from openai import OpenAI
  checks=[]
  for endpoint in ['income-statement','balance-sheet-statement','cash-flow-statement','ratios','key-metrics']:
   try:
    r=requests.get(f'https://financialmodelingprep.com/stable/{endpoint}',params={'apikey':keys[0],'symbol':'NVDA','limit':1},timeout=30)
    checks.append({'endpoint':endpoint,'http_status':r.status_code,'nonempty_list':isinstance(r.json(),list) and len(r.json())>0})
   except Exception as e:checks.append({'endpoint':endpoint,'error_type':type(e).__name__})
  client=OpenAI(api_key=keys[1],base_url=config.get('API_KEYS','openai_base_url',fallback='https://api.openai.com/v1'))
  try:
   client.models.retrieve(config.get('API_KEYS','openai_model',fallback='gpt-4.1'));checks.append({'OpenAI_model_access':'available','billing_status':'not tested by model retrieval'})
  except Exception as e:checks.append({'OpenAI_model_access':'failed','error_type':type(e).__name__})
  print(json.dumps(checks,indent=2));return
 upstream=ROOT/'upstream';src=upstream/'finrobot_equity/core/src'
 if not src.exists():sys.exit('Run code/prepare_upstream.py first')
 for ticker in ([args.ticker] if args.ticker else [t for t in UNIVERSE if t!='GOOGL']):
  name,peers=UNIVERSE[ticker];label='improved' if args.improved else 'baseline';out=ROOT/'actual_runs'/label/ticker
  analysis=out/'analysis';report=out/'report';analysis.mkdir(parents=True,exist_ok=True)
  commands=[[sys.executable,str(src/'generate_financial_analysis.py'),'--company-ticker',ticker,'--company-name',name,
      '--config-file',str(config_path),'--peer-tickers',*peers,'--generate-text-sections','--enable-sensitivity-analysis',
      '--enable-catalyst-analysis','--enable-enhanced-news','--output-dir',str(analysis)]]
  if args.improved:commands[0].append('--data-driven-forecasts')
  step2=[sys.executable,str(src/'create_equity_report.py'),'--company-ticker',ticker,'--company-name',name,
    '--analysis-csv',str(analysis/'financial_metrics_and_forecasts.csv'),'--ratios-csv',str(analysis/'ratios_raw_data.csv'),
    '--config-file',str(config_path),'--output-dir',str(report)]
  for flag,filename in [('tagline','tagline'),('company-overview','company_overview'),('investment-overview','investment_overview'),('valuation-overview','valuation_overview'),('risks','risks'),('competitor-analysis','competitor_analysis'),('major-takeaways','major_takeaways'),('news-summary','news_summary')]:step2 += ['--'+flag+'-file',str(analysis/(filename+'.txt'))]
  step2 += ['--peer-ev-ebitda-csv',str(analysis/'peer_ev_ebitda_comparison.csv')]
  commands.append(step2)
  times=[]
  for i,cmd in enumerate(commands,1):
   start=time.perf_counter();r=subprocess.run(cmd,cwd=upstream,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env={**os.environ,'FMP_CACHE_DIR':str(ROOT/'actual_runs/fmp_cache'),'FINROBOT_EVIDENCE_DIR':str(out/'agent_evidence')})
   (out/f'step{i}.log').write_text(clean(r.stdout));times.append(time.perf_counter()-start)
   # Upstream sometimes returns exit 0 on configuration/data failures. Require artifacts.
   expected=[analysis/'financial_metrics_and_forecasts.csv',analysis/'ratios_raw_data.csv'] + [analysis/(x+'.txt') for x in ['tagline','company_overview','investment_overview','valuation_overview','risks','competitor_analysis','major_takeaways','news_summary']] if i==1 else list(report.glob('*.html'))
   if i==1 and len(list((out/'agent_evidence').glob('*.json')))!=8:sys.exit(f'{ticker}: actual evidence for all 8 agents missing; no experiment claimed')
   if r.returncode or not expected or any(not x.exists() or x.stat().st_size==0 for x in expected):sys.exit(f'{ticker} step {i} failed or missing required artifacts; see redacted log')
  (out/'runner_manifest.json').write_text(json.dumps({'ticker':ticker,'mode':label,'step_seconds':times,'model':config.get('API_KEYS','openai_model'),
    'commands':commands,'api_spend':'not yet measured; collect provider token usage','status':'generated; audits pending'},indent=2))
  print(ticker,label,'completed; verify agent text and report artifacts before interpreting')
if __name__=='__main__':main()
