"""Frozen-input repetitions and exhaustive automated claim audit; no publishing."""
from pathlib import Path
import sys,os,json,asyncio,configparser,time,re
import pandas as pd
from openai import AsyncOpenAI
from agents import Runner,RunConfig,set_default_openai_client,set_tracing_disabled
from dataclasses import asdict,is_dataclass
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'upstream/finrobot_equity/core/src'))
from modules.equity_agents.agent_manager import EquityResearchAgentManager
c=configparser.ConfigParser();c.read(sys.argv[1]);key=c.get('API_KEYS','openai_api_key');model=c.get('API_KEYS','openai_model',fallback='gpt-4.1')
OUT=R/'actual_runs/evaluation';OUT.mkdir(parents=True,exist_ok=True)
def default(x):return x.model_dump() if hasattr(x,'model_dump') else str(x)
async def main():
 async with AsyncOpenAI(api_key=key,base_url=c.get('API_KEYS','openai_base_url')) as client:
  set_default_openai_client(client,use_for_tracing=False);set_tracing_disabled(True)
  mgr=EquityResearchAgentManager();records=[]
  # Original NVDA successful run is repeat1; later runs use exact saved user prompts.
  for run in [2,3]:
   folder=OUT/f'NVDA_run{run}';folder.mkdir(exist_ok=True)
   for p in sorted((R/'actual_runs/baseline/NVDA/agent_evidence').glob('*.json')):
    dest=folder/p.name
    if dest.exists():continue
    original=json.loads(p.read_text());start=time.perf_counter()
    result=await Runner.run(mgr.agents[original['section']],original['user_prompt'],run_config=RunConfig(model=model))
    final=result.final_output
    original.update(output=final.model_dump() if hasattr(final,'model_dump') else str(final),provider_usage=[asdict(x.usage) for x in result.raw_responses],seconds=time.perf_counter()-start)
    dest.write_text(json.dumps(original,indent=2,default=default))
    print('repeat',run,original['section'],flush=True)
  instruction='''You are an evidence auditor. Return JSON only. Extract EVERY factual and quantitative atomic claim from the complete section; split compound assertions. Exclude purely normative recommendations, explicit conditional scenarios and opinions, but include causal explanations asserted as facts. Classify Supported ONLY when directly derivable from supplied financial prompt; Unsupported when evidence is absent (including business history, market leadership, regulation, causal explanations, PB/PS, share price or supply-chain facts not actually supplied); Contradicted when evidence directly conflicts. Do not use outside knowledge. Keep each claim and reason concise. Evidence excerpts must be no more than150 characters; never copy whole tables. Quote a precise prompt excerpt for every Supported/Contradicted claim; give transparent reasoning. Do not classify a forecast as realized, confuse fiscal-year end labels with calendar years, or use model assumptions as history. Output {"claims":[{"claim":"...","class":"Supported|Unsupported|Contradicted","evidence":"exact excerpt or empty","reason":"..."}]}. This is automated provisional classification, not human-verified truth.'''
  for ticker in ['NVDA','AMD','INTC']:
   for p in sorted((R/f'actual_runs/baseline/{ticker}/agent_evidence').glob('*.json')):
    obj=json.loads(p.read_text());dest=OUT/f'{ticker}_{obj["section"]}_claims.json'
    if not dest.exists():
     res=await client.chat.completions.create(model=model,messages=[{'role':'system','content':instruction},{'role':'user','content':json.dumps({'financial_prompt':obj['user_prompt'],'complete_section':obj['output']})}],response_format={'type':'json_object'},temperature=0,max_tokens=18000)
     data=json.loads(res.choices[0].message.content);data.update(usage=res.usage.model_dump(),source_file=str(p.relative_to(R)),method='LLM extraction and classification; conservative exact-excerpt validation; not independent human adjudication')
     dest.write_text(json.dumps(data,indent=2));print('audit',ticker,obj['section'],len(data['claims']),flush=True)
    data=json.loads(dest.read_text())
    for i,x in enumerate(data['claims'],1):
     # Enforce support excerpt is actually present (allow whitespace normalization).
     normalize=lambda t:' '.join(t.split()).lower()
     valid=bool(x.get('evidence')) and normalize(x['evidence']) in normalize(obj['user_prompt'])
     label=x['class'];flag=False
     if label in ['Supported','Contradicted'] and not valid:flag=True
     records.append(dict(ticker=ticker,section=obj['section'],claim_id=i,claim=x['claim'],classification=label,evidence=x.get('evidence',''),reason=x.get('reason',''),invalid_excerpt_overridden=flag,review_status='Automated provisional; manual adjudication pending'))
  frame=pd.DataFrame(records);frame.to_csv(R/'audit/grounding_claims.csv',index=False)
  rates=frame.groupby(['ticker','section']).classification.agg(total='size',supported=lambda x:(x=='Supported').sum(),unsupported=lambda x:(x=='Unsupported').sum(),contradicted=lambda x:(x=='Contradicted').sum()).reset_index();rates['grounding_rate']=rates.supported/rates.total;rates.to_csv(R/'audit/grounding_rates.csv',index=False)
  # Extract comparable decisions from the actual text; retain explicit absence.
  instruction2='''Extract only explicitly stated investment rating and numeric target price from these agent outputs. Return null if absent; never infer a rating or manufacture a target. Extract the first three distinct risk themes in order from risks output, with short normalized labels and verbatim supporting quotes. Return JSON {"rating":null,"target_price":null,"top3_risks":[{"label":"...","quote":"..."}]}.'''
  summaries=[]
  for run,folder in [(1,R/'actual_runs/baseline/NVDA/agent_evidence'),(2,OUT/'NVDA_run2'),(3,OUT/'NVDA_run3')]:
   dest=OUT/f'NVDA_run{run}_decisions.json'
   objs={p.stem:json.loads(p.read_text())['output'] for p in folder.glob('*.json')}
   if not dest.exists():
    r=await client.chat.completions.create(model=model,messages=[{'role':'system','content':instruction2},{'role':'user','content':json.dumps(objs)}],response_format={'type':'json_object'},temperature=0,max_tokens=1300)
    data=json.loads(r.choices[0].message.content);data['usage']=r.usage.model_dump();dest.write_text(json.dumps(data,indent=2))
   data=json.loads(dest.read_text());summaries.append(dict(run=run,rating=data.get('rating'),target_price=data.get('target_price'),top3_risks=json.dumps(data['top3_risks']),input='Exact frozen user_prompt + same upstream agent system instructions',model=model))
  pd.DataFrame(summaries).to_csv(R/'audit/stability_actual.csv',index=False)
  nums=[x['target_price'] for x in summaries if isinstance(x['target_price'],(int,float))]
  (R/'audit/stability_summary.json').write_text(json.dumps({'runs':3,'target_price_variance':float(pd.Series(nums).var(ddof=0)) if len(nums)==3 else None,'missing_decisions':'Missing rating/target is a pipeline output limitation, not zero variance','risk_labels_per_run':[[x['label'] for x in json.loads(y['top3_risks'])] for y in summaries],'decision_extraction':'Automated extraction with verbatim quotes; review required'},indent=2))
asyncio.run(main())
