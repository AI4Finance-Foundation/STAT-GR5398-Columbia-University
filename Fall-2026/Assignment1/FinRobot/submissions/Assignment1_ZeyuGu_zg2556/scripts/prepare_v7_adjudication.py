"""Carry independent labels across exact unchanged units; never read verifier labels."""
import argparse
import copy
import json
from pathlib import Path
from grounding_audit import load_inputs, source_units
from prepare_after_adjudication import normalized, BEFORE
from audit_io import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]
AFTER={'NVDA':'NVDA_after_v6_adjudicated','AMD':'AMD_after_v6_adjudicated_v2','INTC':'INTC_after_v6_adjudicated'}

def prepare(ticker, version='v7'):
    baseline=ROOT/'runs'/f'{ticker}_baseline_1'
    target=ROOT/'runs'/f'{ticker}_final_{version}'
    manifest,inputs=load_inputs(target,baseline)
    known={s:{} for s in inputs}
    for known_version,folder in [('baseline_1',BEFORE[ticker]),('final_v6',AFTER[ticker])]:
        audit=ROOT/'audits/grounding'/folder
        spec=json.loads((audit/'reviewed_specification.json').read_text(encoding='utf-8'))
        _,old=load_inputs(ROOT/'runs'/f'{ticker}_{known_version}',baseline)
        for s,item in old.items():
            units={u['unit_id']:u for u in spec['sections'][s]['units']}
            for span in source_units(item['text']):
                key=normalized(span['text'])
                known[s][key]=(units[span['unit_id']],folder,span)
    spec={'protocol_version':'machine-adjudication-v1','run_id':manifest['run_id'],
        'evidence_run_id':baseline.name,'raw_judge_audit':None,'sections':{}}
    todo=[];mapping=[]
    for s,item in inputs.items():
        section={'source_sha256':item['metadata']['source_sha256'],
            'evidence_sha256':item['metadata']['evidence_prompt_sha256'],'units':[]}
        for span in source_units(item['text']):
            previous=known[s].get(normalized(span['text']))
            if previous:
                unit,folder,old=previous
                record=copy.deepcopy(unit);record['unit_id']=span['unit_id']
                section['units'].append(record)
                mapping.append({'section':s,'target_unit_id':span['unit_id'],
                    'source_audit':folder,'source_unit_id':old['unit_id'],
                    'source_text_sha256':sha256(old['text']),'target_text_sha256':sha256(span['text']),
                    'basis':'Exact text after warning/bullet, apostrophe and whitespace normalization only; same immutable writer evidence. No verifier judgments read.'})
            else: todo.append({'section':s,**span})
        spec['sections'][s]=section
    output=ROOT/'audits/grounding'/f'{ticker}_after_{version}_working'
    output.mkdir(exist_ok=True)
    write_json(output/'partial_specification.json',spec)
    write_json(output/'exact_source_carry_forward.json',mapping)
    write_json(output/'requires_explicit_review.json',todo)
    print(ticker,'carried',len(mapping),'changed',len(todo))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tickers',nargs='+',choices=list(BEFORE));p.add_argument('--version',default='v7')
    args=p.parse_args()
    for ticker in args.tickers:prepare(ticker,args.version)
