"""Carry independent labels only across exact unchanged sentences; flag all others.

No verifier classification or citation is read. Warning prefixes are presentation,
not evidence. Every changed unit requires a new explicit adjudication decision.
"""
import copy
import json
from pathlib import Path
import re
from grounding_audit import load_inputs,source_units
from audit_io import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]
BEFORE={'NVDA':'NVDA_before_adjudicated','AMD':'AMD_before_adjudicated_v2','INTC':'INTC_before_adjudicated'}
def normalized(text):
    text=re.sub(r'^\[(?:UNVERIFIED|NONFACTUAL COMMENTARY)[^\]]*\]\s*','',text.strip())
    text=re.sub(r'^[-*]\s+','',text)
    text=text.replace('\u2019',"'").replace('\u2018',"'")
    return re.sub(r'\s+',' ',text).strip()
def main():
    for t,folder in BEFORE.items():
        baseline=ROOT/'runs'/f'{t}_baseline_1';final=ROOT/'runs'/f'{t}_final_v6'
        old=json.loads((ROOT/'audits/grounding'/folder/'reviewed_specification.json').read_text(encoding='utf-8'))
        _,bi=load_inputs(baseline);manifest,ai=load_inputs(final,baseline)
        spec={'protocol_version':'machine-adjudication-v1','run_id':manifest['run_id'],'evidence_run_id':baseline.name,'raw_judge_audit':None,'sections':{}}
        todo=[];mapping=[];count=0
        for s,item in ai.items():
            bu=source_units(bi[s]['text']);old_units={u['unit_id']:u for u in old['sections'][s]['units']}
            exact={normalized(u['text']):u['unit_id'] for u in bu}
            section={'source_sha256':item['metadata']['source_sha256'],'evidence_sha256':item['metadata']['evidence_prompt_sha256'],'units':[]}
            for u in source_units(item['text']):
                key=normalized(u['text'])
                if key in exact:
                    previous=exact[key];record=copy.deepcopy(old_units[previous]);record['unit_id']=u['unit_id'];section['units'].append(record);count+=1
                    mapping.append({'section':s,'after_unit_id':u['unit_id'],'before_unit_id':previous,'basis':'exact unchanged sentence after removing warning/bullet prefixes, typographic apostrophe and whitespace normalization only','before_text_sha256':sha256(bu[previous]['text']),'after_text_sha256':sha256(u['text'])})
                else:todo.append({'section':s,**u})
            spec['sections'][s]=section
        output=ROOT/'audits/grounding'/f'{t}_after_v6_working';output.mkdir(exist_ok=True)
        write_json(output/'partial_specification.json',spec);write_json(output/'exact_source_carry_forward.json',mapping);write_json(output/'requires_explicit_review.json',todo)
        print(t,'exact inherited units',count,'changed units needing review',len(todo))
if __name__=='__main__':main()
