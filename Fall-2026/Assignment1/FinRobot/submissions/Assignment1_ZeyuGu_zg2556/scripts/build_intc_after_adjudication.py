"""Explicit independent review of every changed INTC V6 source unit.

Unchanged sentences retain prior independently adjudicated atomic judgments.
Changed paraphrases and subsets below were compared with their original source
and writer evidence; verifier verdicts are not inputs to this evaluation.
"""
import copy
import json
from pathlib import Path
from grounding_audit import load_inputs,source_units
from audit_io import write_json
ROOT=Path(__file__).resolve().parents[1]
OLD=json.loads((ROOT/'audits/grounding/INTC_before_adjudicated/reviewed_specification.json').read_text(encoding='utf-8'))
WORK=ROOT/'audits/grounding/INTC_after_v6_working'
SPEC=json.loads((WORK/'partial_specification.json').read_text(encoding='utf-8'))
REVIEW=[]
def atoms(s,i,select=None):
    unit=next(u for u in OLD['sections'][s]['units'] if u['unit_id']==i)
    values=unit['atoms'] if select is None else [unit['atoms'][j] for j in select]
    return copy.deepcopy(values)
def put(s,i,values,reason='Independently reviewed changed wording preserves the selected original atomic propositions.',nonfactual=''):
    SPEC['sections'][s]['units'].append({'unit_id':i,'atoms':values,'nonfactual_reason':nonfactual})
    REVIEW.append({'section':s,'unit_id':i,'review_reason':reason})
def cp(s,i,j,select=None,nonfactual=''):put(s,i,atoms(s,j,select),nonfactual=nonfactual)
def A(c,k,refs=(),why='',num='not_numeric'):
    return {'claim':c,'classification':k,'evidence':[{'line_start':a,'line_end':b} for a,b in refs],'reason':why,'numeric_status':num,'claim_type':'factual'}
def U(*cs):return [A(c,'Unsupported',why='The original financial/peer tables and short earnings excerpt do not establish this specific causal, business, risk or temporal assertion.') for c in cs]
def S(c,refs,why,num='not_numeric'):return A(c,'Supported',refs,why,num)
def C(c,refs,why,num='not_numeric'):return A(c,'Contradicted',refs,why,num)

cp('tagline',0,0,[0]);cp('tagline',1,0,[1,2])
cp('company_overview',16,16);cp('company_overview',18,18,[1,2,3,4])
put('investment_overview',3,[S('Intel reported a recovery in quarterly revenue growth in Q2 2026.',((42,45),),'Release reports 25% YoY revenue growth; recovery is an evaluative description of that positive change.','correct')])
cp('investment_overview',6,3,[2])
cp('investment_overview',9,4,[2])
put('investment_overview',14,[S('Intel recent EBITDA margin improved.',((4,4),(12,12)),'2024A2.3% to2025A27.2%; challenging is evaluative.')])
cp('investment_overview',22,14,[0,1]);cp('investment_overview',27,19,[0,1]);cp('investment_overview',30,22)
cp('valuation_overview',15,15);cp('valuation_overview',18,18);cp('valuation_overview',20,20)
put('risks',1,[],nonfactual='Taxonomic organization of this risk discussion, not a new externally verifiable company claim.')
cp('risks',2,2,[0,1]);cp('risks',3,2,[0,2,3]);cp('risks',4,3);cp('risks',5,4)
put('risks',6,[S('AMD has stronger multiyear EBITDA growth than Intel.',((19,25),),'AMD2021–2025 EBITDA rises while Intel falls.'),S('NVIDIA has stronger multiyear EBITDA growth than Intel.',((19,25),),'NVDA2022–2025 EBITDA rises while Intel falls.'),*U('AMD superior EV/EBITDA establishes better competitive quality than Intel.','NVIDIA superior EV/EBITDA establishes better competitive quality than Intel.')],reason='Split companies; superior multiple is a quality inference, not evidence of business superiority. Latest values are higher, but2024 is a counterexample to uniformly higher multiples.')
cp('risks',8,7);cp('risks',9,8);cp('risks',10,9,[0,1,2]);cp('risks',11,10);cp('risks',12,11)
cp('risks',15,14,[0]);cp('risks',16,14);cp('risks',17,15);cp('risks',18,16);cp('risks',19,17);cp('risks',20,18);cp('risks',21,19);cp('risks',22,20);cp('risks',23,21);cp('risks',24,22)
put('risks',25,[],nonfactual='Normative mitigation recommendations; the removed preceding claims about brand and resources are not in this final unit.')
cp('risks',26,24);cp('risks',27,25)
put('competitor_analysis',1,atoms('competitor_analysis',1,[0,1,3,4])+U('TSMC is an Intel competitor.'),reason='The final wording removes emerging; do not carry that deleted qualifier.')
cp('competitor_analysis',5,5);cp('competitor_analysis',15,15)
put('competitor_analysis',16,[S('Intel revenue was$79,024,000,000 in2021.',((4,4),(6,6)),'Exact2021A observation.','correct'),S('Intel revenue value for2025 is$52,853,000,000.',((4,4),(6,6)),'Exact2025A observation.','correct'),C('Intel2025 revenue value is an estimate.',((4,4),(6,6)),'Column2025A denotes actual, not forecast.','incorrect')],reason='Evaluate the newly exact numbers and preserve the actual-versus-estimate contradiction; do not inherit only approximate values.')
cp('competitor_analysis',19,20);cp('competitor_analysis',21,22);cp('competitor_analysis',22,23);cp('competitor_analysis',23,24)
cp('major_takeaways',1,1)
put('major_takeaways',9,[S('Intel SG&A margin declines overall after2022.',((4,4),(13,13)),'11.1% in2022A falls to8.7% in2025A, with a2023–2024 plateau.','correct'),*atoms('major_takeaways',9,[]) ,*atoms('major_takeaways',9)],reason='The final rewrite adds the explicit post-2022 trend premise, which is separately supported; causal management assertions remain unsupported.')
cp('news_summary',2,2,[0,1])
put('news_summary',4,[S('Intel reported negative GAAP earnings in its latest disclosed quarter.',((42,46),),'Q2 release GAAP EPS -2.16.'),S('Intel reported positive non-GAAP earnings in its latest disclosed quarter.',((42,46),),'Q2 release non-GAAP EPS0.42.'),*U('Intel has this GAAP-negative/non-GAAP-positive pattern across multiple recent quarters.')],reason='The evidence establishes one quarter only; the new plural temporal claim is not established.')
cp('news_summary',5,3,[3]);cp('news_summary',7,5,[0,1,3]);cp('news_summary',8,6)

def main():
    _,inputs=load_inputs(ROOT/'runs/INTC_final_v6',ROOT/'runs/INTC_baseline_1')
    for s,item in inputs.items():
        units=SPEC['sections'][s]['units'];ids=[u['unit_id'] for u in units];expected={u['unit_id'] for u in source_units(item['text'])}
        if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError(f'Missing/duplicate after units in {s}: {set(ids)^expected}')
        units.sort(key=lambda u:u['unit_id'])
    write_json(WORK/'explicit_changed_unit_review.json',REVIEW)
    write_json(ROOT/'audits/grounding/INTC_after_v6_reviewed_specification.json',SPEC)
    print('All INTC after units have explicit independent decisions:',sum(len(x['units']) for x in SPEC['sections'].values()))
if __name__=='__main__':main()
