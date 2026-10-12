"""Independent explicit review of changed AMD V6 sentences, not verifier labels."""
import copy,json,re
from pathlib import Path
from grounding_audit import load_inputs,source_units
from audit_io import write_json
ROOT=Path(__file__).resolve().parents[1]
OLD=json.loads((ROOT/'audits/grounding/AMD_before_adjudicated_v2/reviewed_specification.json').read_text(encoding='utf-8'))
WORK=ROOT/'audits/grounding/AMD_after_v6_working'
SPEC=json.loads((WORK/'partial_specification.json').read_text(encoding='utf-8'));REV=[]
def atoms(s,j,select=None):
    u=next(u for u in OLD['sections'][s]['units'] if u['unit_id']==j)
    return copy.deepcopy(u['atoms'] if select is None else [u['atoms'][k] for k in select])
def put(s,i,values,why='Changed wording independently reviewed as the same selected source propositions.',nonfactual=''):
    SPEC['sections'][s]['units'].append({'unit_id':i,'atoms':values,'nonfactual_reason':nonfactual});REV.append({'section':s,'unit_id':i,'review_reason':why})
def cp(s,i,j,select=None,nonfactual=''):put(s,i,atoms(s,j,select),nonfactual=nonfactual)
def A(c,k,refs=(),why='',num='not_numeric'):
    return {'claim':c,'classification':k,'evidence':[{'line_start':a,'line_end':b} for a,b in refs],'reason':why,'numeric_status':num,'claim_type':'factual'}
def U(*cs):return [A(c,'Unsupported',why='The exact original financial/peer tables and single quarterly earnings excerpt do not establish this proposition.',num='unverifiable' if re.search(r'\d',c) else 'not_numeric') for c in cs]
def S(c,refs,why,num='not_numeric'):return A(c,'Supported',refs,why,num)

put('tagline',0,[S('AMD actual revenue CAGR is20.5%.',((4,4),(6,6)),'2021A–2025A CAGR is20.491%, rounded20.5%.','correct')])
cp('tagline',1,0,[1,2])
put('tagline',3,[S('AMD EBITDA margin is projected to rise.',((4,4),(12,12)),'2025A21% to2028E42%.'),*atoms('tagline',1,[2,3])])
cp('tagline',4,2)
cp('company_overview',18,18);cp('company_overview',20,20)
put('investment_overview',4,U('AMD Q2 2026 revenue beat market expectations.','AMD Q2 2026 non-GAAP diluted EPS beat market expectations.'),why='Split the two expectation comparisons; prompt contains no consensus for either.')
cp('investment_overview',6,3,[1,2,3]);cp('investment_overview',7,4,[0]);cp('investment_overview',8,4,[1,2]);cp('investment_overview',10,5)
cp('investment_overview',12,6);cp('investment_overview',14,8,[0]);cp('investment_overview',15,8,[1]);cp('investment_overview',16,8,[2]);cp('investment_overview',18,9,[0,1]);cp('investment_overview',19,9,[1,2,3]);cp('investment_overview',23,10,[3]);cp('investment_overview',24,11);cp('investment_overview',25,12,[0,1]);cp('investment_overview',26,12)
cp('investment_overview',28,13,[1]);cp('investment_overview',29,13,[1,2]);cp('investment_overview',36,19);cp('investment_overview',37,20)
cp('valuation_overview',2,2)
put('valuation_overview',5,[S('AMD current EV/EBITDA is high relative to displayed peers.',((30,36),),'Latest common2025 AMD exceeds both peers.'),*atoms('valuation_overview',4,[4,5])])
cp('valuation_overview',8,7);cp('valuation_overview',9,8);cp('valuation_overview',10,9);cp('valuation_overview',11,10);cp('valuation_overview',13,12);cp('valuation_overview',14,13);cp('valuation_overview',15,14,[0,2,3,4])
cp('risks',1,0,[2,3,4]);cp('risks',2,1);cp('risks',3,2,[0]);cp('risks',4,2,[1,2]);cp('risks',5,3);cp('risks',7,5);cp('risks',9,7);cp('risks',11,9);cp('risks',12,10);cp('risks',13,11);cp('risks',14,12);cp('risks',15,13);cp('risks',16,14,[0]);cp('risks',17,14,[1,2]);cp('risks',18,14,[3,4]);cp('risks',21,17);cp('risks',23,19);cp('risks',24,20);cp('risks',25,21);cp('risks',28,24);cp('risks',29,25)
cp('competitor_analysis',24,21);cp('competitor_analysis',25,22);cp('competitor_analysis',27,24)
put('major_takeaways',2,[S('AMD revenue rises across actual2021 to forecast2028.',((4,4),(6,6)),'16.434 billion2021A to40.0953 billion2028E; future endpoint is a forecast.','correct'),*atoms('major_takeaways',0,[1,2,3,4])],why='The rewritten sentence drops the false20.5% seven-year CAGR, but its causal demand/product explanations remain unsupported.')
cp('major_takeaways',3,1)
put('major_takeaways',4,[S('AMD actual annual revenue grew strongly across the visible horizon.',((4,4),(6,6)),'2021A–2025A revenue more than doubles.'),*atoms('major_takeaways',2,[0,1,2,4])],why='Remove the dropped highly-competitive qualifier; explicitly include revenue-growth premise.')
cp('major_takeaways',5,3);cp('major_takeaways',6,4);cp('major_takeaways',7,5);cp('major_takeaways',8,6);cp('major_takeaways',9,7);cp('major_takeaways',10,8);cp('major_takeaways',11,9);cp('major_takeaways',12,10,[0,1]);cp('major_takeaways',13,11);cp('major_takeaways',14,12)
cp('news_summary',1,1)
put('news_summary',5,[S('AMD released Q2 2026 financial results.',((42,47),),'The quarter is explicitly identified in the release.','correct'),*atoms('news_summary',5)],why='Added explicit quarter is supported; efficiency and quarterly-growth inference remain unsupported.')
put('news_summary',9,[S('AMD EPS has an improving recent and projected trend.',((4,4),(15,15)),'From2023A0.53 through2025A2.67 and2028E3.18, including forecasts; no claim of full-history monotonic increase.')],nonfactual='A strong investment case is normative; improving EPS premise judged separately.')
cp('news_summary',10,9);cp('news_summary',11,10,[0]);cp('news_summary',12,11);cp('news_summary',13,12);cp('news_summary',14,13)
put('news_summary',15,[],nonfactual='Recommendation to monitor future results and competitive developments; no new factual premise.')
cp('news_summary',16,15)

def main():
    _,inputs=load_inputs(ROOT/'runs/AMD_final_v6',ROOT/'runs/AMD_baseline_1')
    for s,item in inputs.items():
        units=SPEC['sections'][s]['units'];ids=[u['unit_id'] for u in units];expected={u['unit_id'] for u in source_units(item['text'])}
        if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError(f'Incomplete changed-unit review {s}: {set(ids)^expected}')
        units.sort(key=lambda u:u['unit_id'])
    write_json(WORK/'explicit_changed_unit_review.json',REV);write_json(ROOT/'audits/grounding/AMD_after_v6_reviewed_specification.json',SPEC)
    print('All AMD after units independently reviewed:',sum(len(s['units']) for s in SPEC['sections'].values()))
if __name__=='__main__':main()
