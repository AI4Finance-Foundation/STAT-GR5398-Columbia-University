"""Explicit independent adjudication of every changed NVDA V6 source unit.

The evaluator receives the original writer prompt and final text only, plus its
own baseline judgments for unchanged/subset propositions. No verifier labels.
"""
import copy,json,re
from pathlib import Path
from grounding_audit import load_inputs,source_units
from audit_io import write_json
ROOT=Path(__file__).resolve().parents[1]
OLD=json.loads((ROOT/'audits/grounding/NVDA_before_adjudicated/reviewed_specification.json').read_text(encoding='utf-8'))
WORK=ROOT/'audits/grounding/NVDA_after_v6_working'
SPEC=json.loads((WORK/'partial_specification.json').read_text(encoding='utf-8'));REV=[]
def atoms(s,j,select=None):
    u=next(u for u in OLD['sections'][s]['units'] if u['unit_id']==j)
    return copy.deepcopy(u['atoms'] if select is None else [u['atoms'][k] for k in select])
def put(s,i,values,why='Independently reviewed paraphrase/subset preserves the selected original propositions.',nonfactual=''):
    SPEC['sections'][s]['units'].append({'unit_id':i,'atoms':values,'nonfactual_reason':nonfactual});REV.append({'section':s,'unit_id':i,'review_reason':why})
def cp(s,i,j,select=None,nonfactual=''):put(s,i,atoms(s,j,select),nonfactual=nonfactual)
def A(c,k,refs=(),why='',num='not_numeric'):
    return {'claim':c,'classification':k,'evidence':[{'line_start':a,'line_end':b} for a,b in refs],'reason':why,'numeric_status':num,'claim_type':'factual'}
def U(*cs):return [A(c,'Unsupported',why='The original financial/peer tables and short quarterly disclosure do not establish this specific assertion.',num='unverifiable' if re.search(r'\d',c) else 'not_numeric') for c in cs]
def S(c,refs,why,num='not_numeric'):return A(c,'Supported',refs,why,num)
def C(c,refs,why,num='not_numeric'):return A(c,'Contradicted',refs,why,num)
F=((4,4),(6,6));CM=((4,4),(10,10));EM=((4,4),(12,12));PEV=((30,37),)

cp('tagline',0,0,[1,2]);cp('tagline',1,1,[0]);cp('tagline',2,1,[1,2])
cp('company_overview',1,1);cp('company_overview',2,2,[1,2])
put('company_overview',3,U('AI-related product demand drives NVIDIA rapid revenue growth.','Data-center product demand drives NVIDIA rapid revenue growth.'))
put('company_overview',4,atoms('company_overview',3,[0,1,2,3])+U('NVIDIA primarily targets data-center applications.','NVIDIA primarily targets artificial-intelligence applications.'),why='The rewrite drops increasingly; do not import that deleted trend qualifier.')
cp('company_overview',5,4)
put('company_overview',6,atoms('company_overview',5,[0,1])+U('NVIDIA supplies developers worldwide.','NVIDIA supplies enterprises worldwide.','NVIDIA supplies consumers worldwide.','These offerings create value for NVIDIA.'),why='The final version no longer claims fulfillment of evolving demand; evaluate its retained customer and value propositions.')
cp('company_overview',7,6);cp('company_overview',8,7);cp('company_overview',10,9);cp('company_overview',12,11,[0]);cp('company_overview',13,12,[0,1,2]);cp('company_overview',14,13,[0]);cp('company_overview',15,13,[1,2,3]);cp('company_overview',16,14);cp('company_overview',17,15,[0,1,2])
put('company_overview',19,atoms('company_overview',17,[0,1,3])+U('NVIDIA executive team supports its growth strategy.'),why='The final claim drops aggressive from the growth-strategy qualifier.')
cp('company_overview',20,18,[0]);cp('company_overview',21,19,[0,1]);cp('company_overview',22,20,[1,2,3,4]);cp('company_overview',23,21,[3,4]);cp('company_overview',24,22)
cp('investment_overview',6,3,[0]);cp('investment_overview',7,3,[3,4]);cp('investment_overview',9,4,[1])
put('investment_overview',10,[S('NVIDIA reported positive substantial gross margin in Q2 FY2027.',((42,46),),'Both reported gross margins75%; strong gross profitability is directly consistent with the reported ratio.','correct')],why='Gross profitability is measured by gross margin itself; do not infer a new cost-management cause.')
cp('investment_overview',13,5,[0]);cp('investment_overview',14,5,[3,4]);cp('investment_overview',15,6,[0]);cp('investment_overview',16,6,[1]);cp('investment_overview',24,12);cp('investment_overview',26,14);cp('investment_overview',27,15,[1,2,3,4]);cp('investment_overview',28,16);cp('investment_overview',29,17);cp('investment_overview',30,18)
cp('valuation_overview',4,4,[0,1])
put('valuation_overview',5,[C('NVIDIA latest comparable EV/EBITDA is higher than AMD.',PEV,'2025 NVIDIA33.7774 is below AMD47.8456.'),S('NVIDIA latest comparable EV/EBITDA is higher than Intel.',PEV,'2025 NVIDIA33.7774 exceeds Intel14.4982.')],why='Separate the two entities for the only supplied peer valuation multiple; no peer PE series is available.')
cp('valuation_overview',7,7,[0,1,2,3])
put('valuation_overview',8,[],nonfactual='Reasonable valuation given fundamentals is a pure investment assessment without an additional specific factual premise.')
cp('valuation_overview',10,10,[0,1,2]);cp('valuation_overview',13,13);cp('valuation_overview',14,14)
cp('risks',1,1);cp('risks',4,4);cp('risks',5,5)
put('risks',6,U('NVIDIA operates in semiconductors.','AMD competes with NVIDIA.','Intel competes with NVIDIA.'),why='A peer-comparison table identifies comparison companies, not necessarily actual product competition; dropped fierce/dynamic/well-capitalized qualifiers are not retained.')
cp('risks',9,9,[0,1,2]);cp('risks',10,9,[3]);cp('risks',15,14);cp('risks',23,23);cp('risks',26,26)
put('risks',27,U('NVIDIA valuation is high relative to a relevant benchmark.'),nonfactual='Ambitious growth assumptions is a subjective assessment; the source forecast exists but no benchmark for ambitious is supplied.')
put('competitor_analysis',4,U('NVIDIA has a robust market presence.','NVIDIA has strong strategic positioning.'),why='The final removes the financial premise but preserves unsupported market-position conclusions.')
put('competitor_analysis',5,[],nonfactual='General analytic statement defining competitive-position assessment, not a new empirical NVIDIA observation.')
put('competitor_analysis',8,U('AMD competes with NVIDIA in consumer GPUs.','AMD competes with NVIDIA in data-center GPUs.','AMD competes with NVIDIA in consumer CPUs.','AMD competes with NVIDIA in data-center CPUs.'),why='The rewritten scope explicitly extends direct competition to both markets and product types; none is established in the prompt.')
cp('competitor_analysis',10,7,[0,2,3]);cp('competitor_analysis',11,8,[0,1])
put('competitor_analysis',12,atoms('competitor_analysis',8,[2,3])+U('Intel GPU and AI investments are intended to challenge NVIDIA.'),why='No longer asserts NVIDIA dominance; retained purpose is still unsupported.')
cp('competitor_analysis',13,9);cp('competitor_analysis',14,10,[0,1]);cp('competitor_analysis',15,11);cp('competitor_analysis',16,12)
put('competitor_analysis',17,U('Smaller AI-chip companies could disrupt NVIDIA markets.','Smaller ASIC companies could disrupt NVIDIA markets.','Smaller AI-chip companies could erode NVIDIA long-term market share.','Smaller ASIC companies could erode NVIDIA long-term market share.'),why='The rewritten sentence removes tailored/specific-application premises and retains two classes of threat.')
cp('competitor_analysis',20,16);cp('competitor_analysis',21,17,[2,3,4])
put('competitor_analysis',22,[C('NVIDIA revenue CAGR is projected at68.3% through2029.',F,'68.3% belongs to2022A–2026A historical CAGR. The three annual revenue forecasts are5%,6%,4%; the2022A–2029E blended CAGR is about37.5%, not68.3%.','incorrect')],why='The final correction still mislabels historical CAGR as forecast.')
cp('competitor_analysis',25,19);cp('competitor_analysis',26,20);cp('competitor_analysis',28,22,[2,3,4])
put('competitor_analysis',30,U('NVIDIA EV/EBITDA is high against an unspecified valuation benchmark.','NVIDIA valuation indicates investor confidence in durable competitive advantages.'),why='The final removes the explicit higher-than-both-peers comparison; do not carry the prior contradicted comparison into absent wording.')
cp('competitor_analysis',31,24);cp('competitor_analysis',32,25,[0,2])
cp('major_takeaways',0,0,[0]);cp('major_takeaways',1,0,[1,2]);cp('major_takeaways',3,2);cp('major_takeaways',4,3);cp('major_takeaways',5,4);cp('major_takeaways',6,5);cp('major_takeaways',9,8);cp('major_takeaways',10,9);cp('major_takeaways',11,10);cp('major_takeaways',12,11);cp('major_takeaways',13,12);cp('major_takeaways',14,13);cp('major_takeaways',15,14);cp('major_takeaways',16,14,[1,2]);cp('major_takeaways',17,15)
put('major_takeaways',20,[S('NVIDIA EBITDA profitability rises alongside recent annual revenue growth.',F+EM,'2023A–2026A revenue and EBITDA margins rise; efficiently is a qualitative description of that observed conversion.')],why='The final drops investor-benefit assertion; retain only the supported financial premise.')
cp('news_summary',0,0,[0])
put('news_summary',1,U('NVIDIA has strong positioning in AI-related markets.'))
cp('news_summary',4,2)
put('news_summary',7,U('NVIDIA dominates AI infrastructure.'))
cp('news_summary',10,6,[0,1])
put('news_summary',11,U('NVIDIA growth is driven by AI demand.','NVIDIA has market leadership.'),nonfactual='The description of the investment thesis is an analytic view; these underlying factual premises remain unsupported.')
cp('news_summary',14,8,[0,1,2]);cp('news_summary',15,8,[0,1,3])
put('news_summary',16,atoms('news_summary',8,[0,1]),nonfactual='The bullish stance is an investment opinion; forecast financial premises are still extracted.')
cp('news_summary',17,9,[0]);cp('news_summary',18,9,[1]);cp('news_summary',19,9,[2]);cp('news_summary',20,9,[2,3,4]);cp('news_summary',21,10);cp('news_summary',22,11,[0])
put('news_summary',23,U('NVIDIA leads AI solutions.','NVIDIA leads data-center solutions.'))
put('news_summary',24,U('NVIDIA has exceptional overall financial performance.'),why='Company-wide financial quality is broader than the supplied revenue, margin and EPS observations; no balance sheet or cash flow evidence establishes overall quality.')
put('news_summary',25,[S('NVIDIA projected revenue has a positive growth trajectory.',F,'Every future revenue column rises; promising is evaluative.')])
cp('news_summary',26,12,[0])
put('news_summary',27,U('NVIDIA has robust overall financial health.'),nonfactual='A favorable investor outlook focused on transformative technology is an opinion; the claimed financial health is the factual premise.')

def main():
    _,inputs=load_inputs(ROOT/'runs/NVDA_final_v6',ROOT/'runs/NVDA_baseline_1')
    for s,item in inputs.items():
        units=SPEC['sections'][s]['units'];ids=[u['unit_id'] for u in units];expected={u['unit_id'] for u in source_units(item['text'])}
        if len(ids)!=len(set(ids)) or set(ids)!=expected:raise ValueError(f'Incomplete final source review {s}: {set(ids)^expected}')
        units.sort(key=lambda u:u['unit_id'])
    write_json(WORK/'explicit_changed_unit_review.json',REV);write_json(ROOT/'audits/grounding/NVDA_after_v6_reviewed_specification.json',SPEC)
    print('Every NVDA final unit independently reviewed:',sum(len(s['units']) for s in SPEC['sections'].values()))
if __name__=='__main__':main()
