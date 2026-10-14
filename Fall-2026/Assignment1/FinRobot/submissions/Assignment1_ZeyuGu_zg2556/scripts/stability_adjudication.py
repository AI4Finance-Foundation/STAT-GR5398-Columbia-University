"""Explicit machine review of three actual writer runs; no model calls or invented runs.

Every risk sentence was read and assigned the exposures below. These are manual
code-authored machine judgments, NOT human validation or paid extraction results.
All quotes/offsets are resolved from saved text; raw failed API extraction is retained.
"""
import argparse
import itertools
import json
from pathlib import Path
from grounding_audit import source_units, csv_write
from audit_io import sha256, write_json, utc_now
from stability_audit import validate_independent_runs, overlap, numeric_summary

# Categories describe exposures in text, without endorsing their factual grounding.
# Revenue/segment concentration is deliberately separate from customer concentration.
ORDER = ('competition','demand_cyclicality','revenue_concentration','supply_capacity',
         'export_controls_geopolitics','regulation_litigation','technology_execution',
         'pricing_margin','financial_liquidity','valuation_market','cybersecurity_ip',
         'macro_currency','esg_reputation')
MAPS = {
1:{0:[],1:[],2:['demand_cyclicality'],3:['demand_cyclicality','macro_currency'],
   4:['technology_execution','competition'],5:['export_controls_geopolitics','supply_capacity'],
   6:['competition'],7:['competition','pricing_margin'],8:['technology_execution','competition'],
   9:['valuation_market'],10:['supply_capacity'],11:['supply_capacity','export_controls_geopolitics'],
   12:['technology_execution'],13:['technology_execution'],14:['technology_execution'],
   15:['financial_liquidity'],16:['financial_liquidity'],17:['valuation_market','financial_liquidity'],
   18:[],19:['export_controls_geopolitics','regulation_litigation','esg_reputation'],
   20:['regulation_litigation'],21:['regulation_litigation','cybersecurity_ip'],22:['esg_reputation'],
   23:['demand_cyclicality','macro_currency','technology_execution','competition','supply_capacity'],
   24:[],25:[],26:['valuation_market','technology_execution']},
2:{0:[],1:['demand_cyclicality','revenue_concentration'],2:['demand_cyclicality','macro_currency','export_controls_geopolitics'],
   3:['technology_execution','competition'],4:['valuation_market'],5:['competition','pricing_margin'],
   6:['technology_execution','competition','pricing_margin'],7:['supply_capacity'],
   8:['supply_capacity','export_controls_geopolitics'],9:['technology_execution'],10:['technology_execution'],
   11:['valuation_market'],12:['valuation_market','pricing_margin'],13:['financial_liquidity'],14:['financial_liquidity'],
   15:['export_controls_geopolitics','supply_capacity'],16:['regulation_litigation','cybersecurity_ip'],
   17:['esg_reputation'],18:['demand_cyclicality','macro_currency'],
   19:['competition','technology_execution','pricing_margin','valuation_market'],
   20:['supply_capacity','technology_execution'],21:['export_controls_geopolitics'],22:['valuation_market']},
3:{0:[],1:['demand_cyclicality','revenue_concentration'],2:[],
   3:['macro_currency','supply_capacity','demand_cyclicality'],4:['technology_execution','demand_cyclicality'],
   5:['competition'],6:['competition','pricing_margin'],7:['competition','pricing_margin'],
   8:['technology_execution','competition'],9:['technology_execution','financial_liquidity'],
   10:['supply_capacity','export_controls_geopolitics'],11:['technology_execution'],12:['technology_execution'],
   13:['pricing_margin','technology_execution'],14:[],15:['financial_liquidity'],16:['valuation_market'],
   17:['financial_liquidity'],18:[],19:['export_controls_geopolitics','regulation_litigation','esg_reputation'],
   20:['regulation_litigation','cybersecurity_ip'],21:['esg_reputation'],
   22:['demand_cyclicality','macro_currency','competition','pricing_margin','supply_capacity','technology_execution'],
   23:['valuation_market','financial_liquidity'],24:['regulation_litigation','esg_reputation'],25:[]}
}
RATING_PHRASES = {1:None,
 2:'suggesting hold or increase in position depending on portfolio strategy',
 3:'supporting a continued overweight position in the stock with a positive outlook over the next 6-12 months'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs=3,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    loaded=validate_independent_runs(args.runs)
    if [x['manifest']['run_id'] for x in loaded] != [f'NVDA_baseline_{i}' for i in (1,2,3)]:
        raise ValueError('This explicit review applies only to the three named, preserved NVDA runs')
    rows,coverage,observations,proof,ratings,inventory=[],[],[],[],[],[]
    risk_lists=[]
    for i,item in enumerate(loaded,1):
        run=item['manifest']['run_id'];units=source_units(item['sections']['risks']['text'])
        if set(MAPS[i])!=set(range(len(units))):raise ValueError('Risk source coverage changed')
        first=[]
        for unit in units:
            cats=sorted(MAPS[i][unit['unit_id']],key=ORDER.index)
            record={'run_id':run,'section':'risks','unit_id':unit['unit_id'],
                'source_start':unit['start'],'source_end':unit['end'],'source_quote':unit['text'],
                'source_sha256':sha256(item['sections']['risks']['text']),
                'taxonomy_categories':json.dumps(cats),
                'no_exposure_reason':('Generic framing, descriptive forecast, backward reference or mitigation; no distinct additional exposure.' if not cats else '')}
            coverage.append(record)
            for cat in cats:
                rows.append({**record,'taxonomy':cat})
                if cat not in first:first.append(cat)
        risk_lists.append(first[:3])
        # Preserve all eight exact report sections for rating/target inspection.
        for section,data in item['sections'].items():
            inventory.append({'run_id':run,'section':section,'source_sha256':sha256(data['text']),
                              'text':data['text']})
        phrase=RATING_PHRASES[i]
        source=item['sections']['investment_overview']['text']
        if phrase:
            if source.count(phrase)!=1:raise ValueError('Reviewed recommendation changed')
            start=source.index(phrase)
            ratings.append({'run_id':run,'section':'investment_overview','source_start':start,
                'source_end':start+len(phrase),'source_quote':phrase,
                'normalized_rating':'buy' if i==3 else 'N/A',
                'classification':'explicit_overweight' if i==3 else 'conditional_alternatives_hold_or_increase',
                'reason':'Overweight maps to buy only as an ordinal convention.' if i==3 else 'Conditional alternatives do not establish a unique standard rating.'})
        market=(item['market'] or {}).get('metrics',item['market'] or {})
        observations.append({'run_id':run,'writer_rating':('not_stated' if i==1 else 'conditional_hold_or_increase' if i==2 else 'overweight'),
            'writer_rating_ordinal':1 if i==3 else 'N/A','writer_target_price':'N/A','writer_target_status':'not_stated',
            'fmp_rating':json.dumps(market.get('rating')),'fmp_target_price':market.get('target_price'),
            'fmp_source_file':item['market_source_file'],
            **{f'first_mentioned_risk_{n+1}':cat for n,cat in enumerate(first[:3])},
            'input_tokens':sum(r['usage']['input_tokens'] for r in item['costs']),
            'output_tokens':sum(r['usage']['output_tokens'] for r in item['costs']),
            'generation_cost_usd':sum(r['cost_usd'] for r in item['costs']),
            'generation_model_runtime_seconds':sum(r['runtime_seconds'] for r in item['costs'])})
        proof+=item['proof']
    pairs=[]
    for i,j in itertools.combinations(range(3),2):
        jac,tau=overlap(risk_lists[i],risk_lists[j])
        pairs.append({'run_a':observations[i]['run_id'],'run_b':observations[j]['run_id'],
            'risk_jaccard':jac,'kendall_tau_shared':tau,
            'shared_categories':json.dumps(sorted(set(risk_lists[i])&set(risk_lists[j]))),
            'rank_basis':'First substantive textual mention; fixed taxonomy order for within-sentence ties. Not importance ranking.'})
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,data in [('stability_runs.csv',observations),('risk_unit_coverage.csv',coverage),
        ('all_risk_observations.csv',rows),('explicit_ratings.csv',ratings),('risk_pairwise.csv',pairs),
        ('generation_independence.csv',proof)]:csv_write(args.output/filename,data)
    write_json(args.output/'full_text_review_inventory.json',inventory)
    summary={'writer_target_price':{'status':'N/A: no explicit per-share target in any writer run'},
        'writer_rating_variance':{'status':'N/A: only one run has a unique standard rating; do not report one-observation zero variance'},
        'writer_rating_labels':[r['writer_rating'] for r in observations],
        'fmp_target_price':numeric_summary([r['fmp_target_price'] for r in observations]),
        'fmp_interpretation':'Frozen provider input; zero variance provides no evidence of LLM stability.',
        'risk_lists':risk_lists,'risk_pairs':pairs,'generation_count':3,'independent_writer_calls':len(proof),
        'classification_method':'Explicit Codex-assisted reading of all risk units and recommendation passages; not human validation.',
        'adjudication_api_cost_usd':None,'adjudication_tokens':None,'adjudication_runtime_seconds':None,
        'accounting_note':'Codex authoring usage is not instrumented by direct API ledger. Legacy API extraction failures are preserved and charged in the global ledger.',
        'limitations':'Three draws are a small sample. Risk taxonomy is subjective, segment concentration is not customer concentration, first mention is not importance, within-sentence ties use arbitrary fixed taxonomy order. Category overlap can hide differences in detailed exposures. Explicit labels alone do not exhaust implicit sentiment.'}
    write_json(args.output/'stability_summary.json',summary)
    write_json(args.output/'manifest.json',{'state':'complete','version':'stability-explicit-machine-review-v1',
        'completed_at_utc':utc_now(),'runs':[str(p.resolve()) for p in args.runs],
        'human_review_performed':False,'source_hashes':{r['run_id']+'|'+r['section']:r['source_sha256'] for r in inventory},
        'script_sha256':sha256(Path(__file__).read_bytes()),
        'file_hashes':{p.name:sha256(p.read_bytes()) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
