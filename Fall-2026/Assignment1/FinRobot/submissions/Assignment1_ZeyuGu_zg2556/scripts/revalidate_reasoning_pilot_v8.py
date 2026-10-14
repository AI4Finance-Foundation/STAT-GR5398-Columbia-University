"""Offline format repair on the real low-effort output; no new generation."""
import json
from pathlib import Path
from audit_io import Budget,sha256,utc_now,write_json
from experiment import setup,contract,validate_success
from grounding_audit import load_inputs
ROOT=Path(__file__).resolve().parents[1]

def main():
    setup(Path.home()/'.cache/finrobot-zg2556/upstream')
    from modules.equity_agents.verifier_agent import verifier_agent,prepare_verifier_prompt,validate_review
    original=ROOT/'evidence/reasoning_pilot/NVDA_company_overview_low'
    out=ROOT/'evidence/reasoning_pilot/NVDA_company_overview_low_v8_1_revalidation'
    req=json.loads((original/'request.json').read_text(encoding='utf-8'))
    cfg={'openai_model':req['model'],'openai_base_url':req['provider_base_url'],'reasoning_effort':'low'}
    _,inputs=load_inputs(ROOT/'runs/NVDA_baseline_1');item=inputs['company_overview']
    prompt=prepare_verifier_prompt(item['evidence'],item['text'])
    with Budget(ROOT/'config/budget.reasoning_pilot.json',ROOT/'evidence/costs.jsonl') as budget:
        rows=budget.rows()
        output=validate_success(original,verifier_agent,contract(verifier_agent,prompt,cfg,None,20000),budget)
        checked=validate_review(output,item['evidence'],item['text'])
        assert rows==budget.rows()
    before=json.loads((original/'checked.json').read_text(encoding='utf-8'))
    write_json(out/'checked.json',checked)
    write_json(out/'manifest.json',{'revalidated_utc':utc_now(),'original_response':str(original.relative_to(ROOT)),
        'response_sha256':sha256((original/'response.json').read_bytes()),
        'source_run_id':'NVDA_baseline_1','section':'company_overview','new_generation':False,'incremental_model_cost':0,
        'source_sha256':sha256((ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py').read_bytes()),
        'before_verified':before['verified_claim_count'],'after_verified':checked['verified_claim_count'],
        'before_deterministic_failures':before['deterministic_failure_count'],'after_deterministic_failures':checked['deterministic_failure_count'],
        'still_needs_semantic_review':True,'not_full_eight_section_experiment':True,
        'known_semantic_counterexample':'The model labels 2023A-to-2025A revenue CAGR >68% Contradicted, but (130497/26974)^(1/2)-1=119.9518%, so that inequality is true. Format acceptance cannot establish classification accuracy.'})
    print(json.dumps({'before_verified':before['verified_claim_count'],'after_verified':checked['verified_claim_count'],
        'before_failures':before['deterministic_failure_count'],'after_failures':checked['deterministic_failure_count'],
        'paid_calls':0,'residual_semantic_error_disclosed':True}))
if __name__=='__main__':main()
