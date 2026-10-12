"""Alternate low-reasoning configuration on one frozen section; not a repeat of the high-effort request."""
import argparse,asyncio,sys
from pathlib import Path
from audit_io import Budget,credentials,call_agent,write_json,sha256
from experiment import contract,validate_success,setup
from grounding_audit import load_inputs
ROOT=Path(__file__).resolve().parents[1]
async def main(args):
    setup(Path.home()/'.cache/finrobot-zg2556/clean-verify-v7-1')
    from modules.equity_agents.verifier_agent import verifier_agent,prepare_verifier_prompt,validate_review
    _,inputs=load_inputs(ROOT/'runs/NVDA_baseline_1')
    item=inputs['company_overview'];prompt=prepare_verifier_prompt(item['evidence'],item['text'])
    cfg=credentials(args.config);cfg.update(openai_model='gpt-5.4-2026-03-05',reasoning_effort='low')
    directory=ROOT/'evidence/reasoning_pilot/NVDA_company_overview_low'
    budget=Budget(ROOT/'config/budget.reasoning_pilot.json',ROOT/'evidence/costs.jsonl')
    try:
        expected=contract(verifier_agent,prompt,cfg,None,20000)
        if (directory/'response.json').exists():
            output=validate_success(directory,verifier_agent,expected,budget)
        else:
            output=await call_agent(verifier_agent,prompt,config=cfg,budget=budget,directory=directory,
                call_id='reasoning_pilot_gpt54_low/NVDA_company_overview_low',temperature=None,max_tokens=20000)
            output=validate_success(directory,verifier_agent,expected,budget)
        checked=validate_review(output,item['evidence'],item['text'])
        write_json(directory/'checked.json',checked)
        write_json(directory/'pilot_scope.json',{'source_run_id':'NVDA_baseline_1','section':'company_overview',
            'writer_evidence_sha256':sha256(item['evidence']),'writer_text_sha256':sha256(item['text']),
            'instructions_unchanged_from':'V7.1','new_model_draw':True,'not_full_eight_section_run':True,
            'purpose':'One-section model/typed-output capability and semantic pilot. No efficacy claim from this single selected example; independent evaluation required.'})
        print('Real one-section reasoning pilot saved; deterministic checks complete; inspect semantic result.')
    finally:budget.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);asyncio.run(main(p.parse_args()))
