"""Reconcile the observed length-limited pilot from saved provider usage only.

No API call, success response, or claim label is manufactured here.
"""
import json
from pathlib import Path
from audit_io import Budget,sha256,utc_now,write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'evidence/reasoning_pilot/NVDA_company_overview'
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    request,wire,http,error=map(read,['request.json','wire_response.json','http_response.json','error.json'])
    assert http['status_code']==200 and wire['model']==request['model']=='gpt-5.4-2026-03-05'
    assert wire['choices'][0]['finish_reason']=='length' and not wire['choices'][0]['message']['content']
    usage=wire['usage'];assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
    assert usage['completion_tokens_details']['reasoning_tokens']==usage['completion_tokens']==26000
    assert not (folder/'response.json').exists() and not (folder/'checked.json').exists()
    with Budget(ROOT/'config/budget.reasoning_pilot.json',ROOT/'evidence/costs.jsonl') as budget:
        settlements=[r for r in budget.rows() if r['event']=='settle' and r['call_id']==request['call_id']]
        if not settlements:
            budget.settle(request['call_id'],usage['prompt_tokens'],usage['completion_tokens'],error['runtime_seconds'])
        else:
            assert len(settlements)==1 and settlements[0]['input_tokens']==usage['prompt_tokens'] and settlements[0]['output_tokens']==usage['completion_tokens']
    write_json(folder/'reconciliation.json',{'at_utc':utc_now(),'call_id':request['call_id'],
        'provider_completion_id':wire['id'],'wire_response_sha256':sha256((folder/'wire_response.json').read_bytes()),
        'outcome':'FAILED_LENGTH_NO_VISIBLE_OUTPUT','input_tokens':usage['prompt_tokens'],
        'output_tokens':usage['completion_tokens'],'reasoning_tokens':26000,'estimated_cost_usd':0.40093,
        'runtime_seconds':error['runtime_seconds'],'new_call_performed':False,
        'basis':'Complete HTTP200 provider usage, including billed hidden reasoning; not an invoice. Original error and response bytes retained unchanged.',
        'rollout_decision':'Do not expand this pilot. No usable verification or efficacy score. Canonical V6 and separately measured V7.1 remain unchanged.'})
    print('Reconciled failed pilot from provider usage: USD0.40093; no further API call.')
if __name__=='__main__':main()
