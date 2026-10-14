"""Revalidate all seven real V7.1 reviews with V8.1, forbidding new paid calls.

Inputs, prompts, model outputs and ledger are preserved. These are derived runs,
not independent generations or stability repetitions.
"""
import argparse,asyncio,json,socket,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import experiment as e
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

async def forbid_paid(*args,**kwargs):
    raise RuntimeError('Offline derivation prohibits any new model call, including repairs.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);args=p.parse_args()
    upstream=args.upstream.resolve();src=e.setup(upstream)
    source=src/'modules/equity_agents/verifier_agent.py'
    assert sha256(source.read_bytes())==sha256((ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py').read_bytes())
    ledger=ROOT/'evidence/costs.jsonl';old=sha256(ledger.read_bytes());records=[]
    actual_connect=socket.socket.connect
    def guarded_connect(sock,address):
        if isinstance(address,tuple) and address[0] in {'127.0.0.1','::1'}:return actual_connect(sock,address)
        raise RuntimeError('External networking prohibited during saved-review revalidation.')
    for ticker in ['NVDA','AMD','INTC','AVGO','QCOM','AAPL','GOOGL']:
        prior=ROOT/'runs'/f'{ticker}_final_v7_1';params=e.read_json(prior/'manifest.json')['verifier_parameters']
        configuration={'openai_model':params['model'],'openai_base_url':params['provider_base_url']}
        target=ROOT/'runs'/f'{ticker}_final_v8_1'
        options=SimpleNamespace(run=ROOT/'runs'/f'{ticker}_baseline_1',final_run=target,
            reuse_review_run=prior,max_tokens=26000,max_validation_repairs=2,reasoning_effort=None,
            config=None,budget=str(ROOT/'config/budget.evaluation.json'),ledger=ledger,
            resume=target.exists(),upstream=upstream)
        started=time.perf_counter()
        with patch.object(e,'credentials',lambda _:configuration),patch.object(e,'call_agent',forbid_paid),patch.object(socket.socket,'connect',guarded_connect):
            asyncio.run(e.verify(options))
        e.render(SimpleNamespace(run=target,upstream=upstream),src)
        sections=[]
        for section in e.FIELDS:
            before=e.read_json(prior/'verifier'/section/'checked.json');after=e.read_json(target/'verifier'/section/'checked.json')
            sections.append({'section':section,'text_changed':before['revised_text']!=after['revised_text'],
                'before_verified':before['verified_claim_count'],'after_verified':after['verified_claim_count'],
                'before_failures':before['deterministic_failure_count'],'after_failures':after['deterministic_failure_count']})
        records.append({'run_id':target.name,'runtime_seconds':time.perf_counter()-started,'sections':sections})
        print(ticker+': offline revalidation and exact-text render complete; '+str(sum(x['text_changed'] for x in sections))+' changed sections.',flush=True)
    assert old==sha256(ledger.read_bytes())
    write_json(ROOT/'evidence/verifier_versions/v8_1_revalidation.json',{'at_utc':utc_now(),
        'source_sha256':sha256(source.read_bytes()),'new_model_calls':0,'ledger_sha256_before_after':old,
        'not_independent_generation':True,'records':records,
        'next':'Independent full-section review of changed text required before efficacy claims or replacing canonical release.'})
if __name__=='__main__':main()
