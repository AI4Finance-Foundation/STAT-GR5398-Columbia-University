"""Derive a versioned follow-up from captured paid V8.1 reviews, without paid capability.

Other companies may still generate serially in the paid driver. A read-only view
of actual settled ledger records verifies provenance without taking its write lock.
"""
import argparse,asyncio,json,time,contextlib,socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import experiment as e
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

class ReadOnlyLedger:
    def __init__(self,policy,ledger):
        self.policy=json.loads(Path(policy).read_text());self.bytes=Path(ledger).read_bytes()
        self.records=[json.loads(line) for line in self.bytes.decode().splitlines()]
    def rows(self):return self.records
    def close(self):pass
    def reserve(self,*args,**kwargs):raise RuntimeError('Read-only derivation cannot reserve a paid call')
    def settle(self,*args,**kwargs):raise RuntimeError('Read-only derivation cannot alter billing')

async def forbid_paid(*args,**kwargs):raise RuntimeError('Offline derivation prohibits new completions or repairs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--tickers',nargs='+',default=['NVDA','AMD','INTC'])
    p.add_argument('--verifier-version',choices=['v8_2','v9','v9_1'],default='v8_2')
    p.add_argument('--output-root',type=Path,default=ROOT/'runs',help='For reproduction use a new directory under reproduction/; original construction used runs/')
    p.add_argument('--disable-network',action='store_true',help='Reject socket connections throughout verification and rendering')
    args=p.parse_args()
    def network_guard():
        stack=contextlib.ExitStack()
        if args.disable_network:
            def forbidden_network(*a,**kw):raise RuntimeError('Network forbidden during offline reasoning replay')
            stack.enter_context(patch.object(socket,'create_connection',forbidden_network))
            stack.enter_context(patch.object(socket.socket,'connect',forbidden_network))
        return stack
    async def guarded_verify(options):
        # Windows asyncio initializes a local wakeup socket before this body runs.
        # All connections during actual verification remain unconditionally blocked.
        with network_guard():
            await e.verify(options)
    output_root=args.output_root.resolve()
    upstream=args.upstream.resolve();src=e.setup(upstream)
    applied_source=(src/'modules/equity_agents/verifier_agent.py').read_bytes()
    archived_source=(ROOT/f'evidence/verifier_versions/verifier_agent_{args.verifier_version}.py').read_bytes()
    # Git may normalize a mixed CRLF/LF archive on clean apply. Accept only that
    # byte transformation, never whitespace, syntax or semantic normalization.
    assert applied_source.replace(b'\r\n',b'\n')==archived_source.replace(b'\r\n',b'\n')
    original_subprocess_run=e.subprocess.run
    def offline_renderer(command,*positional,**kwargs):
        if not args.disable_network:
            return original_subprocess_run(command,*positional,**kwargs)
        if Path(command[1]).resolve()!=src/'create_equity_report.py':
            raise RuntimeError('Unexpected subprocess during guarded renderer replay')
        wrapper="import runpy,socket,sys,os\ndef forbidden(*a,**k): raise RuntimeError('Network forbidden in replay renderer')\nsocket.create_connection=forbidden\nsocket.socket.connect=forbidden\nsys.argv=sys.argv[1:]\nsys.path.insert(0,os.path.dirname(sys.argv[0]))\nrunpy.run_path(sys.argv[0],run_name='__main__')"
        return original_subprocess_run([command[0],'-c',wrapper,*command[1:]],*positional,**kwargs)
    for ticker in args.tickers:
        prior=ROOT/'runs'/f'{ticker}_final_reasoning_v8_1';manifest=e.read_json(prior/'manifest.json')
        if manifest['state'] not in {'generated','rendered'}:raise ValueError('Paid source run is not complete: '+ticker)
        params=manifest['verifier_parameters'];cfg={'openai_model':params['model'],'openai_base_url':params['provider_base_url']}
        target=output_root/f'{ticker}_final_reasoning_{args.verifier_version}'
        options=SimpleNamespace(run=ROOT/'runs'/f'{ticker}_baseline_1',final_run=target,reuse_review_run=prior,
            max_tokens=20000,max_validation_repairs=2,reasoning_effort='low',config=None,
            budget=str(ROOT/'config/budget.reasoning_followup.json'),ledger=ROOT/'evidence/costs.jsonl',
            resume=target.exists(),upstream=upstream)
        start=time.perf_counter()
        with patch.object(e,'Budget',ReadOnlyLedger),patch.object(e,'credentials',lambda _:cfg),patch.object(e,'call_agent',forbid_paid):
            asyncio.run(guarded_verify(options))
        with network_guard(),patch.object(e.subprocess,'run',offline_renderer):
            e.render(SimpleNamespace(run=target,upstream=upstream),src)
        rows=[]
        for section in e.FIELDS:
            old=e.read_json(prior/'verifier'/section/'checked.json');new=e.read_json(target/'verifier'/section/'checked.json')
            copied=e.read_json(target/'verifier'/section/'reused_review.json')
            rows.append({'section':section,'source_paid_call_id':copied['source_call_id'],
                'text_changed':old['revised_text']!=new['revised_text'],
                'before_verified':old['verified_claim_count'],'after_verified':new['verified_claim_count'],
                'before_failures':old['deterministic_failure_count'],'after_failures':new['deterministic_failure_count']})
        write_json(target/'offline_derivation.json',{'at_utc':utc_now(),'runtime_seconds':time.perf_counter()-start,
            'new_model_calls':0,'new_model_cost_usd':0,'source_run_id':prior.name,'records':rows,
            'network_disabled_during_verification_and_render':args.disable_network,
            'output_root':str(output_root),
            'replay_script_sha256':sha256(Path(__file__).read_bytes()),
            'applied_verifier_source_sha256':sha256(applied_source),
            'archived_verifier_source_sha256':sha256(archived_source),
            'source_comparison':'Exact bytes after CRLF-to-LF only; both raw hashes preserved.',
            'billing_control':'ReadOnlyLedger has no reserve/settle capability; call_agent replaced with unconditional failure. Actual captured provider bytes and original settlement records validated.',
            'not_independent_generation':True,'independent_semantic_audit':'Required before any efficacy claim.'})
        print(ticker+': derived and rendered '+args.verifier_version+'; original paid reviews preserved.',flush=True)
if __name__=='__main__':main()
