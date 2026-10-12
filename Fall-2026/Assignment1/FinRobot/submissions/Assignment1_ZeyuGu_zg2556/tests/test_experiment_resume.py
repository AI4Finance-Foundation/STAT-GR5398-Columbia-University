"""Offline resume control fixtures; no fixture is an assignment experiment result."""
import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import Agent, AgentOutputSchema
from pydantic import BaseModel

SUBMISSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION/'scripts'))
sys.path.insert(0, str(Path(os.environ.get('FINROBOT_UPSTREAM',Path.home()/'.cache/finrobot-zg2556/upstream'))/'finrobot_equity/core/src'))
import experiment as e
from audit_io import Budget, write_json
from modules.equity_agents.verifier_agent import verifier_agent, prepare_verifier_prompt, validate_review


class Output(BaseModel):
    tagline: str


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    for name in ('connect', 'connect_ex'):
        original = getattr(socket.socket, name)
        def guarded(sock, address, original=original):
            if isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}:
                return original(sock, address)
            raise AssertionError('Offline tests forbid external network')
        monkeypatch.setattr(socket.socket, name, guarded)


@pytest.fixture
def context(tmp_path):
    policy = tmp_path/'policy.json'
    write_json(policy, {'approved_cap_usd':20, 'input_usd_per_million':0.4,
        'output_usd_per_million':1.6, 'model':'fixture-model', 'price_source':'test fixture'})
    budget = Budget(policy, tmp_path/'ledger.jsonl')
    cfg = {'openai_model':'fixture-model', 'openai_base_url':'https://example.invalid/v1',
           'openai_api_key':'fixture-secret'}
    agent = Agent(name='Fixture', instructions='Use only evidence.', output_type=Output)
    yield SimpleNamespace(root=tmp_path, policy=policy, budget=budget, cfg=cfg, agent=agent)
    budget.close()


def fixture_call(directory, agent, prompt, config, budget, call_id, temperature, max_tokens, output, fail=False):
    directory.mkdir(parents=True)
    request = dict(e.contract(agent, prompt, config, temperature, max_tokens), call_id=call_id,
                   timestamp_utc='2026-10-11T00:00:00+00:00')
    write_json(directory/'request.json', request)
    (directory/'prompt.txt').write_text(prompt, encoding='utf-8')
    (directory/'instructions.txt').write_text(agent.instructions, encoding='utf-8')
    budget.reserve(call_id, json.dumps(request), max_tokens)
    wire = {'model':config['openai_model'], 'temperature':temperature, 'max_tokens':max_tokens,
        'messages':[{'role':'system','content':agent.instructions},{'role':'user','content':prompt}],
        'response_format':{'type':'json_schema', 'json_schema':{'schema':AgentOutputSchema(agent.output_type).json_schema()}}}
    write_json(directory/'wire_request.json', wire)
    if fail:
        write_json(directory/'error.json', {'call_id':call_id,'message':'Offline fixture interruption'})
        raise RuntimeError('fixture interruption')
    cost = budget.settle(call_id, 20, 10, 0.5)
    usage = {'requests':1, 'input_tokens':20, 'output_tokens':10, 'total_tokens':30}
    write_json(directory/'response.json', {'call_id':call_id, 'structured_output':output,
        'usage':usage,'runtime_seconds':0.5,'cost_usd':cost,'completed_at_utc':'2026-10-11T00:00:01+00:00','attempts':1})
    write_json(directory/'wire_response.json', {'id':'fixture-'+call_id,
        'choices':[{'message':{'content':json.dumps(output)}}],
        'usage':{'prompt_tokens':20, 'completion_tokens':10, 'total_tokens':30}})
    write_json(directory/'http_response.json', {'status_code':200,'request_id':'fixture-http-'+call_id})
    return agent.output_type.model_validate(output)


def persist(c, directory=None):
    directory = directory or c.root/'call'
    fixture_call(directory,c.agent,'Evidence',c.cfg,c.budget,'fixture/call',0.7,4000,{'tagline':'Supported content.'})
    return directory, e.contract(c.agent,'Evidence',c.cfg,0.7,4000)


def test_paid_success_revalidated_without_network(context):
    directory, expected = persist(context)
    assert e.validate_success(directory, context.agent, expected, context.budget).tagline == 'Supported content.'


@pytest.mark.parametrize('mutation',[None,'effort','temperature','legacy_limit','completion_limit'])
def test_reasoning_wire_contract_reuse_rejects_parameter_drift(context,mutation):
    directory,_=persist(context)
    cfg=dict(context.cfg,reasoning_effort='high')
    expected=e.contract(context.agent,'Evidence',cfg,None,4000)
    request=e.read_json(directory/'request.json');request.update(expected)
    write_json(directory/'request.json',request)
    wire=e.read_json(directory/'wire_request.json')
    wire.pop('temperature');wire['max_completion_tokens']=wire.pop('max_tokens');wire['reasoning_effort']='high'
    if mutation=='effort':wire['reasoning_effort']='low'
    if mutation=='temperature':wire['temperature']=None
    if mutation=='legacy_limit':wire['max_tokens']=4000
    if mutation=='completion_limit':wire['max_completion_tokens']=3999
    write_json(directory/'wire_request.json',wire)
    if mutation is None:
        assert e.validate_success(directory,context.agent,expected,context.budget).tagline=='Supported content.'
    else:
        with pytest.raises(ValueError):e.validate_success(directory,context.agent,expected,context.budget)


@pytest.mark.parametrize('mutation', ['model','prompt','instructions','wire_output','usage','schema','cost'])
def test_corrupted_success_is_not_reused(context, mutation):
    directory, expected = persist(context)
    if mutation == 'prompt':
        (directory/'prompt.txt').write_text('changed')
    elif mutation == 'instructions':
        (directory/'instructions.txt').write_text('changed')
    else:
        filename = {'model':'request.json','wire_output':'wire_response.json','usage':'response.json',
                    'schema':'wire_request.json','cost':'response.json'}[mutation]
        record = e.read_json(directory/filename)
        if mutation == 'model': record['model']='changed'
        if mutation == 'wire_output': record['choices'][0]['message']['content']='{"tagline":"changed"}'
        if mutation == 'usage': record['usage']['input_tokens']=21
        if mutation == 'schema': record['response_format']['json_schema']['schema']={}
        if mutation == 'cost': record['cost_usd']=0
        write_json(directory/filename,record)
    with pytest.raises(ValueError): e.validate_success(directory, context.agent, expected, context.budget)


def test_explicit_resume_identity_and_same_run_semantics(tmp_path):
    run = tmp_path/'run'; manifest={'run_id':'run','model':'M','max_tokens':4000}
    e.begin_run(run,manifest,False,tuple(manifest))
    with pytest.raises(FileExistsError): e.begin_run(run,manifest,False,tuple(manifest))
    with pytest.raises(ValueError,match='identity changed'): e.begin_run(run,dict(manifest,model='N'),True,tuple(manifest))
    resumed=e.begin_run(run,manifest,True,tuple(manifest))
    assert resumed['run_id']=='run' and len(resumed['resume_events'])==1
    assert 'never an independent' in resumed['resume_events'][0]['meaning']
    with pytest.raises(FileNotFoundError): e.begin_run(tmp_path/'missing',manifest,True,tuple(manifest))


def test_failed_attempt_preserved_and_unique_id(context):
    current=context.root/'run/agents/tagline'
    with pytest.raises(RuntimeError):
        fixture_call(current,context.agent,'Evidence',context.cfg,context.budget,
                     'run/tagline',0.7,4000,{},fail=True)
    old_hashes=e.hashes(current)
    archived=e.archive_attempt(context.root/'run','agents','tagline')
    assert e.hashes(archived)==old_hashes and not current.exists()
    number=e.attempt_number(context.root/'run','agents','tagline')
    assert number==2
    fixture_call(current,context.agent,'Evidence',context.cfg,context.budget,
                 f'run/tagline/attempt_{number:03d}',0.7,4000,{'tagline':'Supported content.'})
    assert e.hashes(archived)==old_hashes
    rows=context.budget.rows()
    assert len([r for r in rows if r['event']=='reserve'])==2
    assert len([r for r in rows if r['event']=='settle'])==1


def test_recover_text_cannot_overwrite_tampered_success(tmp_path):
    path=tmp_path/'output.txt'
    e.same_or_write(path,'valid')
    assert path.read_text()=='valid'
    path.write_text('tampered')
    with pytest.raises(ValueError,match='text changed'): e.same_or_write(path,'valid')


def good_review():
    return {'units':[{'source_id':0,'nonfactual_reason':'','claims':[{
        'original_quote':'Revenue is 10.','claim':'Revenue is 10.','classification':'Supported',
        'evidence':[{'line_number':1}], 'reason':'Exact fixture evidence.',
        'action':'keep','revised_claim':'Revenue is 10.',
        'number_bindings':[{'claim_number':'10','evidence_line':1,'evidence_number':'10','transform':'identity'}]}]}]}


def verifier_args(c):
    args=SimpleNamespace(final_run=c.root/'final', max_tokens=26000,max_validation_repairs=2)
    args.final_run.mkdir()
    manifest={'state':'verifying','run_id':'final','resume_events':[]}
    write_json(args.final_run/'manifest.json',manifest)
    return args,manifest


def test_review_derivation_preserves_wire_billing_and_rejects_input_change(context,monkeypatch):
    args,manifest=verifier_args(context)
    args.reuse_review_run=context.root/'prior'
    source=args.reuse_review_run/'verifier/tagline'
    prompt=prepare_verifier_prompt('Revenue is 10.','Revenue is 10.')
    fixture_call(source,verifier_agent,prompt,context.cfg,context.budget,'prior/verifier/tagline',0,26000,good_review())
    before_hashes=e.hashes(args.reuse_review_run);before_ledger=context.budget.rows()
    e.adopt_prior_review(args,'tagline',verifier_agent,'Revenue is 10.','Revenue is 10.',
                        context.cfg,context.budget,validate_review,prompt)
    async def forbid(*a,**k):raise AssertionError('Review derivation must not make a paid call')
    monkeypatch.setattr(e,'call_agent',forbid)
    checked,reused,call_id=asyncio.run(e.verify_section(args,manifest,'tagline',verifier_agent,
        'Revenue is 10.','Revenue is 10.',context.cfg,context.budget,validate_review,prompt))
    assert reused and call_id is None and checked['revised_text']=='Revenue is 10.'
    assert context.budget.rows()==before_ledger and e.hashes(args.reuse_review_run)==before_hashes
    record=e.read_json(args.final_run/'verifier/tagline/reused_review.json')
    assert record['source_call_id']=='prior/verifier/tagline' and not record['new_model_generation']
    with pytest.raises(ValueError,match='source/evidence changed'):
        asyncio.run(e.verify_section(args,manifest,'tagline',verifier_agent,
            'Revenue is 99.','Revenue is 10.',context.cfg,context.budget,validate_review,prompt))


def test_verifier_repairs_are_bounded_across_resumes(context,monkeypatch):
    args,manifest=verifier_args(context);calls=[]
    async def fake(agent,prompt,**kw):
        calls.append(kw['call_id'])
        return fixture_call(kw['directory'],agent,prompt,kw['config'],kw['budget'],kw['call_id'],
                            kw['temperature'],kw['max_tokens'],{'units':[]})
    monkeypatch.setattr(e,'call_agent',fake)
    base=prepare_verifier_prompt('Revenue is 10.','Revenue is 10.')
    def run():
        return asyncio.run(e.verify_section(args,manifest,'tagline',verifier_agent,'Revenue is 10.',
                         'Revenue is 10.',context.cfg,context.budget,validate_review,base))
    with pytest.raises(ValueError,match='repair limit reached'):run()
    assert len(calls)==3 and len(set(calls))==3
    with pytest.raises(ValueError,match='repair limit reached'):run()
    assert len(calls)==3
    assert len(list((args.final_run/'attempt_history/verifier/tagline').glob('attempt_*')))==2


def test_verifier_failed_repair_resumes_exact_prompt_and_keeps_costs(context,monkeypatch):
    args,manifest=verifier_args(context);calls=[];prompts=[]
    async def fake(agent,prompt,**kw):
        calls.append(kw['call_id']);prompts.append(prompt)
        output=good_review() if len(calls)>1 else {'units':[]}
        return fixture_call(kw['directory'],agent,prompt,kw['config'],kw['budget'],kw['call_id'],
                            kw['temperature'],kw['max_tokens'],output,fail=len(calls)==2)
    monkeypatch.setattr(e,'call_agent',fake)
    base=prepare_verifier_prompt('Revenue is 10.','Revenue is 10.')
    def run():
        return asyncio.run(e.verify_section(args,manifest,'tagline',verifier_agent,'Revenue is 10.',
                         'Revenue is 10.',context.cfg,context.budget,validate_review,base))
    with pytest.raises(RuntimeError,match='fixture interruption'):run()
    checked,reused,_=run()
    assert checked['revised_text']=='Revenue is 10.' and not reused
    assert len(set(calls))==3 and prompts[1]==prompts[2] and prompts[0]!=prompts[1]
    assert len([x for x in context.budget.rows() if x['event']=='reserve'])==3
    assert run()[1] is True and len(calls)==3
    history=args.final_run/'attempt_history/verifier/tagline'
    assert (history/'attempt_001/validation_error.json').exists()
    assert (history/'attempt_002/error.json').exists()


def test_changed_archived_validation_diagnostic_is_rejected(context,monkeypatch):
    args,manifest=verifier_args(context);count=0
    async def fake(agent,prompt,**kw):
        nonlocal count
        count+=1
        return fixture_call(kw['directory'],agent,prompt,kw['config'],kw['budget'],kw['call_id'],
                            kw['temperature'],kw['max_tokens'],{'units':[]},fail=count==2)
    monkeypatch.setattr(e,'call_agent',fake)
    base=prepare_verifier_prompt('Revenue is 10.','Revenue is 10.')
    def run():return asyncio.run(e.verify_section(args,manifest,'tagline',verifier_agent,'Revenue is 10.',
                          'Revenue is 10.',context.cfg,context.budget,validate_review,base))
    with pytest.raises(RuntimeError):run()
    path=args.final_run/'attempt_history/verifier/tagline/attempt_001/validation_error.json'
    record=e.read_json(path);record['message']='tampered';write_json(path,record)
    with pytest.raises(ValueError,match='diagnostics changed'):run()
    assert count==2


def test_frozen_render_sibling_inputs_preserve_wrapper_and_reject_changes(tmp_path,monkeypatch):
    snapshot=tmp_path/'snapshot';(snapshot/'analysis').mkdir(parents=True)
    monkeypatch.setattr(e,'SUBMISSION_ROOT',tmp_path)
    (snapshot/'analysis'/'accounts.csv').write_text('year,revenue\n2025,100\n',encoding='utf-8')
    inputs=e.hashes(snapshot/'analysis')
    market={'metrics':{'share_price':123.0},'as_of_utc':'2026-10-11T21:10:00+00:00'}
    write_json(snapshot/'market_snapshot.json',market)
    write_json(snapshot/'valuation_inputs.json',{'shares':10})
    write_json(snapshot/'valuation/engine_valuation.json',{'target_price':25})
    write_json(snapshot/'manifest.json',{'status':'frozen','upstream_sha':e.UPSTREAM_SHA,
       'run_id':'fixture-snapshot','file_hashes':e.hashes(snapshot)})
    run=tmp_path/'run';run.mkdir();manifest={'source_analysis':str(snapshot/'analysis'),'input_hashes':inputs}
    output=e.freeze_render_inputs(run,manifest)
    assert e.read_json(output)==market and manifest['render_input_hashes']
    assert e.hashes(snapshot/'analysis')==inputs
    write_json(snapshot/'market_snapshot.json',{'metrics':{'share_price':999.0}})
    # A completed run is self-contained, but a new copy must reject changed source bytes.
    assert e.read_json(e.freeze_render_inputs(run,manifest))==market
    with pytest.raises(ValueError,match='supplement changed'):
        e.freeze_render_inputs(tmp_path/'new_run',{'source_analysis':str(snapshot/'analysis'),'input_hashes':inputs})
    write_json(output,{'metrics':{'share_price':999.0}})
    with pytest.raises(ValueError,match='Copied render inputs changed'):e.freeze_render_inputs(run,manifest)

def test_full_baseline_resume_reuses_success_and_retries_only_failed_call(context,monkeypatch):
    from modules.equity_agents import agent_manager
    original_manager=agent_manager.EquityResearchAgentManager
    class Manager(original_manager):
        def _prepare_financial_data_prompt(self,*args):return 'Immutable fixture evidence.'
    monkeypatch.setattr(agent_manager,'EquityResearchAgentManager',Manager)
    monkeypatch.setattr(e,'load_data',lambda path:{})
    monkeypatch.setattr(e,'credentials',lambda path:dict(context.cfg))
    context.budget.close()
    analysis=context.root/'source/analysis';analysis.mkdir(parents=True)
    (analysis/'data.csv').write_text('metric,value\nRevenue,10\n')
    args=SimpleNamespace(run=context.root/'baseline',analysis=analysis,ticker='FIX',company_name='Fixture',
        config=None,budget=context.policy,ledger=context.root/'ledger.jsonl',temperature=0.7,max_tokens=4000,resume=False)
    calls=[]
    async def fake(agent,prompt,**kw):
        calls.append(kw['call_id'])
        field=next(iter(agent.output_type.model_fields))
        return fixture_call(kw['directory'],agent,prompt,kw['config'],kw['budget'],kw['call_id'],
                  kw['temperature'],kw['max_tokens'],{field:'Fixture source-supported prose.'},fail=len(calls)==3)
    monkeypatch.setattr(e,'call_agent',fake)
    with pytest.raises(RuntimeError,match='fixture interruption'):asyncio.run(e.generate(args))
    completed={p.name:e.hashes(p) for p in (args.run/'agents').iterdir() if (p/'response.json').exists()}
    assert len(completed)==2
    args.resume=True
    asyncio.run(e.generate(args))
    manifest=e.read_json(args.run/'manifest.json')
    assert manifest['state']=='generated' and len(calls)==9 and len(set(calls))==9
    assert len(manifest['resume_events'][0]['reused_sections'])==2
    for section,signature in completed.items():assert e.hashes(args.run/'agents'/section)==signature
    assert len(list((args.run/'attempt_history/agents').glob('*/attempt_001/error.json')))==1
    asyncio.run(e.generate(args))
    assert len(calls)==9 and len(e.read_json(args.run/'manifest.json')['resume_events'][-1]['reused_sections'])==8
    args.temperature=0.8
    with pytest.raises(ValueError,match='identity changed'):asyncio.run(e.generate(args))
    assert len(calls)==9

def test_crash_after_settlement_recovers_without_new_reservation(context):
    directory,expected=persist(context)
    original=e.read_json(directory/'response.json')
    # Test-only simulated interruption after settlement, before response save.
    (directory/'response.json').unlink()
    before=context.budget.rows()
    e.recover_settled_response(directory,context.agent,expected,context.budget)
    result=e.validate_success(directory,context.agent,expected,context.budget)
    assert result.tagline==original['structured_output']['tagline']
    assert e.read_json(directory/'response.json')['recovered_from_settled_wire']
    assert (directory/'recovery.json').exists() and context.budget.rows()==before


def test_unknown_failed_attempt_is_not_automatically_settled(context):
    directory=context.root/'failure';expected=e.contract(context.agent,'Evidence',context.cfg,0.7,4000)
    with pytest.raises(RuntimeError):
        fixture_call(directory,context.agent,'Evidence',context.cfg,context.budget,
                     'failure',0.7,4000,{},fail=True)
    before=context.budget.rows()
    e.recover_settled_response(directory,context.agent,expected,context.budget)
    assert not (directory/'response.json').exists() and context.budget.rows()==before
