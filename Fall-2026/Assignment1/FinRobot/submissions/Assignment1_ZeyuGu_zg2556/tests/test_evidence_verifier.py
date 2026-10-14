"""Offline control fixtures only: none of these are empirical equity results."""
import asyncio
import json
import os
import socket
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

SUBMISSION = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('FINROBOT_UPSTREAM', Path.home()/'.cache/finrobot-zg2556/upstream'))
sys.path.insert(0, str(SUBMISSION/'scripts'))
sys.path.insert(0, str(UPSTREAM/'finrobot_equity/core/src'))
from audit_io import Budget, SnapshotHTTP, call_agent, public_url, redact, sha256, write_json
from experiment import hashes, validate_hashes
from modules.equity_agents.verifier_agent import (
    AtomicClaim, EvidenceQuote, NumberBinding, UnitReview, VerifierResponse,
    validate_review, source_units, NUMBER,
)


@pytest.fixture(autouse=True)
def block_actual_network(monkeypatch):
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def guarded(method):
        def call(sock, address):
            # Windows asyncio implements its internal self-pipe with loopback sockets.
            if isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}:
                return method(sock, address)
            raise AssertionError('Tests forbid external network access')
        return call
    def fail(*args, **kwargs):
        raise AssertionError('Tests forbid actual network access')
    monkeypatch.setattr(socket.socket, 'connect', guarded(original_connect))
    monkeypatch.setattr(socket.socket, 'connect_ex', guarded(original_connect_ex))


def claim(text='Revenue is 10.', **changes):
    values = dict(original_quote=text, claim=text, classification='Supported',
                  evidence=[EvidenceQuote(line_number=1)], reason='Exact fixture evidence.',
                  action='keep', revised_claim=text,
                  number_bindings=[NumberBinding(claim_number='10', evidence_line=1,
                                  evidence_number='10', transform='identity')])
    values.update(changes)
    return AtomicClaim(**values)


def review(*claims):
    return VerifierResponse(units=[UnitReview(source_id=i, claims=[c], nonfactual_reason='')
                                 for i, c in enumerate(claims)])


def test_supported_fact_retained_with_exact_numeric_evidence():
    text = 'Revenue is 10.'
    result = validate_review(review(claim()), text, text)
    assert result['revised_text'] == text
    assert result['claims'][0]['retained']
    assert result['claims'][0]['numeric_results'][0]['passed']


def archived_verifier(filename):
    import importlib.util
    spec=importlib.util.spec_from_file_location(filename[:-3],SUBMISSION/'evidence/verifier_versions'/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_v7_identical_release_is_printed_once_with_both_original_mappings():
    text = 'Revenue is 10.\nRevenue is 10.'
    result = archived_verifier('verifier_agent_v7.py').validate_review(review(claim(), claim()), 'Revenue is 10.', text)
    assert result['revised_text'] == 'Revenue is 10.'
    assert len(result['claims']) == 2
    assert all(row['retained'] for row in result['claims'])
    assert result['claims'][0]['source_start'] != result['claims'][1]['source_start']


def test_v7_1_only_neutral_connectors_can_be_unquoted():
    verifier=archived_verifier('verifier_agent_v7_1.py')
    for prefix in ['In summary, ', 'In conclusion, ', 'Overall, ']:
        result=verifier.validate_review(review(claim()),'Revenue is 10.',prefix+'Revenue is 10.')
        assert result['revised_text']=='Revenue is 10.'
        assert result['unquoted_nonfactual_connectors'][0]['text']==prefix
    for prefix in ['Not ', 'Only ', 'Because ', 'Revenue is declining. ']:
        with pytest.raises(ValueError,match='omitted text'):
            verifier.validate_review(review(claim()),'Revenue is 10.',prefix+'Revenue is 10.')


def test_company_identity_header_cannot_support_business_description():
    text='AMD serves gaming enthusiasts worldwide.'
    candidate=claim(text=text,number_bindings=[])
    result=validate_review(review(candidate),'Financial data for Advanced Micro Devices, Inc. (AMD):',text)
    assert not result['claims'][0]['retained_as_verified']
    assert text in result['revised_text']
    assert 'structural_citation_only_does_not_establish_claim' in result['claims'][0]['deterministic_errors']


def test_period_header_requires_substantive_metric_row():
    text='Revenue increased.'
    candidate=claim(text=text,number_bindings=[])
    result=validate_review(review(candidate),'| metrics | 2024A | 2025A |',text)
    assert not result['claims'][0]['retained_as_verified']
    candidate.evidence.append(EvidenceQuote(line_number=2))
    result=validate_review(review(candidate),'| metrics | 2024A | 2025A |\n| Revenue | 10 | 12 |',text)
    assert result['claims'][0]['retained_as_verified']


def test_unit_omission_or_duplicate_is_rejected():
    text = 'Revenue is 10.\nUnreviewed factual assertion.'
    with pytest.raises(ValueError, match='omitted or duplicated'):
        validate_review(review(claim()), 'Revenue is 10.', text)
    unit = UnitReview(source_id=0, claims=[claim()], nonfactual_reason='')
    with pytest.raises(ValueError, match='omitted or duplicated'):
        validate_review(VerifierResponse(units=[unit, unit]), 'Revenue is 10.', 'Revenue is 10.')


def test_uncovered_second_clause_is_rejected():
    with pytest.raises(ValueError, match='omitted text within'):
        validate_review(review(claim()), 'Revenue is 10.', 'Revenue is 10. EBITDA is also positive.')


def test_exact_rhetorical_gap_is_recorded_without_weakening_factual_coverage():
    source = 'Market Risks: Revenue is 10.'
    checked = validate_review(review(claim()), 'Revenue is 10.', source)
    assert checked['unquoted_nonfactual_connectors'][0]['text'] == 'Market Risks: '
    assert checked['revised_text'] == 'Revenue is 10.'
    for prefix in ('Not ', 'Revenue is declining. ', 'Market Risks: Revenue declines. ', 'Because ', 'Only '):
        with pytest.raises(ValueError, match='omitted text within'):
            validate_review(review(claim()), 'Revenue is 10.', prefix + 'Revenue is 10.')


def test_supported_fact_cannot_be_deleted_to_raise_grounding():
    result=validate_review(review(claim(action='withhold', revised_claim='')), 'Revenue is 10.', 'Revenue is 10.')
    assert result['claims'][0]['retained']
    assert not result['claims'][0]['retained_as_verified']
    assert 'Revenue is 10.' in result['revised_text']
    assert 'supported_fact_withhold_action_rejected' in result['claims'][0]['deterministic_errors']


def test_supported_correct_action_retains_checked_fact():
    result=validate_review(review(claim(action='correct')), 'Revenue is 10.', 'Revenue is 10.')
    assert result['claims'][0]['retained_as_verified']
    assert result['revised_text']=='Revenue is 10.'


@pytest.mark.parametrize('change', [
    {'evidence': [EvidenceQuote(line_number=0)]},
    {'evidence': [EvidenceQuote(line_number=7)]},
    {'number_bindings': [NumberBinding(claim_number='10', evidence_line=1, evidence_number='99', transform='identity')]},
    {'number_bindings': []},
    {'number_bindings': [NumberBinding(claim_number='10', evidence_line=1, evidence_number='10', transform='identity')]*2},
    {'classification': 'Unsupported'},
    {'classification': 'Contradicted'},
])
def test_invalid_evidence_and_bindings_never_release_as_verified(change):
    result=validate_review(review(claim(**change)), 'Revenue is 10.', 'Revenue is 10.')
    row=result['claims'][0]
    assert not row['retained_as_verified'] and row['deterministic_errors']
    if change.get('classification')=='Contradicted':
        assert not row['retained'] and '[FACTUAL CONTENT WITHHELD - N/A]' in result['revised_text']
    else:
        assert row['retained'] and result['revised_text']=='[UNVERIFIED - deterministic evidence/number check failed] Revenue is 10.'


def test_evidence_lines_are_resolved_exactly_and_model_cannot_supply_quote():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvidenceQuote(line_number=1, quote='Invented citation')
    result = validate_review(review(claim()), 'Revenue is 10.   Exact spacing.', 'Revenue is 10.')
    assert result['claims'][0]['evidence'][0]['quote'] == 'Revenue is 10.   Exact spacing.'
    bad = claim(evidence=[EvidenceQuote(line_number=2)])
    result=validate_review(review(bad), 'Revenue is 10.\nA different source row.', 'Revenue is 10.')
    assert 'number_not_in_cited_evidence_line' in json.dumps(result['deterministic_validation_failures'])
    assert result['verified_claim_count']==0


def test_percent_identity_and_half_rounding_are_not_falsely_accepted():
    text = 'Margin is 1%.'
    bad = claim(text, evidence=[EvidenceQuote(line_number=1)],
                number_bindings=[NumberBinding(claim_number='1%', evidence_line=1, evidence_number='1', transform='identity')])
    result=validate_review(review(bad), 'Margin is 1.', text)
    assert not result['claims'][0]['retained_as_verified']
    assert 'identity_changes_percent_unit' in json.dumps(result['deterministic_validation_failures'])
    bad = claim('Revenue is 1.', evidence=[EvidenceQuote(line_number=1)],
                number_bindings=[NumberBinding(claim_number='1', evidence_line=1, evidence_number='1.5', transform='identity')])
    result=validate_review(review(bad), 'Revenue is 1.5.', 'Revenue is 1.')
    assert not result['claims'][0]['retained_as_verified']
    assert 'numeric_mismatch' in json.dumps(result['deterministic_validation_failures'])


def test_numeric_product_and_fiscal_labels_are_not_omitted():
    assert NUMBER.findall('FY2025 Q1 H100 and 10-K') == ['2025', '1', '100', '10']


def policy(tmp_path, cap=1):
    path = tmp_path/'policy.json'
    write_json(path, {'approved_cap_usd': cap, 'input_usd_per_million': 1,
        'output_usd_per_million': 2, 'model': 'fixture-model', 'price_source': 'test only; not a provider quote'})
    return path


def test_budget_reservation_failure_stays_committed_and_cap_is_enforced(tmp_path):
    with Budget(policy(tmp_path, .02), tmp_path/'ledger.jsonl') as budget:
        budget.reserve('failed-unknown', '', 1)
        budget.reserve('second-unknown', '', 1)
        with pytest.raises(RuntimeError, match='Budget guard'):
            budget.reserve('would-exceed', '', 1)
        assert len(budget.rows()) == 2
    assert not (tmp_path/'ledger.jsonl.lock').exists()


def test_budget_invalid_settlements_do_not_free_spending(tmp_path):
    with Budget(policy(tmp_path), tmp_path/'ledger.jsonl') as budget:
        with pytest.raises(ValueError):
            budget.settle('unreserved', 1, 1, 1)
        budget.reserve('one', 'fixture', 10)
        with pytest.raises(ValueError):
            budget.settle('one', -1, 1, 1)
        with pytest.raises(ValueError):
            budget.reserve('one', 'fixture', 10)
        budget.settle('one', 5, 2, .1)
        with pytest.raises(ValueError):
            budget.settle('one', 0, 0, .1)
        assert sum(r['event']=='settle' for r in budget.rows()) == 1


def test_budget_bound_violation_records_actual_and_blocks_future(tmp_path):
    with Budget(policy(tmp_path), tmp_path/'ledger.jsonl') as budget:
        budget.reserve('one', '', 10)
        with pytest.raises(RuntimeError, match='beyond'):
            budget.settle('one', 100, 11, .1)
        assert [r['event'] for r in budget.rows()] == ['reserve', 'settle', 'bound_exceeded']
        with pytest.raises(RuntimeError, match='reconcile'):
            budget.reserve('two', '', 10)


def test_budget_ledger_lock_prevents_parallel_reservations(tmp_path):
    p = policy(tmp_path)
    with Budget(p, tmp_path/'ledger.jsonl'):
        with pytest.raises(RuntimeError, match='locked'):
            Budget(p, tmp_path/'ledger.jsonl')


def test_secret_redaction_and_public_url_preserve_valid_json():
    secret = 'test-secret-with-+/symbols'
    source = json.dumps({'api_key': secret, 'normal': 'data', 'authorization': 'Bearer testsecret'})
    sanitized = redact(source, [secret])
    assert secret not in sanitized and 'testsecret' not in sanitized
    assert json.loads(sanitized)['normal'] == 'data'
    url = public_url('https://example.org/x?symbol=NVDA&api-key=secret&authorization=secret&apikey=secret#fragment')
    assert url == 'https://example.org/x?symbol=NVDA'
    with pytest.raises(ValueError):
        public_url('https://user:secret@example.org/')


def test_snapshot_replay_blocks_missing_network_and_detects_tampering(tmp_path):
    import requests
    url = 'https://financialmodelingprep.com/api/v3/quote/NVDA'
    with SnapshotHTTP(tmp_path, 'replay').installed():
        with pytest.raises(RuntimeError, match='Frozen snapshot missing'):
            requests.get(url, params={'apikey': 'test-secret'})
        with pytest.raises(RuntimeError, match='host is forbidden'):
            requests.get('https://unrecorded.example.org/')
    key = sha256(url)
    body = b'[{"symbol":"TEST_FIXTURE"}]'
    directory = tmp_path/key
    directory.mkdir()
    (directory/'body.bin').write_bytes(body)
    write_json(directory/'response.json', {'status_code':200, 'body_sha256':sha256(body), 'content_type':'application/json'})
    with SnapshotHTTP(tmp_path, 'replay').installed():
        assert requests.get(url, params={'apikey':'test-secret'}).json()[0]['symbol']=='TEST_FIXTURE'
        (directory/'body.bin').write_bytes(b'tampered')
        with pytest.raises(RuntimeError, match='integrity failure'):
            requests.get(url)


def test_frozen_input_detects_added_and_modified_files(tmp_path):
    path=tmp_path/'one.json'; path.write_text('{}')
    expected=hashes(tmp_path)
    validate_hashes(tmp_path,expected)
    (tmp_path/'injected.json').write_text('{}')
    with pytest.raises(ValueError, match='file set changed'):
        validate_hashes(tmp_path,expected)
    (tmp_path/'injected.json').unlink(); path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Frozen input changed'):
        validate_hashes(tmp_path,expected)


def test_sdk_usage_attributes_and_real_typed_call_in_memory(tmp_path, monkeypatch):
    """Exercise installed SDK 0.3.3 on an in-memory HTTP response; no live API."""
    import httpx
    from agents import Agent
    from agents.usage import Usage
    from pydantic import BaseModel
    fields = asdict(Usage())
    assert {'requests','input_tokens','output_tokens','total_tokens'}.issubset(fields)
    requests_seen=[]
    def transport(request):
        requests_seen.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'fixture-response', 'object':'chat.completion', 'created':0,
            'model':'fixture-model', 'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'{"ok":true}'}}],
            'usage':{'prompt_tokens':15,'completion_tokens':5,'total_tokens':20}}, request=request)
    actual=httpx.AsyncClient
    class MemoryClient(actual):
        def __init__(self,*a,**kw):
            kw['transport']=httpx.MockTransport(transport)
            super().__init__(*a,**kw)
    monkeypatch.setattr(httpx,'AsyncClient',MemoryClient)
    class Response(BaseModel):
        ok: bool
    agent=Agent(name='TestOnly',instructions='Return fixture output.',output_type=Response)
    cfg={'openai_api_key':'unit-test-secret','openai_model':'fixture-model','openai_base_url':'https://fixture.invalid/v1'}
    with Budget(policy(tmp_path),tmp_path/'ledger.jsonl') as budget:
        result=asyncio.run(call_agent(agent,'Test only.',config=cfg,budget=budget,directory=tmp_path/'call',call_id='fixture',max_tokens=32))
        assert result.ok
        assert len(requests_seen)==1
        assert requests_seen[0]['response_format']['type']=='json_schema'
        response=json.loads((tmp_path/'call'/'response.json').read_text())
        assert response['usage']['total_tokens']==20
        assert response['usage']['requests']==1
        assert isinstance(response['usage']['input_tokens_details'],dict)
        assert budget.rows()[-1]['event']=='settle'
        for path in (tmp_path/'call').iterdir():
            assert 'unit-test-secret' not in path.read_text(encoding='utf-8')


def test_failed_sdk_response_preserves_reservation_and_redacts_echo(tmp_path, monkeypatch):
    import httpx
    from agents import Agent
    from pydantic import BaseModel
    requests_seen=[]
    def transport(request):
        requests_seen.append(request)
        return httpx.Response(500,json={'error':{'message':'unit-test-secret rejected',
                    'type':'fixture_error','code':'fixture'}},request=request)
    actual=httpx.AsyncClient
    class MemoryClient(actual):
        def __init__(self,*a,**kw):
            kw['transport']=httpx.MockTransport(transport)
            super().__init__(*a,**kw)
    monkeypatch.setattr(httpx,'AsyncClient',MemoryClient)
    class Response(BaseModel):
        ok: bool
    agent=Agent(name='FailureTestOnly',instructions='Return fixture output.',output_type=Response)
    cfg={'openai_api_key':'unit-test-secret','openai_model':'fixture-model','openai_base_url':'https://fixture.invalid/v1'}
    with Budget(policy(tmp_path),tmp_path/'ledger.jsonl') as budget:
        with pytest.raises(RuntimeError,match='Model call failed'):
            asyncio.run(call_agent(agent,'Test only.',config=cfg,budget=budget,
                        directory=tmp_path/'failed',call_id='failed-fixture',max_tokens=32))
        assert len(requests_seen)==1
        assert [row['event'] for row in budget.rows()]==['reserve']
        error=json.loads((tmp_path/'failed'/'error.json').read_text())
        assert 'unsettled_reservation_retained' in error['billing_status']
        for path in (tmp_path/'failed').iterdir():
            assert 'unit-test-secret' not in path.read_text(encoding='utf-8')

def test_unsupported_content_is_visible_and_never_counted_verified():
    text='The company is an industry leader.'
    candidate=AtomicClaim(original_quote=text,claim=text,classification='Unsupported',evidence=[],
        reason='Leadership is absent from the original evidence.',action='withhold',revised_claim='',number_bindings=[])
    result=validate_review(review(candidate),'Revenue is 10.',text)
    assert result['revised_text']=='[UNVERIFIED - evidence not supplied] '+text
    assert result['verified_claim_count']==0 and result['unverified_annotation_count']==1
    assert result['claims'][0]['retained'] and not result['claims'][0]['retained_as_verified']
    assert result['claims'][0]['classification']=='Unsupported'


def test_invalid_second_claim_cannot_silently_discard_supported_fact():
    good=claim();bad=claim('Revenue is 99.',
        number_bindings=[NumberBinding(claim_number='99',evidence_line=1,evidence_number='10',transform='identity')])
    result=validate_review(review(good,bad),'Revenue is 10.','Revenue is 10.\nRevenue is 99.')
    assert result['claims'][0]['retained_as_verified']
    assert not result['claims'][1]['retained_as_verified']
    assert result['claim_retention_rate']==1
    assert result['revised_text']=='Revenue is 10.\n\n[UNVERIFIED - deterministic evidence/number check failed] Revenue is 99.'
    assert 'numeric_check_failed:99:numeric_mismatch' in result['claims'][1]['deterministic_errors']


def test_forecast_value_and_year_require_metric_and_header_binding():
    evidence='| metric | 2029E |\n| EBITDA Margin | 73.5% |'
    original='EBITDA margin is forecast to be over 70% by 2029.'
    candidate=AtomicClaim(original_quote=original,claim=original,classification='Supported',
        evidence=[EvidenceQuote(line_number=1),EvidenceQuote(line_number=2)],
        reason='Render direct forecast value instead of attempting threshold equality.',action='keep',
        revised_claim='Forecast EBITDA margin is 73.5% in 2029E.',number_bindings=[
            NumberBinding(claim_number='73.5%',evidence_line=2,evidence_number='73.5%',transform='identity'),
            NumberBinding(claim_number='2029',evidence_line=1,evidence_number='2029',transform='identity')])
    result=validate_review(review(candidate),evidence,original)
    assert result['revised_text']=='Forecast EBITDA margin is 73.5% in 2029E.'
    assert result['verified_claim_count']==1
    candidate.number_bindings=candidate.number_bindings[:1]
    result=validate_review(review(candidate),evidence,original)
    assert result['verified_claim_count']==1 and result['automatic_year_binding_count']==1
    assert result['claims'][0]['automatic_number_bindings'][0]['evidence_quote']=='| metric | 2029E |'
    candidate.evidence=[EvidenceQuote(line_number=2)]
    result=validate_review(review(candidate),evidence,original)
    assert result['verified_claim_count']==0 and result['automatic_year_binding_count']==0
    assert 'numeric_check_failed:2029:missing_binding' in result['claims'][0]['deterministic_errors']


def test_real_v1_failure_preserved_and_v2_diagnostics_identify_actual_faults():
    import importlib.util
    run=SUBMISSION/'runs/NVDA_final_1/verifier/tagline'
    archive=SUBMISSION/'evidence/verifier_versions/verifier_agent_v1.py'
    if not (run/'response.json').exists() or not archive.exists():
        pytest.skip('Optional real historical regression artifacts are not available in this checkout')
    spec=importlib.util.spec_from_file_location('archived_verifier_v1',archive)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    data=json.loads((run/'response.json').read_text(encoding='utf-8'))['structured_output']
    writer=SUBMISSION/'runs/NVDA_baseline_1'
    evidence=(writer/'agents/tagline/prompt.txt').read_text(encoding='utf-8')
    original=(writer/'output/NVDA/analysis/tagline.txt').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='No verified factual content remains'):
        old.validate_review(old.VerifierResponse.model_validate(data),evidence,original)
    for unit in data['units']:
        for row in unit['claims']:
            for ref in row['evidence']:ref.pop('quote',None)
    v2path=SUBMISSION/'evidence/verifier_versions/verifier_agent_v2.py'
    spec=importlib.util.spec_from_file_location('archived_verifier_v2',v2path)
    v2=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v2;spec.loader.exec_module(v2)
    with pytest.raises(ValueError) as error:
        v2.validate_review(v2.VerifierResponse.model_validate(data),evidence,original)
    assert 'duplicate_number_binding' in str(error.value)
    assert 'numeric_check_failed:2029:missing_binding' in str(error.value)
    assert 'numeric_check_failed:70%:numeric_mismatch' in str(error.value)
    assert 'empty_atomic_claim' in str(error.value)


def test_contradicted_invalid_correction_is_withheld_but_audit_is_complete():
    original='Revenue is 99.'
    candidate=claim(original,classification='Contradicted',action='correct',revised_claim='Revenue is 12.',
        number_bindings=[NumberBinding(claim_number='12',evidence_line=1,evidence_number='10',transform='identity')])
    result=validate_review(review(candidate),'Revenue is 10.',original)
    row=result['claims'][0]
    assert not row['retained'] and not row['retained_as_verified']
    assert row['original_quote']==original and row['revised_claim']=='Revenue is 12.'
    assert row['verification_status']=='contradicted_correction_rejected'
    assert result['withheld_claim_count']==1 and result['claim_retention_rate']==0
    assert result['revised_text'].startswith('[FACTUAL CONTENT WITHHELD - N/A]')


def test_contradicted_valid_correction_released_with_original_mapping():
    original='Revenue is 99.'
    candidate=claim(original,classification='Contradicted',action='correct',revised_claim='Revenue is 10.')
    result=validate_review(review(candidate),'Revenue is 10.',original)
    assert result['revised_text']=='Revenue is 10.' and result['verified_claim_count']==1
    assert result['claims'][0]['original_quote']==original


def test_failed_check_annotation_uses_original_quote_not_failed_model_rewrite():
    candidate=claim(claim='Revenue is 999.',revised_claim='Revenue is 11.',
        number_bindings=[NumberBinding(claim_number='11',evidence_line=1,evidence_number='10',transform='identity')])
    result=validate_review(review(candidate),'Revenue is 10.','Revenue is 10.')
    assert result['revised_text']=='[UNVERIFIED - deterministic evidence/number check failed] Revenue is 10.'
    assert result['claims'][0]['release_text_source']=='original_quote'


def test_automatic_year_binding_never_replaces_wrong_binding_or_binds_financial_amount():
    original='Revenue is 2029.'
    candidate=claim(original,number_bindings=[])
    result=validate_review(review(candidate),'Revenue is 2029.',original)
    assert result['automatic_year_binding_count']==0 and result['verified_claim_count']==0
    original='Revenue is 10 in 2029E.'
    candidate=claim(original,evidence=[EvidenceQuote(line_number=1),EvidenceQuote(line_number=2)],number_bindings=[
        NumberBinding(claim_number='10',evidence_line=2,evidence_number='10',transform='identity'),
        NumberBinding(claim_number='2029',evidence_line=1,evidence_number='2028',transform='identity')])
    result=validate_review(review(candidate),'| metric | 2028E | 2029E |\n| Revenue | 10 | 10 |',original)
    assert result['automatic_year_binding_count']==0 and result['verified_claim_count']==0


def test_source_quote_error_remains_structural_and_rejects_section():
    candidate=claim(original_quote='Not present.')
    with pytest.raises(ValueError,match='omitted text|absent'):
        validate_review(review(candidate),'Revenue is 10.','Revenue is 10.')
