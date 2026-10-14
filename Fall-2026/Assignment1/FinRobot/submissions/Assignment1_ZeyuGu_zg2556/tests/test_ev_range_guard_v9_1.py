"""Synthetic adversarial controls only; real saved-row probe is separate."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ev_range_guard_v9_1_test',ROOT/'evidence/verifier_versions/ev_range_guard_v9_1.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

PROMPT="""Financial data for Example Corporation (EXM):

## Example Corporation Financial Metrics and Forecasts
| metrics | 2021A | 2022A | 2023A | 2024E |
|---|---:|---:|---:|---:|
| Revenue | 100 | 110 | 120 | 150 |

## Peer EV/EBITDA Comparison
| year | OTHER | EXM |
|---:|---:|---:|
| 2021 | 999 | 18.25 |
| 2022 | 1 | 52.5 |
| 2023 | 99 | 31.0 |
| 2024 | 0 | 500 |
"""
SENTENCE="EXM's EV/EBITDA ratio has ranged from the mid-40s to near 60 in recent years."


def row(text=SENTENCE,classification='Unsupported'):
    return {'released_text':text,'retained_as_verified':False,'verification_status':'unverified_annotation',
        'release_text_source':'atomic_claim','classification':classification,'retained':True,
        'claim':text,'revised_claim':text,'action':'withhold','evidence':[{'old':'retained'}],
        'numeric_results':[{'passed':False}],'original_quote':text}


def test_explicit_actual_scope_extrema_and_exact_provenance():
    out=v.clarify_ev_ebitda_range(SENTENCE,PROMPT)
    assert out and out['calculation']['minimum']=='18.25'
    assert out['calculation']['maximum']=='52.5'
    assert out['calculation']['minimum_years']==['2021A']
    assert out['calculation']['maximum_years']==['2022A']
    assert out['first_actual']=='2021A' and out['last_actual']=='2023A'
    assert out['calculation']['forecast_values_used'] is False
    assert out['calculation']['interpolation'] is False
    assert out['original_released_text']==SENTENCE
    assert out['original_range_expression']['window']=='recent years'
    assert 'Not inferred' in out['original_semantic_verdict']
    assert '2021A through 2023A' in out['revised_released_text']
    assert '500' not in out['revised_released_text'] and '999' not in out['revised_released_text']
    for e in out['evidence']:assert e['quote']==PROMPT.splitlines()[e['line_number']-1]
    assert {e['line_number'] for e in out['evidence']}=={1,4,8,9,11,12,13}


@pytest.mark.parametrize('sentence',[
    "Example Corporation's EV/EBITDA multiple has ranged from 18.25 to 52.5 over recent years.",
    "EXM EV/EBITDA has ranged from approximately 18.25 to near 53 in recent years.",
    "EXM’s EV/EBITDA multiples have ranged from low-20s to high-50s in recent years.",
    '[UNVERIFIED - evidence not supplied] '+SENTENCE,
    '[UNVERIFIED - deterministic evidence/number check failed] '+SENTENCE,
])
def test_known_subject_and_closed_range_grammar(sentence):
    assert v.clarify_ev_ebitda_range(sentence,PROMPT)


@pytest.mark.parametrize('sentence',[
    "OTHER's EV/EBITDA ratio has ranged from the mid-40s to near 60 in recent years.",
    "Example's EV/EBITDA ratio has ranged from the mid-40s to near 60 in recent years.",
    "Its EV/EBITDA ratio has ranged from the mid-40s to near 60 in recent years.",
    SENTENCE.replace('EV/EBITDA','PE'), SENTENCE.replace('EV/EBITDA','EV/EBITA'),
    SENTENCE.replace('EV/EBITDA','valuation multiple'),
    SENTENCE[:-1]+', reflecting competitive strength.',
    SENTENCE[:-1]+', above OTHER.',
    SENTENCE+' Peers are below this range.',
    SENTENCE+' 42.', SENTENCE.replace('recent years','2021 to 2023'),
    SENTENCE.replace('recent years','forecast years'),
    SENTENCE.replace('has ranged','will range'),
    SENTENCE.replace('mid-40s','negative 40'),
    SENTENCE.replace('mid-40s','-40'),
    SENTENCE.replace('near 60','None'),
    'Our strong company '+SENTENCE,
    '[VERIFIED] '+SENTENCE,
    '', None,
])
def test_extra_predicates_ambiguous_identity_metric_and_forecasts_abstain(sentence):
    assert v.clarify_ev_ebitda_range(sentence,PROMPT) is None


@pytest.mark.parametrize('prompt',[
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | nan |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | None |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | N/A |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | -5 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | 0 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | Infinity |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022 | 1 | 1,2 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |\n',''),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2021 | 1 | 52.5 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022E | 1 | 52.5 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022-12-31 | 1 | 52.5 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| FY2022 | 1 | 52.5 |'),
    PROMPT.replace('| 2022 | 1 | 52.5 |','| 2022Q4 | 1 | 52.5 |'),
    PROMPT.replace('2022A','2022E'),
    PROMPT.replace('2021A','2021E').replace('2022A','2022E').replace('2023A','2023E'),
    PROMPT.replace('2022A','2021A'),
    PROMPT.replace('2022A','2024A'),
    PROMPT.replace('| year | OTHER | EXM |','| year | EXM | EXM |'),
    PROMPT.replace('| year | OTHER | EXM |','| year | OTHER | UNKNOWN |'),
    PROMPT.replace('## Peer EV/EBITDA Comparison','## Peer PE Comparison'),
    PROMPT+'\n## Peer EV/EBITDA Comparison\n| year | EXM |\n| 2023 | 1 |\n',
    PROMPT.replace('| metrics |','| measures |'),
    '',None,
])
def test_bad_actual_values_dates_missing_forecast_or_conflicting_tables_abstain(prompt):
    assert v.clarify_ev_ebitda_range(SENTENCE,prompt) is None


def test_forecast_outlier_or_missing_value_never_enters_actual_range():
    for future in ('nan','None','-500','9000000'):
        out=v.clarify_ev_ebitda_range(SENTENCE,PROMPT.replace('| 2024 | 0 | 500 |',f'| 2024E | 0 | {future} |'))
        assert out and out['calculation']['maximum']=='52.5'
        assert all(o['year']<=2023 for o in out['observations'])


def test_actual_row_labels_ties_and_scientific_notation_are_exact():
    prompt=PROMPT.replace('| 2021 | 999 | 18.25 |','| 2021A | 999 | 1.825e1 |').replace('| 2022 | 1 | 52.5 |','| 2022A | 1 | 52.5 |').replace('| 2023 | 99 | 31.0 |','| 2023A | 99 | 52.5 |')
    out=v.clarify_ev_ebitda_range(SENTENCE,prompt)
    assert out and out['calculation']['minimum']=='18.25'
    assert out['calculation']['maximum_years']==['2022A','2023A']


def test_subject_years_cannot_be_inferred_from_peer_rows_or_calendar_dates():
    assert v.clarify_ev_ebitda_range(SENTENCE,PROMPT.replace('2021A','2021-12-31')) is None
    assert v.clarify_ev_ebitda_range(SENTENCE,PROMPT.replace('2022A','TTM')) is None
    swapped=PROMPT.replace('| 2021 | 999 | 18.25 |\n| 2022 | 1 | 52.5 |','| 2022 | 1 | 52.5 |\n| 2021 | 999 | 18.25 |')
    assert v.clarify_ev_ebitda_range(SENTENCE,swapped) is None


@pytest.mark.parametrize('classification',['Supported','Unsupported','Contradicted'])
def test_trigger_is_independent_of_paid_label_and_preserves_original_paid_content(classification):
    sentence='[UNVERIFIED - evidence not supplied] '+SENTENCE
    original={'claims':[row(sentence,classification)],'revised_text':sentence,'limitations':'Control.'}
    snapshot=copy.deepcopy(original)
    checked=v.apply_ev_ebitda_range_guard(original,PROMPT)
    assert original==snapshot
    new=checked['claims'][0]
    assert new['classification']==classification and new['action']=='withhold'
    assert new['claim']==sentence and new['revised_claim']==sentence
    assert new['evidence']==snapshot['claims'][0]['evidence']
    assert new['numeric_results']==snapshot['claims'][0]['numeric_results']
    assert new['pre_ev_range_guard_release']['released_text']==sentence
    assert new['ev_ebitda_range_check']['original_released_text']==sentence
    assert new['released_text'] in checked['revised_text']
    assert checked['verified_claim_count']==1 and checked['ev_ebitda_range_check_count']==1


def test_no_substring_replacement_no_information_deletion_and_deduplication_is_exact_only():
    compound=SENTENCE+' Other additional information.'
    old={'claims':[row(SENTENCE)],'revised_text':compound,'limitations':'Control.'}
    assert v.apply_ev_ebitda_range_guard(old,PROMPT)['revised_text']==compound
    old={'claims':[row(SENTENCE),row(SENTENCE),row('Different independent assertion.')],
        'revised_text':SENTENCE+'\n\nDifferent independent assertion.','limitations':'Control.'}
    new=v.apply_ev_ebitda_range_guard(old,PROMPT)
    assert len(new['claims'])==3
    assert new['revised_text'].count('minimum is')==1
    assert 'Different independent assertion.' in new['revised_text']
    assert new['ev_ebitda_range_check_count']==2


def test_idempotent_on_already_explicit_range_text():
    original={'claims':[row()],'revised_text':SENTENCE,'limitations':'Control.'}
    first=v.apply_ev_ebitda_range_guard(original,PROMPT)
    second=v.apply_ev_ebitda_range_guard(first,PROMPT)
    assert first['revised_text']==second['revised_text']
    assert first['claims']==second['claims']


def load_archive(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'evidence/verifier_versions'/(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def test_composed_archive_preserves_exact_parent_and_paid_contract(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('Network forbidden')))
    parent=ROOT/'evidence/verifier_versions/verifier_agent_v9.py'
    candidate=ROOT/'evidence/verifier_versions/verifier_agent_v9_1.py'
    assert candidate.read_bytes().startswith(parent.read_bytes())
    before=load_archive('verifier_agent_v9')
    after=load_archive('verifier_agent_v9_1')
    assert before.VERIFIER_INSTRUCTIONS==after.VERIFIER_INSTRUCTIONS
    assert before.VerifierResponse.model_json_schema()==after.VerifierResponse.model_json_schema()


def test_composed_validator_replays_all_saved_paid_reviews_without_new_generation(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket,'connect',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('Network forbidden')))
    before=load_archive('verifier_agent_v9')
    after=load_archive('verifier_agent_v9_1')
    paths=sorted((ROOT/'runs').glob('*v9/verifier/*/checked.json'))
    assert paths, 'Authentic saved V9 evidence is required for this integration test'
    changed=0
    for path in paths:
        saved=json.loads(path.read_text(encoding='utf-8'))
        response=json.loads(path.with_name('response.json').read_text(encoding='utf-8'))['structured_output']
        manifest=json.loads((path.parents[2]/'manifest.json').read_text(encoding='utf-8'))
        baseline=ROOT/'runs'/manifest['parent_run_id']
        prompt=(baseline/'agents'/path.parent.name/'prompt.txt').read_text(encoding='utf-8')
        source=(baseline/'output'/manifest['ticker']/'analysis'/(path.parent.name+'.txt')).read_text(encoding='utf-8')
        replay=before.validate_review(before.VerifierResponse.model_validate(response),prompt,source)
        assert replay['revised_text']==saved['revised_text']
        direct=v.apply_ev_ebitda_range_guard(replay,prompt)
        composed=after.validate_review(after.VerifierResponse.model_validate(response),prompt,source)
        assert composed['revised_text']==direct['revised_text']
        assert composed['claims']==direct['claims']
        changed+=sum(a['released_text']!=b['released_text'] for a,b in zip(replay['claims'],composed['claims']))
    probe=json.loads((ROOT/'evidence/verifier_versions/v9_1_development.json').read_text(encoding='utf-8'))
    assert changed==probe['changed_rows'] and len(paths)==probe['saved_v9_files']
