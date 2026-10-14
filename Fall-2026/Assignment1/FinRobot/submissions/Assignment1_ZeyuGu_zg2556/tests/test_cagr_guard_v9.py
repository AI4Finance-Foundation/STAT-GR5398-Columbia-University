"""Arithmetic/abstention controls; synthetic unit fixtures are not experiments."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cagr_guard_v9',ROOT/'evidence/verifier_versions/cagr_guard_v9.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

CONTROL="""Financial data for Example Corporation (EXM):

## Example Corporation Financial Metrics and Forecasts
| metrics | 2020A | 2022A | 2024E | CAGR |
|---|---:|---:|---:|---:|
| Revenue | 100 | 121 | 146.41 | 10.0% |
"""

def test_year_difference_not_column_count_and_explicit_forecast():
    checked=v.clarify_revenue_cagr('EXM Revenue CAGR is 10.0% from 2020A to 2024E.',CONTROL)
    assert checked
    assert checked['historical_calculation']['year_intervals']==2
    assert checked['table_endpoint_calculation']['year_intervals']==4
    assert checked['historical_calculation']['display_percent']=='10.00'
    assert '2024E is a model forecast' in checked['revised_released_text']
    assert checked['evidence'][0]['line_number']==4 and checked['evidence'][1]['line_number']==6
    assert 'Not inferred' in checked['original_semantic_verdict']

@pytest.mark.parametrize('sentence',[
    "OtherCompany Revenue CAGR is 10.0% from 2020A to 2024E.",
    "EXM Revenue CAGR is 10.0% from 2020A to 2024E because its moat dominates peers.",
    "EXM Revenue CAGR exceeds 10.0% from 2020A to 2024E.",
    "EXM Revenue CAGR is 11.0% from 2020A to 2024E.",
    "EXM Revenue CAGR is 10.0% from 2020A to 2022A.",
    "EXM Revenue CAGR is 10.0% from 2021A to 2024E.",
    "EXM Revenue CAGR is 10.0% from 2022A to 2024E.",
    "EXM Revenue CAGR is 10.0% and EBITDA CAGR is 20.0% through 2024E.",
    "EXM Revenue CAGR is 10.0% through 2024E. 42.",
])
def test_abstains_from_extra_claims_unknown_periods_inequalities_or_other_entities(sentence):
    assert v.clarify_revenue_cagr(sentence,CONTROL) is None

@pytest.mark.parametrize('prompt',[
    CONTROL.replace('| 100 |','| 0 |'), CONTROL.replace('| 100 |','| -100 |'),
    CONTROL.replace('| 121 |','| nan |'), CONTROL.replace('| 10.0% |','| 11.0% |'),
    CONTROL.replace('2024E','2024A'), CONTROL.replace('2022A','2022E'),
    CONTROL.replace('2022A','2020A'),
])
def test_bad_or_missing_source_values_are_not_repaired(prompt):
    assert v.clarify_revenue_cagr('EXM Revenue CAGR is 10.0% from 2020A to 2024E.',prompt) is None

def test_labeled_original_is_preserved_and_explicit_formula_is_exported():
    text='[UNVERIFIED - evidence not supplied] EXM Revenue CAGR is 10.0% in the table through 2024E.'
    row={'released_text':text,'retained_as_verified':False,'verification_status':'unverified_annotation',
         'release_text_source':'atomic_claim','classification':'Unsupported','retained':True}
    original={'claims':[row],'revised_text':text,'limitations':'Unit control.'}
    checked=v.apply_cagr_horizon_guard(original,CONTROL)
    assert checked['claims'][0]['classification']=='Unsupported'
    assert checked['claims'][0]['pre_semantic_guard_release']['released_text']==text
    assert checked['claims'][0]['semantic_horizon_check']['historical_calculation']['formula'].startswith('100 *')
    assert checked['semantic_horizon_check_count']==1 and checked['verified_claim_count']==1
    assert checked['unverified_annotation_count']==0
