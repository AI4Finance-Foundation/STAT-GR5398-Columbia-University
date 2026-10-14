"""Observed token-shape regression controls; not financial experiment outcomes."""
import importlib.util,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v8_2_test',ROOT/'evidence/verifier_versions/verifier_agent_v8_2.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)

@pytest.mark.parametrize('year,tag',[('2023','A'),('2027','E')])
def test_bare_numeric_binding_keeps_exact_displayed_actual_forecast_wrapper(year,tag):
    token=year+tag;b=v.NumberBinding(claim_number=year,evidence_number=token,evidence_line=1,transform='identity')
    revised='Revenue is 10 in '+token+'.';lines=['| metrics | '+token+' |','| Revenue | 10 |']
    normalized,alias=v._binding_alias(b,revised,lines)
    assert alias and normalized.claim_number==year and normalized.evidence_number==year
    raw={'original_quote':revised,'claim':revised,'classification':'Supported','reason':'Control fixture.',
         'evidence':[{'line_number':1},{'line_number':2}],'action':'keep','revised_claim':revised,
         'number_bindings':[b.model_dump(),{'claim_number':'10','evidence_number':'10','evidence_line':2,'transform':'identity'}]}
    review=v.VerifierResponse.model_validate({'units':[{'source_id':0,'claims':[raw],'nonfactual_reason':''}]})
    result=v.validate_review(review,'\n'.join(lines),revised)
    assert result['verified_claim_count']==1 and result['revised_text']==revised

@pytest.mark.parametrize('a,b,text,line',[
    ('2025','2025A','Revenue in2025E.','| metrics | 2025A |'),
    ('2025','2025A','Revenue in 2025.','| metrics | 2025A |'),
    ('2025','2025A','Revenue in 2025A.','| metrics | 2025E |'),
    ('2024','2025A','Revenue in 2025A.','| metrics | 2025A |'),
    ('2025E','2025A','Revenue in 2025E.','| metrics | 2025A |'),
    ('2025','2025A','Revenue in 2025AB.','| metrics | 2025A |'),
])
def test_no_wrapper_is_added_or_changed_by_alias(a,b,text,line):
    raw=v.NumberBinding(claim_number=a,evidence_number=b,evidence_line=1,transform='identity')
    normalized,alias=v._binding_alias(raw,text,[line])
    assert alias is None and normalized==raw
