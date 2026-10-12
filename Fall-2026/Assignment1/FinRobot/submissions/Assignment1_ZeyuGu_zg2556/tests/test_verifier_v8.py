"""Offline format-adversarial fixtures, not synthetic assignment experiments."""
import importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verifier_v8_test',ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)

def run(text,evidence,bindings,refs=(1,2)):
    review=v.VerifierResponse.model_validate({'units':[{'source_id':0,'nonfactual_reason':'','claims':[{
        'original_quote':text,'claim':text,'classification':'Supported','reason':'Offline test fixture.',
        'action':'keep','revised_claim':text,'evidence':[{'line_number':n} for n in refs],
        'number_bindings':[{'claim_number':a,'evidence_number':b,'evidence_line':n,'transform':'identity'} for a,b,n in bindings]}]}]})
    return v.validate_review(review,evidence,text)['claims'][0]

def test_exact_fiscal_wrapper_preserved_and_normalization_traced():
    row=run('Revenue is 10 in 2025A.','| metrics | 2025A |\n| Revenue | 10 |',
            [('10','10',2),('2025A','2025A',1)])
    assert row['retained_as_verified'] and row['released_text']=='Revenue is 10 in 2025A.'
    assert row['number_bindings'][1]['claim_number']=='2025A'
    assert row['numeric_binding_aliases'][0]['normalized_binding']['claim_number']=='2025'

@pytest.mark.parametrize('binding,evidence_label,text_label',[
    (('2025E','2025A',1),'2025A','2025E'),
    (('2025E','2025E',1),'2025A','2025E'),
    (('2025A','2025A',1),'2025A','2025E'),
    (('2025A','2025A',1),'2024A','2025A'),
])
def test_fiscal_wrapper_cannot_change_actual_forecast_or_year(binding,evidence_label,text_label):
    row=run('Revenue is 10 in '+text_label+'.','| metrics | '+evidence_label+' |\n| Revenue | 10 |',
            [('10','10',2),binding])
    assert not row['retained_as_verified']

def test_normalization_does_not_hide_duplicate_bindings():
    row=run('Revenue is 10 in 2025A.','| metrics | 2025A |\n| Revenue | 10 |',
            [('10','10',2),('2025A','2025A',1),('2025','2025',1)])
    assert not row['retained_as_verified'] and 'duplicate_number_binding' in row['deterministic_errors']

@pytest.mark.parametrize('amount',['$89.0','$10',chr(163)+'10',chr(8364)+'1,234.5'])
def test_exact_currency_wrapper_with_terminal_punctuation(amount):
    row=run('Revenue is '+amount+'.','Revenue is '+amount+'.',[(amount,amount,1)],refs=(1,))
    assert row['retained_as_verified']

@pytest.mark.parametrize('a,b,evidence,refs',[
    ('$10',chr(8364)+'10','Revenue is '+chr(8364)+'10.',(1,)),
    ('$10','$10','Revenue is $100.',(1,)),
    ('$10','$10','Revenue is $10.5.',(1,)),
    ('$10','$10','Unrelated line.\nRevenue is $10.',(1,)),
])
def test_currency_wrappers_cannot_create_missing_or_uncited_values(a,b,evidence,refs):
    line=2 if '\n' in evidence else 1
    row=run('Revenue is '+a+'.',evidence,[(a,b,line)],refs=refs)
    assert not row['retained_as_verified']

def test_number_tokenizer_excludes_sentence_commas_and_preserves_grouping():
    assert v.NUMBER.findall('Article 1, revenue 1,234.5, EPS 2.0. FY2025 Q1 H100 and 10-K')==['1','1,234.5','2.0','2025','1','100','10']

def test_exact_wrapper_cannot_manufacture_nonidentity_conversion():
    b=v.NumberBinding(claim_number='2025A',evidence_number='2025A',evidence_line=1,transform='divide_100')
    result,alias=v._binding_alias(b,'Value in 2025A.',['| metrics | 2025A |'])
    assert result==b and alias is None
