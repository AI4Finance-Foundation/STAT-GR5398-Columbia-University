"""Preserve and correct one independently observed baseline annotation error.

No financial input or generated baseline text changes. The original evidence
already supplied Data Center sales; no later external fact is introduced.
"""
import json,copy
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
def main():
    original=ROOT/'audits/grounding/NVDA_before_adjudicated/reviewed_specification.json'
    spec=json.loads(original.read_text(encoding='utf-8'));changes=[]
    for unit in spec['sections']['company_overview']['units']:
        for atom in unit['atoms']:
            if atom['claim']=='NVIDIA serves data-center markets.':
                before=copy.deepcopy(atom)
                assert atom['classification']=='Unsupported'
                atom.update(classification='Supported',evidence=[{'line_start':45,'line_end':45}],
                    reason='Original writer news already reports Data Center revenue of$89bn. This establishes the narrow service-to-market assertion, but not leadership, primary strategic focus or a specific product catalog. Corrected evaluator omission; no later evidence added.')
                changes.append({'section':'company_overview','unit_id':unit['unit_id'],'before':before,'after':atom})
    assert len(changes)==1
    output=ROOT/'audits/grounding/NVDA_before_reasoning_erratum_working'
    write_json(output/'reviewed_specification.json',spec)
    write_json(output/'annotation_erratum.json',{'at_utc':utc_now(),'original_spec_sha256':sha256(original.read_bytes()),
        'changes':changes,'baseline_generation_unchanged':True,'original_evidence_unchanged':True,
        'no_new_external_evidence':True,'reason':'Maintain identical interpretation of original supplied Data Center sales before/after; correcting this baseline label narrows, not inflates, apparent improvement.',
        'human_review_performed':False,'earlier_audits_preserved':True})
    print('One baseline annotation erratum recorded; original audit retained.')
if __name__=='__main__':main()
