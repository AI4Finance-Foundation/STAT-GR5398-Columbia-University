"""Add a narrow endpoint clarification to V8.2; never alter frozen predecessors."""
from pathlib import Path
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]
parent=ROOT/'evidence/verifier_versions/verifier_agent_v8_2.py'
fragment=ROOT/'evidence/verifier_versions/cagr_guard_v9.py'
text=parent.read_text(encoding='utf-8')+'\n\n'+fragment.read_text(encoding='utf-8')+'''

# Preserve the exact paid-review schema and instructions. This is a separate
# deterministic release check applied only to original writer-prompt evidence.
_validate_review_before_horizon_guard = validate_review

def validate_review(review, evidence, text):
    return apply_cagr_horizon_guard(
        _validate_review_before_horizon_guard(review, evidence, text), evidence)
'''
archive=ROOT/'evidence/verifier_versions/verifier_agent_v9.py'
if archive.exists() and archive.read_text(encoding='utf-8')!=text:
    raise ValueError('Existing immutable V9 source differs; use a new version')
if not archive.exists():
    archive.write_text(text,encoding='utf-8')
target=Path.home()/'.cache/finrobot-zg2556/upstream/finrobot_equity/core/src/modules/equity_agents/verifier_agent.py'
assert sha256(target.read_bytes()) in {sha256(parent.read_bytes()),sha256(archive.read_bytes())}
target.write_bytes(archive.read_bytes())
write_json(ROOT/'evidence/verifier_versions/v9_development.json',{
    'at_utc':utc_now(),'parent_sha256':sha256(parent.read_bytes()),
    'fragment_sha256':sha256(fragment.read_bytes()),'source_sha256':sha256(archive.read_bytes()),
    'scope':'Option A deterministic release clarification; unchanged paid instructions/schema/model responses.',
    'trigger':'Observed repeated association of displayed historical Revenue CAGR with a table containing forecasts.',
    'evaluation_separation':'No audit labels are consumed by the rule. Separate review found ambiguous table-scope atoms in the existing evaluation; those measurement errata must be reported separately from candidate output changes.',
    'new_paid_calls':0,'held_out_evaluation':False,
    'inputs':'Original writer prompt and released standalone sentence only; positive endpoint values, exact fiscal columns and displayed rounding. Extra predicates or unknown entities abstain.',
    'rule_does_not_claim':'It does not infer that an ambiguous original table-span sentence was false, and is not a general semantic verifier.'
})
print('Prepared separate V9 endpoint clarification; archived V8.2 unchanged.')
