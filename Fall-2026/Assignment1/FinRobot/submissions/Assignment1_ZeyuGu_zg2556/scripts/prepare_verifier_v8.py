"""Narrow V8 numeric-binding transport repair from observed genuine output.

No semantic audit labels are imported. Original V7.1 and API responses stay intact.
"""
import argparse
from pathlib import Path
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]

HELPER=r'''
def _binding_alias(binding, claim_text, lines):
    """Normalize exact display wrappers, never a different period or amount.

    A/E suffixes and currency symbols remain in the released text and raw model
    binding. Only the arithmetic token identity is normalized. Both explicit
    wrapped tokens must occur exactly at the claimed/cited locations. No citation
    is added and no number, unit conversion or judgment is inferred here.
    """
    if binding.transform != 'identity' or not 1 <= binding.evidence_line <= len(lines):
        return binding, None
    a, b = binding.claim_number, binding.evidence_number
    year = re.fullmatch(r'((?:19|20)\d{2})([AE])', a)
    currency = re.fullmatch(r'([$\u20ac\u00a3\u00a5])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', a)
    if year and a == b:
        normalized, kind = year[1], 'exact_actual_forecast_period_wrapper'
    elif currency and a == b:
        normalized, kind = currency[2], 'exact_currency_symbol_wrapper'
    else:
        return binding, None
    exact = re.compile(r'(?<![\w.])' + re.escape(a) + r'(?!\w|\.\d)')
    if not exact.search(claim_text) or not exact.search(lines[binding.evidence_line-1]):
        return binding, None
    alias = binding.model_copy(update={'claim_number':normalized,'evidence_number':normalized})
    return alias, {'original_binding':binding.model_dump(), 'normalized_binding':alias.model_dump(),
                   'rule':kind,'evidence_quote':lines[binding.evidence_line-1]}

'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');args=p.parse_args()
    parent=ROOT/'evidence/verifier_versions/verifier_agent_v7_1.py';text=parent.read_text(encoding='utf-8')
    old='NUMBER = re.compile(r"(?<![\\d])[-+]?\\d[\\d,]*(?:\\.\\d+)?(?:[eE][-+]?\\d+)?%?")'
    new='NUMBER = re.compile(r"(?<![\\d])[-+]?\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:[eE][-+]?\\d+)?%?")'
    assert old in text;text=text.replace(old,new)
    text=text.replace('def _substantive_evidence_line(line):',HELPER+'\ndef _substantive_evidence_line(line):')
    text=text.replace('number_results, automatic_bindings = [], []','number_results, automatic_bindings, binding_aliases = [], [], []')
    old='bindings = {binding.claim_number: binding for binding in claim.number_bindings}'
    new='''bindings = {}
                for raw_binding in claim.number_bindings:
                    normalized, alias = _binding_alias(raw_binding, claim.revised_claim, lines)
                    if alias:
                        binding_aliases.append(alias)
                    bindings[normalized.claim_number] = normalized'''
    assert old in text;text=text.replace(old,new)
    text=text.replace('"automatic_number_bindings": automatic_bindings,','"automatic_number_bindings": automatic_bindings,\n                "numeric_binding_aliases": binding_aliases,')
    text=text.replace('7.1-semantic-review-neutral-connector-repair','8.1-exact-display-wrapper-binding')
    archive=ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py'
    if archive.exists() and archive.read_text(encoding='utf-8')!=text:raise ValueError('Existing V8 archive differs')
    if not archive.exists():archive.write_text(text,encoding='utf-8')
    write_json(ROOT/'evidence/verifier_versions/v8_1_development.json',{'created_utc':utc_now(),
        'parent_source_sha256':sha256(parent.read_bytes()),'source_sha256':sha256(archive.read_bytes()),
        'observed_trigger':'Real low-reasoning pilot emitted exact 2023A/2025A bindings; old numeric tokenizer stripped suffix and automatic repair then left duplicate token identities. Trailing sentence commas were also included in numeric tokens.',
        'change':'Normalize only identity bindings with identical A/E or currency wrappers present exactly in both revised text and stated evidence line; preserve raw bindings and mappings. Numeric regex accepts grouping commas only before exactly three digits.',
        'not_changed':'No labels, new evidence, output wording, model instructions or correction policy supplied by this repair. Wrong suffix, currency or uncited evidence still fails.',
        'limitations':'Format compatibility does not establish entity/year column alignment or semantic correctness. All candidates still require independent evaluation.'})
    if args.activate:
        target=Path.home()/'.cache/finrobot-zg2556/upstream/finrobot_equity/core/src/modules/equity_agents/verifier_agent.py'
        assert sha256(target.read_bytes()) in {sha256(parent.read_bytes()),sha256(archive.read_bytes()),sha256((ROOT/'evidence/verifier_versions/verifier_agent_v8.py').read_bytes())}
        target.write_bytes(archive.read_bytes())
    print('Prepared observed-format V8 repair; activated='+str(args.activate))
if __name__=='__main__':main()
