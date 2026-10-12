"""Handle the second observed fiscal-token representation without changing semantics."""
from pathlib import Path
from audit_io import sha256,write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]
parent=ROOT/'evidence/verifier_versions/verifier_agent_v8_1.py';text=parent.read_text(encoding='utf-8')
text=text.replace("year = re.fullmatch(r'((?:19|20)\\d{2})([AE])', a)","year = re.fullmatch(r'((?:19|20)\\d{2})([AE])', b)")
text=text.replace('if year and a == b:', 'if year and a in {b, year[1]}:')
text=text.replace("re.escape(a) + r'(?!\\w|\\.\\d)'", "re.escape(b) + r'(?!\\w|\\.\\d)'")
text=text.replace('8.1-exact-display-wrapper-binding','8.2-exact-display-wrapper-binding')
archive=ROOT/'evidence/verifier_versions/verifier_agent_v8_2.py'
if archive.exists() and archive.read_text(encoding='utf-8')!=text:raise ValueError('Existing immutable V8.2 source differs')
if not archive.exists():archive.write_text(text,encoding='utf-8')
target=Path.home()/'.cache/finrobot-zg2556/upstream/finrobot_equity/core/src/modules/equity_agents/verifier_agent.py'
assert sha256(target.read_bytes()) in {sha256(parent.read_bytes()),sha256(archive.read_bytes())}
target.write_bytes(archive.read_bytes())
write_json(ROOT/'evidence/verifier_versions/v8_2_development.json',{'at_utc':utc_now(),
    'parent_sha256':sha256(parent.read_bytes()),'source_sha256':sha256(archive.read_bytes()),
    'observed_trigger':'New real NVDA company-overview response uses claim_number=2023 and evidence_number=2023A while both revised sentence and cited header explicitly say2023A.',
    'change':'Permit bare numeric claim binding to the identical fully displayed A/E evidence token only if that complete wrapped token appears in BOTH revised text and stated source line. Preserve raw binding and alias trail. A/E mismatches or omitted suffix in displayed text remain rejected.',
    'active_paid_runs':'Do not modify clean-verify-v8-1 or active V8.1 calls. Revalidate saved results into separately named V8.2 runs after they complete.',
    'no_semantic_label_input':True})
print('Prepared separate V8.2; active V8.1 checkout unchanged.')
