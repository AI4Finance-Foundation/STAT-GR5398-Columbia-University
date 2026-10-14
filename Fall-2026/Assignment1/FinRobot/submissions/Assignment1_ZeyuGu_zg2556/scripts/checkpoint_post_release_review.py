"""Record the reopened local review without overwriting historical releases."""
import json
from pathlib import Path
from audit_io import utc_now, write_json
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'evidence/progress/current_continuation.json'
prior=json.loads(path.read_text(encoding='utf-8'))
archive=ROOT/'evidence/progress/pre_post_release_review.json'
if not archive.exists(): write_json(archive,prior)
write_json(path,{
    'updated_at_utc':utc_now(),'phase':'post_release_measurement_correction_and_CAGR_candidate_validation',
    'assignment_complete':False,'settled_cost_usd':10.0450715,'approved_cap_usd':20,
    'live_paid_driver_session_id':None,'new_paid_call_needed':False,
    'publication':'User deferred Medium and course PR. Do not publish or ask again.',
    'historical_releases':'V6, V7.1 and V8.2 runs, annotations, PDF and scoped export preserved; old readiness statements superseded by this review.',
    'measurement_errata':'Symmetric before/after scope corrections complete in config/grounding_reasoning_v8_2_corrected_pairs.json; no model output changed.',
    'corrected_v8_2':'Before: 1479 claims,286 supported,1162 unsupported,31 contradicted. After:1958 claims,530 supported,1420 unsupported,8 contradicted.',
    'true_relocation':'Passed: copied submission, source-root Python filesystem access denied, no network or paid calls, future execution clock;3 identical HTML,24 sections,9 HTML chart assets. evidence/reproducibility/true_submission_relocation/verified/manifest.json',
    'candidate':'V9 explicit historical-versus-table Revenue CAGR clarification. Clean upstream apply and160 tests+6 subtests passed. Offline output generation and independent full-section audit pending.',
    'remaining':['Independent V9 output audit and before/after measurement','Update English report/blog/README/readiness to distinguish annotation corrections from output changes','Refresh PDF visual QA, integrity scan, artifact inventory and scoped local export','Publication intentionally deferred'],
    'next':'Run scripts/revalidate_reasoning_offline.py --upstream <clean V9 checkout> --verifier-version v9 --disable-network; audit exact rendered text before release.'
})
print('Recoverable post-release review checkpoint saved.')
