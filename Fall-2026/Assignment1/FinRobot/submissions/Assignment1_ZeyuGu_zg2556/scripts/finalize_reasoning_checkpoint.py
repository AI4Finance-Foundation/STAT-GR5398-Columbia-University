"""Consolidate completed local reasoning evidence; never publish or call a model."""
import json
from pathlib import Path
from audit_io import write_json, sha256, utc_now
ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))

def main():
    comparisons = []
    for name, canonical in [('comparison', 'comparison_reasoning_v8_2'), ('annotation', 'annotation_reasoning_v8_2')]:
        regenerated = ROOT / 'reproduction/reasoning_release_check' / name
        original = ROOT / 'audits/grounding' / canonical
        expected = sorted(p.name for p in original.glob('*.csv'))
        assert expected and expected == sorted(p.name for p in regenerated.glob('*.csv'))
        for filename in expected:
            before, after = (original / filename).read_bytes(), (regenerated / filename).read_bytes()
            assert before == after, f'Recomputed CSV differs: {name}/{filename}'
            comparisons.append({'file': f'audits/grounding/{canonical}/{filename}', 'sha256': sha256(before), 'byte_identical': True})
    write_json(ROOT / 'evidence/reproducibility/reasoning_release_metrics.json', {
        'checked_utc': utc_now(), 'network_or_model_calls': 0, 'checks': comparisons,
        'passed': True, 'scope': 'All saved comparison and warning-analysis CSVs recomputed from selected complete audit pairs; exact byte comparison. This reproduces metrics from labels, not independent proof of semantic labels.'
    })
    status_path = ROOT / 'docs/reasoning_followup_status.md'
    current = status_path.read_text(encoding='utf-8')
    marker = '<!-- COMPLETED-REASONING-CHECKPOINT -->'
    if marker not in current:
        current = marker + '\n# Completed follow-up: historical checkpoint below\n\nThe full three-company experiment and independent machine audit are now complete. The following original status is retained for chronology and is superseded by [the complete study](verifier_reasoning_v8_followup.md), the updated research report, and evidence/reasoning_study_summary.json. All 24 paid source calls settled; offline derivations are not new draws. Publication remains deferred.\n\n---\n\n' + current
        status_path.write_text(current, encoding='utf-8')
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    link = '\n\nAcademic submission readiness, residual experimental errors and the intentionally deferred publication steps are distinguished in [submission readiness](docs/submission_readiness.md). Final local integrity results are in evidence/delivery_validation.json; all reasoning scorecard/annotation CSVs were independently recomputed in evidence/reproducibility/reasoning_release_metrics.json.\n'
    if 'docs/submission_readiness.md' not in readme:
        readme_path.write_text(readme.rstrip() + link, encoding='utf-8')
    cost = read('audits/resources/cost_summary.json')
    study = read('evidence/reasoning_study_summary.json')
    replay = read('evidence/reproducibility/reasoning_replay/manifest.json')
    assert study['study_scope_complete'] and not cost['unsettled_call_ids']
    assert replay['passed'] and len(replay['records']) == 3 and replay['new_model_calls'] == 0
    path = ROOT / 'evidence/progress/current_continuation.json'
    previous = path.read_bytes()
    archive = path.parent / ('continuation_before_' + sha256(previous)[:12] + '.json')
    if not archive.exists():
        archive.write_bytes(previous)
    write_json(path, {
        'updated_at_utc': utc_now(),
        'phase': 'all_experiments_audits_PDF_QA_and_relocated_replay_complete_local_export',
        'assignment_complete': False,
        'live_paid_driver_session_id': None, 'new_paid_call_needed': False,
        'settled_cost_usd': cost['cost_usd'], 'approved_cap_usd': 20,
        'canonical_release': 'V6 preserved; full V7.1 and reasoning V8.2 measured studies alongside it.',
        'study': 'docs/verifier_reasoning_v8_followup.md',
        'complete_pairs': 'config/grounding_reasoning_v8_2_pairs.json',
        'AMD_selected_audit': 'audits/grounding/AMD_after_reasoning_v8_2_adjudicated_v2',
        'NVDA_selected_baseline': 'audits/grounding/NVDA_before_reasoning_erratum_adjudicated',
        'annotation_errata': 'Original labels preserved; original input and generated text unchanged.',
        'PDF': '15 pages covered by actual visual inspections or exact previously inspected PNG identity.',
        'development_patch': 'evidence/verifier_versions/patches_v8_2; clean apply132 tests +6 subtests.',
        'relocated_replay': 'Three companies replayed with model calls and verifier/renderer network connections disabled;24 exact sections and9 exact chart assets, all analysis files identical. See evidence/reproducibility/reasoning_replay/manifest.json.',
        'publication': 'User explicitly deferred. Do not publish Medium or course PR, and do not ask again unless user resumes.',
        'browser_cleanup': 'Agent report tab closed and local server stopped; user browser tabs untouched.',
        'remaining': 'Publication deferred. Residual narrative errors are research findings; error-free investment-use readiness is not claimed. Final scan/inventory/diff metadata record the local export after this checkpoint.',
        'next': 'Read evidence/delivery_validation.json and evidence/course_submission_diff.json for final export status. No experimental work or paid call remains necessary for the current local research package. Before a later user-authorized publication, refresh scan/inventory/scoped diff and insert only real publication URLs.'
    })
    print(f'All {len(comparisons)} recomputed CSVs byte-identical; local progress and reading guide refreshed.')

if __name__ == '__main__':
    main()
