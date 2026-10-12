"""Checkpoint the measured local revision; publication and truth guarantees stay open."""
import json
from pathlib import Path
from audit_io import utc_now,sha256,write_json
from validate_delivery import validate_semantic_followup,validate_submission_relocation_evidence
from checkpoint_release import publication_context
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def main():
    publication,medium_verified,publication_note,publication_remaining=publication_context()
    semantic=validate_semantic_followup();validate_submission_relocation_evidence('v9_1');validate_submission_relocation_evidence('v8_2')
    study=semantic['study'];s=study['scorecard'];a=study['warnings'];cost=read('audits/resources/cost_summary.json')
    build=read('output/pdf/research_report_build.json');qa=read('evidence/pdf_release_visual_qa.json')
    assert build['pdf_sha256']==qa['pdf_sha256'] and build['source_sha256']==qa['source_sha256']
    assert qa['pages_inspected']==build['pages'] and build['pages']>0
    assert read('evidence/reproducibility/semantic_release_metrics.json')['passed']
    assert read('evidence/reproducibility/course_git_bytes_after.json')['passed']
    assert '235 passed, 6 subtests passed' in (ROOT/'evidence/semantic_final_tests.log').read_text(encoding='utf-8-sig')
    if (ROOT/'evidence/forecast_scenario_release.json').exists():
        from validate_forecast_scenario_release import validate as validate_scenario
        checks=validate_scenario()
        scenario=read('evidence/forecast_scenario_release.json')
        scenario['release_checks']=checks
        scenario['current_pdf_sha256']=build['pdf_sha256']
        scenario['current_pdf_pages']=build['pages']
        scenario['pdf_visual_qa_sha256']=sha256((ROOT/'evidence/pdf_release_visual_qa.json').read_bytes())
        scenario['next']='Local research package and separate forecast audit complete. Refresh final secret/inventory/export checks after all edits. Latest-master applicability is recorded separately in evidence/course_submission_diff.json to avoid self-referential patch hashes. '+publication_note
        write_json(ROOT/'evidence/forecast_scenario_release.json',scenario)
    path=ROOT/'evidence/progress/current_continuation.json';old=path.read_bytes()
    archive=path.parent/('continuation_before_'+sha256(old)[:12]+'.json')
    if not archive.exists():archive.write_bytes(old)
    write_json(path,{'updated_at_utc':utc_now(),'phase':'local_research_revision_complete_integrity_and_export_recorded_separately',
        'assignment_complete':False,'live_paid_driver_session_id':None,'new_paid_call_needed':False,
        'settled_cost_usd':cost['cost_usd'],'approved_cap_usd':20,'unsettled_calls':cost['unsettled_call_ids'],
        'current_followup':'V9.1; original V6/V7.1/V8.2 and initial evaluator labels remain preserved.',
        'study':'docs/verifier_semantic_v9_followup.md','complete_pairs':'config/grounding_reasoning_v9_1_pairs.json',
        'after_counts':{'supported':int(s['after_supported']),'unsupported':int(s['after_unsupported']),
            'contradicted':int(s['after_contradicted']),'claims':int(s['after_total_claims']),
            'numeric_unverifiable':int(s['after_numeric_unverifiable'])},
        'measurement_boundary':'Evaluator scope corrections change labels only; V9/V9.1 then clarify original-prompt horizons/ranges. Development cohort, machine-assisted audit, no human review or held-out guarantee.',
        'verification':f"Clean patch apply;235 tests+6 subtests;24 CSVs byte-identical on offline metric reproduction;true copied-submission V8.2/V9.1 replay with original-root Python reads denied;{build['pages']}-page current PDF visually inspected.",
        'offline_forecast_parameter_followup':'audits/forecast_parameter_scenarios/run_v2;separate from preserved Option A;see evidence/forecast_scenario_release.json and independent review evidence.',
        'publication_preparation':'docs/publication_preparation.md;blog/medium_metadata.json;docs/course_pr_body.md;see current publication facts in config/publication_status.json.',
        'publication':publication_note,'publication_status':publication,
        'remaining':publication_remaining+['Strict zero-error certification unavailable: historical experiment errors and unresolved narratives remain disclosed'],
        'delivery_integrity':'evidence/delivery_validation.json','scoped_local_export':'evidence/course_submission_diff.json',
        'git_bytes':'Submission * -text prevents newline rewriting. Saved initial byte check passed; scoped export independently verifies every final staged and actual clean-index applied blob. Historical conversion failure is retained.',
        'next':'No new paid experiment is needed for saved local reproduction. Read README and current validation/export manifests. Refresh scan/inventory/scoped export only if files change. Follow recorded publication and assistant-write authority separately; a user-published article does not authorize assistant remote writes. Do not confuse replay with new independent generation.'})
    print('Local research revision checkpoint saved; full assignment remains incomplete.')
if __name__=='__main__':main()
