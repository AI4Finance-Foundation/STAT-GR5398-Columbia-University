"""Read-only current-file validation of the isolated forecast-parameter follow-up."""
import csv
import json
import os
from pathlib import Path
from audit_io import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = 'audits/forecast_parameter_scenarios/run_v2'
REVIEW = 'evidence/forecast_parameter_scenarios_review'


def raw(relative):
    path = (ROOT/relative).resolve()
    assert path == ROOT or ROOT in path.parents
    if os.name == 'nt': path = Path('\\\\?\\'+str(path))
    return path.read_bytes()


def read(relative):
    return json.loads(raw(relative).decode('utf-8-sig'))


def rows(relative):
    return list(csv.DictReader(raw(relative).decode('utf-8-sig').splitlines()))


def validate():
    run = read(RUN+'/manifest.json')
    assert run['status'] == 'PASS' and run['all_checks_passed'] and run['network_disabled']
    assert run['new_api_calls'] == run['new_model_calls'] == run['tokens'] == run['api_cost_usd'] == 0
    assert run['original_data_and_verifier_experiment_unchanged']
    assert sha256(raw('scripts/forecast_parameter_scenarios.py')) == run['driver_sha256']
    for name, digest in run['input_hashes'].items(): assert sha256(raw(name)) == digest, name
    for name, digest in run['output_hashes'].items(): assert sha256(raw(RUN+'/'+name)) == digest, name
    assert len(rows(RUN+'/forecast_comparison.csv')) == 6
    assert len(rows(RUN+'/target_price_comparison.csv')) == 2
    assert all(r['verdict'] == 'MATCH' for r in rows(RUN+'/baseline_cell_reconciliation.csv'))
    network = read(RUN+'/network_guard.json')
    assert network['unexpected_attempts'] == 0 and len(network['probes']) == 4
    assert all(p['result'] == 'REJECTED_BEFORE_IO' for p in network['probes'])
    reference = read(REVIEW+'/independent_reference/reference_manifest.json')
    assert sha256(raw('scripts/independent_forecast_scenario_review.py')) == reference['source_sha256']
    for name, digest in reference['inputs'].items(): assert sha256(raw(name)) == digest, name
    for name, digest in reference['outputs'].items(): assert sha256(raw(REVIEW+'/independent_reference/'+name)) == digest, name
    review = read(REVIEW+'/review_manifest.json')
    assert review['passed'] and review['failure_count'] == 0 and review['checks'] == 670
    assert sha256(raw('scripts/compare_forecast_scenario_review.py')) == review['source_sha256']
    # Saved provenance may name the original checkout; resolve within this copy.
    for name, digest in review['inputs'].items():
        suffix = name.replace('\\', '/').split('/Assignment1_ZeyuGu_zg2556/', 1)[-1]
        assert not Path(suffix).is_absolute()
        assert sha256(raw(suffix)) == digest, name
    checks = rows(REVIEW+'/independent_reconciliation.csv')
    assert len(checks) == review['checks'] and all(r['passed'].lower() == 'true' for r in checks)
    before = read(REVIEW+'/before_integrity.json')
    preserved = read(REVIEW+'/preservation_review.json')
    assert preserved['passed'] and not preserved['changed_or_removed'] and not preserved['added_within_protected_scope']
    assert sha256(raw(REVIEW+'/before_integrity.json')) == preserved['before_manifest_sha256']
    submitted = [r for r in before['bindings'] if r.get('last_export_git_blob_sha1')]
    assert len(submitted) == preserved['submitted_files'] == 9083
    for entry in submitted:
        content = raw(entry['path'])
        assert len(content) == entry['bytes'] and sha256(content) == entry['sha256'], entry['path']
    replay = read('audits/forecast_parameter_scenarios/reproduction_validation.json')
    assert replay['status'] == 'PASS' and replay['scientific_artifact_files_byte_identical'] == 33
    assert sha256(raw(RUN+'/manifest.json')) == replay['original_manifest_sha256']
    summary = read('evidence/forecast_scenario_release.json')
    assert sha256(raw(RUN+'/manifest.json')) == summary['scenario_manifest_sha256']
    for name, digest in summary['document_sha256'].items(): assert sha256(raw(name)) == digest, name
    metadata = read('blog/medium_metadata.json')
    assert metadata['article_source_hash'] == sha256(raw('blog/medium_article.md'))
    assert metadata['course_pr']['body_sha256'] == sha256(raw('docs/course_pr_body.md'))
    from validate_medium_publication import validate as validate_publication
    validate_publication()
    return {'passed': True, 'scenario_directory': RUN, 'default_fields': run['baseline_fields_compared'],
        'independent_review_checks': review['checks'], 'protected_original_files_unchanged_now': len(submitted),
        'reproduced_byte_identical_artifacts': replay['scientific_artifact_files_byte_identical'],
        'new_api_cost_usd': 0, 'original_verifier_comparison_changed': False,
        'scope': 'Source/input/output bindings, prior independent review, current original-file hashes and updated manuscript bindings; no forecast accuracy or human-review claim.'}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
