"""Validate the separately selected, user-provided research PDF without rebuilding it."""
import json
import fnmatch
import subprocess
from pathlib import Path
from audit_io import sha256

ROOT = Path(__file__).resolve().parents[1]


def validate():
    path = ROOT / 'evidence/submission_pdf_review/selection.json'
    selection = json.loads(path.read_text(encoding='utf-8'))
    pdf = (ROOT / selection['selected_pdf']).resolve()
    assert ROOT in pdf.parents and pdf.is_file()
    content = pdf.read_bytes()
    assert content.startswith(b'%PDF-')
    assert sha256(content) == selection['pdf_sha256'] == selection['attachment_sha256']
    assert len(content) == selection['pdf_bytes']
    assert selection['pages'] == 16
    assert selection['visually_reviewed_pages'] == list(range(1, 17))
    assert selection['scientific_body_matches_saved_latex_edition']
    assert not selection['exact_build_from_current_source_claimed']
    assert not selection['publication_performed']
    policy = json.loads((ROOT / 'config/research_report_submission_policy.json').read_text(encoding='utf-8'))
    assert policy['selected_pdf'] == selection['selected_pdf']
    assert selection['source_and_comparison_artifacts_scope'] == 'local_only_historical_provenance'
    # These were checked during the earlier comparison, but are deliberately
    # not course deliverables. Validate them locally when available, not as
    # dependencies of the submitted package.
    local_history_checked = []
    for relative, digest in selection['source_and_comparison_sha256'].items():
        assert any(fnmatch.fnmatchcase(relative, pattern) for pattern in policy['excluded_globs'])
        if (ROOT / relative).is_file():
            assert sha256((ROOT / relative).read_bytes()) == digest, relative
            local_history_checked.append(relative)
    for page in selection['page_renders']:
        assert sha256((ROOT / page['path']).read_bytes()) == page['sha256']
    for relative, digest in selection['content_comparison_artifacts'].items():
        assert sha256((ROOT / relative).read_bytes()) == digest, relative
    comparison = json.loads((ROOT / 'evidence/submission_pdf_review/content_comparison/comparison.json').read_text(encoding='utf-8'))
    assert comparison['comparison_pass'] and comparison['all_raw_differences_classified']
    assert comparison['candidate']['file_sha256'] == selection['pdf_sha256']
    assert comparison['reference']['file_sha256'] == selection['source_and_comparison_sha256']['output/pdf/research_report_latex.pdf']
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert '(' + selection['selected_pdf'] + ')' in readme
    assert len(selection['presentation_findings']) == 2
    listed = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'], cwd=ROOT).decode('utf-8').split('\0')
    violations = [name for name in listed if name and any(fnmatch.fnmatchcase(name, pattern) for pattern in policy['excluded_globs'])]
    assert not violations, 'Local-only report artifacts would be submitted: ' + repr(violations)
    return {'passed': True, 'selected_pdf': selection['selected_pdf'],
            'pdf_sha256': selection['pdf_sha256'], 'pages': selection['pages'],
            'preserved_attachment_bytes': True,
            'presentation_findings': selection['presentation_findings'],
            'review_manifest': path.relative_to(ROOT).as_posix(),
            'local_history_files_optionally_checked': local_history_checked,
            'excluded_report_artifacts_in_submission': violations,
            'scope': 'Selected PDF identity, saved page review and comparison records. Manuscript sources and prior PDFs are local-only historical provenance, not required submission dependencies. No human financial validation or reproduced external PDF build is claimed.'}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
