"""Check a real copy of submitted files without local-only report sources."""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from audit_io import sha256, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def disk(path):
    return Path('\\\\?\\' + str(path.resolve())) if os.name == 'nt' else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    target = args.output_directory.resolve()
    assert target != ROOT and ROOT not in target.parents
    target.mkdir(parents=True, exist_ok=False)
    policy = json.loads((ROOT/'config/research_report_submission_policy.json').read_text(encoding='utf-8'))
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.'], cwd=ROOT).decode('utf-8')
    names = sorted(set(filter(None, paths.split('\0'))))
    excluded_export_metadata = {'evidence/course_submission_diff.json', 'docs/course_submission_diff_stat.txt'}
    names = [name for name in names if name not in excluded_export_metadata]
    assert not [name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in policy['excluded_globs'])]
    copied = {}
    for name in names:
        src, dst = disk(ROOT/name), disk(target/name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[name] = sha256(src.read_bytes())
        assert sha256(dst.read_bytes()) == copied[name]
    assert not (target/'docs/research_report.md').exists()
    assert not (target/'docs/latex').exists()
    assert not (target/'evidence/document_history').exists()
    assert not (target/'output/pdf/research_report.pdf').exists()
    assert not (target/'output/pdf/research_report_latex.pdf').exists()
    assert (target/policy['selected_pdf']).is_file()
    # Only the disposable copy gets a Git repository. Its index remains empty.
    subprocess.run(['git', 'init', '--quiet', str(target)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(target), 'config', 'core.longpaths', 'true'], check=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, 'scripts/validate_delivery.py']
    result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True, encoding='utf-8')
    (target/'validation_stdout.log').write_text(result.stdout, encoding='utf-8')
    (target/'validation_stderr.log').write_text(result.stderr, encoding='utf-8')
    result.check_returncode()
    validation = json.loads((target/'evidence/delivery_validation.json').read_text(encoding='utf-8'))
    assert validation['technical_scope_checks_passed']
    assert validation['selected_research_pdf']['local_history_files_optionally_checked'] == []
    proof = {'checked_at_utc': utc_now(), 'passed': True, 'copied_files': len(copied),
             'copied_file_sha256': copied, 'source_sha256': sha256(Path(__file__).read_bytes()),
             'selected_pdf': policy['selected_pdf'],
             'selected_pdf_sha256': sha256((target/policy['selected_pdf']).read_bytes()),
             'local_only_sources_and_older_reports_absent': True,
             'command': command, 'validation_exit_code': result.returncode,
             'stdout': result.stdout, 'stderr': result.stderr,
             'new_api_calls': 0, 'output_directory_local_only': str(target),
             'scope': 'Exact-byte copy of the current scoped Git file list, excluding the same two metadata files as the course exporter. Full delivery validation passed without local report sources or older report editions. This is a packaging/dependency check, not a new experiment or independent semantic review.'}
    write_json(ROOT/'evidence/reproducibility/report_pdf_only_package.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in {'copied_file_sha256','command','stdout','stderr'}}, indent=2))


if __name__ == '__main__':
    main()
