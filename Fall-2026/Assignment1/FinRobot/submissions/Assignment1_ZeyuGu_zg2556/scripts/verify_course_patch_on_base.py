"""Apply an exported course patch to a supplied isolated bare-repository base.

Read-only toward GitHub and the shared course worktree. Does not commit or push.
The verification result stays beside the external patch to avoid recursive hashes.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from audit_io import sha256, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bare-repo', type=Path, required=True)
    parser.add_argument('--base-sha', required=True)
    parser.add_argument('--export-directory', type=Path, required=True)
    args = parser.parse_args()
    bare = args.bare_repo.resolve()
    out = args.export_directory.resolve()
    assert re.fullmatch(r'[0-9a-f]{40}', args.base_sha), 'Supply the observed full commit SHA'
    assert ROOT != out and ROOT not in out.parents
    assert ROOT != bare and ROOT not in bare.parents
    assert subprocess.check_output(['git', 'rev-parse', '--is-bare-repository'], cwd=bare, text=True).strip() == 'true'
    assert subprocess.check_output(['git', 'rev-parse', args.base_sha+'^{commit}'], cwd=bare, text=True).strip() == args.base_sha
    manifest_path = ROOT/'evidence/course_submission_diff.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    patch = out/'course_submission.patch'
    assert Path(manifest['patch_path_local_only']).resolve() == patch
    assert sha256(patch.read_bytes()) == manifest['patch_sha256']
    proof_path = out/'course_index_bytes.json'
    assert sha256(proof_path.read_bytes()) == manifest['git_byte_proof_sha256']
    expected = json.loads(proof_path.read_text(encoding='utf-8'))['git_blob_sha1']
    prefix = manifest['scope']
    repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip())
    # Reject a stale export before comparing the applied Git objects.
    for name, oid in expected.items():
        path = repo/name
        if os.name == 'nt': path = Path('\\\\?\\'+str(path.resolve()))
        raw = path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == oid, name
    index = out/'latest_master_index'
    assert not index.exists(), 'Use a fresh export directory; preserve prior proofs'
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    subprocess.run(['git', 'read-tree', args.base_sha], cwd=bare, env=env, check=True, capture_output=True)
    checked = subprocess.run(['git', 'apply', '--check', '--cached', '--binary', str(patch)], cwd=bare, env=env, capture_output=True)
    (out/'latest_master_apply_check.log').write_bytes(checked.stdout+checked.stderr)
    checked.check_returncode()
    subprocess.run(['git', 'apply', '--cached', '--binary', str(patch)], cwd=bare, env=env, check=True, capture_output=True)
    actual = {}
    raw_entries = subprocess.check_output(['git', 'ls-files', '--stage', '--full-name', '-z', '--', prefix], cwd=bare, env=env).decode('utf-8')
    for entry in raw_entries.split('\0'):
        if not entry: continue
        metadata, name = entry.split('\t', 1)
        mode, oid, stage = metadata.split()
        assert stage == '0' and name.startswith(prefix+'/')
        actual[name] = oid
    assert actual == expected, 'Applied scoped blobs differ from the verified export'
    changed = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--no-textconv', '--cached', args.base_sha, '--name-only', '-z'], cwd=bare, env=env).decode('utf-8').split('\0')
    changed = [name for name in changed if name]
    assert changed and all(name.startswith(prefix+'/') for name in changed)
    result = {'checked_at_utc': utc_now(), 'passed': True, 'base_sha': args.base_sha,
        'base_source': 'Caller-supplied observed course master commit, fetched into an isolated bare repository.',
        'patch_sha256': manifest['patch_sha256'], 'actual_application_passed': True,
        'exact_scoped_blob_count': len(actual), 'changed_paths_only_in_submission': True,
        'scope': prefix, 'bare_repository_local_only': str(bare),
        'script_sha256': sha256(Path(__file__).read_bytes()), 'committed': False, 'pushed': False,
        'limitation': 'Proves application to this observed master SHA, not future master states or public submission.'}
    result_path = out/'latest_master_patch_check.json'
    write_json(result_path, result)
    manifest['observed_master_patch_check'] = {**result, 'proof_path_local_only': str(result_path),
        'proof_sha256': sha256(result_path.read_bytes())}
    write_json(manifest_path, manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
