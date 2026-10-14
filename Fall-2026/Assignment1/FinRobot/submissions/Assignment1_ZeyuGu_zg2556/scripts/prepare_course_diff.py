"""Prepare a reviewable course-folder patch using an isolated temporary Git index.

Does not stage the shared index, switch branches, commit, push, or create a PR.
The large binary patch stays outside the course submission to avoid recursion.
"""
import argparse,os,subprocess,json,hashlib,fnmatch
from collections import Counter
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]

def assert_index_bytes(repo,prefix,env):
    entries=subprocess.check_output(['git','ls-files','--stage','--full-name','-z','--',prefix],cwd=repo,env=env).decode('utf-8')
    bindings={}
    for entry in entries.split('\0'):
        if not entry:continue
        meta,name=entry.split('\t',1);mode,oid,stage=meta.split()
        assert name.startswith(prefix+'/') and stage=='0'
        file=repo/name
        if os.name=='nt':file=Path('\\\\?\\'+str(file.resolve()))
        raw=file.read_bytes();expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert oid==expected,'Git transformed saved bytes; check submission .gitattributes: '+name
        bindings[name]=oid
    assert bindings
    return bindings
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-directory',type=Path,required=True)
    p.add_argument('--credential-config',help='Optional private config outside submission, used only for exact secret-byte scanning of exported patch')
    a=p.parse_args();out=a.output_directory.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Diff/index must stay outside submission')
    out.mkdir(parents=True,exist_ok=False);repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip());prefix=ROOT.relative_to(repo).as_posix()
    shared_index=Path(subprocess.check_output(['git','rev-parse','--git-path','index'],cwd=repo,text=True).strip());shared_index=shared_index if shared_index.is_absolute() else repo/shared_index
    before=sha256(shared_index.read_bytes());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    env=dict(os.environ,GIT_INDEX_FILE=str(out/'index'))
    subprocess.run(['git','read-tree','HEAD'],cwd=repo,env=env,check=True,capture_output=True)
    add=subprocess.run(['git','add','--',prefix],cwd=repo,env=env,capture_output=True,text=True)
    (out/'git_add.log').write_text(add.stderr,encoding='utf-8');add.check_returncode()
    # These describe this very patch; including them creates circular/stale hashes.
    exclusions=['evidence/course_submission_diff.json','docs/course_submission_diff_stat.txt']
    subprocess.run(['git','update-index','--force-remove','--',*[prefix+'/'+x for x in exclusions]],cwd=repo,env=env,check=True,capture_output=True)
    names=subprocess.check_output(['git','diff','--no-ext-diff','--no-textconv','--cached','--name-only','-z'],cwd=repo,env=env).decode().split('\0');names=[n for n in names if n]
    assert names and all(n.startswith(prefix+'/') for n in names)
    policy=json.loads((ROOT/'config/research_report_submission_policy.json').read_text(encoding='utf-8'))
    assert not [n for n in names if any(fnmatch.fnmatchcase(n[len(prefix)+1:],pattern) for pattern in policy['excluded_globs'])], 'Local-only report source/history appeared in course patch'
    blob_bindings=assert_index_bytes(repo,prefix,env)
    write_json(out/'course_index_bytes.json',{'passed':True,'files':len(blob_bindings),'git_blob_sha1':blob_bindings,
        'method':'Every scoped staged object ID equals the Git blob SHA-1 reconstructed from exact unfiltered working-file bytes. Submission * -text prevents platform newline conversion.'})
    short=subprocess.check_output(['git','diff','--no-ext-diff','--no-textconv','--cached','--shortstat'],cwd=repo,env=env,text=True).strip()
    patch=out/'course_submission.patch'
    with patch.open('wb') as f:subprocess.run(['git','diff','--no-ext-diff','--no-textconv','--cached','--binary','--full-index'],cwd=repo,env=env,stdout=f,check=True)
    assert sha256(shared_index.read_bytes())==before
    check_env=dict(os.environ,GIT_INDEX_FILE=str(out/'check_index'))
    subprocess.run(['git','read-tree','HEAD'],cwd=repo,env=check_env,check=True,capture_output=True)
    checked=subprocess.run(['git','apply','--check','--cached','--binary',str(patch)],cwd=repo,env=check_env,capture_output=True)
    (out/'course_patch_apply_check.log').write_bytes(checked.stdout+checked.stderr)
    checked.check_returncode()
    # Prove actual application to a clean HEAD index, not merely --check.
    subprocess.run(['git','apply','--cached','--binary',str(patch)],cwd=repo,env=check_env,capture_output=True,check=True)
    assert assert_index_bytes(repo,prefix,check_env)==blob_bindings
    credential_scan={'performed':False}
    if a.credential_config:
        from audit_io import credentials
        from urllib.parse import quote
        private=Path(a.credential_config).resolve()
        assert private!=ROOT and ROOT not in private.parents
        cfg=credentials(private)
        secrets=[cfg[k] for k in ('openai_api_key','fmp_api_key') if cfg.get(k)]
        assert secrets
        raw=patch.read_bytes()
        matches=sum(any(x.encode() in raw for x in (value,quote(value,safe=''))) for value in secrets)
        assert matches==0, 'Exported patch contains credential bytes; do not publish'
        credential_scan={'performed':True,'known_secret_count':len(secrets),'matching_secret_count':matches,'passed':True,'scope':'Actual FMP/OpenAI key bytes and URL-encoded forms; values never printed.'}
    stats=Counter(n[len(prefix)+1:].split('/')[0] for n in names)
    m={'prepared_utc':utc_now(),'course_base_sha':head,'upstream_course_url':'https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University','required_pr_base':'master','scope':prefix,'changed_files':len(names),'groups':dict(sorted(stats.items())),
       'shortstat':short,'patch_path_local_only':str(patch),'patch_sha256':sha256(patch.read_bytes()),'patch_bytes':patch.stat().st_size,
       'shared_index_unchanged':True,'shared_branch_unchanged':True,'committed':False,'pushed':False,'pull_request_created':False,
       'course_patch_apply_check_passed':True,'course_patch_actual_apply_to_clean_index_passed':True,
       'all_scoped_git_blobs_equal_original_bytes':True,'byte_checked_files':len(blob_bindings),
       'patch_textconv_disabled':True,'patch_external_diff_disabled':True,
       'git_byte_proof_path':str(out/'course_index_bytes.json'),'git_byte_proof_sha256':sha256((out/'course_index_bytes.json').read_bytes()),
       'credential_scan':credential_scan,
       'excluded_local_metadata':exclusions,
       'note':'Two self-referential local diff metadata files are explicitly excluded from the temporary index. All other scoped files reflect this local export. Refresh immediately before any later authorized publication. Shared index and branch untouched.'}
    write_json(ROOT/'evidence/course_submission_diff.json',m)
    (ROOT/'docs/course_submission_diff_stat.txt').write_text(short+'\n\n'+ '\n'.join(f'{name}: {count} files' for name,count in sorted(stats.items()))+'\nOnly '+prefix+' changed. No shared index or branch was modified.\n',encoding='utf-8')
    print(json.dumps(m,indent=2))
if __name__=='__main__':main()
