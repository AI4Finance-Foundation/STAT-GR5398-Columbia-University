"""Compare every scoped temporary-index blob with actual unfiltered file bytes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from audit_io import utc_now,write_json,sha256
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-mismatch',action='store_true');a=p.parse_args()
    env=dict(os.environ,GIT_INDEX_FILE=str(a.index.resolve()))
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    prefix=ROOT.relative_to(repo).as_posix()+'/'
    assert subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=repo,text=True).strip()=='sha1'
    output=subprocess.check_output(['git','ls-files','--stage','--full-name','-z','--','.'],cwd=ROOT,env=env).decode('utf-8')
    records=[];mismatches=[]
    for entry in output.split('\0'):
        if not entry:continue
        meta,name=entry.split('\t',1);mode,oid,stage=meta.split()
        assert name.startswith(prefix) and stage=='0'
        file=repo/name
        if os.name=='nt':file=Path('\\\\?\\'+str(file.resolve()))
        raw=file.read_bytes();computed=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        row={'path':name[len(prefix):],'bytes':len(raw),'raw_sha256':sha256(raw),'raw_git_blob_sha1':computed,'index_git_blob_sha1':oid,'exact_bytes':computed==oid}
        records.append(row)
        if computed!=oid:mismatches.append(row)
    assert records
    write_json(a.output,{'checked_at_utc':utc_now(),'scope':prefix,'temporary_index':str(a.index.resolve()),
        'passed':not mismatches,'files':len(records),'mismatch_count':len(mismatches),'mismatches':mismatches,
        'all_blob_bindings':records,'script_sha256':sha256(Path(__file__).read_bytes()),
        'method':'Reconstruct Git SHA-1 blob object ID from unfiltered saved bytes and compare every staged object ID. No checkout line-ending transformation is assumed away.',
        'scope_limit':'Exact scoped index bytes; separate clean-apply and offline replay tests establish those stages.'})
    print(json.dumps({'passed':not mismatches,'files':len(records),'mismatch_count':len(mismatches)}))
    if mismatches and not a.allow_mismatch:raise SystemExit(1)
if __name__=='__main__':main()
