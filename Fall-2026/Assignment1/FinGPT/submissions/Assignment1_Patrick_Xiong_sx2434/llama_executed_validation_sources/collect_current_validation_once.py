"""Collect this one authorized GPU run when it finishes; no scheduler or new work."""
import hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
W=Path(r'D:\AI4FinRuntime\assignment1_three_model_submission_work');B=Path(__file__).resolve().parent
REMOTE='/root/shared-nvme/AI4FinRuntime/setup/llama_retrospective_validation_v1_20261012/FINALIZATION_RECEIPT.json'
OUT_REMOTE='/root/shared-nvme/AI4FinRuntime/import_20261009/assignment1_20261008/runs/server4090_llama_base_int8_validation240_gen1920_retrospective_20261012'
DEST=W/'llama_validation_results';RUN=DEST/Path(OUT_REMOTE).name
SHA='4409db129bd049e4cca3251182a270fb06895bf2bfa3fa4cf4c6dc656349b572'
def save(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def main():
    deadline=time.monotonic()+12*3600
    if RUN.exists():raise ValueError('Do not overwrite existing collected evidence')
    DEST.mkdir(exist_ok=True)
    while time.monotonic()<deadline:
        # Return only the existing safe completion receipt, never credentials or raw data.
        cmd="python3 -c \"import pathlib; p=pathlib.Path('"+REMOTE+"'); print(p.read_text() if p.exists() else '{}')\""
        r=subprocess.run(['ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=15','paratera-4090',cmd],capture_output=True,text=True,timeout=25)
        try:x=json.loads(r.stdout) if r.returncode==0 else {}
        except ValueError:x={}
        if x.get('stage')=='worker_failed_no_restart':
            save(B/'LOCAL_COLLECTION_STATUS.json',{'stage':'worker_failed_no_restart','automation_resumed':False});return
        if x.get('stage')!='validation480_completed_and_hash_verified':time.sleep(30);continue
        if x.get('plan_sha256')!=SHA or x.get('test_read') is not False or x.get('verified_outputs')!=480:raise ValueError('Completion receipt scope differs')
        subprocess.run(['scp.exe','-r','paratera-4090:'+OUT_REMOTE,str(DEST)],check=True,timeout=600)
        p=RUN/'COMPLETED_OUTPUTS_SHA256.json'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=x['completed_manifest_sha256']:raise ValueError('Collected manifest differs')
        manifest=json.loads(p.read_text())
        for n,h in manifest['files'].items():
            if Path(n).name!=n or hashlib.sha256((RUN/n).read_bytes()).hexdigest()!=h:raise ValueError('Collected file differs')
        receipt={'stage':'collected_and_verified','directory':str(RUN),'output_records':480,'files_verified':len(manifest['files']),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'remote_completion':x,'test_read':False,'automation_resumed':False,'independent_audit_pending':True,'course_ready':False}
        save(B/'LOCAL_COLLECTION_STATUS.json',receipt)
        # Record only mechanical completion. Any new user scope still takes precedence.
        p=W/'SUBMISSION_READINESS_STATUS.json';state=json.loads(p.read_text())
        if state['llama_base_preparation'].get('validation_plan_sha256')==SHA:
            save(B/'READINESS_BEFORE_COLLECTION.json',state)
            state['llama_base_preparation'].update(stage='training_and_validation_completed',validation_completed=True,saved_validation_outputs=480,validation_local_directory=str(RUN),validation_independent_audit_pending=True)
            state['existing_validation_outputs']=1440;state['checklist']['raw_outputs_metrics']='1440 original-prompt validation outputs complete; separate HITL48 not mixed in; final-test not read.';state['course_ready']=False;state['updated_at_utc']=receipt['completed_at_utc'];save(p,state)
        return
    raise TimeoutError('One-run collection wait ended; no restart or new work performed')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(B/'LOCAL_COLLECTION_ERROR.json',{'error_type':type(e).__name__,'automation_resumed':False,'test_read':False});raise
