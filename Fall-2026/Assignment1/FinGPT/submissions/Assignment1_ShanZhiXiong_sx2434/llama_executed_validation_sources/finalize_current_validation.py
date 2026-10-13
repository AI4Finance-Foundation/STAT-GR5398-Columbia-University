"""One-run completion handler; never launches GPU work or resumes an automation."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
R=Path('/root/shared-nvme/AI4FinRuntime');B=Path(__file__).resolve().parent
PLAN_SHA='4409db129bd049e4cca3251182a270fb06895bf2bfa3fa4cf4c6dc656349b572'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main():
    if sha(B/'FROZEN_VALIDATION_PLAN.json')!=PLAN_SHA:raise ValueError('Pinned current run differs')
    plan=json.loads((B/'FROZEN_VALIDATION_PLAN.json').read_text());out=Path(plan['output_directory']);launch=json.loads((B/'ACTUAL_VALIDATION_LAUNCH.json').read_text());deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:
        try:status=json.loads((out/'status.json').read_text())
        except (OSError,ValueError):time.sleep(30);continue
        if status['stage']=='failed':
            save(B/'FINALIZATION_RECEIPT.json',{'stage':'worker_failed_no_restart','worker_pid':launch['pid'],'test_read':False,'automation_resumed':False});return
        if status['stage']!='completed':time.sleep(30);continue
        manifest=json.loads((out/'COMPLETED_OUTPUTS_SHA256.json').read_text())
        if manifest['output_records']!=480 or manifest['test_read'] is not False:raise ValueError('Completion count/scope differs')
        for n,s in manifest['files'].items():
            if Path(n).name!=n or sha(out/n)!=s:raise ValueError('Saved completion file differs')
        if sha(plan['evaluation_data'])!=plan['evaluation_data_sha256']:raise ValueError('Frozen generation source differs')
        rows=json.loads(Path(plan['evaluation_data']).read_text());ids=[r['sample_id'] for r in rows]
        for family in ['base','adapter']:
            records=[json.loads(x) for x in (out/f'{family}_outputs.jsonl').read_text().splitlines()]
            if [r['sample_id'] for r in records]!=ids:raise ValueError('Completed validation identity differs')
            for row,r in zip(rows,records):
                if r['reference']!=row['reference'] or r['previously_inspected_test'] is not False or hashlib.sha256(json.dumps(r['generated_token_ids']).encode()).hexdigest()!=r['generated_token_sha256']:raise ValueError('Output provenance differs')
        pipeline_path=R/'setup/three_model_validation240_pipeline.json';pipeline=json.loads(pipeline_path.read_text())
        if pipeline.get('current_run')==str(out) and pipeline.get('child_pid')==launch['pid']:
            pipeline.update(stage='completed',child_pid=None,pid=None,completed_model_evaluations=['qwen','deepseek','llama'],updated_at_utc=datetime.now(timezone.utc).isoformat(),note='All1440 original-prompt validation outputs complete. Separate HITL48 excluded from count. Final review/test remain pending.',report_pending=True,full_assignment_complete=False)
            save(pipeline_path,pipeline)
        save(B/'FINALIZATION_RECEIPT.json',{'stage':'validation480_completed_and_hash_verified','plan_sha256':PLAN_SHA,'completed_manifest_sha256':sha(out/'COMPLETED_OUTPUTS_SHA256.json'),'worker_pid':launch['pid'],'verified_outputs':480,'updated_at_utc':datetime.now(timezone.utc).isoformat(),'test_read':False,'automation_resumed':False,'course_ready':False})
        return
    raise TimeoutError('Completion wait ended; no restart or new work performed')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(B/'FINALIZATION_ERROR.json',{'error_type':type(e).__name__,'test_read':False,'automation_resumed':False});raise
