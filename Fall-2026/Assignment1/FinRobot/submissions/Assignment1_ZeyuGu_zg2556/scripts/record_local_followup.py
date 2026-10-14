"""Record completed offline checks and a local-only continuation state."""
import json,subprocess,os,sys
from pathlib import Path
from audit_io import sha256,utc_now,write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    records=[]
    for regenerated,canonical in [('local_final_v7_comparison','comparison_v7_1'),('local_final_v7_annotations','annotation_v7_1')]:
        for original in (ROOT/'audits/grounding'/canonical).glob('*.csv'):
            repeated=ROOT/'reproduction'/regenerated/original.name
            assert original.read_bytes()==repeated.read_bytes(),original.name
            records.append({'file':str(original.relative_to(ROOT)),'sha256':sha256(original.read_bytes()),'byte_identical_offline_recompute':True})
    write_json(ROOT/'evidence/reproducibility/v7_final_recompute.json',{'checked_utc':utc_now(),'csvs':records,'paid_calls':0})
    clean=Path.home()/'.cache/finrobot-zg2556/clean-verify-7'
    env=dict(os.environ,FINROBOT_UPSTREAM=str(clean),FINROBOT_UPSTREAM_SRC=str(clean/'finrobot_equity/core/src'))
    result=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,env=env,capture_output=True,text=True)
    log=ROOT/'evidence/reproducibility/local_final_tests.log';log.write_text(result.stdout+result.stderr,encoding='utf-8')
    assert result.returncode==0
    write_json(ROOT/'evidence/reproducibility/local_final_tests.json',{'checked_utc':utc_now(),'upstream':str(clean),
        'returncode':result.returncode,'log_sha256':sha256(log.read_bytes()),'test_files':{str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in (ROOT/'tests').glob('test_*.py')},
        'scope':'Offline regression tests including five reasoning-contract drift cases; not experimental financial evidence.'})
    text=(ROOT/'docs/verifier_v7_followup.md').read_text(encoding='utf-8')
    text=text.replace('while further validation is in progress','with the completed follow-up reported separately')
    if '## Closed reasoning pilot' not in text:
        text+='\n## Closed reasoning pilot\n\nThe one-section GPT-5.4 pilot failed with an empty visible response after its 26,000 completion-token limit. Saved provider usage supports $0.400930 and278.07seconds, now reconciled in the shared ledger. There was no usable review and no rollout. See `evidence/reasoning_pilot/NVDA_company_overview/reconciliation.json`. The consolidated research PDF reports this negative result and both full V6/V7.1 studies. Main V6 release selection is retained; neither treatment satisfies strict factual-correctness certification.\n'
    (ROOT/'docs/verifier_v7_followup.md').write_text(text,encoding='utf-8')
    history=ROOT/'docs/verifier_version_history.md';text=history.read_text(encoding='utf-8')
    if '## Completed V7.1 follow-up' not in text:
        text+='\n## Completed V7.1 follow-up\n\nSee `verifier_v7_followup.md` for seven rendered candidates, all-section paired NVDA/AMD/INTC audits, separate applicable development patches, improved warnings and residual semantic failures. Exact source SHA and reuse lineage remain in each run. The main patches remain V6. A subsequent GPT-5.4 single-section pilot failed at its completion-token limit and was not expanded. All charges are in the common ledger; final summaries include failures. Publication is explicitly deferred by the user.\n'
    history.write_text(text,encoding='utf-8')
    write_json(ROOT/'evidence/progress/current_continuation.json',{'updated_at_utc':utc_now(),
        'phase':'local_release_consolidation','live_paid_driver_session_id':None,'new_paid_call_needed':False,
        'canonical_release':'V6 preserved; full V7.1 separate measured study',
        'reasoning_pilot':'Failed; complete provider usage reconciled; no review; no rollout.',
        'publication':'User explicitly deferred; do not publish Medium or course PR and do not ask again unless user resumes.',
        'assignment_complete':False,'remaining':'Strict zero-factual-error condition is unmet. Formal publication deferred.',
        'next':'Finish current PDF visual inspection, secret scan, scoped inventory/diff and local checkpoint. Do not repeat paid historical failures.'})
    print('Saved exact CSV replay and current offline regression checks.')
if __name__=='__main__':main()
