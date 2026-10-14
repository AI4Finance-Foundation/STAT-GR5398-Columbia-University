"""Archive a V9.1 candidate and probe saved V9 releases offline; never install/run it.

This script reads no audits, credentials or paid-review labels to select changes.
It does not write upstream, generation runs, reports or historical archives.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA="ff9c5553330db88ae262bb69e6a799b7ed75c71ae8c76331526ea0d510c4691b"


def offline(*args,**kwargs):
    raise RuntimeError("V9.1 preparation and saved-row probe are offline")


socket.create_connection=offline
socket.socket.connect=offline
socket.socket.connect_ex=offline
from audit_io import sha256,write_json,utc_now


def save_immutable(path,content):
    if path.exists():
        if path.read_bytes()!=content:
            raise ValueError("Existing immutable candidate differs: "+str(path))
        return
    path.write_bytes(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-output',type=Path,default=ROOT/'evidence/verifier_versions/v9_1_development.json')
    args=parser.parse_args()
    parent=ROOT/'evidence/verifier_versions/verifier_agent_v9.py'
    fragment=ROOT/'evidence/verifier_versions/ev_range_guard_v9_1.py'
    archive=ROOT/'evidence/verifier_versions/verifier_agent_v9_1.py'
    if sha256(parent.read_bytes())!=PARENT_SHA:raise ValueError('Immutable V9 parent changed')
    # Start with the exact original byte prefix, including its newline convention.
    wrapper='''

# The paid-review schema and instructions are unchanged. This independent,
# deterministic release guard consumes original writer-prompt evidence only.
_validate_review_before_ev_range_guard = validate_review

def validate_review(review, evidence, text):
    return apply_ev_ebitda_range_guard(
        _validate_review_before_ev_range_guard(review, evidence, text), evidence)
'''
    candidate=parent.read_bytes()+b'\n\n'+fragment.read_bytes()+wrapper.encode('utf-8')
    compile(candidate,str(archive),'exec')
    save_immutable(archive,candidate)
    spec=importlib.util.spec_from_file_location('ev_range_guard_v9_1_probe',fragment)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=sorted((ROOT/'runs').glob('*v9/verifier/*/checked.json'))
    if not files:raise ValueError('No authentic V9 checked rows available')
    inventory,changes=[],[]
    for path in files:
        run=path.parents[2]
        manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        source=json.loads(path.read_text(encoding='utf-8'))
        section=path.parent.name
        prompt=ROOT/'runs'/manifest['parent_run_id']/'agents'/section/'prompt.txt'
        evidence=prompt.read_text(encoding='utf-8')
        evidence_sha=sha256(evidence)
        for row in source['claims']:
            if row['evidence_prompt_sha256']!=evidence_sha:raise ValueError('Saved V9 row evidence changed')
        before=copy.deepcopy(source)
        changed=module.apply_ev_ebitda_range_guard(source,evidence)
        if source!=before:raise ValueError('Probe mutated original in-memory result')
        assert len(source['claims'])==len(changed['claims'])
        row_changes=[]
        for old,new in zip(source['claims'],changed['claims']):
            if old['released_text']!=new['released_text']:
                # The paid review fields must survive the deterministic release change.
                for key in ('claim','revised_claim','classification','action','evidence','numeric_results','original_quote'):
                    if old[key]!=new[key]:raise ValueError('Paid-review field changed: '+key)
                row_changes.append({'claim_id':old['claim_id'],'old_release':old['released_text'],
                    'new_release':new['released_text'],'source_start':old['source_start'],'source_end':old['source_end'],
                    'clarification':new['ev_ebitda_range_check']})
        if bool(row_changes)!=(source['revised_text']!=changed['revised_text']):raise ValueError('Row/text change mismatch')
        record={'run_id':manifest['run_id'],'ticker':manifest['ticker'],'section':section,
            'checked_path':path.relative_to(ROOT).as_posix(),'checked_sha256':sha256(path.read_bytes()),
            'prompt_path':prompt.relative_to(ROOT).as_posix(),'prompt_sha256':evidence_sha,
            'row_count':len(source['claims']),'changed_rows':len(row_changes),
            'prior_text_sha256':sha256(source['revised_text']),'candidate_text_sha256':sha256(changed['revised_text'])}
        inventory.append(record)
        if row_changes:changes.append({**record,'changes':row_changes})
    output={'created_at_utc':utc_now(),'candidate_version':'9.1','parent_sha256':PARENT_SHA,
        'fragment_sha256':sha256(fragment.read_bytes()),'source_sha256':sha256(archive.read_bytes()),
        'saved_v9_files':len(files),'saved_v9_claim_rows':sum(x['row_count'] for x in inventory),
        'changed_rows':sum(x['changed_rows'] for x in inventory),'changed_sections':len(changes),
        'inventory':inventory,'changes':changes,'new_paid_calls':0,'upstream_written':False,
        'authentic_runs_written':False,'audit_labels_read':False,'held_out_evaluation':False,
        'scope':'Narrow additional release clarification within Option A. The rule was designed after inspecting a residual development failure. This probe is not an independent after audit or a perfect-score claim.',
        'trigger':'Only a complete isolated explicit-subject EV/EBITDA recent-years range sentence. Exact focal identity, actual columns and complete positive finite EV/EBITDA rows are required. Other predicates, unknown scope/metric, dates and malformed rows abstain.'}
    if args.probe_output.exists():raise ValueError('Probe destination already exists; supply a new --probe-output')
    write_json(args.probe_output,output)
    print(json.dumps({k:output[k] for k in ('saved_v9_files','saved_v9_claim_rows','changed_rows','changed_sections','new_paid_calls','upstream_written','authentic_runs_written')},indent=2))


if __name__=='__main__':main()
