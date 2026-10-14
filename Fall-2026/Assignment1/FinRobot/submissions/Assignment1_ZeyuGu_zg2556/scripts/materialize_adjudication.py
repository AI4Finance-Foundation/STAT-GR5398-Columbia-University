"""Validate and materialize an explicitly reviewed machine-adjudication specification.

This program does not invent labels or call any model. Every source unit and
atomic judgment must already be present in the reviewed JSON specification.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
from grounding_audit import FIELDS,load_inputs,source_units,validate_audit,metrics,csv_write,read_json
from audit_io import sha256,write_json,utc_now


def materialize(args):
    spec=read_json(args.spec)
    run,inputs=load_inputs(args.run,args.evidence_run)
    if spec['run_id']!=run['run_id'] or spec['evidence_run_id']!=next(iter(inputs.values()))['metadata']['evidence_run_id']:
        raise ValueError('Reviewed specification references a different run')
    if set(spec['sections'])!=set(FIELDS):
        raise ValueError('All eight reviewed sections are required')
    protocol=Path(__file__).resolve().parents[1]/'docs/machine_adjudication_protocol.md'
    protocol_sha=sha256(protocol.read_bytes())
    rows,coverage,summaries,mappings=[],[],[],[]
    proposal_rows=[]
    if spec.get('raw_judge_audit'):
        proposal_path=Path(spec['raw_judge_audit'])/'claims.csv'
        if proposal_path.exists():
            with proposal_path.open(encoding='utf-8',newline='') as f:proposal_rows=list(csv.DictReader(f))
    source_hashes,evidence_hashes={},{}
    fallback_fact_keys=0
    for section,item in inputs.items():
        review=spec['sections'][section]
        source_hashes[section]=item['metadata']['source_sha256']
        evidence_hashes[section]=item['metadata']['evidence_prompt_sha256']
        if review['source_sha256']!=source_hashes[section] or review['evidence_sha256']!=evidence_hashes[section]:
            raise ValueError('Source/evidence changed since explicit review: '+section)
        units=[]
        for reviewed in review['units']:
            atoms=[]
            for atom in reviewed['atoms']:
                if not atom.get('fact_key'):
                    # An explicitly disclosed lexical identity, not inferred semantic equivalence.
                    atom=dict(atom,fact_key=run['ticker'].lower()+'|'+re.sub(r'\s+',' ',atom['claim'].lower()).strip())
                    fallback_fact_keys+=1
                atoms.append(atom)
            units.append({'unit_id':reviewed['unit_id'],'claims':atoms,'nonfactual_reason':reviewed['nonfactual_reason']})
        section_rows,section_coverage=validate_audit({'units':units},item['text'],item['evidence'],item['metadata'])
        for row in section_rows:
            row['claim_id']='adjudicated.'+row['claim_id']
            row['evaluation']='codex_assisted_explicit_adjudication_not_human_validated'
            row['reviewer']='Codex task agent; exact underlying model identifier not exported'
            overlaps=[x for x in proposal_rows if x['section']==section and int(x['source_start'])<row['source_end'] and int(x['source_end'])>row['source_start']]
            mappings.append({'claim_id':row['claim_id'],'ticker':run['ticker'],'section':section,
                'raw_judge_claim_ids':json.dumps([x['claim_id'] for x in overlaps]),
                'raw_judge_classifications':json.dumps([x['classification'] for x in overlaps]),
                'adjudicated_classification':row['classification'],'adjudicated_claim':row['claim'],
                'mapping_basis':'Source-unit span overlap; not one-to-one semantic agreement.'})
        rows+=section_rows;coverage+=section_coverage
        summaries.append({'ticker':run['ticker'],'run_id':run['run_id'],'section':section,**metrics(section_rows)})
    summaries.append({'ticker':run['ticker'],'run_id':run['run_id'],'section':'ALL',**metrics(rows)})
    args.output.mkdir(parents=True,exist_ok=False)
    csv_write(args.output/'adjudicated_claims.csv',rows)
    # The identical compatibility name lets offline grounding summarize/compare consume this dataset.
    (args.output/'claims.csv').write_bytes((args.output/'adjudicated_claims.csv').read_bytes())
    csv_write(args.output/'text_coverage.csv',coverage)
    csv_write(args.output/'grounding_summary.csv',summaries)
    csv_write(args.output/'raw_judge_mapping.csv',mappings)
    (args.output/'reviewed_specification.json').write_bytes(args.spec.read_bytes())
    (args.output/'adjudication_instructions.md').write_bytes(protocol.read_bytes())
    write_json(args.output/'manifest.json',{'state':'complete','auditor_version':'machine-adjudication-v1',
        'ticker':run['ticker'],'run_id':run['run_id'],'source_run':str(args.run.resolve()),
        'evidence_run':str((args.evidence_run or args.run).resolve()),'sections':list(FIELDS),
        'completed_at_utc':utc_now(),'source_hashes':source_hashes,'evidence_hashes':evidence_hashes,
        'instructions_sha256':protocol_sha,'auditor_model':'Codex-assisted source adjudication; model identifier not exported',
        'human_review_performed':False,'claim_csv_sha256':sha256((args.output/'claims.csv').read_bytes()),
        'text_coverage_sha256':sha256((args.output/'text_coverage.csv').read_bytes()),
        'reviewed_specification_sha256':sha256(args.spec.read_bytes()),'raw_judge_audit':spec.get('raw_judge_audit'),
        'review_scope':'Every deterministic source unit and explicitly decomposed factual premise; no sampling.',
        'adjudication_tokens':None,'adjudication_cost_usd':None,'adjudication_runtime_seconds':None,
        'cost_limitation':'Codex task-agent authoring is not instrumented by the direct API ledger; unavailable does not mean zero. Raw API judge and verifier usage remains in the shared ledger.',
        'lexical_fallback_fact_key_count':fallback_fact_keys,
        'retention_limitation':'Without explicit fact_key, retention uses lowercased claim text; this lexical proxy can miss paraphrases. Inspect evidence-span retention and original claim mappings.',
        'resume_fingerprint':{'source_hashes':source_hashes,'evidence_hashes':evidence_hashes,
            'provider':'Codex task-agent session','schema_sha256':sha256('explicit_reviewed_units_and_atomic_claims_v1'),
            'temperature':'not_exported','max_tokens':'not_exported'},
        'limitations':'Independent machine judgment can still miss implicit facts or misclassify evidence. This is not human validation. Measurement protocol was revised after raw-judge pilot failures.'})
    print(json.dumps({'ticker':run['ticker'],'units':len(coverage),'claims':len(rows),**metrics(rows)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--evidence-run',type=Path)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    materialize(p.parse_args())


if __name__=='__main__':main()
