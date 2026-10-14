"""Recompute cost/runtime and run inventory from immutable response records and ledger."""
import argparse
import csv
import json
from itertools import chain
from collections import defaultdict
from pathlib import Path
from audit_io import write_json, utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve()
    ledger=[json.loads(line) for line in (root/'evidence/costs.jsonl').read_text(encoding='utf-8').splitlines()]
    settled=[r for r in ledger if r['event']=='settle']
    ids=[r['call_id'] for r in settled]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate settlement')
    reservations={r['call_id']:r['reserved_usd'] for r in ledger if r['event']=='reserve'}
    unknown={k:v for k,v in reservations.items() if k not in ids}
    groups=defaultdict(list)
    for row in settled:
        groups[row['call_id'].split('/')[0]].append(row)
    result=[]
    for group, rows in sorted(groups.items()):
        result.append({'run_or_audit_id':group,'paid_call_count':len(rows),
            'input_tokens':sum(r['input_tokens'] for r in rows),'output_tokens':sum(r['output_tokens'] for r in rows),
            'cost_usd':sum(r['cost_usd'] for r in rows),'model_runtime_seconds':sum(r['runtime_seconds'] for r in rows)})
    output=root/'audits/resources'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'cost_runtime_by_run.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]) if result else ['run_or_audit_id'])
        w.writeheader();w.writerows(result)
    models=set()
    for path in chain((root/'runs').rglob('wire_response.json'),(root/'evidence').rglob('wire_response.json')):
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('model'): models.add(record['model'])
        except (UnicodeError,json.JSONDecodeError):
            continue
    policy=json.loads((root/'config/budget.approved.json').read_text(encoding='utf-8'))
    summary={'as_of_utc':utc_now(),'settled_call_count':len(settled),
        'cost_usd':sum(r['cost_usd'] for r in settled),'unsettled_reserved_usd':sum(unknown.values()),
        'unsettled_call_ids':list(unknown),'approved_cap_usd':policy['approved_cap_usd'],
        'input_tokens':sum(r['input_tokens'] for r in settled),'output_tokens':sum(r['output_tokens'] for r in settled),
        'model_runtime_seconds':sum(r['runtime_seconds'] for r in settled),'resolved_response_models':sorted(models),
        'cost_basis':'Recorded usage at configured list prices; cached input charged full price conservatively. Includes failed validation and repair attempts, preflight and evaluation. This is a reproducible upper estimate, not a provider invoice.',
        'runtime_basis':'Sum of paid model call elapsed seconds, including failed validation attempts; excludes local processing and human login delays.'}
    write_json(output/'cost_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
