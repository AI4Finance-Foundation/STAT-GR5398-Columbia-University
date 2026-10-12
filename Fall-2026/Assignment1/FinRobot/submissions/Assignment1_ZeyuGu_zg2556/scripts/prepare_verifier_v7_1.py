"""Prepare a narrowly derived transport repair; activate only after V7 exits."""
import argparse,json
from pathlib import Path
from audit_io import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');args=p.parse_args()
old=ROOT/'evidence/verifier_versions/verifier_agent_v7.py'
text=old.read_text(encoding='utf-8')
text=text.replace('"and", "additionally", "furthermore", "however", "among these",',
    '"and", "additionally", "furthermore", "however", "among these",\n    "in summary", "in conclusion", "overall",')
text=text.replace('7-semantic-period-peer-horizon-review','7.1-semantic-review-neutral-connector-repair')
archive=ROOT/'evidence/verifier_versions/verifier_agent_v7_1.py'
if archive.exists() and archive.read_text(encoding='utf-8')!=text:raise ValueError('Immutable V7.1 source differs')
if not archive.exists():archive.write_text(text,encoding='utf-8')
plan=json.loads((ROOT/'config/verifier_v7_plan.json').read_text(encoding='utf-8'))
for step in plan:
    old_id=step['id'];new_id=old_id.replace('_v7','_v7_1')
    step['id']=new_id;step['completion_manifest']=step['completion_manifest'].replace(old_id,new_id)
    step['argv']=[x.replace(old_id,new_id) for x in step['argv']]+['--reuse-review-run',f'runs/{old_id}']
write_json(ROOT/'config/verifier_v7_1_plan.json',plan)
write_json(ROOT/'evidence/verifier_versions/v7_1_development.json',{
    'version':'7.1-semantic-review-neutral-connector-repair','parent_source_sha256':sha256(old.read_bytes()),
    'source_sha256':sha256(archive.read_bytes()),'change':'Only add three finite discourse connectors: in summary, in conclusion, overall. No factual gaps become optional.',
    'generation_note':'Explicit reuse of wire-confirmed V7 responses where valid. Missing sections require fresh calls; reuse is not independent generation.',
    'semantic_instruction_change':False})
if args.activate:
    history=[json.loads(x) for x in (ROOT/'evidence/progress/verifier_v7_plan.jsonl').read_text().splitlines()]
    if not any(x['step']=='GOOGL_final_v7' and x['event']=='exit' for x in history):
        raise ValueError('Do not replace source until the whole V7 paid driver exits')
    (Path.home()/'.cache/finrobot-zg2556/upstream/finrobot_equity/core/src/modules/equity_agents/verifier_agent.py').write_bytes(archive.read_bytes())
print('V7.1 prepared; activated='+str(args.activate))
