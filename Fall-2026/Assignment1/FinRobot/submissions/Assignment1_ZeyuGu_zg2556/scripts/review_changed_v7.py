"""Independent changed-unit proposals; exact unchanged-unit labels remain traceable.

No verifier verdicts, citations or checked.json are inputs. This is a machine
proposal stage, not human review or an automatic promotion to canonical results.
"""
import argparse,asyncio,copy,json,re
from pathlib import Path
from audit_io import Budget,call_agent,credentials,sha256,write_json
from grounding_audit import build_agent,load_inputs,source_units,validate_audit,INSTRUCTIONS
from experiment import contract,validate_success
ROOT=Path(__file__).resolve().parents[1]

async def main(args):
    t=args.ticker;directory=ROOT/'audits/grounding'/f'{t}_after_{args.version}_working'
    spec=json.loads((directory/'partial_specification.json').read_text(encoding='utf-8'))
    _,inputs=load_inputs(ROOT/'runs'/f'{t}_final_{args.version}',ROOT/'runs'/f'{t}_baseline_1')
    agent=build_agent();cfg=credentials(args.config)
    budget=Budget(ROOT/'config/budget.evaluation.json',ROOT/'evidence/costs.jsonl')
    cfg['openai_model']='gpt-4.1'
    proposals={}
    suffix='_blind' if args.blind_warnings else ''
    try:
        for section,item in inputs.items():
            carried=spec['sections'][section]['units'];ids={x['unit_id'] for x in carried}
            todo=[u for u in source_units(item['text']) if u['unit_id'] not in ids]
            if not todo:continue
            shown=copy.deepcopy(todo)
            if args.blind_warnings:
                for unit in shown:
                    unit['text']=re.sub(r'^\s*\[(?:UNVERIFIED|NONFACTUAL COMMENTARY)[^\]]*\]\s*','',unit['text'])
                write_json(directory/'blinding'/f'{section}.json',[
                    {'unit_id':raw['unit_id'],'source_start':raw['start'],'source_end':raw['end'],
                     'original_text':raw['text'],'judge_text':new['text'],
                     'transformation':'Remove only leading machine-warning prefix; all underlying proposition words remain unchanged. Labels are judged on the assertion, not the warning.'}
                    for raw,new in zip(todo,shown)])
            prompt='ORIGINAL WRITER EVIDENCE (numbered lines):\n'+'\n'.join(f'{i}: {line}' for i,line in enumerate(item['evidence'].splitlines(),1))
            prompt+='\n\nSOURCE UNITS (immutable unit_id and source offsets; assertion text):\n'+json.dumps(shown,ensure_ascii=False)
            prompt+='\nUnit IDs are intentionally noncontiguous. Return every supplied ID exactly once. Other units have exactly unchanged text and independent saved labels; their labels and all verifier judgments are withheld from you.'
            diagnostic=''
            for attempt_number in range(1,4):
                location=directory/('changed_unit_calls'+suffix)/section/f'attempt_{attempt_number:03d}'
                actual=prompt+ ('\n\nSCHEMA/COVERAGE REPAIR: '+diagnostic if diagnostic else '')
                expected=contract(agent,actual,cfg,0,16000)
                if (location/'response.json').exists():
                    output=validate_success(location,agent,expected,budget)
                else:
                    if location.exists():raise ValueError('Unresolved prior attempt; inspect before another paid call.')
                    output=await call_agent(agent,actual,config=cfg,budget=budget,directory=location,
                        call_id=f'{t}_{args.version}_changed_audit{suffix}/{section}/{attempt_number}',temperature=0,max_tokens=16000)
                    output=validate_success(location,agent,expected,budget)
                data=output.model_dump()
                try:
                    got=[u['unit_id'] for u in data['units']]
                    if len(got)!=len(set(got)) or set(got)!={u['unit_id'] for u in todo}:
                        raise ValueError('Return exactly the supplied changed-unit IDs, without duplicates.')
                    if args.blind_warnings:
                        for unit in data['units']:
                            if not unit['claims'] and re.search(r'annotation|unverified|unsubstantiated',unit['nonfactual_reason'],re.I):
                                raise ValueError('A warning is not a reason to omit underlying factual assertions. Re-evaluate every supplied assertion; no source contains only a warning.')
                    combined=[{'unit_id':u['unit_id'],'claims':u['atoms'],'nonfactual_reason':u['nonfactual_reason']} for u in carried]+data['units']
                    # Reviewed specs can omit lexical fact keys; materialize uses
                    # this same disclosed lexical fallback, never inferred identity.
                    for u in combined:
                        for atom in u['claims']:
                            if not atom.get('fact_key'):atom['fact_key']=t.lower()+'|'+atom['claim'].lower()
                    validate_audit({'units':combined},item['text'],item['evidence'],item['metadata'])
                except ValueError as exc:
                    diagnostic=str(exc)
                    write_json(location/'validation_error.json',{'error':diagnostic})
                    if attempt_number==3:raise
                    continue
                proposals[section]={'selected_attempt':str(location.relative_to(ROOT)),
                    'response_sha256':sha256((location/'response.json').read_bytes()),
                    'units':data['units']}
                break
        candidate=copy.deepcopy(spec)
        for section,proposal in proposals.items():
            candidate['sections'][section]['units'] += [{'unit_id':u['unit_id'],'atoms':u['claims'],
                'nonfactual_reason':u['nonfactual_reason']} for u in proposal['units']]
            candidate['sections'][section]['units'].sort(key=lambda u:u['unit_id'])
        write_json(directory/('changed_unit_proposals'+suffix+'.json'),proposals)
        write_json(directory/('candidate_specification'+suffix+'.json'),candidate)
        write_json(directory/('proposal_status'+suffix+'.json'),{'state':'complete','ticker':t,
            'human_review_performed':False,'not_canonical_until_explicit_review':True,
            'instructions_sha256':sha256(INSTRUCTIONS),'changed_units':sum(len(x['units']) for x in proposals.values()),'warning_prefixes_blinded':args.blind_warnings,
            'source':'Independent GPT-4.1 only receives original evidence and changed final source units; exact known sentences retain independent earlier labels.'})
        print(t,'changed-unit independent proposals complete')
    finally:budget.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ticker',choices=['NVDA','AMD','INTC']);p.add_argument('--config',required=True);p.add_argument('--version',default='v7');p.add_argument('--blind-warnings',action='store_true')
    asyncio.run(main(p.parse_args()))
