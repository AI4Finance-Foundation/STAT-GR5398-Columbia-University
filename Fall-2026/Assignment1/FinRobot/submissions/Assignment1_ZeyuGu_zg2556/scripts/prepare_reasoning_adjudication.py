"""Carry independent labels only when identical text/evidence has no label conflict."""
import argparse,copy,json
from pathlib import Path
from grounding_audit import load_inputs,source_units
from prepare_after_adjudication import normalized,BEFORE
from prepare_v7_adjudication import AFTER
from audit_io import sha256,write_json
ROOT=Path(__file__).resolve().parents[1]

def prepare(ticker,version='reasoning_v8_2'):
    baseline=ROOT/'runs'/f'{ticker}_baseline_1';target=ROOT/'runs'/f'{ticker}_final_{version}'
    manifest,inputs=load_inputs(target,baseline);known={s:{} for s in inputs}
    for prior_version,folder in [('baseline_1',BEFORE[ticker]),('final_v6',AFTER[ticker]),
                           ('final_v7_1',f'{ticker}_after_v7_1_adjudicated')]:
        audit=ROOT/'audits/grounding'/folder;spec=json.loads((audit/'reviewed_specification.json').read_text(encoding='utf-8'))
        _,old=load_inputs(ROOT/'runs'/f'{ticker}_{prior_version}',baseline)
        for section,item in old.items():
            assert item['metadata']['evidence_prompt_sha256']==inputs[section]['metadata']['evidence_prompt_sha256']
            units={u['unit_id']:u for u in spec['sections'][section]['units']}
            for span in source_units(item['text']):
                unit=units[span['unit_id']]
                semantic=copy.deepcopy(unit);semantic.pop('unit_id')
                for atom in semantic.get('atoms',[]):atom.pop('fact_key',None)
                key=normalized(span['text'])
                known[section].setdefault(key,[]).append((unit,folder,span,json.dumps(semantic,sort_keys=True),'source_unit'))
                for atom in unit.get('atoms',[]):
                    # A complete new assertion can exactly equal a previously
                    # independently adjudicated atomic assertion. Carry only
                    # that atom, never its neighboring facts or verifier label.
                    atomic_unit={'unit_id':unit['unit_id'],'atoms':[copy.deepcopy(atom)],'nonfactual_reason':''}
                    atomic_semantic=copy.deepcopy(atomic_unit);atomic_semantic.pop('unit_id')
                    atomic_semantic['atoms'][0].pop('fact_key',None)
                    atom_span=dict(span,text=atom['claim'])
                    known[section].setdefault(normalized(atom['claim']),[]).append((atomic_unit,folder,atom_span,json.dumps(atomic_semantic,sort_keys=True),'adjudicated_atom'))
    spec={'protocol_version':'machine-adjudication-v1','run_id':manifest['run_id'],
          'evidence_run_id':baseline.name,'raw_judge_audit':None,'sections':{}}
    todo=[];mapping=[]
    for section,item in inputs.items():
        target_section={'source_sha256':item['metadata']['source_sha256'],
                        'evidence_sha256':item['metadata']['evidence_prompt_sha256'],'units':[]}
        for span in source_units(item['text']):
            candidates=known[section].get(normalized(span['text']),[])
            if candidates and len({x[3] for x in candidates})==1:
                unit,folder,old,_,match_kind=candidates[0];record=copy.deepcopy(unit);record['unit_id']=span['unit_id']
                target_section['units'].append(record)
                mapping.append({'section':section,'target_unit_id':span['unit_id'],'source_audit':folder,
                    'source_unit_id':old['unit_id'],'source_text_sha256':sha256(old['text']),
                    'target_text_sha256':sha256(span['text']),'evidence_sha256':item['metadata']['evidence_prompt_sha256'],
                    'match_kind':match_kind,
                    'basis':'Complete new source assertion equals either a prior complete source unit or a prior independently adjudicated atomic assertion after warning/bullet, whitespace and apostrophe normalization; exact writer evidence hash; no conflicting preserved independent labels. No verifier labels read.'})
            else:
                todo.append({'section':section,**span,'reason':'conflicting_prior_labels' if candidates else 'new_text'})
        spec['sections'][section]=target_section
    out=ROOT/'audits/grounding'/f'{ticker}_after_{version}_working';out.mkdir(exist_ok=True)
    if (out/'reviewed_specification.json').exists():raise ValueError('Do not replace a completed explicit review')
    write_json(out/'partial_specification.json',spec);write_json(out/'exact_source_carry_forward.json',mapping)
    write_json(out/'requires_explicit_review.json',todo)
    print(ticker,'independent units carried',len(mapping),'units needing explicit review',len(todo))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tickers',nargs='+',choices=list(BEFORE));p.add_argument('--version',default='reasoning_v8_2');a=p.parse_args()
    for ticker in a.tickers:prepare(ticker,a.version)
