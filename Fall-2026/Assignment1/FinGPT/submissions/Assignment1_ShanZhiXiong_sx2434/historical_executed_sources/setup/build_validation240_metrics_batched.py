"""Completed validation240 runs only; CPU summaries, no GPU use or test access."""
import argparse,collections,hashlib,json,pathlib,statistics,sys
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen','deepseek','llama'],required=True);args=ap.parse_args()
R=pathlib.Path('/root/shared-nvme/AI4FinRuntime');P=R/'import_20261009/assignment1_20261008';sys.path.insert(0,str(P))
from metrics import parse_final_answer,direction,course
registry=json.loads((R/'setup/three_model_runtime_registry_20261010.json').read_text())
O=pathlib.Path(registry[args.model].get('evaluation_run',str(P/f'runs/server4090_{args.model}_int8_validation240_gen1920_20261010')))
assert json.loads((O/'status.json').read_text())['stage']=='completed','Do not summarize incomplete run as final results'
C=json.loads((O/'config.json').read_text());assert C['split']=='validation' and C['samples']==240 and C['max_new_tokens']==1920 and C['max_length']==6144
registry=json.loads((R/'setup/three_model_runtime_registry_20261010.json').read_text());data=pathlib.Path(registry[args.model]['evaluation_data']);evidence=json.loads(data.read_text())
assert C['canonical_evidence_sha256']==hashlib.sha256(data.read_bytes()).hexdigest()
rows={k:[json.loads(line) for line in (O/f'{k}_outputs.jsonl').read_text().splitlines() if line.strip()] for k in ['base','adapter']}
for variant,records in rows.items():
    assert len(records)==240 and len({r['sample_id'] for r in records})==240
    assert [r['sample_id'] for r in records]==[r['sample_id'] for r in evidence]
    for record,source in zip(records,evidence):
        for field in ['sample_id','symbol','period','label','reference']:assert record[field]==source[field]
        assert record['prompt_tokens']==len(source['input_ids']) and record['generated_tokens']<=1920
        assert not record.get('previously_inspected_test',False)
parsed={k:[parse_final_answer(r['generated']) for r in records] for k,records in rows.items()}
refs=[parse_final_answer(r['reference']) for r in evidence];assert all(r['format_complete'] for r in refs)
targets=[direction(r['label']) for r in evidence];assert all(d in [-1,1] for d in targets)
matched=[i for i in range(240) if all(parsed[k][i]['format_complete'] for k in parsed)]
reuse=json.loads((O/'reuse_manifest.json').read_text()) if (O/'reuse_manifest.json').exists() else {}
count=reuse.get('reused_per_kind',0);assert count in [0,30]
if count:
    for kind in rows:
        original=pathlib.Path(reuse['source'])/f'{kind}_outputs.jsonl'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==reuse['sha256'][kind]
        old=[json.loads(line) for line in original.read_text().splitlines()]
        for oldrow,newrow in zip(old,rows[kind][:count]):
            for field in ['sample_id','generated','seconds','prompt_tokens','generated_tokens']:assert oldrow[field]==newrow[field]
labels=collections.Counter(targets)
report={'model':args.model,'split':'validation','validation_only':True,'test_used':False,'full_denominator':240,
    'generation_max_new_tokens':1920,'inference_context_tokens':6144,'training_context_tokens':4096,
    'matched_structure_count':len(matched),'label_counts':{str(k):v for k,v in labels.items()},
    'always_up_accuracy':labels[1]/240,'always_down_accuracy':labels[-1]/240,
    'reused_per_variant':count,'new_per_variant':240-count,'generation_batch_size':C.get('generation_batch_size',1),'timing_semantics':C.get('timing_semantics','single-sample latency'),
    'policy':'Original course metrics are preserved in comparison.json. Normalized final-answer headings are evaluated over all240 with missing sections scoring zero; matched-structure ROUGE uses the same intersection for base/adapter. Validation was used for checkpoint selection, so these are development results.'}
correct={}
for kind,ps in parsed.items():
    records=rows[kind];correct[kind]=[p['direction']==target for p,target in zip(ps,targets)]
    recall={str(d):{'correct':sum(correct[kind][i] for i,t in enumerate(targets) if t==d),'total':labels[d]} for d in [-1,1]}
    item={'samples':240,'direction_correct':sum(correct[kind]),'direction_accuracy_all240':sum(correct[kind])/240,
        'direction_counts':dict(collections.Counter(str(p['direction']) for p in ps)),
        'direction_unparsed':sum(p['direction'] not in [-1,1] for p in ps),
        'structure_complete':sum(p['format_complete'] for p in ps),'structure_complete_rate':sum(p['format_complete'] for p in ps)/240,
        'nonempty_outputs':sum(bool(r['generated'].strip()) for r in records),'eos_count':sum(r['ended_eos'] for r in records),
        'length_limit_count':sum(r['hit_generation_limit'] for r in records),'recall_counts':recall,
        'balanced_accuracy':statistics.mean(v['correct']/v['total'] for v in recall.values() if v['total']),
        'peak_allocated_gib':max(r['peak_allocated_gib'] for r in records),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in records),
        'seconds_sum_all240':sum(r['seconds'] for r in records),'seconds_mean_all240':statistics.mean(r['seconds'] for r in records),
        'seconds_median_all240':statistics.median(r['seconds'] for r in records),
        'seconds_sum_new_only':sum(r['seconds'] for r in records[count:]),
        'seconds_mean_new_only':statistics.mean(r['seconds'] for r in records[count:])}
    for label,key in [('pros','positive_developments'),('cons','potential_concerns'),('analysis','analysis')]:
        item[label+'_rouge_all240']=course.calc_rouge_score([r[key] for r in refs],[p.get(key,'') for p in ps])
        item[label+'_rouge_matched']=course.calc_rouge_score([refs[i][key] for i in matched],[ps[i][key] for i in matched]) if matched else None
    report[kind]=item
report['paired_direction_change_counts']={f'{a}->{b}':sum(correct['base'][i]==a and correct['adapter'][i]==b for i in range(240)) for a in [False,True] for b in [False,True]}
def save_once(path,value):
    if path.exists():assert json.loads(path.read_text())==value,'Previous metric results preserved'
    else:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
save_once(O/'supplemental_paired_metrics240.json',report)
verification={'same_240_ids_references_labels':True,'prepared_inputs_match_actual_lengths':True,'reused_outputs_verified':bool(count),
    'input_evidence_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
    'sha256':{name:hashlib.sha256((O/name).read_bytes()).hexdigest() for name in ['config.json','comparison.json','base_outputs.jsonl','adapter_outputs.jsonl','supplemental_paired_metrics240.json']},
    'test_used':False,'manual_quality_review_pending':True}
save_once(O/'verification240.json',verification)
print(json.dumps(report,ensure_ascii=False,allow_nan=False))
