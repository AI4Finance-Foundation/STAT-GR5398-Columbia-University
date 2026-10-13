"""Recompute full-test metrics and distinguish single-example from batch latency."""
from pathlib import Path
import argparse,json,hashlib,statistics,sys
import numpy as np
import pandas as pd
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import calc_metrics,parse_answer

def main(a):
    root=Path(a.run_dir).resolve();course=Path(__file__).resolve().parents[1]
    test=pd.read_parquet(course/'data/raw/test.parquet')
    test['start']=test.period.str[:10];test=test.sort_values(['symbol','start']).reset_index(drop=True)
    training=json.loads((root/'results/training.json').read_text())
    assert training['changed_adapter_tensors']>0
    assert training['frozen_parameter_hash_before']==training['frozen_parameter_hash_after']
    assert np.isfinite(training['training_metrics']['train_loss']) and np.isfinite(training['validation_metrics']['eval_loss'])
    weight_files=list((root/'finetuned_models').glob('*/adapter_model.safetensors'));assert len(weight_files)==1
    with safe_open(weight_files[0],framework='pt',device='cpu') as f:
        keys=list(f.keys());assert all('lora_' in k for k in keys)
        assert all(f.get_tensor(k).isfinite().all() for k in keys)
        trainable=sum(f.get_tensor(k).numel() for k in keys)
    paired={};checks={}
    saved=json.loads((root/'results/metrics.json').read_text()) if (root/'results/metrics.json').exists() else []
    for variant in ['base','lora']:
        p=root/'results'/f'{variant}_answers.jsonl'
        rows=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        assert len(rows)<=len(test) and len({(r['symbol'],r['period']) for r in rows})==len(rows)
        for i,r in enumerate(rows):
            original=test.iloc[i]
            assert r['sample_index']==i and r['symbol']==original.symbol and r['period']==original.period
            assert r['label']==original.label and r['reference_answer']==original.answer
            assert r['parse_valid']==(parse_answer(r['generated_answer']) is not None)
            assert r['elapsed_s']>0 and np.isfinite(r['elapsed_s'])
            if r.get('generation_batch_size',1)>1:
                assert np.isclose(r['elapsed_s']*r['generation_batch_size'],r['batch_wall_s'])
        if not a.allow_partial:assert len(rows)==300,f'{variant}: incomplete {len(rows)}/300'
        metrics=calc_metrics([r['generated_answer'] for r in rows],[r['reference_answer'] for r in rows]) if rows else None
        if len(rows)==300:
            entry=next(m for m in saved if m['model']==variant)
            for k,v in metrics.items():assert entry[k]==v,(variant,k)
            valid=[(parse_answer(r['generated_answer']),parse_answer(r['reference_answer'])) for r in rows if parse_answer(r['generated_answer']) is not None and parse_answer(r['reference_answer']) is not None]
            if valid:
                assert np.isclose(metrics['bin_acc'],np.mean([x['prediction_binary']==y['prediction_binary'] for x,y in valid]))
                assert np.isclose(metrics['mse'],np.mean([(x['prediction']-y['prediction'])**2 for x,y in valid]))
        checks[variant]={'n':len(rows),'complete':len(rows)==300,'reference_identity_verified':True,'metrics_recomputed':len(rows)==300,'valid_count':None if not metrics else metrics['valid_count']}
        paired[variant]=rows
    # Compare latency only across identical batch composition/attention, never mixed means.
    b={r['sample_index']:r for r in paired['base']};l={r['sample_index']:r for r in paired['lora']}
    matched=[i for i in b.keys()&l.keys() if b[i].get('generation_batch_size',1)==l[i].get('generation_batch_size',1) and b[i].get('batch_start_index',i)==l[i].get('batch_start_index',i) and b[i].get('attention_implementation','eager')==l[i].get('attention_implementation','eager')]
    timing={'matched_case_count':len(matched),'interpretation':'batch wall time divided by batch size is amortized throughput cost, not single-request latency'}
    if matched:
        timing.update({v:statistics.mean(paired[v][i]['elapsed_s'] for i in matched) for v in ['base','lora']})
    result={'run':root.name,'adapter_tensor_count':len(keys),'trainable_adapter_elements':trainable,'adapter_values_finite':True,'adapter_sha256':hashlib.sha256(weight_files[0].read_bytes()).hexdigest(),'frozen_hash_check':'first4096-element sample only; requires_grad audited during training','test_checks':checks,'comparable_latency':timing,'complete':all(x['complete'] for x in checks.values())}
    (root/'results/full_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--allow_partial',action='store_true');main(p.parse_args())
