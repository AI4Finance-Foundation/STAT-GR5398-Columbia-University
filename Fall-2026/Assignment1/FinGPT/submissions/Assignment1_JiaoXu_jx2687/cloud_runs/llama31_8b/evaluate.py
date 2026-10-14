"""Greedy base-vs-LoRA generation, raw outputs, strict coverage-aware metrics."""
import argparse,json,time,sys,statistics,os
from pathlib import Path
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import torch,pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer,AutoModelForCausalLM,set_seed
from peft import PeftModel
from utils import make_prompt,calc_metrics,parse_answer
ROOT=Path(__file__).resolve().parent

def main(a):
    set_seed(5398);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='cpu' and not a.allow_cpu:raise RuntimeError('No GPU visible; run with Apple GPU access or explicit --allow_cpu')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.eos_token
    data=load_from_disk(str(ROOT/'data/local_subset'))['test']
    dtype=torch.float16 if device!='cpu' else torch.float32
    results=[]
    for variant in ['base','lora']:
        path=ROOT/'results'/f'{variant}_answers.jsonl'
        from transformers import BitsAndBytesConfig
        model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,trust_remote_code=False,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True),device_map={'':0},attn_implementation='sdpa')
        if variant=='lora':model=PeftModel.from_pretrained(model,str(ROOT/'finetuned_models'/ROOT.name))
        model.eval();model.config.use_cache=True
        # Explicit warmup excluded from measured inference latency.
        with torch.inference_mode():model.generate(**tokenizer('Hello',return_tensors='pt').to(device),max_new_tokens=2,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        for i,r in enumerate(rows):
            expected,info=make_prompt(data[i]['prompt'],tokenizer,896)
            assert r['sample_index']==i and r['input_prompt']==expected and r['reference_answer']==data[i]['answer']
        print(f'{variant}: resuming from {len(rows)} saved cases',flush=True)
        with open(path,'a') as f:
            i=len(rows);batch_size=a.batch_size
            while i<len(data):
                size=min(batch_size-(i%batch_size),len(data)-i)
                prompts=[];infos=[]
                for j in range(i,i+size):
                    prompt,info=make_prompt(data[j]['prompt'],tokenizer,896)
                    prompts.append(prompt);infos.append(info)
                inputs=tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to(device)
                if device=='cuda':torch.cuda.synchronize()
                elif device=='mps':torch.mps.synchronize()
                t=time.perf_counter()
                try:
                    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
                    if device=='cuda':torch.cuda.synchronize()
                    elif device=='mps':torch.mps.synchronize()
                except torch.cuda.OutOfMemoryError:
                    del inputs
                    torch.cuda.empty_cache()
                    if batch_size==1:raise
                    batch_size=max(1,batch_size//2)
                    print('OOM fallback batch_size',batch_size,flush=True)
                    continue
                elapsed=time.perf_counter()-t
                for offset,j in enumerate(range(i,i+size)):
                    row=data[j];ids=out[offset,inputs['input_ids'].shape[1]:].tolist()
                    if tokenizer.eos_token_id in ids:ids=ids[:ids.index(tokenizer.eos_token_id)+1]
                    answer=tokenizer.decode(ids,skip_special_tokens=True)
                    record={'sample_index':j,'model':variant,'symbol':row['symbol'],'period':row['period'],'label':row['label'],'reference_answer':row['answer'],'input_prompt':prompts[offset],'generated_answer':answer,'elapsed_s':elapsed/size,'generation_batch_size':size,'batch_wall_s':elapsed,'batch_start_index':i,'latency_measurement':'amortized_batch' if size>1 else 'single_example','attention_implementation':'sdpa','generated_tokens':len(ids),'hit_token_limit':len(ids)>=a.max_new_tokens,'parse_valid':parse_answer(answer) is not None,**infos[offset]}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();rows.append(record)
                print(f'{variant}: saved {i+size}/{len(data)}; batch={size}; batch_wall={elapsed:.2f}s; valid={sum(r["parse_valid"] for r in rows)}',flush=True)
                i+=size
                del inputs,out
        metrics=calc_metrics([r['generated_answer'] for r in rows],[r['reference_answer'] for r in rows])
        metrics.update({'model':variant,'mean_inference_s':statistics.mean(r['elapsed_s'] for r in rows),'median_inference_s':statistics.median(r['elapsed_s'] for r in rows),'token_limit_count':sum(r['hit_token_limit'] for r in rows),'mean_generated_tokens':statistics.mean(r['generated_tokens'] for r in rows),'device':device})
        metrics['latency_note']='elapsed_s is single-example time for old records, amortized batch time for batch>1; do not compare mixed overall means as single-example latency'
        metrics['latency_strata']={str(b):{'n':sum(r.get('generation_batch_size',1)==b for r in rows),'mean_elapsed_s':statistics.mean(r['elapsed_s'] for r in rows if r.get('generation_batch_size',1)==b)} for b in sorted({r.get('generation_batch_size',1) for r in rows})}
        results.append(metrics)
        (ROOT/'results/metrics.json').write_text(json.dumps(results,indent=2))
        del model
        if device=='mps':torch.mps.empty_cache()
        elif device=='cuda':torch.cuda.empty_cache()
    flat=[]
    for row in results:
        r={k:v for k,v in row.items() if not isinstance(v,dict)}
        for k in ['pros_rouge_scores','cons_rouge_scores','anal_rouge_scores']:
            r.update({f'{k}_{n}':v for n,v in row[k].items()})
        flat.append(r)
    pd.DataFrame(flat).to_csv(ROOT/'results/metrics.csv',index=False)
    baseline=sum(r['label'].startswith('up') for r in rows)/len(rows)
    (ROOT/'results/baselines.json').write_text(json.dumps({'n':len(rows),'always_up_accuracy':baseline,'always_down_accuracy':1-baseline,'train_majority_direction':'up','note':'Fixed train-majority benchmark; test majority is descriptive only. No teacher API calls made.'},indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--max_new_tokens',type=int,default=512);p.add_argument('--batch_size',type=int,default=16);p.add_argument('--allow_cpu',action='store_true');main(p.parse_args())
