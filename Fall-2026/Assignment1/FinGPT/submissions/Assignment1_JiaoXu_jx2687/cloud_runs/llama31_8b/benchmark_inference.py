"""Fixed six-case single-request latency benchmark, separate from batched scoring."""
from pathlib import Path
import os,json,time,statistics,argparse
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
from peft import PeftModel
from utils import make_prompt,parse_answer
ROOT=Path(__file__).resolve().parent
INDICES=[0,9,70,149,230,299]

def main(a):
    assert torch.cuda.is_available();set_seed(5398);torch.set_num_threads(4)
    data=load_from_disk(str(ROOT/'data/local_subset'))['test'];assert len(data)==300
    tokenizer=AutoTokenizer.from_pretrained(a.model);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,trust_remote_code=False,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True),device_map={'':0},attn_implementation='sdpa')
    output=ROOT/'results/single_request_benchmark.jsonl';summary=[]
    with output.open('w') as f:
        for variant in ['base','lora']:
            if variant=='lora':model=PeftModel.from_pretrained(model,str(ROOT/'finetuned_models'/ROOT.name))
            model.eval();model.config.use_cache=True
            with torch.inference_mode():model.generate(**tokenizer('Hello',return_tensors='pt').to('cuda'),max_new_tokens=2,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            rows=[]
            for i in INDICES:
                r=data[i];prompt,info=make_prompt(r['prompt'],tokenizer,896)
                inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                ids=out[0,inputs['input_ids'].shape[1]:];answer=tokenizer.decode(ids,skip_special_tokens=True)
                record={'model':variant,'sample_index':i,'symbol':r['symbol'],'period':r['period'],'input_prompt':prompt,'reference_answer':r['answer'],'generated_answer':answer,'elapsed_s':elapsed,'generated_tokens':len(ids),'hit_token_limit':len(ids)>=512,'parse_valid':parse_answer(answer) is not None,'batch_size':1,'attention_implementation':'sdpa',**info}
                f.write(json.dumps(record)+'\n');f.flush();rows.append(record)
                print(f'benchmark {variant} index{i}: {elapsed:.2f}s tokens{len(ids)}',flush=True)
            summary.append({'variant':variant,'n':len(rows),'sample_indices':INDICES,'mean_single_request_s':statistics.mean(x['elapsed_s'] for x in rows),'median_single_request_s':statistics.median(x['elapsed_s'] for x in rows),'mean_generated_tokens':statistics.mean(x['generated_tokens'] for x in rows),'scope':'Six predeclared cases; synchronized prefill+decode, batch1SDPA, loading/warmup excluded; observational latency, no confidence claim'})
    (ROOT/'results/single_request_benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);main(p.parse_args())
