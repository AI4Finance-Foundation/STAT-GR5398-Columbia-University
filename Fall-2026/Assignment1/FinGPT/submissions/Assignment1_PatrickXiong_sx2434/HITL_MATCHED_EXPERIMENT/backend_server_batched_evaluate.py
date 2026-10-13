"""Isolated, full-input batched validation evaluation. One GPU job at a time."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
R=Path('/root/shared-nvme/AI4FinRuntime'); S=R/'setup'; P=R/'import_20261009/assignment1_20261008'
sys.path.insert(0,str(P))
from data_processing import SYSTEM
from metrics import summarize, baselines
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def status(out,stage,**kw):
    v=dict(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw);save(out/'status.json',v);print(json.dumps(v),flush=True)
class Engine:
    def __init__(self,key):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
        self.torch=torch;set_seed(42);self.info=read(S/'three_model_runtime_registry_20261010.json')[key]
        self.rows=read(self.info['evaluation_data']);self.selected=read(Path(self.info['training_run'])/'selected_checkpoint.json');self.checkpoint=Path(self.selected['checkpoint'])
        self.tok=AutoTokenizer.from_pretrained(self.info['cache'],local_files_only=True,padding_side='left');self.dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tok.pad_token=self.tok.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(self.info['cache'],local_files_only=True,quantization_config=BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0),device_map={'':0},torch_dtype=self.dtype,low_cpu_mem_usage=True,attn_implementation='sdpa')
        self.model.eval();self.model.config.use_cache=True
        self.guard=min(21.,torch.cuda.get_device_properties(0).total_memory/2**30*.9)
    def adapter(self):
        from peft import PeftModel
        self.model=PeftModel.from_pretrained(self.model,str(self.checkpoint),is_trainable=False);self.model.eval()
    def ids(self,row):
        text=row['prompt_audit']['compressed_text']
        if self.tok.chat_template: ids=self.tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
        else: ids=self.tok.encode(SYSTEM+'\n\n'+text+'\n\n',add_special_tokens=True)
        assert ids==row['input_ids'],'actual input differs from frozen evidence'
        return ids
    def generate(self,rows,budget,callback=None):
        from transformers import StoppingCriteria,StoppingCriteriaList
        torch=self.torch;ids=[self.ids(r) for r in rows];width=max(map(len,ids));assert width+budget<=6144
        x=torch.tensor([[self.tok.pad_token_id]*(width-len(v))+v for v in ids],device='cuda')
        mask=torch.tensor([[0]*(width-len(v))+[1]*len(v) for v in ids],device='cuda')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        class Heartbeat(StoppingCriteria):
            last=start
            def __call__(self,input_ids,scores,**kw):
                now=time.monotonic()
                if callback and now-self.last>=5:
                    self.last=now;callback(input_ids.shape[-1]-width,now-start)
                return False
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=self.dtype):
            y=self.model.generate(input_ids=x,attention_mask=mask,max_new_tokens=budget,do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,pad_token_id=self.tok.eos_token_id,eos_token_id=self.tok.eos_token_id,stopping_criteria=StoppingCriteriaList([Heartbeat()]))
        torch.cuda.synchronize();elapsed=time.monotonic()-start;tails=y[:,width:].tolist();allocated=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30;results=[]
        for row,inp,tail in zip(rows,ids,tails):
            eos=self.tok.eos_token_id in tail
            if eos:tail=tail[:tail.index(self.tok.eos_token_id)+1]
            results.append(dict(sample_id=row['sample_id'],symbol=row['symbol'],period=row['period'],label=row['label'],reference=row['reference'],generated=self.tok.decode(tail,skip_special_tokens=True),prompt_tokens=len(inp),generated_tokens=len(tail),ended_eos=eos,hit_generation_limit=len(tail)>=budget and not eos,seconds=elapsed/len(rows),generation_latency_seconds=elapsed,batch_seconds=elapsed,batch_size=len(rows),timing_semantics='seconds is amortized batch time; generation_latency_seconds is whole batch latency',peak_allocated_gib=allocated,peak_reserved_gib=reserved,previously_inspected_test=False,generated_token_sha256=hashlib.sha256(json.dumps(tail).encode()).hexdigest(),generated_token_ids=tail))
        del x,mask,y
        assert allocated<=self.guard,f'allocated guard exceeded: {allocated}'
        return results
def evaluate(key,out,batch_size,budget=1920):
    status(out,'loading_cached_model',batch_size=batch_size)
    e=Engine(key);rows=e.rows;assert len(rows)==240;config=dict(model_id=e.info['model_id'],split='validation',samples=240,do_sample=False,max_new_tokens=budget,canonical_evidence_sha256=hashlib.sha256(Path(e.info['evaluation_data']).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((e.checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),checkpoint=str(e.checkpoint),prompt_policy='full frozen240 input; left padding with attention mask; no truncation',quantization_bits=8,max_length=6144,test_used=False,generation_batch_size=batch_size,timing_semantics='amortized batch seconds per answer; latency separately recorded',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:save(out/'config.json',config)
    all_rows={};all_metrics={}
    for kind in ['base','adapter']:
        if kind=='adapter':e.adapter()
        f=out/f'{kind}_outputs.jsonl';records=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        assert [r['sample_id'] for r in records]==[r['sample_id'] for r in rows[:len(records)]]
        while len(records)<len(rows):
            group=rows[len(records):len(records)+batch_size]
            def beat(tokens=0,seconds=0):
                status(out,'evaluating_'+kind,completed=len(records),total=240,current_symbol=','.join(r['symbol'] for r in group),max_new_tokens=budget,batch_size=len(group),in_progress=True,current_generated_tokens=tokens,elapsed_generation_seconds=seconds,allocated_gib=e.torch.cuda.max_memory_allocated()/2**30,reserved_gib=e.torch.cuda.max_memory_reserved()/2**30)
            beat();result=e.generate(group,budget,beat)
            with f.open('a',encoding='utf-8') as stream:
                stream.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));stream.flush();os.fsync(stream.fileno())
            records.extend(result);status(out,'evaluating_'+kind,completed=len(records),total=240,batch_size=len(group),last_seconds=result[-1]['seconds'],last_batch_seconds=result[-1]['batch_seconds'],allocated_gib=result[-1]['peak_allocated_gib'],max_new_tokens=budget)
            if (out/'stop_after_checkpoint.flag').exists():raise RuntimeError('Intentional boundary stop; saved records preserved')
        all_rows[kind]=records;all_metrics[kind]=summarize(records);save(out/f'{kind}_metrics.json',all_metrics[kind])
    save(out/'comparison.json',dict(config=config,**all_metrics,simple_baselines=baselines(read(P/'data/train_raw.json'),rows),analysis_quality_requires_manual_review=True,test_used_for_checkpoint_selection=False))
    review=['# Batched validation output review','']
    for i,row in enumerate(rows):
        review.extend([f"## {i+1}. {row['symbol']} / {row['period']}",f"Label: {row['label']}",'### Supplied input',row['prompt_audit']['compressed_text'],'### Base',all_rows['base'][i]['generated'],'### Adapter',all_rows['adapter'][i]['generated'],'### Reference',row['reference']])
    (out/'manual_review.md').write_text('\n\n'.join(review));status(out,'completed',base_evaluated=240,adapter_evaluated=240,batch_size=batch_size)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch-size',type=int,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:evaluate(a.model,a.out,a.batch_size)
    except BaseException as exc:
        status(a.out,'failed',error_type=type(exc).__name__,error=str(exc));raise
