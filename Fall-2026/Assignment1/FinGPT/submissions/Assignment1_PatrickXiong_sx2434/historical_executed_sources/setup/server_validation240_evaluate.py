"""Identical frozen evidence for base/selected-adapter; resumable per-answer records."""
import argparse,ctypes,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path('/root/shared-nvme/AI4FinRuntime/import_20261009/assignment1_20261008')
sys.path.insert(0,str(ROOT))
from data_processing import SYSTEM
from metrics import summarize,baselines
MODEL_IDS={'qwen':'Qwen/Qwen2.5-7B-Instruct','deepseek':'deepseek-ai/DeepSeek-R1-Distill-Llama-8B','llama':'meta-llama/Llama-3.1-8B'}
TRAIN=ROOT/'runs/server4090_qwen_int8_4096_epoch_20261010'
DATA=Path('/root/shared-nvme/AI4FinRuntime/setup/full_int8_4096_20261010')
CACHE=Path('/root/shared-nvme/AI4FinRuntime/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')
OUT=None
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(p)
def status(stage,**kw):
    obj={'stage':stage,'updated_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'pid':os.getpid(),**kw};save('status.json',obj);print(json.dumps(obj),flush=True)
def render(tok,text):
    if tok.chat_template:
        return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
    return tok.encode(SYSTEM+'\n\n'+text+'\n\n',add_special_tokens=True)
def main():
    global OUT,TRAIN,CACHE,DATA
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODEL_IDS,required=True)
    ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--max-new-tokens',type=int,default=640)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--limit',type=int,help='Validation-only prefix probe; never applies to final test')
    a=ap.parse_args()
    registry=json.loads(Path('/root/shared-nvme/AI4FinRuntime/setup/three_model_runtime_registry_20261010.json').read_text());info=registry[a.model]
    TRAIN=Path(info['training_run']);CACHE=Path(info['cache']);DATA=Path(info['evaluation_data']).parent
    if not 1<=a.max_new_tokens<=2048:raise ValueError('Generation budget must be between 1 and 2048')
    if a.limit is not None and (a.split!='validation' or a.limit<1):raise ValueError('Probe limit requires validation and a positive count')
    if a.split=='test':
        frozen=ROOT/'final_test_authorized.json'
        if not frozen.exists():raise RuntimeError('Final model configurations have not been frozen; test generation refused')
        final_config=json.loads(frozen.read_text(encoding='utf-8'))
        if a.max_new_tokens!=final_config.get('generation',{}).get('max_new_tokens',640):
            raise RuntimeError('Final test generation budget differs from the frozen validation configuration')
    OUT=a.out or ROOT/f'runs/{a.model}_{a.split}_evaluation';OUT.mkdir(parents=True,exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed,StoppingCriteria,StoppingCriteriaList
    from peft import PeftModel
    assert a.model in MODEL_IDS and a.split=='validation', 'Only development validation is authorized here'
    selected=json.loads((TRAIN/'selected_checkpoint.json').read_text(encoding='utf-8'))
    checkpoint=Path(selected['checkpoint']);assert checkpoint.is_dir()
    if a.split=='test':
        expected=final_config['model_configs_frozen'][a.model]
        assert str(checkpoint)==expected['checkpoint']
        assert hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()==expected['adapter_sha256']
    datafile=Path(info['evaluation_data'])
    rows=json.loads(datafile.read_text(encoding='utf-8'))
    if a.split=='test':assert hashlib.sha256(datafile.read_bytes()).hexdigest()==final_config['test_sha256']
    if a.limit is not None:rows=rows[:a.limit]
    config={'model_id':MODEL_IDS[a.model],'split':a.split,'samples':len(rows),'do_sample':False,'max_new_tokens':a.max_new_tokens,
      'canonical_evidence_sha256':hashlib.sha256(datafile.read_bytes()).hexdigest(),
      'adapter_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
      'checkpoint':str(checkpoint),'prompt_policy':'same full original validation input for base and adapter; all frozen 240 validation IDs; complete original evidence; model-specific template; 6144 inference cap','quantization_bits':8,'max_length':6144,'test_used':False,
      'reference_comparison':'unmodified course parser; supplemental full-denominator label metrics'}
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text(encoding='utf-8'))==config
    save('config.json',config);set_seed(42);assert torch.cuda.is_available()
    tok=AutoTokenizer.from_pretrained(str(CACHE),local_files_only=True)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0)
    status('loading_cached_model')
    model=AutoModelForCausalLM.from_pretrained(str(CACHE),local_files_only=True,quantization_config=quant,device_map={'':0},
      torch_dtype=dtype,low_cpu_mem_usage=True,attn_implementation='sdpa')
    guard=min(torch.cuda.get_device_properties(0).total_memory/2**30*.9,torch.cuda.get_device_properties(0).total_memory/2**30-.75)
    all_records={};all_metrics={}
    for kind in ('base','adapter'):
        if kind=='adapter':model=PeftModel.from_pretrained(model,str(checkpoint),is_trainable=False)
        model.eval();model.config.use_cache=True;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        path=OUT/f'{kind}_outputs.jsonl'
        records=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        assert [r['sample_id'] for r in records]==[r['sample_id'] for r in rows[:len(records)]]
        for i,row in enumerate(rows[len(records):],start=len(records)):
            ids=render(tok,row['prompt_audit']['compressed_text'])
            assert ids==row['input_ids'], 'Prepared and actual input tokens differ'
            assert len(ids)+a.max_new_tokens<=6144,'Input plus generation exceeds approved context cap'
            # For the untemplated Llama base model the assistant prefix is part of the prompt.
            prefix=''  # Un-templated Llama uses the same prompt boundary as training.
            x=torch.tensor([ids],device='cuda');torch.cuda.synchronize();start=time.monotonic()
            status('evaluating_'+kind,completed=len(records),total=len(rows),current_symbol=row['symbol'],max_new_tokens=a.max_new_tokens,in_progress=True)
            class Heartbeat(StoppingCriteria):
                last=start
                def __call__(self,input_ids,scores,**kwargs):
                    now=time.monotonic()
                    if now-self.last>=5:
                        self.last=now
                        status('evaluating_'+kind,completed=len(records),total=len(rows),current_symbol=row['symbol'],
                          max_new_tokens=a.max_new_tokens,in_progress=True,elapsed_generation_seconds=now-start,
                          current_generated_tokens=input_ids.shape[-1]-len(ids),allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                    return False
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=dtype):
                  y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=a.max_new_tokens,do_sample=False,
                  temperature=None,top_p=None,top_k=None,repetition_penalty=1.,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([Heartbeat()]))
            torch.cuda.synchronize();seconds=time.monotonic()-start;new=y[0,len(ids):].tolist();eos=bool(new and new[-1]==tok.eos_token_id)
            record={'sample_id':row['sample_id'],'symbol':row['symbol'],'period':row['period'],'label':row['label'],
              'reference':row['reference'],'generated':prefix+tok.decode(new,skip_special_tokens=True),'prompt_tokens':len(ids),
              'generated_tokens':len(new),'ended_eos':eos,'hit_generation_limit':len(new)>=a.max_new_tokens and not eos,'seconds':seconds,
              'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
              'previously_inspected_test':row.get('previously_inspected_test',False)}
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            records.append(record)
            if len(records)==1 or len(records)%10==0 or len(records)==len(rows):
                save(f'{kind}_metrics.json',summarize(records))
            status('evaluating_'+kind,completed=len(records),total=len(rows),last_seconds=seconds,allocated_gib=record['peak_allocated_gib'],max_new_tokens=a.max_new_tokens)
            del x,y;torch.cuda.empty_cache()
            if record['peak_allocated_gib']>guard:raise RuntimeError('Evaluation allocated-memory guard exceeded; outputs retained')
            if record['peak_reserved_gib']>torch.cuda.get_device_properties(0).total_memory/2**30-.1:raise RuntimeError('Evaluation reserved-memory guard exceeded; outputs retained')
            if (OUT/'stop_after_checkpoint.flag').exists():raise RuntimeError('Stop requested; generated records retained')
        all_records[kind]=records;all_metrics[kind]=summarize(records)
        if a.split=='test':
            for exposed,name in ((True,'exposed20'),(False,'not_individually_inspected280')):
                save(f'{kind}_{name}_metrics.json',summarize([r for r in records if r['previously_inspected_test']==exposed]))
    train=json.loads((ROOT/'data/train_raw.json').read_text(encoding='utf-8'))
    save('comparison.json',{'config':config,**all_metrics,'simple_baselines':baselines(train,rows),
       'analysis_quality_requires_manual_review':True,'test_used_for_checkpoint_selection':False})
    review=['# Validation output review' if a.split=='validation' else '# Final test output review','',
      'Inspect evidence support, omitted information, reasoning, and unsupported claims. ROUGE measures text overlap only.','']
    for i,row in enumerate(rows):
        review += [f"## {i+1}. {row['symbol']} / {row['period']}",f"Label: {row['label']}",'','### Supplied input',row['prompt_audit']['compressed_text'],
          '','### Base output',all_records['base'][i]['generated'],'','### Selected adapter output',all_records['adapter'][i]['generated'],
          '','### Reference',row['reference'],'']
    (OUT/'manual_review.md').write_text('\n'.join(review),encoding='utf-8')
    status('completed',base_evaluated=len(rows),adapter_evaluated=len(rows));return 0
if __name__=='__main__':
    pass
    try:code=main()
    except BaseException as e:
        if OUT:status('failed',error_type=type(e).__name__,error=str(e))
        import traceback;traceback.print_exc();code=1
    finally:pass
    sys.exit(code)
