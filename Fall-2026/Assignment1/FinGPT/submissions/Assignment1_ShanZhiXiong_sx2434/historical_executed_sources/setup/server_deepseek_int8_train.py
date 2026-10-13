"""One GPU, one fresh adapter, one epoch; select with a chronological validation set."""
import argparse,ctypes,hashlib,json,math,os,signal,sys,time
from pathlib import Path
ROOT=Path('/root/shared-nvme/AI4FinRuntime/import_20261009/assignment1_20261008')
sys.path.insert(0,str(ROOT))
DATA=Path('/root/shared-nvme/AI4FinRuntime/setup/deepseek_full_int8_4096_20261010')
from training_core import prepare_model,sparse_loss
MODEL_IDS={'qwen':'Qwen/Qwen2.5-7B-Instruct','llama':'meta-llama/Llama-3.1-8B','deepseek':'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'}
OUT=None;STOP=False
def write(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(p)
def status(stage,**kw):
    v={'stage':stage,'updated_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'pid':os.getpid(),**kw};write('status.json',v);print(json.dumps(v),flush=True)
def stop(*args):
    global STOP
    STOP=True
def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODEL_IDS,required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();OUT=Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'run_config.json').exists():raise RuntimeError('Refusing to overwrite previous GPU run')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    status('imports')
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback,DataCollatorForSeq2Seq,set_seed
    from peft import LoraConfig,get_peft_model
    from safetensors.torch import load_file
    assert torch.cuda.is_available(),'No CPU fallback'
    assert a.model=='deepseek'
    trpath=DATA/'deepseek_train_4096.json';vapath=DATA/'deepseek_validation_4096.json'
    train=json.loads(trpath.read_text(encoding='utf-8'));valid=json.loads(vapath.read_text(encoding='utf-8'))
    assert len(train)==990 and len(valid)==240
    assert not ({r['sample_id'] for r in train}&{r['sample_id'] for r in valid})
    proto=json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))
    write('run_config.json',{'model_id':MODEL_IDS[a.model],'protocol':proto,'fresh_base_no_old_adapter':True,'purpose':'one epoch training with chronological validation; not complete Assignment 1','max_length':4096,'quantization_bits':8,'batch_size':1,'gradient_accumulation':4,'learning_rate':2e-4,'lora_r':8,'full_answers':True,'test_used':False,
      'train_sha256':hashlib.sha256(trpath.read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256(vapath.read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':math.ceil(len(train)/4)})
    gpu=torch.cuda.get_device_properties(0);limit=min(gpu.total_memory/2**30*.9,gpu.total_memory/2**30-.75)
    write('hardware.json',{'gpu':gpu.name,'dedicated_gib':gpu.total_memory/2**30,'allocated_guard_gib':limit,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'python':sys.version})
    set_seed(42);cache=Path('/root/shared-nvme/AI4FinRuntime/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1');tok=AutoTokenizer.from_pretrained(str(cache),local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=6.0)
    status('loading_cached_model',model=MODEL_IDS[a.model])
    model=AutoModelForCausalLM.from_pretrained(str(cache),local_files_only=True,quantization_config=quant,
      device_map={'':0},torch_dtype=dtype,low_cpu_mem_usage=True,attn_implementation='sdpa')
    model=get_peft_model(prepare_model(model),LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    write('memory_audit.json',{'loaded_allocated_gib':torch.cuda.memory_allocated()/2**30,'embedding_dtype':str(model.get_input_embeddings().weight.dtype),'head_dtype':str(model.get_output_embeddings().weight.dtype)})
    data=Dataset.from_list([{k:v for k,v in r.items() if k in ('input_ids','attention_mask','labels')} for r in train])
    validation_records=[]
    def validate(step,checkpoint=None):
        was_training=model.training;model.eval();start=time.monotonic();total=0.;tokens=0;sample_loss=[]
        for i,row in enumerate(valid):
            inputs={k:torch.tensor([row[k]],device='cuda') for k in ('input_ids','attention_mask','labels')}
            count=sum(v!=-100 for v in row['labels'][1:]);assert count>0
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=dtype):res=sparse_loss(model,inputs)
            loss=float(res.loss);assert math.isfinite(loss)
            total+=loss*count;tokens+=count;sample_loss.append(loss)
            del res,inputs;torch.cuda.empty_cache()
            if (i+1)%40==0:status('validation_loss',step=step,completed_steps=step,max_steps=248,completed=i+1,total=len(valid))
        if was_training:model.train()
        item={'step':step,'checkpoint':str(checkpoint) if checkpoint else None,'samples':len(valid),
          'supervised_tokens':tokens,'token_weighted_loss':total/tokens,'sample_mean_loss':sum(sample_loss)/len(sample_loss),'seconds':time.monotonic()-start}
        validation_records.append(item);write('validation_loss.json',validation_records)
        print(json.dumps({'validation_result':item}),flush=True)
        return item
    class SparseTrainer(Trainer):
        def training_step(self,model,inputs,num_items_in_batch=None):
            result=super().training_step(model,inputs,num_items_in_batch);torch.cuda.empty_cache();return result
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            res=sparse_loss(model,inputs,num_items_in_batch);self.seen_microbatches+=1
            return (res.loss,res) if return_outputs else res.loss
    class Progress(TrainerCallback):
        def __init__(self):self.reason=None;self.times=[]
        def on_step_begin(self,args,state,control,**kw):self.begin=time.monotonic()
        def on_step_end(self,args,state,control,**kw):
            torch.cuda.synchronize();seconds=time.monotonic()-self.begin;self.times.append(seconds)
            peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
            if peak>limit or reserved>gpu.total_memory/2**30-.1:self.reason='stopped_memory_guard'
            if state.global_step>=2 and seconds>120:self.reason='stopped_speed_guard'
            if STOP or (OUT/'stop_after_checkpoint.flag').exists():self.reason='stopped_requested'
            if state.global_step in (1,2,124,248) or state.global_step%50==0:control.should_save=True
            if self.reason:control.should_save=True;control.should_training_stop=True
            item={'step':state.global_step,'max_steps':state.max_steps,'step_seconds':seconds,'seen_microbatches':trainer.seen_microbatches,'allocated_gib':peak,'reserved_gib':reserved,'peak_allocated_gib':peak,'peak_reserved_gib':reserved}
            with (OUT/'step_telemetry.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item)+'\n')
            status(self.reason or 'training',**item)
        def on_save(self,args,state,control,**kw):
            path=OUT/'checkpoints'/f'checkpoint-{state.global_step}'
            tensors=load_file(str(path/'adapter_model.safetensors'),device='cpu')
            assert tensors and all(torch.isfinite(x).all() for x in tensors.values())
            # The first optimizer update uses LR=0 during warmup; initialized B=0 is valid.
            if state.global_step>=2:
                assert any(torch.count_nonzero(x) for k,x in tensors.items() if 'lora_B' in k),'Adapter updates absent after warmup'
            write(f'checkpoint_checks/step_{state.global_step}.json',{'passed':True,'step':state.global_step,'checkpoint':str(path),'tensors':len(tensors),
              'sha256':hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()})
            del tensors
            if state.global_step in (124,248) and not self.reason:validate(state.global_step,path)
    progress=Progress()
    args=TrainingArguments(output_dir=str(OUT/'checkpoints'),max_steps=248,num_train_epochs=1,
      per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=2e-4,warmup_ratio=.03,lr_scheduler_type='cosine',
      bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
      optim='adamw_8bit',logging_steps=5,save_strategy='steps',save_steps=50,report_to='none',dataloader_num_workers=0,
      remove_unused_columns=False,seed=42,label_names=['labels'])
    trainer=SparseTrainer(model=model,args=args,train_dataset=data,data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),callbacks=[progress])
    trainer.seen_microbatches=0;trainer.model_accepts_loss_kwargs=True
    validate(0)
    torch.cuda.reset_peak_memory_stats()
    status('training_start',completed_steps=0,max_steps=248,samples=990,validation_samples=240,steps=248)
    result=trainer.train();trainer.save_model(str(OUT/'final_adapter'));tok.save_pretrained(str(OUT/'final_adapter'))
    write('training_metrics.json',{**result.metrics,'completed_steps':trainer.state.global_step,'seen_microbatches':trainer.seen_microbatches,
      'step_seconds':progress.times,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
    if progress.reason:status(progress.reason,completed_steps=trainer.state.global_step);return 2
    assert trainer.seen_microbatches==990 and trainer.state.global_step==248
    eligible=[r for r in validation_records if r['checkpoint']]
    best=min(eligible,key=lambda r:(r['token_weighted_loss'],r['step']))
    write('selected_checkpoint.json',{**best,'selection_rule':'minimum full validation token-weighted cross-entropy; earlier checkpoint on tie','test_used':False})
    status('completed',max_steps=248,completed_steps=248,seen_microbatches=990,selected_checkpoint=best['checkpoint'],validation_loss=best['token_weighted_loss'])
    return 0
if __name__=='__main__':
    pass
    try:code=main()
    except BaseException as e:
        if OUT:status('failed',error_type=type(e).__name__,error=str(e))
        import traceback;traceback.print_exc();code=1
    finally:pass
    sys.exit(code)
