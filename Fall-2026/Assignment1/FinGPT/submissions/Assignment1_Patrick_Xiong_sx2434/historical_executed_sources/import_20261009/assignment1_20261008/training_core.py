"""Offline full-answer context probes and a single pass over the training set."""
import argparse,ctypes,hashlib,json,math,os,platform,signal,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STOP=False
OUT=None
def save(name,value):
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(path)
def status(stage,**kw):
    value=dict(stage=stage,timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    save('status.json',value);print(json.dumps(value,default=str),flush=True)
def stop(signum,frame):
    global STOP
    STOP=True
def prepare_model(model):
    import torch
    for p in model.parameters():p.requires_grad=False
    for module in model.modules():
        if 'norm' in type(module).__name__.lower():
            for p in module.parameters(recurse=False):p.data=p.data.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    return model
def sparse_loss(model,inputs,num_items_in_batch=None):
    import torch
    labels=inputs['labels']
    assert labels.shape[0]==1,'This memory profile requires microbatch size 1'
    shifted=labels[:,1:]
    positions=torch.nonzero(shifted[0]!=-100,as_tuple=False).flatten()
    # Keep every supervised answer/EOS token, and skip only masked prompt logits.
    outputs=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=labels,
                  logits_to_keep=positions,shift_labels=shifted[:,positions],num_items_in_batch=num_items_in_batch)
    return outputs
def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True)
    ap.add_argument('--mode',choices=['probe','epoch'],required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();OUT=Path(a.out);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'run_config.json').exists():raise RuntimeError('Use a new output directory; old results are preserved')
    data_path=ROOT/f'{"probe" if a.mode=="probe" else "train"}_{a.length}.json'
    save('run_config.json',dict(mode=a.mode,max_length=a.length,learning_rate=2e-4,lora_r=8,accumulation=4,
                              microbatch=1,full_answers=True,sparse_answer_logits=True,
                              release_inactive_cache_after_microbatch=True,
                              prepared_dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                              preprocessor_sha256=hashlib.sha256((ROOT/'data_processing.py').read_bytes()).hexdigest(),
                              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    signal.signal(signal.SIGINT,stop)
    status('imports')
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,TrainerCallback,DataCollatorForSeq2Seq,set_seed
    from peft import LoraConfig,get_peft_model
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    total=torch.cuda.get_device_properties(0).total_memory/2**30;limit=min(.9*total,total-.75)
    save('hardware.json',dict(gpu=torch.cuda.get_device_name(0),dedicated_gib=total,guard_gib=limit,torch=torch.__version__))
    if platform.system()=='Windows':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    data=json.loads(data_path.read_text())
    if a.mode=='epoch' and len(data)!=1230:raise RuntimeError('One-pass run requires all 1230 samples, without silently dropping long answers')
    steps=8 if a.mode=='probe' else math.ceil(len(data)/4)
    dataset=Dataset.from_list([{k:v for k,v in row.items() if k!='source_index'} for row in data])
    save('data_selection.json',dict(samples=len(data),source_indices=[x['source_index'] for x in data],steps=steps,
                                   max_sequence_tokens=max(len(x['input_ids']) for x in data),answer_truncated=0))
    set_seed(42);tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    compute=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=compute)
    status('loading_cached_model',steps=steps)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',local_files_only=True,
        quantization_config=quant,device_map={'':0},torch_dtype=compute,attn_implementation='sdpa',low_cpu_mem_usage=True)
    model=get_peft_model(prepare_model(model),LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    save('memory_audit.json',dict(loaded_allocated_gib=torch.cuda.memory_allocated()/2**30,
         input_embedding_dtype=str(model.get_input_embeddings().weight.dtype),output_head_dtype=str(model.get_output_embeddings().weight.dtype),guard_gib=limit))
    class SparseTrainer(Trainer):
        def training_step(self,model,inputs,num_items_in_batch=None):
            loss=super().training_step(model,inputs,num_items_in_batch)
            # Variable full-answer lengths otherwise retain many large vocabulary-logit blocks
            # in the Windows CUDA caching allocator, spilling unused pool pages into shared RAM.
            torch.cuda.empty_cache()
            return loss
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            result=sparse_loss(model,inputs,num_items_in_batch)
            self.seen_microbatches+=1
            return (result.loss,result) if return_outputs else result.loss
    class Progress(TrainerCallback):
        def __init__(self):self.reason=None;self.times=[]
        def on_step_begin(self,args,state,control,**kw):self.start=time.monotonic()
        def on_step_end(self,args,state,control,**kw):
            torch.cuda.synchronize();seconds=time.monotonic()-self.start;self.times.append(seconds)
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>limit:self.reason='stopped_memory_guard'
            elif state.global_step>=2 and seconds>120:self.reason='stopped_speed_guard'
            elif STOP:self.reason='paused'
            if state.global_step==1 or state.global_step==state.max_steps:control.should_save=True
            if self.reason:control.should_save=True;control.should_training_stop=True
            value=dict(step=state.global_step,max_steps=state.max_steps,step_seconds=round(seconds,3),
                       peak_allocated_gib=round(peak,4),peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,4),
                       microbatches_seen=trainer.seen_microbatches)
            with (OUT/'step_telemetry.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value)+'\n')
            status(self.reason or 'training',**value)
    progress=Progress()
    args=TrainingArguments(output_dir=str(OUT/'checkpoints'),max_steps=steps,num_train_epochs=1,
        per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=2e-4,warmup_ratio=.03,
        lr_scheduler_type='cosine',bf16=compute==torch.bfloat16,fp16=compute==torch.float16,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_8bit',
        logging_steps=5,save_steps=8 if a.mode=='probe' else 50,report_to='none',dataloader_num_workers=0,
        remove_unused_columns=False,seed=42,label_names=['labels'])
    trainer=SparseTrainer(model=model,args=args,train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),callbacks=[progress])
    trainer.seen_microbatches=0;trainer.model_accepts_loss_kwargs=True
    status('training_start',samples=len(data),steps=steps,length=a.length)
    result=trainer.train();trainer.save_model(str(OUT/'adapter'));tok.save_pretrained(str(OUT/'adapter'))
    metrics=dict(result.metrics,completed_steps=trainer.state.global_step,microbatches_seen=trainer.seen_microbatches,
                 actual_epoch=trainer.state.epoch,step_seconds=progress.times,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    save('training_metrics.json',metrics)
    if progress.reason:
        status(progress.reason,completed_steps=trainer.state.global_step,adapter=str(OUT/'adapter'));return 2
    if a.mode=='epoch':assert trainer.seen_microbatches==1230 and trainer.state.global_step==308,'Full single-pass coverage check failed'
    status('completed',completed_steps=trainer.state.global_step,microbatches_seen=trainer.seen_microbatches,adapter=str(OUT/'adapter'))
    return 0
if __name__=='__main__':
    try:code=main()
    except BaseException as e:
        if OUT is not None:status('failed',error_type=type(e).__name__,error=str(e))
        import traceback;traceback.print_exc();code=1
    finally:
        if platform.system()=='Windows':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    sys.exit(code)
