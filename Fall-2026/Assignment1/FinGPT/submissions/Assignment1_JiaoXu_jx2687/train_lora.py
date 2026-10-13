"""Portable Forecaster LoRA training; course code reviewed in sources/train_lora.py.
Local 0.5B run is explicitly a reduced substitute, not the required 7B/8B run.
"""
import argparse,json,time,os,platform,sys,hashlib
from pathlib import Path
import torch
from datasets import load_from_disk,Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer,DataCollatorForSeq2Seq,set_seed
from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
from utils import make_prompt,compact_answer,lora_module_dict
ROOT=Path(__file__).resolve().parent

def main(a):
    set_seed(5398);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype=torch.float16 if device in ['cuda','mps'] else torch.float32
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=a.offline,trust_remote_code=False)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    kwargs={'torch_dtype':dtype,'trust_remote_code':False,'local_files_only':a.offline,'attn_implementation':'eager'}
    if a.quant8:
        if device!='cuda':raise RuntimeError('8-bit + DeepSpeed course configuration requires CUDA')
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_8bit=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,**kwargs)
    if a.quant8:model=prepare_model_for_kbit_training(model)
    else:model=model.to(device)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.1,target_modules=lora_module_dict[a.arch],bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    for n,p in model.named_parameters():
        assert p.requires_grad==('lora_' in n),n
    data=load_from_disk(str(ROOT/a.dataset))
    counts=[]
    def encode(row):
        prompt,info=make_prompt(row['prompt'],tokenizer,a.prompt_budget)
        target=compact_answer(row['answer'],tokenizer) if not a.full_answers else row['answer']
        p=tokenizer.encode(prompt,add_special_tokens=False);t=tokenizer.encode(target,add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(p)+len(t)>a.max_length:raise ValueError(f'Combined length {len(p)+len(t)} exceeds {a.max_length}; enlarge max_length')
        counts.append({**info,'prompt_tokens':len(p),'target_tokens':len(t),'symbol':row['symbol'],'period':row['period']})
        return {'input_ids':p+t,'attention_mask':[1]*(len(p)+len(t)),'labels':[-100]*len(p)+t}
    encoded={s:Dataset.from_list([encode(row) for row in data[s]]) for s in ['train','validation']}
    out=ROOT/'finetuned_models'/a.run_name;out.mkdir(parents=True,exist_ok=True)
    # Hash a frozen parameter and all adapter values to verify an actual update.
    frozen=next(p for n,p in model.named_parameters() if not p.requires_grad)
    frozen_before=hashlib.sha256(frozen.detach().cpu().numpy().tobytes()).hexdigest()
    adapter_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters())
    args=TrainingArguments(output_dir=str(out),num_train_epochs=a.epochs,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-5,weight_decay=.01,warmup_ratio=.03,lr_scheduler_type='constant',logging_steps=1,save_strategy='no',eval_strategy='no',report_to=['tensorboard'],logging_dir=str(out/'tensorboard'),seed=5398,data_seed=5398,dataloader_num_workers=0,dataloader_pin_memory=False,fp16=device=='cuda',deepspeed=a.ds_config,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    trainer=Trainer(model=model,args=args,train_dataset=encoded['train'],eval_dataset=encoded['validation'],processing_class=tokenizer,data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,label_pad_token_id=-100,return_tensors='pt'))
    t=time.perf_counter();result=trainer.train();elapsed=time.perf_counter()-t
    validation=trainer.evaluate()
    frozen_after=hashlib.sha256(frozen.detach().cpu().numpy().tobytes()).hexdigest()
    changed=sum(not torch.equal(adapter_before[n],p.detach().cpu()) for n,p in model.named_parameters() if p.requires_grad)
    assert changed>0 and frozen_before==frozen_after
    model.save_pretrained(out,safe_serialization=True)
    tokenizer.save_pretrained(out/'tokenizer')
    state={'args':vars(a),'model_revision':json.load(open(ROOT/'provenance.json'))['model_revision'],'device':device,'dtype':str(dtype),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'cuda_available':torch.cuda.is_available(),'mps_available':torch.backends.mps.is_available(),'trainable_parameters':trainable,'total_parameters':total,'trainable_percent':100*trainable/total,'train_runtime_wall_s':elapsed,'training_metrics':result.metrics,'validation_metrics':validation,'train_log':trainer.state.log_history,'frozen_parameter_hash_before':frozen_before,'frozen_parameter_hash_after':frozen_after,'changed_adapter_tensors':changed,'max_length_observed':max(x['prompt_tokens']+x['target_tokens'] for x in counts),'context_shortened_count':sum(x['context_shortened'] for x in counts),'tokenization':counts,'compact_training_answers':not a.full_answers}
    (ROOT/'results/training.json').write_text(json.dumps(state,indent=2));print(json.dumps({k:v for k,v in state.items() if k not in ['tokenization','train_log']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--arch',choices=list(lora_module_dict),default='qwen');p.add_argument('--dataset',default='data/local_subset');p.add_argument('--run_name',default='qwen05_local');p.add_argument('--max_length',type=int,default=1536);p.add_argument('--prompt_budget',type=int,default=896);p.add_argument('--epochs',type=float,default=1);p.add_argument('--offline',action='store_true');p.add_argument('--quant8',action='store_true');p.add_argument('--ds_config');p.add_argument('--full_answers',action='store_true');main(p.parse_args())
