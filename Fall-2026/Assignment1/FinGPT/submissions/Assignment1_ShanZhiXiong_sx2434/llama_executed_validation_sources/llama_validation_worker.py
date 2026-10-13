"""Only frozen retrospective Llama validation240, serialized by the GPU lock."""
import argparse,fcntl,hashlib,importlib.util,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
R=Path('/root/shared-nvme/AI4FinRuntime');B=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def check(v,s):
    if not v:raise ValueError(s)
def save(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def direction(records):
    values=[]
    for r in records:
        lines=re.findall(r'(?im)^\s*(?:-\s*)?Prediction\s*:\s*(.*)$',r['generated'].replace('*','').replace('_',''))
        directions=set('up' if x.lower() in ['up','increase','rise'] else 'down' for x in re.findall(r'\b(up|down|increase|decrease|rise|fall|decline)\b',lines[-1],re.I)) if lines else set()
        predicted=next(iter(directions)) if len(directions)==1 else None;expected='up' if r['label'].lower().startswith('up') else 'down'
        values.append({'sample_id':r['sample_id'],'predicted':predicted,'expected':expected})
    valid=sum(x['predicted'] is not None for x in values);correct=sum(x['predicted']==x['expected'] for x in values)
    return {'metric_variant':'source_checked_final_direction_markdown_tolerant_v1','total_outputs':len(values),'valid_direction_outputs':valid,'correct_direction_count':correct,'accuracy_over_all_outputs':correct/len(values),'accuracy_over_valid_outputs':correct/valid if valid else None,'rows':values}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);a=ap.parse_args()
    lock=(R/'setup/gpu_training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=None
    try:
        check(sha(B/'FROZEN_VALIDATION_PLAN.json')==a.plan_sha,'External validation plan SHA differs');plan=json.loads((B/'FROZEN_VALIDATION_PLAN.json').read_text());out=Path(plan['output_directory'])
        check(not out.exists(),'Fresh validation directory required');out.mkdir(parents=True)
        for n,s in plan['dependency_sha256'].items():check(Path(n).name==n and sha(B/n)==s,'Source dependency changed')
        check(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU worker present')
        scope=json.loads((R/'setup/submission_readiness_scope_20261011.json').read_text());check(scope['experiment_repair_decision']=='retrospective_original_data' and scope['llama_training_allowed'] and 'llama' in scope['active_models'],'User route differs')
        check(plan['test_read'] is False and plan['samples']==240,'Validation only')
        train=Path(plan['training_run']);status=json.loads((train/'status.json').read_text());check(status['stage']=='completed' and status['completed_steps']==248 and status['seen_microbatches']==990,'Training incomplete')
        selected=json.loads((train/'selected_checkpoint.json').read_text());check(selected==plan['selected_checkpoint'],'Selected checkpoint differs')
        checkpoint=Path(selected['checkpoint']);check(sha(checkpoint/'adapter_model.safetensors')==plan['adapter_sha256'],'Adapter changed')
        marker=plan['cache_verified'];cache=Path(marker['cache']);check(marker['model_id']=='meta-llama/Llama-3.1-8B' and cache.name==plan['revision'],'Wrong model cache')
        for n,s in {**marker['weight_sha256'],**marker['nonweight_sha256']}.items():check(sha(cache/n)==s,'Verified cache changed')
        sys.path.insert(0,str(B));from split_target_contract import load_bound_tokenizer,load_frozen_split
        tok=load_bound_tokenizer(cache,model_key='llama',specification=plan['tokenizer_binding']);check(not tok.chat_template,'Native BASE prefix changed')
        pp=B/'SOURCE_POLICY_REMOTE.json';policy=json.loads(pp.read_text());receipts={};source_rows={}
        for split in ['train','validation']:
            rows,receipt=load_frozen_split(policy['splits'][split]['path'],split=split,policy_path=pp,policy_sha256=plan['source_policy_sha256']);receipts[split]=receipt;source_rows[split]={r['sample_id']:r for r in rows}
        save(out/'DATA_READ_RECEIPTS.json',receipts)
        data=Path(plan['evaluation_data']);check(sha(data)==plan['evaluation_data_sha256'],'Prepared evaluation data changed');rows=json.loads(data.read_text());check(len(rows)==240 and set(r['sample_id'] for r in rows)==set(source_rows['validation']),'Validation identity differs')
        bindings=json.loads((B/'RUNTIME_SOURCE_BINDINGS.json').read_text());root=R/'import_20261009/assignment1_20261008'
        for n,s in bindings['core_sources'].items():check(sha(root/n)==s,'Historical runtime source changed')
        sys.path.insert(0,str(root));from data_processing import SYSTEM
        for row in rows:
            src=source_rows['validation'][row['sample_id']];text=re.sub(r'\[/INST\]\s*$','',src['prompt'].split('<</SYS>>',1)[-1].strip()).replace('cryptocurrency price movement','stock price movement')
            ids=tok.encode(SYSTEM+'\n\n'+text+'\n\n',add_special_tokens=True)
            check(text==row['prompt_audit']['compressed_text'] and ids==row['input_ids'] and row['reference']==src['answer'].strip(),'Native input/reference changed')
            check(row['previously_inspected_test'] is False and len(ids)+plan['max_new_tokens']<=6144,'Budget/exposure differs')
        registry=json.loads((R/'setup/three_model_runtime_registry_20261010.json').read_text())['llama']
        for key in ['cache','training_run','evaluation_data']:check(registry[key]==plan[key],'Registry binding changed')
        module_spec=importlib.util.spec_from_file_location('pinned_llama_backend',B/'backend_server_chunked_prefill_evaluate.py');backend=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(backend)
        config={k:plan[k] for k in ['model_id','revision','samples','batch_size','prefill_chunk_tokens','max_new_tokens','context_tokens','adapter_sha256','target_policy','route','test_read']};config.update(plan_sha256=a.plan_sha,prompt_policy=plan['prompt_policy'],speed_comparison_claim=False);save(out/'config.json',config)
        save(out/'MODEL_IDENTITY_RECEIPT.json',{'weight_sha256':marker['weight_sha256'],'tokenizer_binding':tok.binding_receipt,'selected_checkpoint':selected,'adapter_sha256':plan['adapter_sha256'],'test_read':False})
        backend.status(out,'loading_cached_model',batch_size=plan['batch_size']);engine=backend.Engine('llama');check(engine.ids(rows[0])==rows[0]['input_ids'],'Loaded tokenizer rendering differs');outputs={}
        for family in ['base','adapter']:
            if family=='adapter':engine.adapter()
            path=out/f'{family}_outputs.jsonl';records=[]
            for offset in range(0,240,plan['batch_size']):
                group=rows[offset:offset+plan['batch_size']]
                def beat(tokens=0,seconds=0):backend.status(out,'evaluating_'+family,completed=len(records),total=240,current_symbol=','.join(x['symbol'] for x in group),current_generated_tokens=tokens,elapsed_generation_seconds=seconds,batch_size=len(group),max_new_tokens=1920)
                beat();result=engine.generate(group,plan['max_new_tokens'],beat)
                for item,row in zip(result,group):
                    check(hashlib.sha256(json.dumps(item['generated_token_ids']).encode()).hexdigest()==item['generated_token_sha256'],'Generated token hash differs')
                    check(item['peak_allocated_gib']<=engine.guard and item['peak_reserved_gib']<=engine.torch.cuda.get_device_properties(0).total_memory/2**30-.1,'Memory guard exceeded')
                    item.update(variant='base_native_base' if family=='base' else 'adapter_full_reference',boundary='generated')
                    item['generated_boundary_receipt']={'model_key':'llama','model_id':plan['model_id'],'revision':plan['revision'],'variant':item['variant'],'boundary':'generated','prompt_token_sha256':hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest(),'generated_token_sha256':item['generated_token_sha256'],'native_bos_id':tok.bos_token_id,'eos_token_id':tok.eos_token_id,'pad_policy':'existing EOS with attention mask','tokenizer_sha256':tok.binding_receipt['file_sha256'],'eos_observed':item['ended_eos'],'chat_template_used':False}
                with path.open('a') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in result));f.flush();os.fsync(f.fileno())
                records.extend(result);backend.status(out,'evaluating_'+family,completed=len(records),total=240,batch_size=len(group),last_seconds=result[-1]['seconds'],last_batch_seconds=result[-1]['batch_seconds'],allocated_gib=result[-1]['peak_allocated_gib'],max_new_tokens=1920)
                if (out/'stop_after_checkpoint.flag').exists():raise RuntimeError('Requested boundary stop; raw outputs preserved')
            outputs[family]=records;save(out/f'{family}_metrics.json',backend.summarize(records));save(out/f'{family}_full_denominator_direction.json',direction(records))
        # Existing baselines read this exact train file, already checked/read with V3 above.
        save(out/'comparison.json',{'config':config,'base':backend.summarize(outputs['base']),'adapter':backend.summarize(outputs['adapter']),'simple_baselines':backend.baselines(list(source_rows['train'].values()),rows),'predictive_advantage_claimed':False,'test_used_for_checkpoint_selection':False})
        save(out/'COMPLETED_OUTPUTS_SHA256.json',{'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='status.json'},'output_records':480,'test_read':False})
        backend.status(out,'completed',base_evaluated=240,adapter_evaluated=240,batch_size=plan['batch_size'],test_read=False)
    except BaseException as e:
        if out and out.exists():save(out/'status.json',{'stage':'failed','pid':os.getpid(),'error_type':type(e).__name__,'message':'Validation preflight or generation failed; frozen evidence preserved','updated_at_utc':datetime.now(timezone.utc).isoformat()})
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
    os.environ.update(HF_HOME=str(R/'huggingface'),HF_ENDPOINT='https://huggingface.co',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TORCH_HOME=str(R/'torch_cache'),TRITON_CACHE_DIR=str(R/'tmp/triton_cache'),TOKENIZERS_PARALLELISM='false')
    main()
