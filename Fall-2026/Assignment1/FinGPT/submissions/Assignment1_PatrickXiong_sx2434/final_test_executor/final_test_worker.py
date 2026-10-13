"""Prospective one-run test executor. Missing real authorization rejects before access.

This file has not evaluated the real test. No resume, parameter search or training.
"""
import argparse, fcntl, gc, hashlib, importlib.util, importlib.metadata, json, os, platform, re, statistics, sys
from pathlib import Path
from datetime import datetime, timezone
import subprocess
from authorization_gate import authorize, hydrate_test_policy, read_pinned, save_new, sha_bytes

R=Path('/root/shared-nvme/AI4FinRuntime')
P=R/'import_20261009/assignment1_20261008'
B=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(path):return sha_bytes(Path(path).read_bytes())

def final_text(key,family,text):
    if key=='deepseek' and family=='base':
        return text.split('</think>',1)[1] if text.count('</think>')==1 else ''
    return text

def common_direction(text):
    lines=re.findall(r'(?im)^\s*(?:-\s*)?Prediction\s*:\s*(.*)$',text.replace('*','').replace('_',''))
    def extract(line):
        words=re.findall(r'\b(up|down|increase|decrease|rise|fall|decline)\b',line,re.I)
        choices={'up' if w.lower() in ['up','increase','rise'] else 'down' for w in words}
        return next(iter(choices)) if len(choices)==1 else None
    return (extract(lines[-1]) if lines else None, extract(lines[0]) if lines else None,len(lines))

def score_group(key,family,rows,course):
    raw=course.calc_metrics([r['generated'] for r in rows],[r['reference'] for r in rows])
    raw.setdefault('valid_count',0)
    accepted=[course.parse_answer(r['reference'])['prediction'] for r in rows
              if course.parse_answer(r['generated']) is not None and course.parse_answer(r['reference']) is not None]
    preds=[common_direction(final_text(key,family,r['generated'])) for r in rows]
    truth=[r['label'].lower().split()[0] for r in rows]
    require(all(t in ['up','down'] for t in truth),'Unknown dataset-label direction')
    counts={t:truth.count(t) for t in ['up','down']}
    recalls={t:sum(p[0]==t for p,y in zip(preds,truth) if y==t)/n if n else None for t,n in counts.items()}
    result={'samples':len(rows),'course_raw':raw,'constant_plus_half_mse_same_course_valid_subset':statistics.mean((.5-v)**2 for v in accepted) if accepted else None,
            'common_final_line':{'variant':'final_line_direction_common_v1','valid':sum(p[0] is not None for p in preds),'correct':sum(p[0]==t for p,t in zip(preds,truth)),
            'invalid_as_wrong':True,'balanced_accuracy':statistics.mean(recalls.values()) if all(counts.values()) else None,'class_counts':counts,
            'multiple_prediction_outputs':sum(p[2]>1 for p in preds),'first_line_correct':sum(p[1]==t for p,t in zip(preds,truth)),
            'first_last_disagreements':sum(p[0]!=p[1] for p in preds)},
            'always_up_correct':counts['up'],'train_majority_up_correct':counts['up'],
            'EOS':sum(r['ended_eos'] for r in rows),'caps':sum(r['hit_generation_limit'] for r in rows),
            'limitation':'Dataset labels and encoded references, not independently verified returns; ROUGE is not factual grounding.'}
    if key=='deepseek' and family=='base':
        texts=[final_text(key,family,r['generated']) for r in rows]
        result['deepseek_final_extracted_separate_variant']=course.calc_metrics(texts,[r['reference'] for r in rows])
        result['deepseek_final_extracted_separate_variant'].setdefault('valid_count',0)
        refs=[course.parse_answer(r['reference'])['prediction'] for r,t in zip(rows,texts) if course.parse_answer(t) is not None and course.parse_answer(r['reference']) is not None]
        result['deepseek_final_extracted_constant_plus_half_same_valid_subset']=statistics.mean((.5-v)**2 for v in refs) if refs else None
    return result

def generation_groups(cohort,batch_size):
    """Frozen input-length-only partition, identical for base and adapter."""
    groups=[]
    for offset in range(0,len(cohort),batch_size):
        original=cohort[offset:offset+batch_size]
        require(all(type(r.get('single_row_batch_required')) is bool for r in original),'Explicit serving flag required')
        groups.extend([[r] for r in original] if any(r['single_row_batch_required'] for r in original) else [original])
    return groups

def preflight(plan,keys):
    for name,h in plan['dependency_sha256'].items():require(Path(name).name==name and digest(B/name)==h,'Frozen source differs')
    for name,h in plan['runtime_core_sha256'].items():require((P/name).resolve().is_relative_to(P) and digest(P/name)==h,'Runtime core differs')
    require(platform.python_version()==plan['runtime_versions']['python'],'Python version differs')
    for package,version in plan['runtime_versions']['packages'].items():require(importlib.metadata.version(package)==version,'Frozen library version differs')
    scope=json.loads((R/'setup/submission_readiness_scope_20261011.json').read_text())
    require(scope['experiment_repair_decision']=='retrospective_original_data' and all(k in scope['active_models'] for k in plan['model_keys']),'Current user scope differs')
    bindings=read_pinned(B/'MODEL_BINDINGS.json',plan['dependency_sha256']['MODEL_BINDINGS.json'])
    registry=json.loads((R/'setup/three_model_runtime_registry_20261010.json').read_text())
    from transformers import AutoTokenizer
    sys.path.insert(0,str(P));from data_processing import SYSTEM
    spec=importlib.util.spec_from_file_location('frozen_backend',B/'backend_server_chunked_prefill_evaluate.py');backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    tokenizers={}
    for key in keys:
        bound=bindings[key];info=registry[key];require(info==bound['registry'],'Registry differs')
        require(digest(info['evaluation_data'])==bound['evaluation_data_sha256'],'Validation preparation differs')
        require(json.loads((Path(info['training_run'])/'selected_checkpoint.json').read_text())==bound['selected_checkpoint'],'Checkpoint selection differs')
        checkpoint=Path(bound['selected_checkpoint']['checkpoint'])
        require(digest(checkpoint/'adapter_model.safetensors')==bound['adapter_sha256'],'Adapter differs')
        for name,h in bound['cache_files_sha256'].items():require(Path(name).name==name and digest(Path(info['cache'])/name)==h,'Pinned weights/tokenizer differ')
        tokenizers[key]=AutoTokenizer.from_pretrained(info['cache'],local_files_only=True)
    return bindings,registry,tokenizers,backend,SYSTEM

def child(a,plan):
    # Every CLI entry point independently checks real authorization before
    # opening even a prepared test derivative. Parent passes its held lock FD.
    key=a.child_model;require(key in plan['model_keys'],'Unknown child model')
    out=Path(plan['output_directory'])
    control=read_pinned(out/'PARENT_CONTROL.json',a.control_sha)
    ph=control['derived_policy_sha256']
    require(control['parent_pid']==os.getppid() and control['plan_sha256']==a.plan_sha,'Not the authorized parent process')
    fd=os.fstat(a.gpu_lock_fd);real=os.stat(R/'setup/gpu_training.lock')
    require((fd.st_dev,fd.st_ino)==(real.st_dev,real.st_ino),'Not the shared GPU lock')
    fcntl.flock(a.gpu_lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU process is active')
    bindings,registry,tokenizers,backend,SYSTEM=preflight(plan,[key])
    cohort=read_pinned(out/(key+'_PREPARED.json'),control['prepared_sha256'][key])
    require(len(cohort)==300 and sum(r['previously_inspected_test'] for r in cohort)==20,'Prepared exposure/count differs')
    prompts=read_pinned(out/(key+'_PROMPT_BINDING.json'),control['prompt_binding_sha256'][key])
    prompt_receipts={r['sample_id']:r for r in prompts['samples']}
    require([r['sample_id'] for r in cohort]==list(prompt_receipts),'Prepared ID order differs')
    save_new(out/(key+'_PREPARED_READ_RECEIPT.json'),{'path':str(out/(key+'_PREPARED.json')),'sha256':control['prepared_sha256'][key],'plan_sha256':a.plan_sha,'authorization_sha256':a.authorization_sha,'read_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'authorized prepared test derivative','samples':300})
    run=out/key;run.mkdir();engine=backend.Engine(key);bound=bindings[key]
    require(engine.checkpoint==Path(bound['selected_checkpoint']['checkpoint']) and all(engine.ids(r)==r['input_ids'] for r in cohort),'Actual loaded adapter/input differs')
    batch=plan['generation']['batch_sizes'][key]
    require(all(r['single_row_batch_required']==(len(r['input_ids'])>plan['validated_prompt_max_tokens'][key]) and len(r['input_ids'])+1920<=6144 for r in cohort),'Length-only serving policy differs')
    schedule=generation_groups(cohort,batch)
    for family in ['base','adapter']:
        if family=='adapter':engine.adapter()
        generated=[];path=run/(family+'_outputs.jsonl')
        offset=0
        for group in schedule:
            def beat(tokens=0,seconds=0):backend.status(run,'test_'+family,completed=offset,total=len(cohort),current_generated_tokens=tokens,elapsed_generation_seconds=seconds)
            result=engine.generate(group,1920,beat)
            require([r['sample_id'] for r in result]==[r['sample_id'] for r in group],'Generated identities differ')
            for source,record in zip(group,result):
                require(record['batch_size']==len(group) and record['reference']==source['reference'] and record['label']==source['label'] and record['peak_allocated_gib']<=engine.guard,'Output/guard binding differs')
                require(record['generated_tokens']<=1920 and sha_bytes(json.dumps(record['generated_token_ids']).encode())==record['generated_token_sha256'],'Raw token receipt differs')
                tail=record['generated_token_ids'];eos=bool(tail and tail[-1]==engine.tok.eos_token_id)
                require(record['generated_tokens']==len(tail) and record['ended_eos']==eos and record['hit_generation_limit']==(len(tail)==1920 and not eos) and record['generated']==engine.tok.decode(tail,skip_special_tokens=True),'Native decode/EOS/cap evidence differs')
                record.update(previously_inspected_test=source['previously_inspected_test'],evaluation_variant=('base_native_reasoning' if key=='deepseek' else 'base_native') if family=='base' else ('adapter_legacy_unclosed' if key=='deepseek' else 'adapter_original_full_reference'),boundary_policy='Native generation; original unclosed DeepSeek adapter supervision retained',plan_sha256=a.plan_sha)
                record['closed_think_markers']=record['generated'].count('</think>')
                record['single_row_batch_required']=source['single_row_batch_required']
                record['serving_rule']='Input-length-only; any over-envelope row makes its original batch use single-row generation'
                record['boundary_receipt']={**prompt_receipts[source['sample_id']], 'variant':record['evaluation_variant'], 'closed_think_markers':record['closed_think_markers'], 'natural_eos':record['ended_eos'], 'generation_limit':record['hit_generation_limit']}
                require(record['peak_reserved_gib']<engine.torch.cuda.get_device_properties(0).total_memory/2**30-.1,'Reserved memory guard exceeded')
            with path.open('a',encoding='utf-8') as stream:stream.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result));stream.flush();os.fsync(stream.fileno())
            generated.extend(result);offset+=len(group)
        require(len(generated)==300,'Incomplete run')
        groups={'all300':generated,'previously_inspected20':[r for r in generated if r['previously_inspected_test']],'other280_not_claimed_pristine':[r for r in generated if not r['previously_inspected_test']]}
        save_new(run/(family+'_metrics.json'),{name:score_group(key,family,g,backend.summarize.__globals__['course']) for name,g in groups.items()})
        save_new(run/(family+'_OUTPUT_RECEIPT.json'),{'file_sha256':digest(path),'samples':300,'plan_sha256':a.plan_sha,'original_authorization_sha256':a.authorization_sha,'derived_policy_sha256':ph,'variant':generated[0]['evaluation_variant'],'boundary_policy':generated[0]['boundary_policy']})
    torch=engine.torch;del engine;gc.collect();torch.cuda.empty_cache()

def main():
    require(sys.flags.optimize==0,'Optimized Python is forbidden: frozen backend cache checks require asserts')
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);ap.add_argument('--authorization');ap.add_argument('--authorization-sha');ap.add_argument('--child-model');ap.add_argument('--control-sha');ap.add_argument('--gpu-lock-fd',type=int);a=ap.parse_args()
    plan,auth=authorize(B/'PLAN.json',a.plan_sha,a.authorization,a.authorization_sha)
    require(plan['model_keys']==['qwen','deepseek','llama'] and plan['new_training'] is False,'Fixed three-model evaluation required')
    require(plan['generation']=={'context_tokens':6144,'max_new_tokens':1920,'do_sample':False,'prefill_chunk_tokens':512,'batch_sizes':{'qwen':16,'deepseek':16,'llama':8}},'Unreviewed serving configuration')
    if a.child_model is not None:
        child(a,plan);return
    out=Path(plan['output_directory']);require(out.is_absolute() and not out.exists(),'Fresh output directory required')
    lock=(R/'setup/gpu_training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Another GPU worker is active')
        # E1: every test-independent check and CPU tokenizer load precedes mkdir/read.
        bindings,registry,tokenizers,backend,SYSTEM=preflight(plan,plan['model_keys'])
        out.mkdir(parents=True);save_new(out/'ORIGINAL_HUMAN_AUTHORIZATION.json',auth)
        pp,ph,aa,ah=hydrate_test_policy(B/'PLAN.json',a.plan_sha,a.authorization,a.authorization_sha,B/'SOURCE_POLICY_REMOTE.json',plan['dependency_sha256']['SOURCE_POLICY_REMOTE.json'],out)
        from split_target_contract import load_frozen_split
        for key in plan['model_keys']:
            rows,receipt=load_frozen_split(plan['test_source']['path'],split='test',policy_path=pp,policy_sha256=ph,authorization_path=aa,authorization_sha256=ah,requested_scope={'model_key':key,'run_id':plan['run_id']})
            save_new(out/(key+'_TEST_READ_RECEIPT.json'),receipt)
        control={'parent_pid':os.getpid(),'plan_sha256':a.plan_sha,'derived_policy_sha256':ph,'prepared_sha256':{},'prompt_binding_sha256':{}}
        for key in plan['model_keys']:
            tok=tokenizers[key];cohort=[];bound=bindings[key];info=registry[key];prompt_receipts=[]
            for row in rows:
                require(isinstance(row.get('answer'),str) and isinstance(row.get('label'),str) and bool(row['answer'].strip()) and bool(row['label'].strip()) and row['label'].lower().split()[0] in ['up','down'],'Nonempty reference/binary label required')
                text=re.sub(r'\[/INST\]\s*$','',row['prompt'].split('<</SYS>>',1)[-1].strip()).replace('cryptocurrency price movement','stock price movement')
                if tok.chat_template:ids=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
                else:ids=tok.encode(SYSTEM+'\n\n'+text+'\n\n',add_special_tokens=True)
                require(len(ids)+1920<=6144,'Full input exceeds hard context budget: abort before any GPU load, no clipping/filtering/tuning')
                cohort.append({**row,'single_row_batch_required':len(ids)>plan['validated_prompt_max_tokens'][key],'input_ids':ids,'reference':row['answer'].strip(),'prompt_audit':{'compressed_text':text,'compressed':False}})
                prompt_receipts.append({'sample_id':row['sample_id'],'model_id':info['model_id'],'revision':Path(info['cache']).name,'tokenizer_sha256':bound['cache_files_sha256']['tokenizer.json'],'tokenizer_config_sha256':bound['cache_files_sha256']['tokenizer_config.json'],'BOS':tok.bos_token_id,'EOS':tok.eos_token_id,'single_row_batch_required':len(ids)>plan['validated_prompt_max_tokens'][key],'prompt_tokens':len(ids),'prompt_ids_sha256':sha_bytes(json.dumps(ids).encode()),'previously_inspected_test':row['previously_inspected_test']})
            prepared=out/(key+'_PREPARED.json');save_new(prepared,cohort);control['prepared_sha256'][key]=digest(prepared)
            prompt_file=out/(key+'_PROMPT_BINDING.json');save_new(prompt_file,{'samples':prompt_receipts,'pad_policy':'existing EOS, masked left padding, no vocabulary resize'});control['prompt_binding_sha256'][key]=digest(prompt_file)
        control_file=out/'PARENT_CONTROL.json';save_new(control_file,control);ch=digest(control_file)
        for key in plan['model_keys']:
            # E2: independent OS process per model, parent holds the SAME lock throughout.
            command=[sys.executable,str(B/'final_test_worker.py'),'--plan-sha',a.plan_sha,'--authorization',str(Path(a.authorization).resolve()),'--authorization-sha',a.authorization_sha,'--child-model',key,'--control-sha',ch,'--gpu-lock-fd',str(lock.fileno())]
            with (out/(key+'_child.log')).open('x') as log:result=subprocess.run(command,pass_fds=(lock.fileno(),),stdout=log,stderr=subprocess.STDOUT)
            require(result.returncode==0,'Isolated child failed; preserve outputs, no automatic restart')
            require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Child GPU process did not terminate')
        save_new(out/'COMPLETED.json',{'outputs':1800,'plan_sha256':a.plan_sha,'test_read':True,'new_training':False,'publish':False,'submit':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        if out.exists():save_new(out/'FAILED.json',{'error_type':type(error).__name__,'no_resume_or_parameter_tuning':True,'review_existing_access_receipts':True})
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    os.environ.update(HF_HOME=str(R/'huggingface'),HF_ENDPOINT='https://huggingface.co',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(R/'tmp'),TOKENIZERS_PARALLELISM='false')
    main()
