"""Fail closed before GPU loading; keep one lock through the entire training run."""
import fcntl,hashlib,importlib.metadata,json,os,re,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

R=Path('/root/shared-nvme/AI4FinRuntime'); B=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def check(value,message):
    if not value:raise ValueError(message)
def locked_run(run):
    lock=(R/'setup/gpu_training.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(sha(B/'FROZEN_TRAINING_PLAN.json')==os.environ.get('FINGPT_TRAINING_PLAN_SHA'),'External plan SHA differs')
        plan=json.loads((B/'FROZEN_TRAINING_PLAN.json').read_text())
        check(B==Path(plan['remote_bundle']).resolve(),'Bundle path differs')
        check(sys.argv==[str(B/'llama_full_int8_train.py'),'--model','llama','--out',plan['output_directory']],'Launch arguments differ')
        out=Path(plan['output_directory']);check(not out.exists(),'Fresh output directory required')
        for n,s in plan['dependency_sha256'].items():check(Path(n).name==n and sha(B/n)==s,'Source dependency differs')
        scope=json.loads((R/'setup/submission_readiness_scope_20261011.json').read_text())
        check(scope['experiment_repair_decision']=='retrospective_original_data' and scope['llama_training_allowed'] and scope['llama_access_user_confirmed'] and 'llama' in scope['active_models'],'User scope differs')
        check(not plan['test_read'] and not plan['test_authorized'],'Train/validation only')
        versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
        check(versions==plan['expected_versions'],'Runtime versions differ')
        check(shutil.disk_usage(R).free>=plan['minimum_free_bytes'],'Insufficient output headroom')
        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        check(not compute.strip(),'Other GPU compute process present')
        marker=json.loads((B/'CACHE_VERIFIED_SNAPSHOT.json').read_text());prep=json.loads((B/'PREPARATION_VERIFIED_SNAPSHOT.json').read_text())
        cache=Path(marker['cache']);check(marker['verified'] and marker['model_id']==plan['model_id'] and cache.name==plan['revision'],'Cache identity differs')
        print(json.dumps({'stage':'preflight_weight_hashes','pid':os.getpid()}),flush=True)
        for n,s in {**marker['weight_sha256'],**marker['nonweight_sha256']}.items():check(sha(cache/n)==s,'Existing verified cache file changed')
        index=json.loads((cache/'model.safetensors.index.json').read_text());check(set(index['weight_map'].values())==set(marker['weight_sha256']),'Index coverage differs')
        from split_target_contract import load_bound_tokenizer,load_frozen_split,encode_full_target
        tok=load_bound_tokenizer(cache,model_key='llama',specification=prep['tokenizer_binding'])
        check(not tok.chat_template,'Native BASE unexpectedly has chat template')
        policy_path=B/'SOURCE_POLICY_REMOTE.json';policy=json.loads(policy_path.read_text());known=policy['known_test_hash_source']
        check(sha(known['path'])==known['sha256'],'Recorded metadata changed')
        check(json.loads(Path(known['path']).read_text())['splits']['test']['file_sha256'] in policy['known_test_sha256'],'Known test hash metadata differs')
        # The exact SYSTEM is pinned by the previous executed HITL plan.
        old=R/'setup/hitl_retrospective_evidence_prompt_v2_20261012'
        bindings=json.loads((B/'RUNTIME_SOURCE_BINDINGS.json').read_text())
        source_root=R/'import_20261009/assignment1_20261008'
        check(sha(source_root/'data_processing.py')==bindings['core_sources']['data_processing.py'],'SYSTEM source changed')
        sys.path.insert(0,str(source_root));from data_processing import SYSTEM
        receipts={};data_dir=R/'setup/llama_retrospective_full_4096_20261012'
        for split in ['train','validation']:
            path=data_dir/f'llama_{split}_4096.json';check(sha(path)==plan['prepared_sha256'][split],'Prepared data changed')
            prepared=json.loads(path.read_text());rows,receipt=load_frozen_split(policy['splits'][split]['path'],split=split,policy_path=policy_path,policy_sha256=plan['source_policy_sha256'])
            check(len(rows)==len(prepared)==plan['samples_'+split],'Row count differs')
            for row,p in zip(rows,prepared):
                check(row['sample_id']==p['sample_id'] and p['previously_inspected_test'] is False,'Prepared identity differs')
                text=re.sub(r'\[/INST\]\s*$','',row['prompt'].split('<</SYS>>',1)[-1].strip()).replace('cryptocurrency price movement','stock price movement')
                ids=tok.encode(SYSTEM+'\n\n'+text+'\n\n',add_special_tokens=True)
                encoded,audit=encode_full_target(tok,model_key='llama',prompt_ids=ids,answer=row['answer'],context_tokens=4096,target_policy=plan['target_policy'])
                check(all(encoded[k]==p[k] for k in ['input_ids','attention_mask','labels']) and audit==p['target_audit'],'Full target recheck differs')
            receipts[split]=receipt
        out.mkdir(parents=True)
        receipt={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':os.environ['FINGPT_TRAINING_PLAN_SHA'],'versions':versions,'weight_sha256':marker['weight_sha256'],'tokenizer_binding':tok.binding_receipt,'read_receipts':receipts,'all_full_targets_rechecked':1230,'test_read':False,'lock_held':True,'free_bytes':shutil.disk_usage(R).free}
        (out/'GPU_PREFLIGHT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({'stage':'preflight_passed','full_targets':1230,'test_read':False}),flush=True)
        return run()
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
