"""Derive a locked Llama run from the preserved historical training backend."""
import hashlib, json, pathlib, shutil

B = pathlib.Path(__file__).resolve().parent
R = pathlib.PurePosixPath('/root/shared-nvme/AI4FinRuntime')
RB = R / 'setup' / B.name
def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Historical source anchor differs: ' + old[:70])
    return text.replace(old, new)

source = (B/'original_server_full_int8_train.py').read_text()
source = replace_once(source, "ROOT=Path('/root/shared-nvme/AI4FinRuntime/import_20261009/assignment1_20261008')", "ROOT=Path(__file__).resolve().parent")
source = replace_once(source, "DATA=Path('/root/shared-nvme/AI4FinRuntime/setup/full_int8_4096_20261010')", "DATA=Path('/root/shared-nvme/AI4FinRuntime/setup/llama_retrospective_full_4096_20261012')")
source = replace_once(source, "    assert a.model=='qwen'", "    if a.model!='llama': raise ValueError('This frozen run is Llama BASE only')")
source = replace_once(source, "trpath=DATA/'qwen_train_4096.json';vapath=DATA/'qwen_validation_4096.json'", "trpath=DATA/'llama_train_4096.json';vapath=DATA/'llama_validation_4096.json'")
source = replace_once(source, "proto=json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))", "proto=json.loads((ROOT/'FROZEN_TRAINING_PLAN.json').read_text(encoding='utf-8'))")
source = replace_once(source, "cache=Path('/root/shared-nvme/AI4FinRuntime/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')", "cache=Path('/root/shared-nvme/AI4FinRuntime/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b')")
source = replace_once(source, 'try:code=main()', 'try:\n        from training_preflight import locked_run\n        code=locked_run(main)')
for name,text in [('llama_full_int8_train.py',source)]:
    if (B/name).exists(): raise ValueError('Preserve existing derived source')
    (B/name).write_text(text,encoding='utf-8')
W=B.parent
shutil.copyfile(W/'code_contract_v3/split_target_contract.py', B/'split_target_contract.py')
shutil.copyfile(W/'hitl_retrospective_evidence_prompt_v2_20261012/SOURCE_POLICY_REMOTE.json', B/'SOURCE_POLICY_REMOTE.json')
shutil.copyfile(W/'llama_base_preparation/20261012T044358Z/llama_cache_verified.json', B/'CACHE_VERIFIED_SNAPSHOT.json')
shutil.copyfile(W/'llama_base_preparation/20261012T044358Z/LLAMA_PREPARATION_VERIFIED.json', B/'PREPARATION_VERIFIED_SNAPSHOT.json')
plan={
 'run_id':B.name,'route':'retrospective','model_key':'llama','model_id':'meta-llama/Llama-3.1-8B',
 'revision':'d04e592bb4f6aa9cfee91e2e20afa771667e1d4b','remote_bundle':str(RB),
 'output_directory':str(R/'import_20261009/assignment1_20261008/runs/server4090_llama_base_int8_4096_epoch_retrospective_20261012'),
 'training_allowed':True,'user_route_statement':'先retrospective吧','user_access_confirmed':True,
 'max_length':4096,'samples_train':990,'samples_validation':240,'max_steps':248,
 'microbatch':1,'gradient_accumulation':4,'learning_rate':2e-4,'warmup_ratio':.03,'scheduler':'cosine',
 'lora_r':8,'lora_alpha':16,'lora_dropout':.05,'target_modules':['q_proj','v_proj'],'seed':42,
 'quantization_bits':8,'target_policy':'full_reference_plus_eos','native_base_without_chat_template':True,
 'boundary':'source SYSTEM + two newlines + canonical full input + two newlines, native BOS',
 'pad_policy':'existing EOS, masked padding, no embedding resize',
 'source_policy_sha256':sha(B/'SOURCE_POLICY_REMOTE.json'),
 'prepared_sha256':{'train':'94464beb45103fa026ef23eb9c4d2a62b16e90a0dc6824e2414e37f7635d1df5','validation':'71828dd44e6a325bedb46b3a72d3a0d5864d1045cb2fb3905ea7b7969d5ff32d'},
 'selection':'minimum full validation token-weighted CE at steps 124/248; earlier on tie',
 'expected_versions':{'torch':'2.7.0a0+7c8ec84dab.nv25.3','transformers':'4.51.3','peft':'0.15.2','accelerate':'1.7.0','bitsandbytes':'0.47.0','datasets':'3.6.0'},
 'minimum_free_bytes':1073741824,'validation_only_selection':True,'test_read':False,'test_authorized':False,
 'publish_authorized':False,'submit_authorized':False,'source_data_repaired':False,'reference_content_repaired':False,
 'comparability_limit':'Llama is BASE with native plain prefix; Qwen/DeepSeek use their native instruction templates. Same training settings do not remove model-family differences.',
 'historical_backend_sha256':sha(B/'original_server_full_int8_train.py'),
 'dependency_sha256':{n:sha(B/n) for n in ['llama_full_int8_train.py','training_core.py','training_preflight.py','split_target_contract.py','SOURCE_POLICY_REMOTE.json','CACHE_VERIFIED_SNAPSHOT.json','PREPARATION_VERIFIED_SNAPSHOT.json']}
}
plan['frozen_at_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
(B/'FROZEN_TRAINING_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'plan_sha256':sha(B/'FROZEN_TRAINING_PLAN.json'),'directory':str(B)}))
