"""Run only after training completes; freeze actual selected adapter before generation."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
R=Path('/root/shared-nvme/AI4FinRuntime');B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=B/'FROZEN_VALIDATION_PLAN.json'
    if p.exists():raise ValueError('Do not refreeze after generation')
    train=R/'import_20261009/assignment1_20261008/runs/server4090_llama_base_int8_4096_epoch_retrospective_20261012';status=json.loads((train/'status.json').read_text())
    if status['stage']!='completed' or status['completed_steps']!=248 or status['seen_microbatches']!=990:raise ValueError('Full training not complete')
    selected=json.loads((train/'selected_checkpoint.json').read_text());marker=json.loads((R/'setup/llama_cache_verified.json').read_text());prep=R/'setup/llama_retrospective_full_4096_20261012';proof=json.loads((prep/'LLAMA_PREPARATION_VERIFIED.json').read_text());data=prep/'llama_validation_generation240_full.json'
    plan={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'route':'retrospective','model_id':marker['model_id'],'revision':marker['revision'],'cache':marker['cache'],'cache_verified':marker,'tokenizer_binding':proof['tokenizer_binding'],'training_run':str(train),'selected_checkpoint':selected,'adapter_sha256':sha(Path(selected['checkpoint'])/'adapter_model.safetensors'),'evaluation_data':str(data),'evaluation_data_sha256':sha(data),'output_directory':str(R/'import_20261009/assignment1_20261008/runs/server4090_llama_base_int8_validation240_gen1920_retrospective_20261012'),'samples':240,'max_new_tokens':1920,'context_tokens':6144,'batch_size':8,'prefill_chunk_tokens':512,'do_sample':False,'quantization_bits':8,'target_policy':'full_reference_plus_eos','prompt_policy':'Native BASE: source SYSTEM + double newline + canonical full input + double newline, native BOS; identical training/inference prefix; no clipping','source_policy_sha256':sha(B/'SOURCE_POLICY_REMOTE.json'),'test_read':False,'test_authorized':False,'data_repaired':False,'speed_comparison_claim':False,'dependency_sha256':{n:sha(B/n) for n in ['llama_validation_worker.py','backend_server_chunked_prefill_evaluate.py','split_target_contract.py','SOURCE_POLICY_REMOTE.json','RUNTIME_SOURCE_BINDINGS.json']}}
    p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'plan_sha256':sha(p),'selected_step':selected['step'],'adapter_sha256':plan['adapter_sha256']}))
if __name__=='__main__':main()
