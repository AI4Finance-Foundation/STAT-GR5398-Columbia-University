"""Prepare validation only: full evidence, common ordering, no reference-based budget."""
import hashlib, json, pathlib, re, sys
from transformers import AutoTokenizer
R = pathlib.Path('/root/shared-nvme/AI4FinRuntime')
P = R / 'import_20261009/assignment1_20261008'
sys.path.insert(0, str(P))
from data_processing import SYSTEM
O = R / 'setup/three_model_validation240_20261010'
O.mkdir(exist_ok=True)
registry_path = R / 'setup/three_model_runtime_registry_20261010.json'
if registry_path.exists():
    registry = json.loads(registry_path.read_text())
else:
    registry = {
      'qwen': {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'cache': str(R/'huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28')},
      'deepseek': {'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'cache': str(R/'huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1')},
      'llama': {'model_id': 'meta-llama/Llama-3.1-8B', 'cache': None}}
verified = R/'setup/llama_cache_verified.json'
if verified.exists():
    info = json.loads(verified.read_text())
    assert info['model_id'] == registry['llama']['model_id'] and info['verified']
    registry['llama']['cache'] = info['cache']
raw = json.loads((P/'data/validation_raw.json').read_text())
assert len(raw) == 240 and all(x['split'] == 'validation' for x in raw)
ids = [x['sample_id'] for x in json.loads((P/'data/validation_generation30.json').read_text())]
lookup = {x['sample_id']: x for x in raw}
assert len(lookup) == 240 and len(ids) == len(set(ids)) == 30
ordered = [lookup[i] for i in ids] + sorted([x for x in raw if x['sample_id'] not in ids], key=lambda x: (-len(x['prompt']), x['sample_id']))
canonical = []
for x in ordered:
    text = re.sub(r'\[/INST\]\s*$', '', x['prompt'].split('<</SYS>>', 1)[-1].strip()).replace('cryptocurrency price movement', 'stock price movement')
    canonical.append({'sample_id': x['sample_id'], 'symbol': x['symbol'], 'period': x['period'], 'label': x['label'], 'reference': x['answer'].strip(), 'input': text})
def save_once(path, value):
    if path.exists(): assert json.loads(path.read_text()) == value, str(path)
    else: path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
save_once(O/'canonical_validation240.json', canonical)
summaries = {}
for key, cfg in registry.items():
    cfg['training_run'] = str(P/f'runs/server4090_{key}_int8_4096_epoch_20261010')
    if not cfg['cache']: continue
    tok = AutoTokenizer.from_pretrained(cfg['cache'], local_files_only=True)
    rows = []
    for x in canonical:
        if tok.chat_template:
            tokens = tok.apply_chat_template([{'role':'system','content':SYSTEM}, {'role':'user','content':x['input']}], tokenize=True, add_generation_prompt=True)
        else:
            tokens = tok.encode(SYSTEM+'\n\n'+x['input']+'\n\n', add_special_tokens=True)
        assert len(tokens)+1920 <= 6144, (key, x['sample_id'], len(tokens))
        rows.append({k:v for k,v in x.items() if k!='input'} | {'input_ids':tokens, 'prompt_audit':{'compressed_text':x['input'], 'prompt_tokens':len(tokens), 'policy':'Full input; first frozen 30 followed by remaining validation rows ordered by input length; no label used in ordering'}})
    dest = O/f'{key}_validation_generation240_full.json'
    save_once(dest, rows)
    cfg['evaluation_data'] = str(dest)
    summaries[key] = {'samples':240, 'max_prompt_tokens':max(len(x['input_ids']) for x in rows), 'input_truncated_count':0, 'max_new_tokens':1920, 'total_context_cap':6144, 'test_used':False, 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
registry_path.write_text(json.dumps(registry, indent=2))
(O/'preflight_summary.json').write_text(json.dumps(summaries, indent=2))
print(json.dumps(summaries, indent=2))
