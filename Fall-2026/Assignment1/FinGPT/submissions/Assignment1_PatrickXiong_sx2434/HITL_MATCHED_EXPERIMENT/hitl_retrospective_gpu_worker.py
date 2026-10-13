"""Serial locked validation inference on unchanged legacy adapters; no training/test."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

R = Path('/root/shared-nvme/AI4FinRuntime')
P = R / 'import_20261009/assignment1_20261008'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def direction_metrics(records, model_key):
    correct = valid = 0
    details = []
    for item in records:
        text = item['generated']
        predicted = None
        native_base = item['variant'].startswith('base_') and model_key == 'deepseek'
        closure_valid = text.count('</think>') == 1 and '<think>' not in text
        if not native_base or closure_valid:
            if closure_valid:
                text = text.rsplit('</think>', 1)[1]
            lines = re.findall(r'(?im)^\s*(?:-\s*)?Prediction\s*:\s*(.*)$', text.replace('*', '').replace('_', ''))
            if lines:
                directions = set('up' if m.lower() in {'up', 'increase', 'rise'} else 'down'
                                 for m in re.findall(r'\b(up|down|increase|decrease|rise|fall|decline)\b', lines[-1], re.I))
                if len(directions) == 1:
                    predicted = directions.pop()
        label = str(item['label']).lower()
        if label.startswith('up'):
            expected = 'up'
        elif label.startswith('down'):
            expected = 'down'
        else:
            raise ValueError('Unexpected validation direction label')
        valid += predicted is not None
        correct += predicted == expected
        details.append({'sample_id': item['sample_id'], 'predicted': predicted, 'expected': expected})
    return {'metric_variant': 'source_checked_final_direction_markdown_tolerant_v1',
            'total_outputs': len(records), 'valid_direction_outputs': valid,
            'correct_direction_count': correct, 'accuracy_over_all_outputs': correct / len(records),
            'accuracy_over_valid_outputs': correct / valid if valid else None,
            'native_deepseek_base_requires_generated_think_closure': True,
            'raw_output_modified': False, 'rows': details}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--plan-sha', required=True)
    ap.add_argument('--model', choices=['qwen', 'deepseek'], required=True)
    ap.add_argument('--bundle-manifest-sha', required=True)
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    out = bundle / ('run_' + args.model)
    out.mkdir(exist_ok=True)
    def status(stage, **fields):
        value = {'stage': stage, 'model_key': args.model, 'pid': os.getpid(),
                 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **fields}
        save(out / 'status.json', value)
        print(json.dumps(value), flush=True)
    gpu_lock = (R / 'setup/gpu_training.lock').open('a+')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        manifest_path = bundle / 'EXECUTION_BUNDLE_MANIFEST.json'
        if sha(manifest_path) != args.bundle_manifest_sha:
            raise ValueError('Externally pinned execution bundle changed')
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            if Path(name).name != name or sha(bundle / name) != expected:
                raise ValueError('Execution source dependency changed')
        plan_path = bundle / 'FROZEN_HITL_VALIDATION_PLAN.json'
        if sha(plan_path) != args.plan_sha:
            raise ValueError('Externally pinned plan SHA differs')
        plan = json.loads(plan_path.read_text())
        if bundle != Path(plan['remote_directory']).resolve() or plan['route'] != 'retrospective':
            raise ValueError('Wrong experiment directory/route')
        scope = json.loads((R / 'setup/submission_readiness_scope_20261011.json').read_text())
        if scope['experiment_repair_decision'] != 'retrospective_original_data' or args.model not in scope['active_models']:
            raise PermissionError('Current user scope does not cover retrospective validation')
        if plan['test_authorized'] or plan['test_read'] or plan['new_training']:
            raise PermissionError('This worker permits development validation only')
        for name, expected in plan['dependency_sha256'].items():
            if Path(name).name != name or sha(bundle / name) != expected:
                raise ValueError('Frozen experiment dependency changed')
        sys.path.insert(0, str(bundle))
        from split_target_contract import load_frozen_split, load_bound_tokenizer
        policy_path = bundle / 'SOURCE_POLICY_REMOTE.json'
        policy = json.loads(policy_path.read_text())
        known_source = policy['known_test_hash_source']
        if sha(known_source['path']) != known_source['sha256']:
            raise ValueError('Recorded test hash metadata source changed')
        metadata = json.loads(Path(known_source['path']).read_text())
        if metadata['splits']['test']['file_sha256'] != policy['known_test_sha256'][0]:
            raise ValueError('Known test hash metadata differs')
        sources, receipt = load_frozen_split(policy['splits']['validation']['path'], split='validation',
                                            policy_path=policy_path, policy_sha256=plan['source_policy_sha256'])
        save(out / 'VALIDATION_READ_RECEIPT.json', receipt)
        allowed = {r['sample_id'] for r in sources}
        spec = plan['models'][args.model]
        data_path = bundle / spec['data_file']
        if sha(data_path) != spec['data_sha256']:
            raise ValueError('Prepared data SHA differs')
        rows = json.loads(data_path.read_text())
        if [r['sample_id'] for r in rows] != plan['sample_ids'] or not set(plan['sample_ids']).issubset(allowed):
            raise ValueError('Prepared rows differ from authorized validation subset')
        control_path = bundle / spec['original_data_file']
        if sha(control_path) != spec['original_data_sha256']:
            raise ValueError('Original matched-control data SHA differs')
        control_rows = json.loads(control_path.read_text())
        if [r['sample_id'] for r in control_rows] != plan['sample_ids']:
            raise ValueError('Original matched-control case IDs differ')
        registry = json.loads((R / 'setup/three_model_runtime_registry_20261010.json').read_text())
        info = registry[args.model]
        runtime_bindings = json.loads((bundle / 'RUNTIME_SOURCE_BINDINGS.json').read_text())
        expected_runtime = runtime_bindings['models'][args.model]
        if Path(info['evaluation_data']).resolve() != Path(expected_runtime['evaluation_data']).resolve():
            raise PermissionError('Historical backend evaluation path is not approved validation')
        if sha(info['evaluation_data']) != expected_runtime['evaluation_sha256']:
            raise ValueError('Historical validation generation file changed')
        if Path(info['training_run']).resolve() != Path(expected_runtime['training_run']).resolve():
            raise PermissionError('Historical training run path differs')
        for name, expected in runtime_bindings['core_sources'].items():
            if Path(name).name != name or sha(P / name) != expected:
                raise ValueError('Historical runtime core source changed')
        cache = Path(info['cache'])
        if info['model_id'] != spec['model_id'] or cache.name != spec['revision']:
            raise ValueError('Runtime model identity/revision differs')
        selected = json.loads((Path(info['training_run']) / 'selected_checkpoint.json').read_text())
        checkpoint = Path(selected['checkpoint'])
        if checkpoint.name != 'checkpoint-248' or sha(checkpoint / 'adapter_model.safetensors') != spec['adapter_sha256']:
            raise ValueError('Original selected adapter changed')
        status('checking_existing_weight_hashes')
        verified = {}
        for name, expected in spec['weight_files'].items():
            path = cache / name
            if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('Existing official weight identity differs; preserved for diagnosis')
            verified[name] = expected['sha256']
        index = json.loads((cache / 'model.safetensors.index.json').read_text())
        if set(index['weight_map'].values()) != set(verified):
            raise ValueError('Weight index coverage differs')
        tok = load_bound_tokenizer(cache, model_key=args.model, specification=spec['tokenizer_binding'])
        save(out / 'MODEL_IDENTITY_RECEIPT.json', {'model_id': spec['model_id'], 'revision': spec['revision'],
             'weight_sha256': verified, 'adapter_sha256': spec['adapter_sha256'], 'checkpoint': str(checkpoint),
             'tokenizer_binding': tok.binding_receipt, 'weights_downloaded': False, 'adapter_changed': False})
        filename = 'backend_server_batched_evaluate.py' if args.model == 'qwen' else 'backend_server_chunked_prefill_evaluate.py'
        module_spec = importlib.util.spec_from_file_location('historical_backend_' + args.model, bundle / filename)
        backend = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(backend)
        if hashlib.sha256(backend.SYSTEM.encode()).hexdigest() != plan['system_sha256']:
            raise ValueError('Original backend SYSTEM differs')
        for row in rows + control_rows:
            if row['previously_inspected_test'] is not False:
                raise ValueError('Invalid validation exposure flag')
            actual = tok.apply_chat_template([{'role': 'system', 'content': backend.SYSTEM},
                                             {'role': 'user', 'content': row['prompt_audit']['compressed_text']}],
                                            tokenize=True, add_generation_prompt=True)
            if actual != row['input_ids'] or len(actual) + plan['max_new_tokens'] > plan['max_context_tokens']:
                raise ValueError('Prepared/actual prompt or budget differs')
        compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if compute.strip():
            raise RuntimeError('Other GPU compute process present')
        config = {'plan_sha256': args.plan_sha, 'model_key': args.model, 'model_id': spec['model_id'],
                  'revision': spec['revision'], 'selected_checkpoint': str(checkpoint), 'samples': len(rows),
                  'batch_size': spec['batch_size'], 'prefill_chunk_tokens': spec['prefill_chunk_tokens'],
                  'max_new_tokens': plan['max_new_tokens'], 'context_cap': plan['max_context_tokens'],
                  'quantization_bits': 8, 'variant_scope': plan['variants'], 'prompt_effect_only': True,
                  'new_training': False, 'test_read': False, 'data_availability_verified': False,
                  'legacy_target_confound_repaired': False, 'speed_comparison_claim': False}
        if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
            raise ValueError('Existing run config differs')
        save(out / 'config.json', config)
        status('loading_cached_model')
        engine = backend.Engine(args.model)
        all_outputs = {}
        for variant_index, (kind, prompt_kind) in enumerate([('base', 'original'), ('base', 'revised'), ('adapter', 'original'), ('adapter', 'revised')]):
            if kind == 'adapter' and prompt_kind == 'original':
                engine.adapter()
            variant_rows = control_rows if prompt_kind == 'original' else rows
            record_key = kind + '_' + prompt_kind
            path = out / (record_key + '_outputs.jsonl')
            records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            if [r['sample_id'] for r in records] != [r['sample_id'] for r in variant_rows[:len(records)]]:
                raise ValueError('Resume output identity differs')
            while len(records) < len(variant_rows):
                group = variant_rows[len(records):len(records) + spec['batch_size']]
                def beat(tokens=0, seconds=0):
                    status('generating_' + record_key, completed=len(records), total=len(variant_rows),
                           current_generated_tokens=tokens, elapsed_seconds=seconds)
                result = engine.generate(group, plan['max_new_tokens'], beat)
                for item, row in zip(result, group):
                    ids = item['generated_token_ids']
                    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != item['generated_token_sha256']:
                        raise ValueError('Generated token receipt differs')
                    if item['peak_allocated_gib'] > engine.guard:
                        raise RuntimeError('Allocated-memory guard exceeded')
                    if item['peak_reserved_gib'] > engine.torch.cuda.get_device_properties(0).total_memory / 2**30 - .1:
                        raise RuntimeError('Reserved-memory guard exceeded')
                    item['variant'] = plan['variants'][variant_index]
                    item['boundary'] = 'generated'
                    family = 'base_native_reasoning' if kind == 'base' and args.model == 'deepseek' else ('base_native' if kind == 'base' else 'adapter_legacy')
                    item['native_variant_family'] = family
                    item['generated_boundary_receipt'] = {
                        'model_key': args.model, 'model_id': spec['model_id'], 'revision': spec['revision'],
                        'variant': family, 'method_variant': item['variant'], 'boundary': item['boundary'], 'think_closure_prefilled': False,
                        'description': 'Only new tokens after the frozen native assistant prefix; raw generated text preserved.',
                        'prompt_token_sha256': hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest(),
                        'generated_token_sha256': item['generated_token_sha256'],
                        'bos_token_id': tok.bos_token_id, 'eos_token_id': tok.eos_token_id,
                        'tokenizer_sha256': tok.binding_receipt['file_sha256'],
                        'closure_is_not_eos': True, 'eos_observed': item['ended_eos'],
                        'think_closed_in_generated_ids': 128014 in ids if args.model == 'deepseek' else None,
                    }
                with path.open('a') as stream:
                    stream.write(''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in result))
                    stream.flush()
                    os.fsync(stream.fileno())
                records.extend(result)
                status('generating_' + record_key, completed=len(records), total=len(variant_rows))
                if (out / 'stop_after_checkpoint.flag').exists():
                    raise RuntimeError('Requested boundary stop; records preserved')
            all_outputs[record_key] = records
            save(out / (record_key + '_course_raw_metrics.json'), backend.summarize(records))
            save(out / (record_key + '_full_denominator_direction.json'), direction_metrics(records, args.model))
        original = json.loads((bundle / 'ORIGINAL_PROMPT_BASELINES.json').read_text())[args.model]
        save(out / 'before_after_course_raw.json', {
            'scope': '6 selected validation cases; feedback adaptation; not a representative accuracy study.',
            'historical_original_prompt_base_context_only': backend.summarize(original['base']),
            'historical_original_prompt_adapter_context_only': backend.summarize(original['adapter']),
            'matched_original_prompt_base': backend.summarize(all_outputs['base_original']),
            'matched_original_prompt_adapter_legacy': backend.summarize(all_outputs['adapter_original']),
            'revised_prompt_base': backend.summarize(all_outputs['base_revised']),
            'revised_prompt_adapter_legacy': backend.summarize(all_outputs['adapter_revised']),
            'qualitative_effect_review_pending': True, 'new_training': False, 'test_read': False,
        })
        status('completed', matched_base_outputs=len(all_outputs['base_original']), revised_base_outputs=len(all_outputs['base_revised']),
               matched_adapter_outputs=len(all_outputs['adapter_original']), revised_adapter_outputs=len(all_outputs['adapter_revised']),
               ground_quality_improvement_claimed=False, new_training=False, test_read=False)
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, message='Preparation or generation failed; outputs and evidence preserved.')
        raise
    finally:
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        gpu_lock.close()

if __name__ == '__main__':
    os.environ['HF_HOME'] = str(R / 'huggingface')
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TMPDIR'] = str(R / 'tmp')
    os.environ['TORCH_HOME'] = str(R / 'torch_cache')
    os.environ['XDG_CACHE_HOME'] = str(R / 'runtime_cache')
    os.environ['TRITON_CACHE_DIR'] = str(R / 'tmp/triton_cache')
    main()
