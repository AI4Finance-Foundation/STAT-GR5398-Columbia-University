"""CPU verification of the completed authorized run. Never starts GPU work.

Recomputes frozen metric functions, checks raw receipts and complete identities.
This is Codex mechanical verification, not an independent Claude audit.
"""
import argparse, ast, contextlib, csv, hashlib, io, json, math, os, re, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

W = Path(r'D:\AI4FinRuntime\assignment1_three_model_submission_work')
B = W / 'prospective_final_test_v4_20261012'
PLAN_SHA = 'f284b5fb9c06a970990d2557057d3ed0036bf4441c8811eb709479bb5d2c5339'
AUTH_SHA = '56296cabe16f4b21d93cd81a091399839fb6560fbe95aa081277b6723534d24a'
COURSE = Path(r'D:\AI4FinRuntime\assignment1_two_model_delivery\delivery_ready_20261011\training_code_adapters\import_20261009\assignment1_20261008\source_snapshot\course_utils.py')
INPUT_SOURCE = COURSE.parent.parent / 'data_processing.py'
GROUPS = ['all300', 'previously_inspected20', 'other280_not_claimed_pristine']

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def save_new(p, value):
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')

def equal(a, b, location='metric'):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), location + ' keys differ')
        for k in a: equal(a[k], b[k], location + '.' + k)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), location + ' length differs')
        for n, (x, y) in enumerate(zip(a, b)): equal(x, y, location + '.' + str(n))
    elif isinstance(a, float):
        require(isinstance(b, (float, int)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), location + ' number differs')
    else: require(type(a) == type(b) and a == b, location + ' value differs')

def selected_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in tree.body} == set(names), 'Frozen functions absent')
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace

def main():
    os.environ.update(HF_HOME=str(W/'local_offline_hf_audit'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True); a = ap.parse_args(); run = a.run.resolve()
    require(sha(B / 'PLAN.json') == PLAN_SHA, 'Pinned plan changed'); plan = load(B / 'PLAN.json')
    for name, digest in plan['dependency_sha256'].items(): require(sha(B / name) == digest, 'Pinned dependency changed')
    completion = load(run / 'COMPLETED.json')
    require(not (run / 'FAILED.json').exists() and completion['outputs'] == 1800 and completion['plan_sha256'] == PLAN_SHA and completion['test_read'] is True, 'Run not complete')
    manifest = load(run / 'COMPLETED_FILES_SHA256.json')
    for name, digest in manifest['files'].items():
        p = (run / name).resolve(); require(p.is_relative_to(run) and sha(p) == digest, 'Collected manifest differs')
    auth = run / 'ORIGINAL_HUMAN_AUTHORIZATION.json'; require(sha(auth) == AUTH_SHA, 'Human authorization differs')
    authorization = load(auth); first = load(run / 'INITIAL_AUTHORIZED_TEST_READ.json')
    require(first['actual_sha256'] == plan['test_source']['sha256'] and first['plan_sha256'] == PLAN_SHA and first['authorization_sha256'] == AUTH_SHA, 'Initial read differs')
    require(first['source_user_message'] == authorization['source_user_message'] == '赶紧收尾完工了', 'Human message differs')
    freeze = datetime.fromisoformat(plan['frozen_at_utc']); authorized = datetime.fromisoformat(authorization['authorized_at_utc']); accessed = datetime.fromisoformat(first['read_at_utc'])
    require(freeze <= authorized <= accessed, 'Chronology differs')
    original_source = run / 'AUDIT_TEST_SOURCE_BYTES.json'
    require(sha(original_source) == plan['test_source']['sha256'], 'Exact authorized test bytes differ')
    original_rows = load(original_source)
    require(len(original_rows) == 300 and sum(r['previously_inspected_test'] for r in original_rows) == 20, 'Exact source exposure differs')
    control = load(run / 'PARENT_CONTROL.json'); bindings = load(B / 'MODEL_BINDINGS.json')
    require(control['plan_sha256'] == PLAN_SHA and sha(run / 'HYDRATED_SOURCE_POLICY.json') == control['derived_policy_sha256'], 'Hydrated policy differs')
    require(sha(COURSE) == plan['runtime_core_sha256']['source_snapshot/course_utils.py'], 'Course parser changed')
    require(sha(INPUT_SOURCE) == plan['runtime_core_sha256']['data_processing.py'], 'Input source changed')
    tree = ast.parse(INPUT_SOURCE.read_text(encoding='utf-8'))
    system = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SYSTEM' for t in n.targets))
    from transformers import AutoTokenizer
    from sklearn.metrics import accuracy_score, mean_squared_error
    from rouge_score import rouge_scorer
    course = selected_functions(COURSE, ['parse_answer', 'calc_rouge_score', 'calc_metrics'], {'re':re, 'defaultdict':defaultdict, 'accuracy_score':accuracy_score, 'mean_squared_error':mean_squared_error, 'rouge_scorer':rouge_scorer})
    frozen = selected_functions(B / 'final_test_worker.py', ['require', 'final_text', 'common_direction', 'score_group', 'generation_groups'], {'re':re, 'statistics':statistics})
    course = SimpleNamespace(**{k: course[k] for k in ['parse_answer', 'calc_rouge_score', 'calc_metrics']})
    summary = []; cohort_identity = None; metric_reproductions = {}
    for key in plan['model_keys']:
        require(sha(run / (key + '_PREPARED.json')) == control['prepared_sha256'][key], 'Prepared hash differs')
        require(sha(run / (key + '_PROMPT_BINDING.json')) == control['prompt_binding_sha256'][key], 'Prompt binding changed')
        cohort = load(run / (key + '_PREPARED.json')); prompts = load(run / (key + '_PROMPT_BINDING.json'))['samples']
        require(len(cohort) == len(original_rows) and all(all(prepared[field] == value for field, value in original.items()) for prepared, original in zip(cohort, original_rows)), 'Prepared raw source fields differ from committed bytes')
        identity = [(r['sample_id'], r['symbol'], r['period'], r['reference'], r['label'], r['previously_inspected_test']) for r in cohort]
        require(len(identity) == 300 and len({r[0] for r in identity}) == 300 and sum(r[-1] for r in identity) == 20, 'Cohort/exposure differs')
        if cohort_identity is None: cohort_identity = identity
        require(identity == cohort_identity, 'Cross-model cohort differs')
        require([r['sample_id'] for r in prompts] == [r[0] for r in identity], 'Prompt order differs')
        schedule = frozen['generation_groups'](cohort, plan['generation']['batch_sizes'][key]); batches = [len(g) for g in schedule for r in g]
        read = load(run / (key + '_TEST_READ_RECEIPT.json'))
        require(read['samples'] == 300 and read['sha256'] == plan['test_source']['sha256'] and read['scope'] == {'model_key':key, 'run_id':plan['run_id']}, 'Scoped access differs')
        bound = bindings[key]
        tokenizer_copy = run / 'AUDIT_TOKENIZER_COPIES' / key
        for file in tokenizer_copy.iterdir():
            require(file.is_file() and file.name in bound['cache_files_sha256'] and sha(file) == bound['cache_files_sha256'][file.name], 'Copied native tokenizer/config differs')
        tok = AutoTokenizer.from_pretrained(str(tokenizer_copy), local_files_only=True)
        for source, prompt in zip(cohort, prompts):
            text = re.sub(r'\[/INST\]\s*$', '', source['prompt'].split('<</SYS>>', 1)[-1].strip()).replace('cryptocurrency price movement', 'stock price movement')
            ids = tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True) if tok.chat_template else tok.encode(system+'\n\n'+text+'\n\n',add_special_tokens=True)
            require(ids == source['input_ids'] and prompt['BOS'] == tok.bos_token_id and prompt['EOS'] == tok.eos_token_id, 'Actual native prefix/BOS/EOS differs')
        for family in ['base', 'adapter']:
            variant = ('base_native_reasoning' if key == 'deepseek' else 'base_native') if family == 'base' else ('adapter_legacy_unclosed' if key == 'deepseek' else 'adapter_original_full_reference')
            path = run / key / (family + '_outputs.jsonl'); raw = path.read_bytes(); require(raw.endswith(b'\n'), 'Partial JSONL tail')
            rows = [json.loads(line) for line in raw.splitlines()]
            receipt = load(run / key / (family + '_OUTPUT_RECEIPT.json'))
            require(receipt['file_sha256'] == sha(path) and receipt['samples'] == 300 and receipt['variant'] == variant and receipt['plan_sha256'] == PLAN_SHA and receipt['original_authorization_sha256'] == AUTH_SHA, 'Output receipt differs')
            require([(r['sample_id'], r['symbol'], r['period'], r['reference'], r['label'], r['previously_inspected_test']) for r in rows] == identity, 'Raw output identities differ')
            for r, source, prompt, batch in zip(rows, cohort, prompts, batches):
                ids = r['generated_token_ids']; eos = bool(ids and ids[-1] == prompt['EOS'])
                require(r['evaluation_variant'] == variant and r['plan_sha256'] == PLAN_SHA and r['batch_size'] == batch, 'Serving/variant differs')
                require(type(r['previously_inspected_test']) is bool and type(r['single_row_batch_required']) is bool, 'Boolean flags missing')
                require(r['single_row_batch_required'] == source['single_row_batch_required'] == (len(source['input_ids']) > plan['validated_prompt_max_tokens'][key]), 'Length fallback differs')
                require(r['prompt_tokens'] == len(source['input_ids']) == prompt['prompt_tokens'] and r['prompt_tokens'] + 1920 <= 6144, 'Prompt budget differs')
                require(prompt['prompt_ids_sha256'] == hashlib.sha256(json.dumps(source['input_ids']).encode()).hexdigest(), 'Input token hash differs')
                require(prompt['model_id'] == bound['registry']['model_id'] and prompt['revision'] == Path(bound['registry']['cache']).name and prompt['tokenizer_sha256'] == bound['cache_files_sha256']['tokenizer.json'] and prompt['tokenizer_config_sha256'] == bound['cache_files_sha256']['tokenizer_config.json'], 'Model/tokenizer binding differs')
                require(r['generated_token_sha256'] == hashlib.sha256(json.dumps(ids).encode()).hexdigest() and r['generated_tokens'] == len(ids) <= 1920, 'Generated tokens differ')
                require(r['generated'] == tok.decode(ids, skip_special_tokens=True), 'Actual native tokenizer decode differs')
                require(r['ended_eos'] == eos and r['hit_generation_limit'] == (len(ids) == 1920 and not eos), 'EOS/cap differs')
                require(r['closed_think_markers'] == r['generated'].count('</think>'), 'Boundary marker count differs')
                require(r['boundary_receipt'] == {**prompt, 'variant':variant, 'closed_think_markers':r['closed_think_markers'], 'natural_eos':eos, 'generation_limit':r['hit_generation_limit']}, 'Complete boundary receipt differs')
                require(0 <= r['peak_allocated_gib'] <= 21 and r['peak_reserved_gib'] < 24, 'Memory evidence differs')
            cohorts = {'all300':rows, 'previously_inspected20':[r for r in rows if r['previously_inspected_test']], 'other280_not_claimed_pristine':[r for r in rows if not r['previously_inspected_test']]}
            with contextlib.redirect_stdout(io.StringIO()): reproduced = {name:frozen['score_group'](key, family, subset, course) for name, subset in cohorts.items()}
            actual = load(run / key / (family + '_metrics.json')); equal(reproduced, actual)
            metric_reproductions[key + '_' + family] = reproduced
            for group in GROUPS:
                result = reproduced[group]; c = result['course_raw']; d = result['common_final_line']; subset = cohorts[group]
                summary.append({'model':key, 'family':family, 'evaluation_variant':variant, 'group':group, 'n':len(subset), 'course_valid':c['valid_count'], 'course_mse':c.get('mse'), 'constant_plus_half_mse_same_course_valid_subset':result['constant_plus_half_mse_same_course_valid_subset'], 'common_direction_correct':d['correct'], 'common_direction_valid':d['valid'], 'balanced_accuracy':d['balanced_accuracy'], 'always_up_correct':result['always_up_correct'], 'EOS':result['EOS'], 'caps':result['caps'], 'amortized_seconds_mean':statistics.mean(r['seconds'] for r in subset), 'peak_allocated_gib':max(r['peak_allocated_gib'] for r in subset), 'actual_batch_sizes':sorted({r['batch_size'] for r in subset}), 'over_envelope_rows':sum(r['single_row_batch_required'] for r in subset), 'multiple_prediction_outputs':d['multiple_prediction_outputs'], 'first_last_disagreements':d['first_last_disagreements'], 'closed_think_outputs':sum(r['closed_think_markers'] == 1 for r in subset)})
    value = {'verified_at_utc':datetime.now(timezone.utc).isoformat(), 'verified_outputs':1800, 'plan_sha256':PLAN_SHA, 'authorization_sha256':AUTH_SHA, 'completed_manifest_sha256':sha(run / 'COMPLETED_FILES_SHA256.json'), 'all18_metric_groups_recomputed':True, 'all_generated_token_count_hash_and_boundary_receipts_verified':True, 'all900_native_prompts_reencoded':True, 'native_decode_recomputed_locally':True, 'native_decode_basis':'All1800 decoded with copied SHA-verified native tokenizer files; independent audit pending', 'new_training':False, 'publish':False, 'submit':False, 'schedule_resumed':False, 'independent_actual_results_audit_pending':True, 'rows':summary, 'metric_reproductions':metric_reproductions, 'verification_kind':'Codex mechanical CPU checks, not independent Claude review'}
    save_new(run / 'LOCAL_VERIFICATION.json', value)
    with (run / 'FINAL_TEST_METRICS.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({'verified_outputs':1800, 'metric_groups':18, 'audit_pending':True}))

if __name__ == '__main__': main()
