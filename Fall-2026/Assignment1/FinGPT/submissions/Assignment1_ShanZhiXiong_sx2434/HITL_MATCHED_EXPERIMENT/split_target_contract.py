"""Prospective contracts; source provenance and real user authorization are external.

No historical adapter/data is repaired by this module. Authorization records are
an auditable workflow binding, NOT a cryptographic proof of human permission.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

MODEL_IDS = {'qwen': 'Qwen/Qwen2.5-7B-Instruct',
             'deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
             'llama': 'meta-llama/Llama-3.1-8B'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def bounds(period):
    if not isinstance(period, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}', period):
        raise ValueError('Invalid forecast period')
    start, end = map(date.fromisoformat, period.split(' to '))
    if end <= start:
        raise ValueError('Forecast period must advance')
    return start, end


def validate_policy(policy):
    if not isinstance(policy, dict) or not isinstance(policy.get('splits'), dict):
        raise ValueError('Policy must contain a split dictionary')
    splits = policy['splits']
    if not {'train', 'validation'}.issubset(splits) or set(splits) - {'train', 'validation', 'test'}:
        raise ValueError('Train and validation required; unknown split forbidden')
    all_ids = []; all_paths = []; spans = {}
    for split, entry in splits.items():
        required = {'path', 'sha256', 'sample_ids', 'samples', 'allowed_periods', 'source_split'}
        if not isinstance(entry, dict) or required - entry.keys():
            raise ValueError('Incomplete policy entry')
        if not isinstance(entry['path'], str) or not Path(entry['path']).is_absolute():
            raise ValueError('Absolute split path required')
        if not valid_sha(entry['sha256']):
            raise ValueError('Full file SHA256 required')
        ids = entry['sample_ids']; n = entry['samples']
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or not isinstance(ids, list):
            raise ValueError('Positive integer samples and ID list required')
        if not all(valid_sha(s) for s in ids) or len(ids) != n or len(set(ids)) != n:
            raise ValueError('Policy ID/count mismatch')
        periods = entry['allowed_periods']
        if not isinstance(periods, list) or not periods or not all(isinstance(p, str) for p in periods):
            raise ValueError('Approved periods required')
        if len(set(periods)) != len(periods):
            raise ValueError('Duplicate approved period')
        parsed = [bounds(p) for p in periods]
        if parsed != sorted(parsed) or any(a[1] > b[0] for a, b in zip(parsed, parsed[1:])):
            raise ValueError('Overlapping/out-of-order forecast periods')
        spans[split] = (parsed[0][0], parsed[-1][1])
        expected_source = 'test' if split == 'test' else 'train'
        if entry['source_split'] != expected_source:
            raise ValueError('Unexpected upstream source split')
        all_ids.extend(ids); all_paths.append(str(Path(entry['path']).resolve()))
    if len(set(all_ids)) != len(all_ids) or len(set(all_paths)) != len(all_paths):
        raise ValueError('Split ID/path overlap')
    if spans['train'][1] > spans['validation'][0]:
        raise ValueError('Training period overlaps validation')
    if 'test' in spans and spans['validation'][1] > spans['test'][0]:
        raise ValueError('Validation period overlaps test')
    prohibited = policy.get('known_test_sha256')
    if not isinstance(prohibited, list) or not prohibited or not all(valid_sha(s) for s in prohibited):
        raise ValueError('Nonempty known-test hash list required')
    for split, entry in splits.items():
        if split != 'test' and entry['sha256'] in prohibited:
            raise PermissionError('Non-test policy is pinned to known test bytes')
    if 'test' in splits and splits['test']['sha256'] not in prohibited:
        raise ValueError('Frozen test hash must be registered as known test')


@dataclass(frozen=True)
class FrozenPolicy:
    path: str
    raw: bytes
    sha256: str

    def decoded(self):
        if digest(self.raw) != self.sha256:
            raise ValueError('Policy bytes changed')
        policy = json.loads(self.raw)
        validate_policy(policy)
        return policy


def load_policy(path, expected_sha256):
    path = Path(path)
    if not path.is_absolute() or not valid_sha(expected_sha256):
        raise ValueError('Absolute frozen policy path and SHA required')
    raw = path.read_bytes()
    if digest(raw) != expected_sha256:
        raise ValueError('Policy file differs from approved SHA')
    policy = FrozenPolicy(str(path.resolve()), raw, expected_sha256)
    policy.decoded()
    return policy


def require_test_authorization(path, expected_sha256, frozen_policy, entry, requested_scope):
    if path is None or not valid_sha(expected_sha256):
        raise PermissionError('Separate recorded user authorization required')
    path = Path(path)
    if not path.is_absolute():
        raise PermissionError('Authorization record path must be absolute')
    raw = path.read_bytes()
    if digest(raw) != expected_sha256:
        raise PermissionError('Authorization record SHA mismatch')
    auth = json.loads(raw)
    if not isinstance(auth, dict):
        raise PermissionError('Malformed authorization record')
    timestamp = auth.get('authorized_at_utc')
    if not isinstance(timestamp, str):
        raise PermissionError('Explicit UTC authorization timestamp required')
    try:
        at = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except (TypeError, ValueError):
        raise PermissionError('Explicit UTC authorization timestamp required')
    if at.tzinfo is None or at.utcoffset().total_seconds() != 0:
        raise PermissionError('Explicit UTC authorization timestamp required')
    if at > datetime.now(timezone.utc):
        raise PermissionError('Authorization cannot be dated in the future')
    if (not isinstance(requested_scope, dict)
            or set(requested_scope) != {'model_key', 'run_id'}
            or requested_scope['model_key'] not in MODEL_IDS
            or not isinstance(requested_scope['run_id'], str) or not requested_scope['run_id'].strip()):
        raise PermissionError('Explicit model and run scope required')
    scope = auth.get('scope')
    if (not isinstance(scope, dict) or scope.get('purpose') != 'final_test'
            or scope.get('run_id') != requested_scope['run_id']
            or not isinstance(scope.get('model_keys'), list) or not scope['model_keys']
            or not all(m in MODEL_IDS for m in scope['model_keys'])
            or requested_scope['model_key'] not in scope['model_keys']):
        raise PermissionError('Authorization scope does not cover this model/run')
    if not (auth.get('authorized') is True and auth.get('policy_sha256') == frozen_policy.sha256
            and auth.get('test_sha256') == entry['sha256']
            and isinstance(auth.get('authorized_by'), str) and auth['authorized_by'].strip()
            and isinstance(auth.get('source_user_message'), str) and auth['source_user_message'].strip()):
        raise PermissionError('No matching recorded human test authorization')
    return {'authorization_sha256': expected_sha256, 'authorized_at_utc': at.isoformat(),
            'source_user_message': auth['source_user_message'], 'scope': requested_scope}


def load_frozen_split(path, *, split, policy_path, policy_sha256, authorization_path=None,
                      authorization_sha256=None, requested_scope=None):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Known split required')
    # The production entry point always verifies external policy bytes and SHA.
    frozen_policy = load_policy(policy_path, policy_sha256)
    policy = frozen_policy.decoded()  # All overlap/structure checks BEFORE opening data.
    entry = policy['splits'].get(split)
    if entry is None:
        if split == 'test':
            raise PermissionError('No frozen test source policy')
        raise ValueError('Split policy missing')
    auth = {}
    if split == 'test':
        auth = require_test_authorization(authorization_path, authorization_sha256, frozen_policy, entry, requested_scope)
    path = Path(path)
    if not path.is_absolute() or path.resolve() != Path(entry['path']).resolve():
        raise ValueError('Input path differs from frozen split')
    raw = path.read_bytes(); source_sha = digest(raw)
    if split != 'test' and source_sha in policy['known_test_sha256']:
        raise PermissionError('Known test bytes offered as training/development data')
    if source_sha != entry['sha256']:
        raise ValueError('Frozen data SHA mismatch')
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != entry['samples'] or not all(isinstance(r, dict) for r in rows):
        raise ValueError('Malformed rows/count')
    for r in rows:
        fields = ['sample_id', 'split', 'source_split', 'symbol', 'period', 'prompt']
        if any(not isinstance(r.get(k), str) for k in fields):
            raise ValueError('Required row identity/period fields missing')
        if r['split'] != split or r['source_split'] != entry['source_split']:
            raise ValueError('Row split/source_split mismatch')
        inspected = r.get('previously_inspected_test')
        if type(inspected) is not bool or (split != 'test' and inspected):
            raise ValueError('Explicit previously-inspected flag is invalid for this split')
        bounds(r['period'])
        if r['period'] not in entry['allowed_periods']:
            raise ValueError('Forecast period not approved for this split')
        calculated = digest((r['symbol']+'|'+r['period']+'|'+r['prompt']).encode())
        if r['sample_id'] != calculated:
            raise ValueError('Sample ID/content mismatch')
    if [r['sample_id'] for r in rows] != entry['sample_ids']:
        raise ValueError('Frozen ID order mismatch')
    receipt = {'path': str(path.resolve()), 'sha256': source_sha, 'samples': len(rows),
               'split': split, 'policy_sha256': frozen_policy.sha256,
               'read_at_utc': datetime.now(timezone.utc).isoformat(), **auth}
    return rows, receipt  # Caller MUST save this receipt into the run before GPU use.


def single_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError('Boundary is not one tokenizer token')
    return ids[0]


@dataclass(frozen=True)
class BoundTokenizer:
    inner: object
    bound_model_key: str
    binding_receipt: dict

    def __getattr__(self, name):
        # During unpickling/deepcopy inner may not have been restored yet.
        if name.startswith('__') or name in {'inner', 'bound_model_key', 'binding_receipt'}:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'inner'), name)


def tokenizer_for_tests(inner, model_key):
    if model_key not in MODEL_IDS:
        raise ValueError('Unknown synthetic tokenizer model')
    return BoundTokenizer(inner, model_key, {'synthetic_test_only': True})


def check_tokenizer_binding(tokenizer, model_key, *, allow_synthetic_for_tests=False):
    if not isinstance(tokenizer, BoundTokenizer) or tokenizer.bound_model_key != model_key:
        raise ValueError('Use the fixed model-specific tokenizer binding')
    receipt = tokenizer.binding_receipt
    if not isinstance(receipt, dict):
        raise ValueError('Tokenizer receipt missing')
    if receipt.get('synthetic_test_only') is True:
        if allow_synthetic_for_tests is not True:
            raise ValueError('Synthetic tokenizer prohibited in production')
        return
    if (receipt.get('model_id') != MODEL_IDS[model_key]
            or not isinstance(receipt.get('revision'), str)
            or not re.fullmatch('[0-9a-f]{40}', receipt['revision'])
            or not isinstance(receipt.get('file_sha256'), dict)
            or not all(valid_sha(receipt['file_sha256'].get(n))
                       for n in ['tokenizer.json', 'tokenizer_config.json'])
            or receipt.get('bos_token_id') != tokenizer.bos_token_id
            or receipt.get('eos_token_id') != tokenizer.eos_token_id):
        raise ValueError('Incomplete or mismatched tokenizer binding receipt')


def load_bound_tokenizer(cache, *, model_key, specification):
    cache = Path(cache)
    if model_key not in MODEL_IDS or specification.get('model_id') != MODEL_IDS[model_key]:
        raise ValueError('Tokenizer model identity mismatch')
    revision = specification.get('revision')
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision) or cache.name != revision:
        raise ValueError('Pinned tokenizer revision/cache mismatch')
    recorded = {}
    for filename in ['tokenizer.json', 'tokenizer_config.json']:
        expected = specification.get('file_sha256', {}).get(filename)
        if not valid_sha(expected):
            raise ValueError('Tokenizer file SHA missing')
        actual = digest((cache/filename).read_bytes())
        if actual != expected:
            raise ValueError('Pinned tokenizer file changed')
        recorded[filename] = actual
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True)
    if tokenizer.eos_token_id != specification.get('eos_token_id') or tokenizer.bos_token_id != specification.get('bos_token_id'):
        raise ValueError('Tokenizer BOS/EOS identity mismatch')
    return BoundTokenizer(tokenizer, model_key,
        {'model_id': MODEL_IDS[model_key], 'revision': revision, 'file_sha256': recorded,
         'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id})


def encode_full_target(tokenizer, *, model_key, prompt_ids, answer, context_tokens, target_policy,
                       allow_synthetic_for_tests=False):
    if model_key not in MODEL_IDS or not prompt_ids or not isinstance(answer, str) or not answer.strip():
        raise ValueError('Known model and nonempty prompt/reference required')
    check_tokenizer_binding(tokenizer, model_key, allow_synthetic_for_tests=allow_synthetic_for_tests)
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
        raise ValueError('Positive context budget required')
    if not all(type(i) is int and i >= 0 for i in prompt_ids):
        raise ValueError('Prompt token IDs invalid')
    reference = answer.strip(); close_id = None
    if model_key in {'llama', 'deepseek'}:
        bos = tokenizer.bos_token_id
        if not isinstance(bos, int) or prompt_ids[0] != bos or prompt_ids.count(bos) != 1:
            raise ValueError('Exactly one leading BOS required')
    if model_key == 'deepseek':
        if target_policy != 'close_implicit_think_then_full_reference':
            raise ValueError('DeepSeek direct-final target policy required')
        open_id = single_id(tokenizer, '<think>'); close_id = single_id(tokenizer, '</think>')
        assistant_id = single_id(tokenizer, '<｜Assistant｜>')
        tail = [assistant_id, open_id] + tokenizer.encode('\n', add_special_tokens=False)
        if (prompt_ids[-len(tail):] != tail or prompt_ids.count(assistant_id) != 1
                or prompt_ids.count(open_id) != 1 or close_id in prompt_ids):
            raise ValueError('Unexpected implicit think token structure')
        if '<think>' in reference or '</think>' in reference:
            raise ValueError('Reference must be final answer only')
        target_text = '</think>\n\n' + reference
    else:
        if target_policy != 'full_reference_plus_eos':
            raise ValueError('Unexpected target policy')
        target_text = reference
    target = tokenizer.encode(target_text, add_special_tokens=False)
    controls = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    if any(i in controls and i != close_id for i in target):
        raise ValueError('Reference contains a tokenizer control/added token')
    if close_id is not None and (not target or target[0] != close_id or target.count(close_id) != 1):
        raise ValueError('Exactly one supervised closing boundary required')
    if tokenizer.decode(target, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target_text:
        raise ValueError('Target tokenizer round-trip changed text')
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or (model_key in {'deepseek', 'llama'} and eos in prompt_ids):
        raise ValueError('Explicit EOS and no EOS inside prompt required')
    target.append(eos)
    if len(prompt_ids) + len(target) > context_tokens:
        raise ValueError('Full input and target do not fit; never silently clip')
    ids = list(prompt_ids) + target
    return {'input_ids': ids, 'attention_mask': [1]*len(ids),
            'labels': [-100]*len(prompt_ids) + target}, {
        'raw_reference_sha256': digest(answer.encode()), 'reference_sha256': digest(reference.encode()),
        'tokenizer_binding': tokenizer.binding_receipt,
        'reference_was_stripped': answer != reference, 'target_text_sha256': digest(target_text.encode()),
        'target_policy': target_policy, 'prompt_tokens': len(prompt_ids), 'supervised_tokens': len(target),
        'supervised_closing_token_id': close_id, 'answer_truncated': False, 'prompt_truncated': False,
        'reasoning_trace_supervision': False, 'direct_final_supervision': model_key == 'deepseek'}


DEEPSEEK_VARIANTS = {
    'base_native_reasoning': 'generated',
    'base_forced_direct': 'prefilled',
    'adapter_direct_final': 'generated',
    'adapter_legacy_unclosed': 'generated',
}


def direct_final_control_prompt(tokenizer, prompt_ids, *, total, new,
                                allow_synthetic_for_tests=False):
    check_tokenizer_binding(tokenizer, 'deepseek', allow_synthetic_for_tests=allow_synthetic_for_tests)
    bos = tokenizer.bos_token_id
    assistant = single_id(tokenizer, '<｜Assistant｜>')
    opened = single_id(tokenizer, '<think>'); closed = single_id(tokenizer, '</think>')
    tail = [assistant, opened] + tokenizer.encode('\n', add_special_tokens=False)
    if (not prompt_ids or any(type(i) is not int or i < 0 for i in prompt_ids)
            or prompt_ids[0] != bos or prompt_ids.count(bos) != 1
            or prompt_ids[-len(tail):] != tail or prompt_ids.count(assistant) != 1
            or prompt_ids.count(opened) != 1 or closed in prompt_ids
            or tokenizer.eos_token_id in prompt_ids):
        raise ValueError('Invalid native DeepSeek prompt for forced-direct control')
    suffix = tokenizer.encode('</think>\n\n', add_special_tokens=False)
    if not suffix or suffix[0] != closed or suffix.count(closed) != 1:
        raise ValueError('Invalid control closing boundary')
    result = list(prompt_ids) + suffix
    assert_full_inference_budget(result, total=total, new=new)
    return result, {'variant': 'base_forced_direct', 'boundary': 'prefilled',
                    'prefilled_ids': suffix, 'prompt_tokens': len(result),
                    'prompt_sha256': digest(json.dumps(result, separators=(',', ':')).encode()),
                    'tokenizer_binding': tokenizer.binding_receipt}


def final_answer_text(generated, *, model_key, boundary='generated', variant=None):
    if model_key not in MODEL_IDS or not isinstance(generated, str):
        raise ValueError('Known model and generation text required')
    if boundary not in {'generated', 'prefilled'}:
        raise ValueError('Unknown boundary policy')
    if model_key == 'deepseek' and variant is not None:
        if DEEPSEEK_VARIANTS.get(variant) != boundary:
            raise ValueError('Variant and boundary policy disagree')
    if boundary == 'prefilled':
        if model_key != 'deepseek' or variant != 'base_forced_direct':
            raise ValueError('Prefilled boundary requires explicit forced-direct variant')
        ok = '</think>' not in generated and '<think>' not in generated
        return generated, {'transform': 'none_boundary_in_prompt', 'think_closed': ok,
            'boundary': boundary, 'variant': variant,
            'raw_generated_sha256': digest(generated.encode()),
            'final_text_sha256': digest(generated.encode())}
    if model_key != 'deepseek':
        return generated, {'transform': 'none', 'think_closed': None}
    n = generated.count('</think>')
    if n != 1 or '<think>' in generated:
        return generated, {'transform': 'none_invalid_boundary', 'closing_markers': n, 'think_closed': False}
    final = generated.split('</think>', 1)[1].lstrip('\n')
    return final, {'transform': 'after_single_closing_think_lstrip_newlines',
                   'closing_markers': 1, 'think_closed': True,
                   'raw_generated_sha256': digest(generated.encode()), 'final_text_sha256': digest(final.encode())}


def assert_full_inference_budget(prompt_ids, *, total, new):
    if (not prompt_ids or type(total) is not int or type(new) is not int
            or total <= 0 or new <= 0 or len(prompt_ids)+new > total):
        raise ValueError('Full input plus generation budget does not fit')
