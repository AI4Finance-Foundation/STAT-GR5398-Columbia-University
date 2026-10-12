import hashlib
import json
import re
import os
from pathlib import Path
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer


LLAMA_LIKE_LORA_MODULES = [
    'q_proj', 'k_proj', 'v_proj',
    'o_proj', 'gate_proj', 'up_proj', 'down_proj',
]

SUPPORTED_MODEL_FAMILIES = {
    'meta-llama/Llama-3.1-8B': 'llama3',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 'deepseek',
    'Qwen/Qwen2.5-7B-Instruct': 'qwen2.5',
}


lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': LLAMA_LIKE_LORA_MODULES,
    'llama3': LLAMA_LIKE_LORA_MODULES,
    'deepseek': LLAMA_LIKE_LORA_MODULES,
    'qwen2.5': LLAMA_LIKE_LORA_MODULES,
}


def validate_lora_target_modules(model, target_modules):
    """Return target-layer counts or raise when a requested module is absent."""

    module_names = [name.rsplit('.', 1)[-1] for name, _ in model.named_modules()]
    target_counts = {name: module_names.count(name) for name in target_modules}
    missing_modules = [name for name, count in target_counts.items() if count == 0]
    if missing_modules:
        raise ValueError(f"LoRA target modules not found: {missing_modules}")
    return target_counts


def validate_adapter_directory(adapter_dir):
    """Validate that a directory contains a LoRA adapter and no full-model weights."""

    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / 'adapter_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing adapter configuration: {config_path}')

    weight_candidates = [
        adapter_dir / 'adapter_model.safetensors',
        adapter_dir / 'adapter_model.bin',
    ]
    weight_path = next((path for path in weight_candidates if path.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f'Missing adapter weights in {adapter_dir}; expected adapter_model.safetensors or '
            'adapter_model.bin.'
        )

    forbidden_patterns = (
        'model.safetensors',
        'model-*.safetensors',
        'model.safetensors.index.json',
        'pytorch_model.bin',
        'pytorch_model-*.bin',
        'pytorch_model.bin.index.json',
    )
    forbidden_files = sorted(
        {
            path.name
            for pattern in forbidden_patterns
            for path in adapter_dir.glob(pattern)
        }
    )
    if forbidden_files:
        raise ValueError(f'Full-model weight files found in adapter directory: {forbidden_files}')

    adapter_config = json.loads(config_path.read_text())
    if str(adapter_config.get('peft_type', '')).upper() != 'LORA':
        raise ValueError(f'Expected a LoRA adapter, found: {adapter_config.get("peft_type")}')
    if str(adapter_config.get('task_type', '')).upper() != 'CAUSAL_LM':
        raise ValueError(
            f'Expected a CAUSAL_LM adapter, found: {adapter_config.get("task_type")}'
        )

    digest = hashlib.sha256()
    with weight_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)

    return {
        'adapter_directory': str(adapter_dir.resolve()),
        'adapter_config_file': config_path.name,
        'adapter_weights_file': weight_path.name,
        'adapter_weights_bytes': weight_path.stat().st_size,
        'adapter_weights_sha256': digest.hexdigest(),
        'base_model_name_or_path': adapter_config.get('base_model_name_or_path'),
        'peft_type': adapter_config.get('peft_type'),
        'task_type': adapter_config.get('task_type'),
        'target_modules': sorted(adapter_config.get('target_modules') or []),
        'rank': adapter_config.get('r'),
        'lora_alpha': adapter_config.get('lora_alpha'),
        'lora_dropout': adapter_config.get('lora_dropout'),
        'full_model_weights_present': False,
    }


def write_adapter_manifest(adapter_dir):
    manifest = validate_adapter_directory(adapter_dir)
    manifest_path = Path(adapter_dir) / 'adapter_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


LEGACY_LLAMA_PROMPT = re.compile(
    r'^\s*\[INST\]\s*<<SYS>>\s*(.*?)\s*<</SYS>>\s*(.*?)\s*\[/INST\]\s*$',
    flags=re.DOTALL,
)


def format_training_prompt(tokenizer, prompt, use_chat_template=True):
    """Convert the dataset's legacy Llama-2 wrapper to the active tokenizer format."""

    match = LEGACY_LLAMA_PROMPT.match(prompt)
    if match:
        messages = [
            {'role': 'system', 'content': match.group(1).strip()},
            {'role': 'user', 'content': match.group(2).strip()},
        ]
    else:
        messages = [{'role': 'user', 'content': prompt.strip()}]

    if use_chat_template and getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts = []
    for message in messages:
        parts.append(f"{message['role'].title()}: {message['content']}")
    parts.append('Assistant:')
    return '\n\n'.join(parts)


def tokenize(args, tokenizer, feature):

    prompt = format_training_prompt(
        tokenizer,
        feature['prompt'],
        use_chat_template=getattr(args, 'use_chat_template', True),
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(
        feature['answer'].strip(),
        add_special_tokens=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define eos_token_id for causal-LM training.')
    if not target_ids or target_ids[-1] != tokenizer.eos_token_id:
        target_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) > args.max_length
    label_ids = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": label_ids,
        "exceed_max_length": exceed_max_length,
        "sequence_length": len(input_ids),
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-chat-hf' # if from_remote else 'base_models/Llama-2-7b-chat-hf'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    
    dataset_names = [d for d in names.split(',')]
    dataset_list = []
    
    for name in dataset_names:
        rep = 1
        if not os.path.exists(name):
            rep = int(name.split('*')[1]) if '*' in name else 1
            name = ('FinGPT/fingpt-forecaster-' if from_remote else 'data/fingpt-forecaster-') + name.split('*')[0]
        tmp_dataset = datasets.load_dataset(name) if from_remote else datasets.load_from_disk(name)
    
        if 'test' not in tmp_dataset:
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)   
        dataset_list.extend([tmp_dataset] * rep)
    
    return dataset_list


ANSWER_FORMAT = re.compile(
    r"^\s*\[Positive Developments\]:\s*(?P<pros>.*?)\s*"
    r"\[Potential Concerns\]:\s*(?P<cons>.*?)\s*"
    r"\[Prediction & Analysis\]:\s*(?P<pna>.*?)\s*$",
    flags=re.DOTALL,
)


def parse_answer_with_reason(answer):
    """Parse the required answer format and explain any format failure."""

    if not isinstance(answer, str) or not answer.strip():
        return None, "empty_answer"

    section_match = ANSWER_FORMAT.match(answer)
    if not section_match:
        return None, "missing_or_misordered_sections"
    if not section_match.group("pros").strip() or not section_match.group("cons").strip():
        return None, "empty_development_or_concern_section"

    detail_match = re.match(
        r"^Prediction:\s*(?P<pred>.*?)\s*Analysis:\s*(?P<analysis>.*?)\s*$",
        section_match.group("pna"),
        flags=re.DOTALL,
    )
    if not detail_match:
        return None, "missing_prediction_or_analysis"

    prediction_text = detail_match.group("pred").strip()
    if not prediction_text or not detail_match.group("analysis").strip():
        return None, "empty_prediction_or_analysis"
    normalized_prediction = prediction_text.lower()
    if re.search(
        r"\b(up|increase|increased|increasing|rise|rises|rising|gain|gains)\b",
        normalized_prediction,
    ):
        prediction_binary = 1
    elif re.search(
        r"\b(down|decrease|decreased|decreasing|decline|declines|fall|falls|drop|drops)\b",
        normalized_prediction,
    ):
        prediction_binary = -1
    else:
        prediction_binary = 0

    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*%",
        prediction_text,
        flags=re.IGNORECASE,
    )
    single_match = re.search(r"(\d+(?:\.\d+)?)\s*%", prediction_text)
    if range_match:
        magnitude = (float(range_match.group(1)) + float(range_match.group(2))) / 2
    elif single_match:
        magnitude = float(single_match.group(1))
        if re.search(r"\bmore than\b", prediction_text, flags=re.IGNORECASE):
            magnitude += 0.5
    else:
        magnitude = 0.0

    return {
        "positive developments": section_match.group("pros").strip(),
        "potential concerns": section_match.group("cons").strip(),
        "prediction": prediction_binary * magnitude,
        "prediction_binary": prediction_binary,
        "analysis": detail_match.group("analysis").strip(),
    }, None


def parse_answer(answer):
    parsed, _ = parse_answer_with_reason(answer)
    return parsed

def calc_rouge_score(references, answers):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
