#!/usr/bin/env python3
"""Run resumable FinGPT adapter inference and save one JSON object per test sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "train_lora.py").is_file() else SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_lora import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_quantization_config,
    load_dataset_source,
    resolve_precision,
    select_samples,
    validate_dataset,
)
from utils import format_training_prompt, parse_answer_with_reason, validate_adapter_directory


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    assignment_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=assignment_dir / "evaluation" / "qwen_outputs.jsonl",
    )
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=assignment_dir / "pretrained-models",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sample indices already present in the JSONL output (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of resuming it.",
    )
    return parser.parse_args()


def load_existing_records(path: Path) -> list[dict]:
    records = []
    if not path.is_file():
        return records
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    indices = [record.get("sample_index") for record in records]
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate sample indices in {path}: {duplicates[:10]}")
    return records


def validate_resume_metadata(records: list[dict], expected: dict) -> None:
    for record in records:
        for key, expected_value in expected.items():
            if record.get(key) != expected_value:
                raise ValueError(
                    f"Cannot resume: sample {record.get('sample_index')} has {key}="
                    f"{record.get(key)!r}, expected {expected_value!r}."
                )


def build_summary(
    records: list[dict],
    *,
    dataset_total: int,
    selected_total: int,
    metadata: dict,
    adapter_manifest: dict,
) -> dict:
    generated = [record for record in records if record.get("error") is None]
    parsable_count = sum(bool(record.get("parseable")) for record in generated)
    generated_count = len(generated)
    completed_count = len(records)
    elapsed = [record["inference_seconds"] for record in generated]
    return {
        "schema_version": SCHEMA_VERSION,
        **metadata,
        "adapter_manifest": adapter_manifest,
        "total_test_samples": dataset_total,
        "selected_test_samples": selected_total,
        "completed_samples": completed_count,
        "generation_success_count": generated_count,
        "generation_error_count": completed_count - generated_count,
        "parsable_samples": parsable_count,
        "valid_count": parsable_count,
        "parsing_success_rate": (
            parsable_count / generated_count if generated_count else 0.0
        ),
        "overall_parse_rate": parsable_count / completed_count if completed_count else 0.0,
        "average_inference_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "parse_failure_reasons": dict(
            Counter(
                record.get("parse_failure_reason")
                for record in generated
                if not record.get("parseable")
            )
        ),
        "complete": completed_count == selected_total,
    }


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: argparse.Namespace) -> None:
    if args.max_input_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive integers.")

    set_seed(args.seed)
    args.adapter_path = args.adapter_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path
        else args.output_path.with_suffix(".summary.json")
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    adapter_manifest = validate_adapter_directory(args.adapter_path)
    hf_token = os.environ.get("HF_TOKEN")
    if args.base_model.startswith("meta-llama/") and not hf_token:
        raise EnvironmentError(
            "Set HF_TOKEN after receiving access to the gated Meta Llama repository."
        )

    dataset = load_dataset_source(args.dataset, args.cache_dir, hf_token)
    validate_dataset(dataset, args.dataset)
    if args.split not in dataset:
        raise ValueError(f"Dataset does not contain split {args.split!r}.")
    full_split = dataset[args.split]
    selected_split = select_samples(full_split, args.max_samples)

    metadata = {
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path),
        "dataset": args.dataset,
        "dataset_split": args.split,
    }
    if args.output_path.exists() and args.overwrite:
        args.output_path.write_text("")
        existing_records = []
    elif args.output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Use --resume or --overwrite."
        )
    else:
        existing_records = load_existing_records(args.output_path)
    validate_resume_metadata(existing_records, metadata)

    selected_indices = set(range(len(selected_split)))
    records_by_index = {
        record["sample_index"]: record
        for record in existing_records
        if record.get("sample_index") in selected_indices
    }
    pending_indices = sorted(selected_indices.difference(records_by_index))
    initial_summary = build_summary(
        list(records_by_index.values()),
        dataset_total=len(full_split),
        selected_total=len(selected_split),
        metadata=metadata,
        adapter_manifest=adapter_manifest,
    )
    write_summary(summary_path, initial_summary)
    if not pending_indices:
        print(f"Evaluation already complete: {args.output_path}")
        print(json.dumps(initial_summary, indent=2, sort_keys=True))
        return

    precision_name, compute_dtype = resolve_precision(args.precision)
    quantization_config = build_quantization_config(args.quantization, compute_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither a padding token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
        "dtype": compute_dtype,
    }
    if quantization_config is not None:
        model_kwargs.update(quantization_config=quantization_config, device_map="auto")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    generation_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "seed": args.seed,
        "precision": precision_name,
        "quantization": args.quantization,
    }
    print(
        f"Evaluating {len(pending_indices)} pending samples; appending to {args.output_path}"
    )
    with args.output_path.open("a") as output_handle:
        for sample_index in pending_indices:
            row = selected_split[sample_index]
            prompt = format_training_prompt(tokenizer, row["prompt"], use_chat_template=True)
            record = {
                "schema_version": SCHEMA_VERSION,
                "sample_index": sample_index,
                "symbol": row["symbol"],
                "period": row["period"],
                "label": row["label"],
                "prompt": prompt,
                "reference_answer": row["answer"],
                "generated_answer": None,
                "parseable": False,
                "parse_failure_reason": None,
                "inference_seconds": None,
                "prompt_token_count": None,
                "generated_token_count": None,
                **metadata,
                "generation_settings": generation_settings,
                "error": None,
            }
            try:
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                prompt_tokens = int(encoded["input_ids"].shape[1])
                record["prompt_token_count"] = prompt_tokens
                if prompt_tokens > args.max_input_length:
                    raise ValueError(
                        f"Prompt has {prompt_tokens} tokens, exceeding --max-input-length "
                        f"{args.max_input_length}."
                    )
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                generate_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if args.do_sample:
                    generate_kwargs["temperature"] = args.temperature
                synchronize_if_needed()
                start_time = time.perf_counter()
                with torch.inference_mode():
                    generated_ids = model.generate(**encoded, **generate_kwargs)
                synchronize_if_needed()
                record["inference_seconds"] = time.perf_counter() - start_time
                continuation = generated_ids[0, prompt_tokens:]
                record["generated_token_count"] = int(continuation.shape[0])
                answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                record["generated_answer"] = answer
                parsed, failure_reason = parse_answer_with_reason(answer)
                record["parseable"] = parsed is not None
                record["parse_failure_reason"] = failure_reason
            except Exception as exc:  # keep a long evaluation resumable after one bad sample
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["parse_failure_reason"] = "generation_error"

            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
            records_by_index[sample_index] = record
            summary = build_summary(
                list(records_by_index.values()),
                dataset_total=len(full_split),
                selected_total=len(selected_split),
                metadata=metadata,
                adapter_manifest=adapter_manifest,
            )
            write_summary(summary_path, summary)
            print(
                f"[{len(records_by_index)}/{len(selected_split)}] sample={sample_index} "
                f"parseable={record['parseable']} error={record['error'] is not None}"
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
