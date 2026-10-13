#!/usr/bin/env python3
"""Calculate FinGPT metrics from one evaluator JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR if (SCRIPT_DIR / "utils.py").is_file() else SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils import calc_metrics, parse_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw evaluator JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON output.")
    parser.add_argument("--evaluation-summary", type=Path, help="Evaluator summary JSON.")
    parser.add_argument("--run-config", type=Path, help="Training run_config.json.")
    parser.add_argument("--trainer-state", type=Path, help="Trainer trainer_state.json.")
    parser.add_argument("--model-label", help="Short display name for tables and plots.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation output not found: {path}")
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Evaluation output is empty: {path}")
    indices = [record.get("sample_index") for record in records]
    duplicates = [index for index, count in Counter(indices).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample indices: {sorted(duplicates)[:10]}")
    return records


def common_value(records: list[dict], key: str):
    values = {json.dumps(record.get(key), sort_keys=True) for record in records}
    if len(values) != 1:
        raise ValueError(f"Evaluation records do not share one {key!r} value.")
    return records[0].get(key)


def load_optional_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def flatten_rouge(metrics: dict) -> dict:
    section_keys = {
        "positive": "pros_rouge_scores",
        "concerns": "cons_rouge_scores",
        "analysis": "anal_rouge_scores",
    }
    flat = {}
    for section, source_key in section_keys.items():
        scores = metrics.get(source_key, {})
        for rouge_name in ("rouge1", "rouge2", "rougeL"):
            flat[f"{section}_{rouge_name}"] = scores.get(rouge_name)
    for rouge_name in ("rouge1", "rouge2", "rougeL"):
        values = [
            flat[f"{section}_{rouge_name}"]
            for section in section_keys
            if flat[f"{section}_{rouge_name}"] is not None
        ]
        flat[rouge_name] = sum(values) / len(values) if values else None
    return flat


def main(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    generation_settings = common_value(records, "generation_settings")
    base_model = common_value(records, "base_model")
    adapter_path = common_value(records, "adapter_path")
    dataset = common_value(records, "dataset")
    dataset_split = common_value(records, "dataset_split")

    successful = [
        record
        for record in records
        if record.get("error") is None and isinstance(record.get("generated_answer"), str)
    ]
    generated_answers = [record["generated_answer"] for record in successful]
    references = [record["reference_answer"] for record in successful]
    repository_metrics = calc_metrics(generated_answers, references)
    valid_count = int(repository_metrics.get("valid_count", 0))
    parsable_count = sum(parse_answer(answer) is not None for answer in generated_answers)
    inference_times = [
        float(record["inference_seconds"])
        for record in successful
        if record.get("inference_seconds") is not None
    ]

    if args.evaluation_summary is not None:
        evaluation_summary = load_optional_json(args.evaluation_summary)
    else:
        inferred_summary = args.input.with_suffix(".summary.json")
        evaluation_summary = (
            json.loads(inferred_summary.read_text()) if inferred_summary.is_file() else None
        )
    run_config = load_optional_json(args.run_config)
    if args.run_config is None and adapter_path:
        inferred_config = Path(adapter_path) / "run_config.json"
        if inferred_config.is_file():
            run_config = json.loads(inferred_config.read_text())
    trainer_state = load_optional_json(args.trainer_state)
    if args.trainer_state is None and adapter_path:
        inferred_state = Path(adapter_path) / "trainer_state.json"
        if inferred_state.is_file():
            trainer_state = json.loads(inferred_state.read_text())

    output = {
        "schema_version": 1,
        "model_label": args.model_label or base_model.rsplit("/", 1)[-1],
        "base_model": base_model,
        "adapter_path": adapter_path,
        "dataset": dataset,
        "dataset_split": dataset_split,
        "generation_settings": generation_settings,
        "total_test_samples": (
            evaluation_summary.get("total_test_samples")
            if evaluation_summary
            else len(records)
        ),
        "selected_test_samples": (
            evaluation_summary.get("selected_test_samples")
            if evaluation_summary
            else len(records)
        ),
        "completed_samples": len(records),
        "generation_success_count": len(successful),
        "generation_error_count": len(records) - len(successful),
        "parsable_samples": parsable_count,
        "valid_count": valid_count,
        "parsing_success_rate": parsable_count / len(successful) if successful else 0.0,
        "binary_accuracy": repository_metrics.get("bin_acc"),
        "mse": repository_metrics.get("mse"),
        **flatten_rouge(repository_metrics),
        "average_inference_seconds": (
            sum(inference_times) / len(inference_times) if inference_times else None
        ),
        "run_config": run_config,
        "trainer_state": trainer_state,
        "evaluation_summary": evaluation_summary,
        "source_jsonl": str(args.input.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
