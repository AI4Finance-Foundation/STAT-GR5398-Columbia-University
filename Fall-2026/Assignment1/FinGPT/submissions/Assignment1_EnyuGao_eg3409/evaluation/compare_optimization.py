#!/usr/bin/env python3
"""Validate and compare one controlled LoRA-rank optimization experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONTROLLED_FIELDS = [
    "model_name",
    "model_family",
    "dataset_name",
    "test_dataset",
    "max_length",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "max_steps",
    "warmup_ratio",
    "scheduler",
    "optim",
    "num_workers",
    "lora_alpha",
    "lora_dropout",
    "quantization",
    "precision",
    "gradient_checkpointing",
    "use_chat_template",
    "trust_remote_code",
    "eval_strategy",
    "max_train_samples",
    "max_eval_samples",
    "seed",
    "resolved_model_family",
    "resolved_precision",
    "effective_batch_size",
    "world_size",
    "lora_target_modules",
    "deepspeed",
    "train_dataset",
    "eval_dataset",
]
METRIC_FIELDS = [
    "valid_count",
    "parsing_success_rate",
    "binary_accuracy",
    "mse",
    "rouge1",
    "rouge2",
    "rougeL",
    "average_inference_seconds",
]
OUTPUT_COLUMNS = [
    "comparison_row",
    "model_label",
    "base_model",
    "lora_rank",
    "trainable_parameter_count",
    "logged_loss_points",
    "initial_training_loss",
    "final_training_loss",
    "minimum_training_loss",
    "training_loss_change",
    "optimizer_steps",
    *METRIC_FIELDS,
    "hitl_completed_reviews",
    "hitl_mean_overall_quality",
    "hitl_formatting_compliance_rate",
    "hitl_hallucination_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline metrics JSON.")
    parser.add_argument("--variant", type=Path, required=True, help="Variant metrics JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hitl-summary", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def validate_complete_metrics(item: dict) -> None:
    label = item.get("model_label", "unnamed run")
    if item.get("completed_samples") != item.get("selected_test_samples"):
        raise ValueError(f"{label} evaluation is incomplete.")
    if item.get("selected_test_samples") != item.get("total_test_samples"):
        raise ValueError(f"{label} did not evaluate the full test split.")
    missing = [key for key in METRIC_FIELDS if item.get(key) is None]
    if missing:
        raise ValueError(f"{label} is missing metrics: {missing}")


def validate_controlled_experiment(baseline: dict, variant: dict) -> None:
    validate_complete_metrics(baseline)
    validate_complete_metrics(variant)
    if baseline.get("base_model") != variant.get("base_model"):
        raise ValueError("Baseline and variant must use the same base model.")
    for key in ("dataset", "dataset_split", "total_test_samples", "generation_settings"):
        if baseline.get(key) != variant.get(key):
            raise ValueError(f"Baseline and variant differ in evaluation field {key!r}.")

    baseline_config = baseline.get("run_config") or {}
    variant_config = variant.get("run_config") or {}
    missing_fields = [
        field
        for field in [*CONTROLLED_FIELDS, "lora_r", "trainable_parameters"]
        if field not in baseline_config or field not in variant_config
    ]
    if missing_fields:
        raise ValueError(
            f"Training configuration is missing controlled fields: {missing_fields}"
        )
    if baseline_config["lora_r"] is None or variant_config["lora_r"] is None:
        raise ValueError("Training configuration must provide lora_r for both runs.")
    if (
        baseline_config["trainable_parameters"] is None
        or variant_config["trainable_parameters"] is None
    ):
        raise ValueError("Training configuration must provide trainable_parameters for both runs.")
    differences = [
        field
        for field in CONTROLLED_FIELDS
        if baseline_config.get(field) != variant_config.get(field)
    ]
    if differences:
        raise ValueError(
            f"The runs change fields besides LoRA rank: {differences}. "
            "Keep all controlled settings fixed."
        )
    if baseline_config["lora_r"] == variant_config["lora_r"]:
        raise ValueError("The optimization variant must use a different lora_r.")


def loss_summary(item: dict) -> dict:
    state = item.get("trainer_state") or {}
    points = [
        entry
        for entry in state.get("log_history", [])
        if "step" in entry and "loss" in entry
    ]
    if not points:
        raise ValueError(f"{item.get('model_label')} has no training-loss history.")
    return {
        "logged_loss_points": len(points),
        "initial_training_loss": float(points[0]["loss"]),
        "final_training_loss": float(points[-1]["loss"]),
        "minimum_training_loss": min(float(point["loss"]) for point in points),
        "training_loss_change": float(points[-1]["loss"]) - float(points[0]["loss"]),
        "optimizer_steps": int(state.get("global_step") or points[-1]["step"]),
    }


def load_hitl(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: row for row in rows}


def result_row(kind: str, item: dict, hitl: dict[str, dict]) -> dict:
    config = item["run_config"]
    human = hitl.get(item["model_label"], {})
    return {
        "comparison_row": kind,
        "model_label": item["model_label"],
        "base_model": item["base_model"],
        "lora_rank": config["lora_r"],
        "trainable_parameter_count": config["trainable_parameters"],
        **loss_summary(item),
        **{field: item[field] for field in METRIC_FIELDS},
        "hitl_completed_reviews": human.get("completed_reviews", ""),
        "hitl_mean_overall_quality": human.get("mean_overall_quality", ""),
        "hitl_formatting_compliance_rate": human.get(
            "formatting_compliance_rate", ""
        ),
        "hitl_hallucination_rate": human.get("hallucination_rate", ""),
    }


def numeric_delta(candidate, baseline):
    if candidate in (None, "") or baseline in (None, ""):
        return ""
    try:
        return round(float(candidate) - float(baseline), 12)
    except (TypeError, ValueError):
        return ""


def main(args: argparse.Namespace) -> None:
    baseline = load_json(args.baseline)
    variant = load_json(args.variant)
    validate_controlled_experiment(baseline, variant)
    hitl = load_hitl(args.hitl_summary)
    rows = [
        result_row("baseline", baseline, hitl),
        result_row("variant", variant, hitl),
    ]
    delta = {
        "comparison_row": "variant_minus_baseline",
        "model_label": f"{variant['model_label']} - {baseline['model_label']}",
        "base_model": baseline["base_model"],
    }
    for column in OUTPUT_COLUMNS[3:]:
        delta[column] = numeric_delta(rows[1].get(column), rows[0].get(column))
    rows.append(delta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Validated a controlled LoRA-rank experiment and wrote {args.output}: "
        f"r={rows[0]['lora_rank']} versus r={rows[1]['lora_rank']}."
    )


if __name__ == "__main__":
    main(parse_args())
