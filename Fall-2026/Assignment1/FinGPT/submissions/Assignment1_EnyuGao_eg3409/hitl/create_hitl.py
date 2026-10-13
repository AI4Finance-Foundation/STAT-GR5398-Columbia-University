#!/usr/bin/env python3
"""Create a deterministic, unrated HITL review CSV from model output JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


RATING_COLUMNS = [
    "positive_development_quality",
    "concern_risk_quality",
    "financial_reasoning_quality",
    "prediction_justification",
    "factual_grounding",
    "clarity_professionalism",
]
HITL_COLUMNS = [
    "review_id",
    "model",
    "sample_index",
    "symbol",
    "period",
    "label",
    *RATING_COLUMNS,
    "formatting_compliance",
    "hallucination_observed",
    "notes",
    "prompt",
    "reference_answer",
    "generated_answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        metavar="MODEL=JSONL",
        help="Named model output, for example qwen=evaluation/qwen_outputs.jsonl.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-model", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-nonstandard-size",
        action="store_true",
        help="Allow a review form outside the recommended 20–30 total response rows.",
    )
    return parser.parse_args()


def parse_output_specs(specs: list[str]) -> dict[str, Path]:
    outputs = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Output must use MODEL=JSONL syntax: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label or label in outputs:
            raise ValueError(f"Model labels must be non-empty and unique: {label!r}")
        outputs[label] = Path(raw_path).expanduser().resolve()
    return outputs


def read_successful_records(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            index = record.get("sample_index")
            if index in records:
                raise ValueError(f"Duplicate sample index {index} in {path}")
            if record.get("error") is None and isinstance(record.get("generated_answer"), str):
                records[index] = record
    if not records:
        raise ValueError(f"No successfully generated records found in {path}")
    return records


def validate_matching_sample(records_by_model: dict[str, dict[int, dict]], index: int) -> None:
    # Chat-template formatting makes the stored prompt model-specific. The source identity fields
    # and reference answer must still match for a fair paired review.
    fields = ("symbol", "period", "label", "reference_answer")
    first_label = next(iter(records_by_model))
    baseline = records_by_model[first_label][index]
    for model_label, records in records_by_model.items():
        for field in fields:
            if records[index].get(field) != baseline.get(field):
                raise ValueError(
                    f"Sample {index} field {field!r} differs between {first_label} and "
                    f"{model_label}."
                )


def main(args: argparse.Namespace) -> None:
    if args.samples_per_model <= 0:
        raise ValueError("--samples-per-model must be positive.")
    output_specs = parse_output_specs(args.outputs)
    total_reviews = args.samples_per_model * len(output_specs)
    if not args.allow_nonstandard_size and not 20 <= total_reviews <= 30:
        raise ValueError(
            f"The review would contain {total_reviews} rows; use settings that create 20–30 "
            "rows or pass --allow-nonstandard-size."
        )
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing review file: {output_path}")

    records_by_model = {
        label: read_successful_records(path) for label, path in output_specs.items()
    }
    common_indices = set.intersection(
        *(set(records) for records in records_by_model.values())
    )
    if len(common_indices) < args.samples_per_model:
        raise ValueError(
            f"Only {len(common_indices)} successful sample indices are shared by all models; "
            f"{args.samples_per_model} are required."
        )
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(sorted(common_indices), args.samples_per_model))

    rows = []
    for index in selected_indices:
        validate_matching_sample(records_by_model, index)
        for model_label, records in records_by_model.items():
            record = records[index]
            rows.append(
                {
                    "model": model_label,
                    "sample_index": index,
                    "symbol": record.get("symbol"),
                    "period": record.get("period"),
                    "label": record.get("label"),
                    "prompt": record.get("prompt"),
                    "reference_answer": record.get("reference_answer"),
                    "generated_answer": record.get("generated_answer"),
                    **{column: "" for column in RATING_COLUMNS},
                    "formatting_compliance": "",
                    "hallucination_observed": "",
                    "notes": "",
                }
            )
    rng.shuffle(rows)
    for number, row in enumerate(rows, start=1):
        row["review_id"] = f"HITL-{number:03d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HITL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} unrated responses across {len(output_specs)} models to "
        f"{output_path}. Selected sample indices: {selected_indices}"
    )


if __name__ == "__main__":
    main(parse_args())
