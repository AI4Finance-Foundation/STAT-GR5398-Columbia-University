#!/usr/bin/env python3
"""Build the FinGPT model-comparison CSV and reproducible plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


CSV_COLUMNS = [
    "model",
    "base_model",
    "trainable_parameter_count",
    "lora_rank",
    "learning_rate",
    "epochs",
    "total_test_samples",
    "parsable_samples",
    "valid_count",
    "parsing_success_rate",
    "binary_accuracy",
    "mse",
    "rouge1",
    "rouge2",
    "rougeL",
    "positive_rouge1",
    "positive_rouge2",
    "positive_rougeL",
    "concerns_rouge1",
    "concerns_rouge2",
    "concerns_rougeL",
    "analysis_rouge1",
    "analysis_rouge2",
    "analysis_rougeL",
    "average_inference_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-mixed-settings",
        action="store_true",
        help="Permit different datasets or generation settings (not recommended).",
    )
    return parser.parse_args()


def load_metrics(paths: list[Path]) -> list[dict]:
    metrics = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        item = json.loads(path.read_text())
        item["metrics_file"] = str(path.resolve())
        metrics.append(item)
    labels = [item["model_label"] for item in metrics]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Model labels must be unique: {labels}")
    return metrics


def comparison_signature(item: dict) -> tuple:
    settings = item.get("generation_settings") or {}
    comparable_settings = {
        key: settings.get(key)
        for key in ("max_new_tokens", "do_sample", "temperature", "seed")
    }
    return (
        item.get("dataset"),
        item.get("dataset_split"),
        item.get("total_test_samples"),
        item.get("selected_test_samples"),
        json.dumps(comparable_settings, sort_keys=True),
    )


def validate_comparability(metrics: list[dict], allow_mixed: bool) -> None:
    signatures = {comparison_signature(item) for item in metrics}
    if len(signatures) != 1 and not allow_mixed:
        raise ValueError(
            "Metrics use different datasets, split sizes, or generation settings. "
            "Re-evaluate consistently or explicitly pass --allow-mixed-settings."
        )
    for item in metrics:
        config = item.get("run_config") or {}
        missing_config = [
            key
            for key in ("trainable_parameters", "lora_r", "learning_rate", "num_epochs")
            if config.get(key) is None
        ]
        if missing_config:
            raise ValueError(
                f"{item['model_label']} is missing training configuration: {missing_config}"
            )
        if item.get("completed_samples") != item.get("selected_test_samples"):
            raise ValueError(
                f"{item['model_label']} evaluation is incomplete: "
                f"{item.get('completed_samples')}/{item.get('selected_test_samples')} samples."
            )
        if item.get("selected_test_samples") != item.get("total_test_samples"):
            raise ValueError(
                f"{item['model_label']} did not evaluate the full test split: "
                f"{item.get('selected_test_samples')}/{item.get('total_test_samples')} samples."
            )
        if item.get("valid_count", 0) <= 0:
            raise ValueError(
                f"{item['model_label']} has valid_count=0; accuracy and text metrics "
                "cannot be compared."
            )
        missing_metrics = [
            key
            for key in (
                "binary_accuracy",
                "mse",
                "rouge1",
                "rouge2",
                "rougeL",
                "average_inference_seconds",
            )
            if item.get(key) is None
        ]
        if missing_metrics:
            raise ValueError(f"{item['model_label']} is missing metrics: {missing_metrics}")
        loss_points = [
            entry
            for entry in (item.get("trainer_state") or {}).get("log_history", [])
            if "loss" in entry and "step" in entry
        ]
        if not loss_points:
            raise ValueError(f"{item['model_label']} has no training-loss history.")


def training_value(item: dict, key: str):
    config = item.get("run_config") or {}
    return config.get(key)


def comparison_row(item: dict) -> dict:
    return {
        "model": item["model_label"],
        "base_model": item.get("base_model"),
        "trainable_parameter_count": training_value(item, "trainable_parameters"),
        "lora_rank": training_value(item, "lora_r"),
        "learning_rate": training_value(item, "learning_rate"),
        "epochs": training_value(item, "num_epochs"),
        **{column: item.get(column) for column in CSV_COLUMNS[6:]},
    }


def write_csv(path: Path, metrics: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(comparison_row(item) for item in metrics)


def import_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "fingpt-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: python -m pip install matplotlib"
        ) from exc
    return plt


def save_bar_plot(plt, metrics, key, title, ylabel, path) -> None:
    labels = [item["model_label"] for item in metrics]
    values = [item.get(key) for item in metrics]
    if any(value is None for value in values):
        raise ValueError(f"Cannot plot {key}: at least one model has no value.")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, values, color="#2F6B8A")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_rouge_plot(plt, metrics, path) -> None:
    labels = [item["model_label"] for item in metrics]
    keys = ("rouge1", "rouge2", "rougeL")
    width = 0.24
    positions = list(range(len(labels)))
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, key in enumerate(keys):
        values = [item.get(key) for item in metrics]
        if any(value is None for value in values):
            raise ValueError(f"Cannot plot {key}: at least one model has no value.")
        shifted = [position + (offset - 1) * width for position in positions]
        axis.bar(shifted, values, width=width, label=key)
    axis.set_xticks(positions, labels)
    axis.set_title("Mean ROUGE by model")
    axis.set_ylabel("F1 score")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_training_loss_plot(plt, metrics, path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for item in metrics:
        state = item.get("trainer_state") or {}
        points = [
            entry
            for entry in state.get("log_history", [])
            if "loss" in entry and "step" in entry
        ]
        if points:
            axis.plot(
                [point["step"] for point in points],
                [point["loss"] for point in points],
                marker="o",
                markersize=3,
                label=item["model_label"],
            )
            plotted = True
    if not plotted:
        plt.close(figure)
        raise ValueError("No training-loss entries were found in the trainer states.")
    axis.set_title("Training loss")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    metrics = load_metrics(args.metrics)
    validate_comparability(metrics, args.allow_mixed_settings)
    write_csv(args.output_csv, metrics)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plt = import_matplotlib()
    save_training_loss_plot(plt, metrics, args.figures_dir / "training_loss.png")
    save_bar_plot(
        plt,
        metrics,
        "binary_accuracy",
        "Binary direction accuracy",
        "Accuracy",
        args.figures_dir / "model_accuracy.png",
    )
    save_bar_plot(
        plt, metrics, "mse", "Prediction magnitude error", "MSE", args.figures_dir / "mse.png"
    )
    save_rouge_plot(plt, metrics, args.figures_dir / "rouge.png")
    save_bar_plot(
        plt,
        metrics,
        "average_inference_seconds",
        "Mean inference time",
        "Seconds per sample",
        args.figures_dir / "inference_time.png",
    )
    print(f"Wrote {args.output_csv} and five plots to {args.figures_dir}")


if __name__ == "__main__":
    main(parse_args())
