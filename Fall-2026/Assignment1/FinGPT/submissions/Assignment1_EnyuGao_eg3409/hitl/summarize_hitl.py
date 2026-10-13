#!/usr/bin/env python3
"""Validate completed HITL ratings and summarize them by model."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from create_hitl import HITL_COLUMNS, RATING_COLUMNS


SUMMARY_COLUMNS = [
    "model",
    "total_review_rows",
    "completed_reviews",
    *[f"mean_{column}" for column in RATING_COLUMNS],
    "mean_overall_quality",
    "formatting_compliance_rate",
    "hallucination_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize completed rows while reporting incomplete rows in total_review_rows.",
    )
    return parser.parse_args()


def parse_rating(value: str, row_number: int, column: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        rating = int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number} {column} must be an integer from 1 to 5.") from exc
    if rating not in range(1, 6):
        raise ValueError(f"Row {row_number} {column} must be from 1 to 5.")
    return rating


def parse_yes_no(value: str, row_number: int, column: str) -> bool | None:
    value = value.strip().lower()
    if not value:
        return None
    if value not in {"yes", "no"}:
        raise ValueError(f"Row {row_number} {column} must be yes or no.")
    return value == "yes"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(HITL_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"HITL CSV is missing columns: {sorted(missing)}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("HITL CSV contains no review rows.")

    grouped = defaultdict(list)
    incomplete_rows = []
    for row_number, row in enumerate(raw_rows, start=2):
        ratings = {
            column: parse_rating(row[column], row_number, column)
            for column in RATING_COLUMNS
        }
        formatting = parse_yes_no(row["formatting_compliance"], row_number, "formatting_compliance")
        hallucination = parse_yes_no(
            row["hallucination_observed"], row_number, "hallucination_observed"
        )
        fields = [*ratings.values(), formatting, hallucination]
        if any(value is None for value in fields):
            incomplete_rows.append(row_number)
            grouped[row["model"]].append(None)
        else:
            grouped[row["model"]].append(
                {
                    "ratings": ratings,
                    "formatting": formatting,
                    "hallucination": hallucination,
                }
            )
    if incomplete_rows and not args.allow_incomplete:
        raise ValueError(
            f"{len(incomplete_rows)} review rows are incomplete; first rows: "
            f"{incomplete_rows[:10]}. Fill every rating or pass --allow-incomplete."
        )

    summary_rows = []
    for model, reviews in sorted(grouped.items()):
        completed = [review for review in reviews if review is not None]
        row = {
            "model": model,
            "total_review_rows": len(reviews),
            "completed_reviews": len(completed),
        }
        for column in RATING_COLUMNS:
            row[f"mean_{column}"] = mean(
                [review["ratings"][column] for review in completed]
            )
        row["mean_overall_quality"] = mean(
            [
                rating
                for review in completed
                for rating in review["ratings"].values()
            ]
        )
        row["formatting_compliance_rate"] = mean(
            [float(review["formatting"]) for review in completed]
        )
        row["hallucination_rate"] = mean(
            [float(review["hallucination"]) for review in completed]
        )
        summary_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        f"Wrote {len(summary_rows)} model summaries to {args.output}; "
        f"completed reviews: {sum(row['completed_reviews'] for row in summary_rows)}/"
        f"{len(raw_rows)}"
    )


if __name__ == "__main__":
    main(parse_args())
