"""
CLI entrypoint for running news-to-signal backtests.
"""

from __future__ import annotations

import argparse

from backtest.backtester import compute_metrics, run_backtest


def _print_metrics_table(metrics: dict) -> None:
    keys = list(metrics.keys())
    key_width = max(len(key) for key in keys) + 2
    print("=" * (key_width + 24))
    print(f"{'Metric':<{key_width}}Value")
    print("=" * (key_width + 24))
    for key in keys:
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key:<{key_width}}{value:.6f}")
        else:
            print(f"{key:<{key_width}}{value}")
    print("=" * (key_width + 24))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FinGPT news-to-signal backtest.")
    parser.add_argument("--dataset", required=True, help="Path to parquet/csv dataset file.")
    parser.add_argument(
        "--output",
        default="output/backtest_results.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to process for quick tests.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print backtest metrics summary after completion.",
    )
    args = parser.parse_args()

    results = run_backtest(
        dataset_path=args.dataset,
        output_path=args.output,
        max_rows=args.max_rows,
    )
    if args.metrics:
        metrics = compute_metrics(results)
        _print_metrics_table(metrics)


if __name__ == "__main__":
    main()

