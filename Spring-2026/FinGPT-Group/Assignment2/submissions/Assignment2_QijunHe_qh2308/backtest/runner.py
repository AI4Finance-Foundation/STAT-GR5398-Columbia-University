"""
backtest/runner.py — Backtester stub.

Reads a signals JSON file from output/ and prints a summary table.
Backtesting logic against historical price data is intentionally absent.
"""

# TODO: wire to historical price data

import json
import logging
import os
import sys

from config import LOG_LEVEL, OUTPUT_DIR

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_signals(filepath: str) -> list[dict]:
    """Load and return a signals list from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {filepath}, got {type(data).__name__}")
    return data


def print_summary(signals: list[dict]) -> None:
    """Print a formatted summary table of ticker / direction / strategy_tag."""
    if not signals:
        print("No signals to display.")
        return

    col_w = (10, 10, 16)
    header = (
        f"{'Ticker':<{col_w[0]}} {'Direction':<{col_w[1]}} {'Strategy Tag':<{col_w[2]}}"
    )
    divider = "-" * (sum(col_w) + 2)

    print(divider)
    print(header)
    print(divider)
    for sig in signals:
        print(
            f"{sig.get('ticker', ''):<{col_w[0]}} "
            f"{sig.get('direction', ''):<{col_w[1]}} "
            f"{sig.get('strategy_tag', ''):<{col_w[2]}}"
        )
    print(divider)
    print(f"Total signals: {len(signals)}")


def run(signals_file: str | None = None) -> None:
    """
    Main entry point.

    If signals_file is None, use the most recently modified file in output/.
    """
    if signals_file is None:
        if not os.path.isdir(OUTPUT_DIR):
            logger.error("output/ directory not found. Run pipeline.py first.")
            sys.exit(1)

        candidates = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("signals_") and f.endswith(".json")
        ]
        if not candidates:
            logger.error("No signal files found in %s.", OUTPUT_DIR)
            sys.exit(1)

        signals_file = max(candidates, key=os.path.getmtime)
        logger.info("Using most recent signals file: %s", signals_file)

    signals = load_signals(signals_file)
    logger.info("Loaded %d signal(s) from %s", len(signals), signals_file)
    print_summary(signals)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run(target)
