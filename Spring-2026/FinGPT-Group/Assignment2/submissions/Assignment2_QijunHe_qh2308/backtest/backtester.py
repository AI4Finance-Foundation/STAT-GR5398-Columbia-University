"""
End-to-end news-to-signal backtester.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import pandas as pd

from agent1.extractor import extract_fingerprint_batch
from agent2.reasoner import generate_signal_batch
from backtest.dataset_parser import build_backtest_rows, load_dataset
from backtest.price_fetcher import direction_from_return, get_realized_return
from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


_BATCH_SIZE = 10


def _position_from_direction(direction: str) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def _make_base_result(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "article_text": row["article_text"][:120],
        "fingpt_label": row["fingpt_label"],
        # Agent 1 sentiment (logits-derived)
        "sentiment_label": None,
        "sentiment_confidence": None,
        "sentiment_probabilities": None,
        # Agent 2 signal (logits-derived)
        "signal_direction": None,
        "signal_confidence": None,
        "signal_strategy_tag": None,
        "signal_logits": None,
        "signal_probabilities": None,
        # Backtest metrics
        "realized_return": None,
        "position": None,
        "strategy_return": None,
        "skipped_reason": "",
    }


def run_backtest(
    dataset_path: str,
    output_path: str = "output/backtest_results.csv",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full news2signal pipeline and persist detailed results.

    vLLM inference runs in batches of ``_BATCH_SIZE`` (default 10).  Each batch
    makes three vLLM calls:
      1. Guided fact-extraction (all articles in the batch, single call).
      2. Sentiment CoT + logits (all articles in the batch, single call).
      3. Strategy CoT + logits (valid fingerprints only, single call).
    Per-item result assembly, price fetching, and CSV serialisation are
    unchanged from the single-item version.
    """
    df = load_dataset(dataset_path)
    rows = build_backtest_rows(df)
    if max_rows is not None:
        rows = rows[:max_rows]

    results: list[dict] = []
    total = len(rows)

    for batch_start in range(0, total, _BATCH_SIZE):
        batch = rows[batch_start : batch_start + _BATCH_SIZE]
        batch_end = batch_start + len(batch)
        logger.info(
            "Backtest progress: rows %d–%d / %d", batch_start + 1, batch_end, total
        )

        # --- Agent 1: batch extract fingerprints (2 vLLM calls) ---
        try:
            fingerprints = extract_fingerprint_batch(
                [r["article_text"] for r in batch]
            )
        except Exception as exc:
            logger.exception(
                "extract_fingerprint_batch failed for rows %d-%d: %s",
                batch_start + 1, batch_end, exc,
            )
            fingerprints = [None] * len(batch)

        # --- Agent 2: batch generate signals (1 vLLM call, valid FPs only) ---
        valid_pairs: list[tuple[int, object]] = [
            (j, fp) for j, fp in enumerate(fingerprints) if fp is not None
        ]
        batch_signals: list = [None] * len(batch)
        if valid_pairs:
            valid_indices, valid_fps = zip(*valid_pairs)
            try:
                raw_signals = generate_signal_batch(list(valid_fps))  # type: ignore[arg-type]
            except Exception as exc:
                logger.exception(
                    "generate_signal_batch failed for rows %d-%d: %s",
                    batch_start + 1, batch_end, exc,
                )
                raw_signals = [None] * len(valid_fps)
            for k, j in enumerate(valid_indices):
                batch_signals[j] = raw_signals[k]

        # --- Per-row result assembly (output logic unchanged) ---
        for j, row in enumerate(batch):
            base_result = _make_base_result(row)
            fingerprint = fingerprints[j]
            signal = batch_signals[j]

            try:
                if fingerprint is None:
                    base_result["skipped_reason"] = "fingerprint_failed"
                    results.append(base_result)
                    continue

                base_result["sentiment_label"] = fingerprint.sentiment_label
                base_result["sentiment_confidence"] = fingerprint.sentiment_confidence
                base_result["sentiment_probabilities"] = str(
                    fingerprint.sentiment_probabilities
                )

                if signal is None:
                    base_result["skipped_reason"] = "signal_failed"
                    results.append(base_result)
                    continue

                realized_return = get_realized_return(
                    ticker=row["ticker"],
                    start_date=row["start_date"],
                    end_date=row["end_date"],
                )

                base_result["signal_direction"] = signal.direction
                base_result["signal_confidence"] = signal.confidence
                base_result["signal_strategy_tag"] = signal.strategy_tag
                base_result["signal_logits"] = str(signal.signal_logits)
                base_result["signal_probabilities"] = str(signal.signal_probabilities)

                if realized_return is None:
                    base_result["skipped_reason"] = "price_fetch_failed"
                    results.append(base_result)
                    continue

                position = _position_from_direction(signal.direction)
                strategy_return = position * realized_return

                base_result["realized_return"] = realized_return
                base_result["position"] = position
                base_result["strategy_return"] = strategy_return
                results.append(base_result)

            except Exception as exc:
                global_idx = batch_start + j + 1
                logger.exception(
                    "Failed to assemble result for row %d: %s", global_idx, exc
                )
                if not base_result["skipped_reason"]:
                    base_result["skipped_reason"] = "signal_failed"
                results.append(base_result)

    results_df = pd.DataFrame(results)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Saved backtest results to %s", output_path)
    return results_df


def compute_metrics(results: pd.DataFrame) -> dict:
    """
    Compute backtest summary metrics over successful rows only.
    """
    total_rows = int(len(results))
    successful = results[results["skipped_reason"] == ""].copy()
    successful_rows = int(len(successful))
    skipped = total_rows - successful_rows
    skip_rate = (skipped / total_rows) if total_rows else 0.0

    metrics = {
        "total_rows": total_rows,
        "successful_rows": successful_rows,
        "skip_rate": skip_rate,
        "direction_accuracy": 0.0,
        "long_accuracy": 0.0,
        "short_accuracy": 0.0,
        "mean_strategy_return": 0.0,
        "std_strategy_return": 0.0,
        "annualized_sharpe": 0.0,
        "total_pnl": 0.0,
        "vs_fingpt_accuracy": 0.0,
    }
    if successful_rows == 0:
        return metrics

    successful["realized_direction"] = successful["realized_return"].astype(float).apply(
        direction_from_return
    )
    successful["signal_market_direction"] = successful["signal_direction"].map(
        {"long": "up", "short": "down", "neutral": "neutral"}
    )
    successful["is_direction_correct"] = (
        successful["signal_market_direction"] == successful["realized_direction"]
    )
    metrics["direction_accuracy"] = float(successful["is_direction_correct"].mean())

    long_rows = successful[successful["signal_direction"] == "long"]
    if not long_rows.empty:
        metrics["long_accuracy"] = float(
            (long_rows["realized_direction"] == "up").mean()
        )

    short_rows = successful[successful["signal_direction"] == "short"]
    if not short_rows.empty:
        metrics["short_accuracy"] = float(
            (short_rows["realized_direction"] == "down").mean()
        )

    strategy_returns = successful["strategy_return"].astype(float)
    mean_return = float(strategy_returns.mean())
    std_return = float(strategy_returns.std(ddof=0))
    metrics["mean_strategy_return"] = mean_return
    metrics["std_strategy_return"] = std_return
    metrics["total_pnl"] = float(strategy_returns.sum())
    if std_return > 0:
        metrics["annualized_sharpe"] = float((mean_return / std_return) * math.sqrt(52.0))

    successful["fingpt_direction"] = successful["fingpt_label"].map(
        {"up": "long", "down": "short", "neutral": "neutral"}
    )
    metrics["vs_fingpt_accuracy"] = float(
        (successful["signal_direction"] == successful["fingpt_direction"]).mean()
    )

    return metrics

