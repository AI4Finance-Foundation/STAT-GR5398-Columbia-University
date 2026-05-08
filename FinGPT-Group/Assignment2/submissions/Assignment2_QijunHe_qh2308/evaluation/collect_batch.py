"""
evaluation/collect_batch.py — Build a held-out evaluation batch.

Fetches recent Alpaca news and runs each article through Agent 1 and Agent 2.
Outputs successful records to output/held_out_batch_<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from config import LOG_LEVEL, OUTPUT_DIR
from ingestion.news_fetcher import fetch_recent_articles
from agent1.extractor import extract_fingerprint
from agent2.reasoner import generate_signal

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD"]
DEFAULT_BATCH_SIZE = 100


def collect_held_out_batch(
    tickers: list[str] | None = None,
    target_count: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """
    Fetch and process a held-out batch through Agent 1 -> Agent 2.

    Returns only successful records in the shape:
    {
      "article": {...},
      "fingerprint": {...},
      "signal": {...}
    }
    """
    chosen_tickers = tickers or DEFAULT_TICKERS
    articles = fetch_recent_articles(chosen_tickers, limit=target_count)
    logger.info(
        "Collected %d article(s) from Alpaca (requested=%d).",
        len(articles),
        target_count,
    )

    records: list[dict[str, Any]] = []
    skipped_agent1 = 0
    skipped_agent2 = 0

    for idx, article in enumerate(articles, start=1):
        headline = str(article.get("headline", "") or "")
        summary = str(article.get("summary", "") or "")
        article_text = f"{headline} {summary}".strip()

        if not article_text:
            skipped_agent1 += 1
            logger.warning("[%d/%d] Skipped: Agent 1 failed (empty article text).", idx, len(articles))
            continue

        fingerprint = extract_fingerprint(article_text)
        if fingerprint is None:
            skipped_agent1 += 1
            logger.warning("[%d/%d] Skipped: Agent 1 failed.", idx, len(articles))
            continue

        signal = generate_signal(fingerprint)
        if signal is None:
            skipped_agent2 += 1
            logger.warning("[%d/%d] Skipped: Agent 2 failed.", idx, len(articles))
            continue

        records.append(
            {
                "article": {
                    "headline": headline,
                    "summary": summary,
                    "source": str(article.get("source", "") or ""),
                    "published_at": str(article.get("created_at", "") or ""),
                },
                "fingerprint": fingerprint.model_dump(),
                "signal": signal.model_dump(),
            }
        )

        logger.info(
            "[%d/%d] Success: %s %s (%s)",
            idx,
            len(articles),
            signal.ticker,
            signal.direction,
            signal.strategy_tag,
        )

    logger.info(
        "Held-out processing complete: %d success, %d skipped (Agent 1), %d skipped (Agent 2).",
        len(records),
        skipped_agent1,
        skipped_agent2,
    )
    return records


def save_held_out_batch(records: list[dict[str, Any]]) -> str:
    """Save held-out batch records to output/held_out_batch_<timestamp>.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = os.path.join(OUTPUT_DIR, f"held_out_batch_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    logger.info("Held-out batch saved to %s", filepath)
    return filepath


def run() -> str:
    records = collect_held_out_batch(DEFAULT_TICKERS, DEFAULT_BATCH_SIZE)
    return save_held_out_batch(records)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect held-out batch for evaluation.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to query from Alpaca.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Target number of recent articles to request.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    records = collect_held_out_batch(tickers=args.tickers, target_count=args.count)
    output_path = save_held_out_batch(records)
    print(f"Saved {len(records)} held-out records to {output_path}")


if __name__ == "__main__":
    main()
