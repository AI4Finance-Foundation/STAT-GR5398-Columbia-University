"""
pipeline.py — Orchestrates Agent 1 → Agent 2 end-to-end.

Entry point: run_pipeline()
Writes the final signals list to output/signals_<timestamp>.json.
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import LOG_LEVEL, OUTPUT_DIR
from ingestion.news_fetcher import fetch_recent_articles
from agent1.extractor import extract_fingerprint
from agent2.reasoner import generate_signal
from agent2.schema import TradingSignal

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def run_pipeline(
    tickers: list[str],
    limit: int = 20,
) -> list[TradingSignal]:
    """
    Fetch news, run Agent 1 (fact extraction) then Agent 2 (signal reasoning)
    for each article. Returns the list of valid TradingSignals produced.

    The full list is also written to output/signals_<timestamp>.json.
    """
    articles = fetch_recent_articles(tickers, limit)
    logger.info("Pipeline received %d articles to process.", len(articles))

    signals: list[TradingSignal] = []

    for i, article in enumerate(articles, start=1):
        article_text = article["headline"] + " " + article.get("summary", "")
        logger.info("[%d/%d] Extracting fingerprint...", i, len(articles))

        fingerprint = extract_fingerprint(article_text)
        if fingerprint is None:
            logger.info("[%d/%d] Fingerprint extraction failed — skipping.", i, len(articles))
            continue

        logger.info("[%d/%d] Generating signal for %s...", i, len(articles), fingerprint.companies_named)

        signal = generate_signal(fingerprint)
        if signal is None:
            logger.info("[%d/%d] Signal generation failed — skipping.", i, len(articles))
            continue

        signals.append(signal)
        logger.info(
            "[%d/%d] Signal: %s %s (%s)",
            i, len(articles),
            signal.ticker, signal.direction, signal.strategy_tag,
        )

    _save_signals(signals)
    logger.info("Pipeline complete. %d signal(s) produced.", len(signals))
    return signals


def _save_signals(signals: list[TradingSignal]) -> None:
    """Serialise signals to output/signals_<timestamp>.json."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filepath = os.path.join(OUTPUT_DIR, f"signals_{timestamp}.json")

    payload = [signal.model_dump() for signal in signals]
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Signals saved to %s", filepath)


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    run_pipeline(tickers, limit=20)
