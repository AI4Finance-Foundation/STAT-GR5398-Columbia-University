"""
evaluation/evaluator.py — Signal quality evaluation without backtesting.

Methods:
1) Directional accuracy from 1-day forward returns via yfinance.
2) Confidence calibration (5 bins + ECE).
3) Self-consistency across repeated pipeline runs (N=3 by default).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from config import LOG_LEVEL, OUTPUT_DIR
from agent1.extractor import extract_fingerprint
from agent2.reasoner import generate_signal

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None  # type: ignore[assignment]


logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_CONF_BUCKETS: list[tuple[float, float, str, float]] = [
    (0.0, 0.2, "0.0-0.2", 0.1),
    (0.2, 0.4, "0.2-0.4", 0.3),
    (0.4, 0.6, "0.4-0.6", 0.5),
    (0.6, 0.8, "0.6-0.8", 0.7),
    (0.8, 1.0000001, "0.8-1.0", 0.9),
]


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _extract_close_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None

    close_col = None
    if "Close" in frame.columns:
        close_col = "Close"
    elif isinstance(frame.columns, pd.MultiIndex):
        for col in frame.columns:
            if isinstance(col, tuple) and col and col[0] == "Close":
                close_col = col
                break

    if close_col is None:
        return None

    close = frame[close_col].dropna()
    if close.empty:
        return None

    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.tz_convert("UTC")
    return close


def _closest_close_at_or_after(close: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    subset = close.loc[close.index >= ts]
    if subset.empty:
        return None
    return float(subset.iloc[0])


def _compute_signal_outcome(signal: dict[str, Any]) -> Optional[dict[str, Any]]:
    if yf is None:
        logger.warning("yfinance is not available; skipping price-based evaluation.")
        return None

    direction = str(signal.get("direction", "")).strip().lower()
    if direction not in {"long", "short"}:
        return None

    ticker = str(signal.get("ticker") or signal.get("source_ticker") or "").strip().upper()
    ts = _parse_timestamp(signal.get("published_at"))
    if not ticker or ts is None:
        logger.warning("Skipping signal with missing ticker/published_at: %s", signal)
        return None

    start = (ts - pd.Timedelta(hours=6)).to_pydatetime()
    end = (ts + pd.Timedelta(days=2)).to_pydatetime()

    try:
        bars = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None

    close = _extract_close_series(bars)
    if close is None:
        logger.warning("Missing close data for %s around %s", ticker, ts.isoformat())
        return None

    price_now = _closest_close_at_or_after(close, ts)
    price_next_day = _closest_close_at_or_after(close, ts + pd.Timedelta(days=1))
    if price_now is None or price_next_day is None:
        logger.warning("Insufficient forward prices for %s around %s", ticker, ts.isoformat())
        return None

    forward_return = (price_next_day - price_now) / price_now
    correct = (direction == "long" and forward_return > 0) or (
        direction == "short" and forward_return < 0
    )

    confidence = signal.get("confidence")
    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = 0.0

    return {
        "ticker": ticker,
        "direction": direction,
        "forward_return": float(forward_return),
        "correct": bool(correct),
        "confidence": max(0.0, min(1.0, confidence_val)),
    }


def evaluate_directional_accuracy(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluate non-neutral directional correctness using 1-day forward returns.
    """
    total_signals = len(signals)
    neutral_excluded = sum(1 for s in signals if str(s.get("direction", "")).lower() == "neutral")

    outcomes = []
    for signal in signals:
        outcome = _compute_signal_outcome(signal)
        if outcome is not None:
            outcomes.append(outcome)

    total_non_neutral = len(outcomes)
    correct = sum(1 for item in outcomes if item["correct"])
    long_outcomes = [o for o in outcomes if o["direction"] == "long"]
    short_outcomes = [o for o in outcomes if o["direction"] == "short"]

    long_correct = sum(1 for item in long_outcomes if item["correct"])
    short_correct = sum(1 for item in short_outcomes if item["correct"])

    return {
        "accuracy": (correct / total_non_neutral) if total_non_neutral else 0.0,
        "total_signals": total_signals,
        "neutral_excluded": neutral_excluded,
        "long_accuracy": (long_correct / len(long_outcomes)) if long_outcomes else 0.0,
        "short_accuracy": (short_correct / len(short_outcomes)) if short_outcomes else 0.0,
        "mean_forward_return_on_long": (
            sum(item["forward_return"] for item in long_outcomes) / len(long_outcomes)
            if long_outcomes
            else 0.0
        ),
        "mean_forward_return_on_short": (
            sum(item["forward_return"] for item in short_outcomes) / len(short_outcomes)
            if short_outcomes
            else 0.0
        ),
    }


def evaluate_calibration(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Bucket by confidence and compute directional accuracy per bucket.
    """
    outcomes = []
    for signal in signals:
        outcome = _compute_signal_outcome(signal)
        if outcome is not None:
            outcomes.append(outcome)

    buckets_payload: list[dict[str, Any]] = []
    ece_weighted_sum = 0.0
    n_total = len(outcomes)

    for low, high, label, midpoint in _CONF_BUCKETS:
        bucket_items = [o for o in outcomes if low <= o["confidence"] < high]
        count = len(bucket_items)
        acc = (sum(1 for item in bucket_items if item["correct"]) / count) if count else 0.0
        buckets_payload.append(
            {
                "confidence_range": label,
                "count": count,
                "accuracy": acc,
            }
        )
        if n_total:
            ece_weighted_sum += (count / n_total) * abs(acc - midpoint)

    return {
        "buckets": buckets_payload,
        "expected_calibration_error": ece_weighted_sum if n_total else 0.0,
    }


def _article_variant(article: str, run_idx: int) -> str:
    # Prompt variation for self-consistency checks without changing core agent code.
    suffix = "\n" * (run_idx + 1)
    return f"{article.strip()}{suffix}"


def evaluate_self_consistency(articles: list[str], n_runs: int = 3) -> dict[str, Any]:
    """
    Run the pipeline multiple times per article and measure direction agreement.

    The current public pipeline does not expose temperature knobs, so this uses
    slight prompt variations as a practical stochasticity proxy.
    """
    subset = [article for article in articles if article and article.strip()][:20]
    per_article: list[dict[str, Any]] = []

    for article in subset:
        directions: list[str] = []
        for run_idx in range(max(1, n_runs)):
            try:
                variant_text = _article_variant(article, run_idx)
                fingerprint = extract_fingerprint(variant_text)
                if fingerprint is None:
                    directions.append("error")
                    continue
                signal = generate_signal(fingerprint)
                directions.append(signal.direction if signal is not None else "error")
            except Exception as exc:
                logger.warning("Self-consistency run failed: %s", exc)
                directions.append("error")

        cleaned = [d for d in directions if d != "error"]
        consistent = bool(cleaned) and len(cleaned) == len(directions) and len(set(cleaned)) == 1
        headline = article.splitlines()[0].strip()[:160]
        per_article.append(
            {
                "headline": headline,
                "directions": directions,
                "consistent": consistent,
            }
        )

    consistent_count = sum(1 for item in per_article if item["consistent"])
    consistency_rate = (consistent_count / len(per_article)) if per_article else 0.0
    return {
        "consistency_rate": consistency_rate,
        "per_article": per_article,
    }


def _extract_articles_for_consistency(signals: list[dict[str, Any]]) -> list[str]:
    articles: list[str] = []
    for signal in signals:
        if isinstance(signal.get("article_text"), str) and signal["article_text"].strip():
            articles.append(signal["article_text"])
            continue
        headline = str(signal.get("headline", "")).strip()
        summary = str(signal.get("summary", "")).strip()
        if headline or summary:
            articles.append(f"{headline} {summary}".strip())
            continue
    return articles


def _print_summary(report: dict[str, Any]) -> None:
    direction = report["directional_accuracy"]
    calibration = report["calibration"]
    consistency = report["self_consistency"]

    print("\n=== FinGPT Signal Evaluation Summary ===")
    print(f"Signals file: {report['signals_file']}")
    print(f"Report file : {report['report_file']}")
    print("")
    print("Directional Accuracy")
    print("--------------------")
    print(f"Accuracy                    : {direction['accuracy']:.3f}")
    print(f"Total signals               : {direction['total_signals']}")
    print(f"Neutral excluded            : {direction['neutral_excluded']}")
    print(f"Long accuracy               : {direction['long_accuracy']:.3f}")
    print(f"Short accuracy              : {direction['short_accuracy']:.3f}")
    print(f"Mean fwd return on long     : {direction['mean_forward_return_on_long']:.4f}")
    print(f"Mean fwd return on short    : {direction['mean_forward_return_on_short']:.4f}")
    print("")
    print("Confidence Calibration")
    print("----------------------")
    for bucket in calibration["buckets"]:
        print(
            f"{bucket['confidence_range']:>8} | count={bucket['count']:>3} | "
            f"accuracy={bucket['accuracy']:.3f}"
        )
    print(f"ECE                         : {calibration['expected_calibration_error']:.3f}")
    print("")
    print("Self-Consistency")
    print("----------------")
    print(f"Consistency rate            : {consistency['consistency_rate']:.3f}")
    print(f"Articles evaluated          : {len(consistency['per_article'])}")
    print("========================================\n")


def run_full_evaluation(signals_file: str) -> dict[str, Any]:
    """
    Run all evaluation methods and save the combined report.
    """
    with open(signals_file, "r", encoding="utf-8") as handle:
        signals = json.load(handle)

    if not isinstance(signals, list):
        raise ValueError("signals_file must contain a JSON list of signal objects.")

    directional = evaluate_directional_accuracy(signals)
    calibration = evaluate_calibration(signals)
    articles = _extract_articles_for_consistency(signals)
    self_consistency = evaluate_self_consistency(articles, n_runs=3)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_file = os.path.join(OUTPUT_DIR, f"evaluation_{timestamp}.json")

    report = {
        "generated_at": timestamp,
        "signals_file": signals_file,
        "directional_accuracy": directional,
        "calibration": calibration,
        "self_consistency": self_consistency,
        "report_file": report_file,
    }

    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_summary(report)
    logger.info("Evaluation report saved to %s", report_file)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FinGPT signals without backtesting.")
    parser.add_argument("signals_file", help="Path to signals JSON file.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_full_evaluation(args.signals_file)


if __name__ == "__main__":
    main()
