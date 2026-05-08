#!/usr/bin/env python
"""Build point-in-time retail/social signals from StockTwits messages."""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINRL = ROOT / "finrl2" / "finrl_stock_selection.csv"
DEFAULT_OUTPUT = ROOT / "finrobot_tracka" / "retail_social_signals.csv"


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def fetch_stocktwits_messages(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_pages: int,
    sleep_seconds: float,
) -> list[dict]:
    """Fetch StockTwits messages by paging backward with max message id."""
    base_url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    messages: list[dict] = []
    max_id = None

    for _ in range(max_pages):
        params = {"limit": 30}
        if max_id is not None:
            params["max"] = max_id

        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(max(5.0, sleep_seconds * 5))
            continue
        response.raise_for_status()
        payload = response.json()
        page_messages = payload.get("messages", [])
        if not page_messages:
            break

        messages.extend(page_messages)
        created_dates = pd.to_datetime([m.get("created_at") for m in page_messages], errors="coerce").tz_localize(None)
        oldest = created_dates.min()
        newest = created_dates.max()
        min_message_id = min(int(m["id"]) for m in page_messages if m.get("id") is not None)
        max_id = min_message_id - 1

        if pd.notna(oldest) and oldest < start_date:
            break
        if pd.isna(oldest) or pd.isna(newest):
            break
        time.sleep(sleep_seconds)

    dedup = {}
    for message in messages:
        if message.get("id") is not None:
            dedup[message["id"]] = message
    return list(dedup.values())


def message_sentiment(message: dict) -> str:
    sentiment = ((message.get("entities") or {}).get("sentiment") or {}).get("basic")
    if str(sentiment).lower() == "bullish":
        return "bullish"
    if str(sentiment).lower() == "bearish":
        return "bearish"
    return "neutral"


def summarize_window(ticker: str, rebalance_date: pd.Timestamp, messages: pd.DataFrame, window_days: int) -> dict:
    start = rebalance_date - timedelta(days=window_days)
    window = messages[(messages["created_ts"] <= rebalance_date) & (messages["created_ts"] >= start)].copy()
    if window.empty:
        return {
            "date": rebalance_date.date().isoformat(),
            "ticker": ticker,
            "retail_window_days": window_days,
            "retail_message_count": 0,
            "retail_bullish_count": 0,
            "retail_bearish_count": 0,
            "retail_neutral_count": 0,
            "retail_bullish_pct": None,
            "retail_buzz_score": 0.0,
            "retail_social_score": 0.5,
            "retail_start_date": start.date().isoformat(),
            "retail_end_date": rebalance_date.date().isoformat(),
            "retail_top_messages": "",
        }

    total = int(len(window))
    bullish = int((window["sentiment"] == "bullish").sum())
    bearish = int((window["sentiment"] == "bearish").sum())
    neutral = int((window["sentiment"] == "neutral").sum())
    directional = 0.5 + 0.5 * (bullish - bearish) / max(bullish + bearish + 2, 1)
    coverage = min(total / 50.0, 1.0)
    social_score = clip01(0.75 * directional + 0.25 * coverage)
    bullish_pct = 100.0 * bullish / max(bullish + bearish, 1) if (bullish + bearish) > 0 else None
    buzz = min(total / 100.0, 1.0) * 100.0
    top_messages = " | ".join(window.sort_values("created_ts", ascending=False)["body"].head(3).str.replace(r"\s+", " ", regex=True))
    return {
        "date": rebalance_date.date().isoformat(),
        "ticker": ticker,
        "retail_window_days": window_days,
        "retail_message_count": total,
        "retail_bullish_count": bullish,
        "retail_bearish_count": bearish,
        "retail_neutral_count": neutral,
        "retail_bullish_pct": round(bullish_pct, 2) if bullish_pct is not None else None,
        "retail_buzz_score": round(buzz, 2),
        "retail_social_score": round(social_score, 4),
        "retail_start_date": start.date().isoformat(),
        "retail_end_date": rebalance_date.date().isoformat(),
        "retail_top_messages": top_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finrl-csv", default=str(DEFAULT_FINRL))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-pages", type=int, default=400)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    args = parser.parse_args()

    finrl = pd.read_csv(args.finrl_csv)
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    selected["date"] = pd.to_datetime(selected["date"]).dt.tz_localize(None)
    tickers = sorted(selected["ticker"].unique())
    global_start = selected["date"].min() - timedelta(days=args.window_days)
    global_end = selected["date"].max()

    rows = []
    for ticker in tickers:
        print(f"Fetching StockTwits messages for {ticker}")
        raw = fetch_stocktwits_messages(ticker, global_start, global_end, args.max_pages, args.sleep_seconds)
        messages = pd.DataFrame(raw)
        if messages.empty:
            messages = pd.DataFrame(columns=["created_at", "body"])
        for col in ["created_at", "body"]:
            if col not in messages.columns:
                messages[col] = ""
        messages["created_ts"] = pd.to_datetime(messages["created_at"], errors="coerce").dt.tz_localize(None)
        messages = messages.dropna(subset=["created_ts"]).copy()
        messages = messages[(messages["created_ts"] >= global_start) & (messages["created_ts"] <= global_end)].copy()
        messages["sentiment"] = [message_sentiment(m) for m in messages.to_dict("records")]

        for date in sorted(selected.loc[selected["ticker"] == ticker, "date"].unique()):
            rows.append(summarize_window(ticker, pd.Timestamp(date), messages, args.window_days))

    output = pd.DataFrame(rows).sort_values(["date", "ticker"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    print(output[["date", "ticker", "retail_message_count", "retail_bullish_count", "retail_bearish_count", "retail_social_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
