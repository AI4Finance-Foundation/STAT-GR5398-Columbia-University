#!/usr/bin/env python
"""Build point-in-time news, catalyst, and risk signals from FMP stock news."""

from __future__ import annotations

import argparse
import configparser
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINRL = ROOT / "finrl2" / "finrl_stock_selection.csv"
DEFAULT_CONFIG = ROOT / "external" / "FinRobot" / "finrobot_equity" / "core" / "config" / "config.ini"
DEFAULT_OUTPUT = ROOT / "finrobot_tracka" / "historical_news_signals.csv"
DEFAULT_COMPANY_MAP = ROOT / "finrobot_tracka" / "company_peer_map.csv"


CATEGORY_KEYWORDS = {
    "earnings": ["earnings", "revenue", "profit", "quarterly", "guidance", "eps", "beat", "miss", "forecast"],
    "product": ["launch", "product", "release", "unveil", "announce", "innovation", "technology", "service"],
    "management": ["ceo", "cfo", "executive", "board", "leadership", "appoint", "resign", "hire"],
    "regulatory": ["fda", "sec", "regulation", "compliance", "lawsuit", "legal", "investigation", "settlement"],
    "market": ["market", "stock", "share", "price", "analyst", "upgrade", "downgrade", "target", "rating"],
    "acquisition": ["acquire", "merger", "acquisition", "deal", "buyout", "partnership", "joint venture"],
    "financial": ["debt", "bond", "dividend", "buyback", "capital", "financing", "credit", "loan"],
}

POSITIVE_KEYWORDS = [
    "growth", "increase", "beat", "exceed", "strong", "upgrade", "success", "gain", "improve", "record",
    "surge", "outperform", "bullish", "optimistic", "overweight", "buy rating", "raises target",
    "price target raised", "top pick", "partnership", "expansion",
]
NEGATIVE_KEYWORDS = [
    "decline", "decrease", "miss", "weak", "downgrade", "loss", "fail", "drop", "concern", "risk",
    "challenge", "plunge", "underperform", "bearish", "warning", "underweight", "sell rating",
    "recall", "investigation", "lawsuit", "cuts target", "lowers target",
]
RISK_TERMS = {
    "lawsuit": 1.3,
    "investigation": 1.3,
    "recall": 1.2,
    "downgrade": 1.2,
    "cuts target": 1.1,
    "lowers target": 1.1,
    "margin": 1.0,
    "guidance cut": 1.3,
    "miss": 1.0,
    "weak": 0.9,
    "regulatory": 1.2,
    "competition": 0.8,
    "supply": 0.8,
    "execution": 1.1,
}

RELEVANCE_ALIASES = {
    "AMAT": ["applied materials", "amat", "semiconductor equipment", "chipmaking equipment"],
    "AMD": [
        "advanced micro devices",
        "amd",
        "semiconductor",
        "chip",
        "gpu",
        "cpu",
        "radeon",
        "ryzen",
        "epyc",
        "instinct",
        "data center",
        "ai accelerator",
    ],
    "AMZN": ["amazon", "amzn", "aws", "amazon web services"],
    "COST": ["costco", "cost"],
    "CSCO": ["cisco", "csco"],
    "GOOGL": ["alphabet", "google", "googl", "youtube", "waymo", "gemini"],
    "ISRG": ["intuitive surgical", "isrg", "da vinci surgical", "da vinci system"],
    "META": ["meta platforms", "meta", "facebook", "instagram", "whatsapp", "threads"],
    "NFLX": ["netflix", "nflx"],
    "TSLA": ["tesla", "tsla", "elon musk", "model y", "model 3", "cybertruck"],
    "TXN": ["texas instruments", "txn", "analog chip", "embedded processor"],
}

IRRELEVANT_PATTERNS = {
    "AMD": [
        r"\bwet amd\b",
        r"\bdry amd\b",
        r"age[- ]related macular degeneration",
        r"\bmacular degeneration\b",
        r"\bbevacizumab\b",
        r"\blytenava\b",
        r"\bophthalm",
    ],
    "META": [r"\bmeta-analysis\b"],
}


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_fmp_key(config_file: Path) -> str:
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.get("API_KEYS", "fmp_api_key")


def fetch_stock_news(ticker: str, start: str, end: str, limit: int, api_key: str) -> list[dict]:
    url = "https://financialmodelingprep.com/api/v3/stock_news"
    params = {"tickers": ticker, "from": start, "to": end, "limit": limit, "apikey": api_key}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def load_company_aliases(company_map: Path) -> dict[str, list[str]]:
    aliases = {ticker: values.copy() for ticker, values in RELEVANCE_ALIASES.items()}
    if not company_map.exists():
        return aliases
    company_df = pd.read_csv(company_map)
    for row in company_df.itertuples(index=False):
        ticker = str(row.ticker).upper()
        company_name = str(row.company_name).lower()
        cleaned_name = re.sub(r"\b(inc|incorporated|corporation|corp|ltd|plc|class a)\b\.?", "", company_name)
        cleaned_name = re.sub(r"[^a-z0-9 ]+", " ", cleaned_name)
        terms = aliases.setdefault(ticker, [ticker.lower()])
        for term in {company_name, cleaned_name.strip(), ticker.lower()}:
            if term and term not in terms:
                terms.append(term)
    return aliases


def is_relevant_article(ticker: str, text: str, aliases: dict[str, list[str]]) -> bool:
    text_lower = text.lower()
    for pattern in IRRELEVANT_PATTERNS.get(ticker, []):
        if re.search(pattern, text_lower):
            return False

    terms = aliases.get(ticker, [ticker.lower()])
    for term in terms:
        term = term.lower().strip()
        if not term:
            continue
        if len(term) <= 5 and term.isalnum():
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text_lower):
                return True
        elif term in text_lower:
            return True
    return False


def build_articles_frame(raw: list[dict], ticker: str, aliases: dict[str, list[str]]) -> tuple[pd.DataFrame, int]:
    articles = pd.DataFrame(raw)
    if articles.empty:
        articles = pd.DataFrame(columns=["title", "text", "publishedDate", "site", "url"])
    for col in ["title", "text", "publishedDate", "site", "url"]:
        if col not in articles.columns:
            articles[col] = ""
    articles["published_ts"] = pd.to_datetime(articles["publishedDate"], errors="coerce").dt.tz_localize(None)
    articles = articles.dropna(subset=["published_ts"]).copy()
    articles["combined_text"] = (articles["title"].fillna("") + " " + articles["text"].fillna("")).str.strip()
    raw_count = len(articles)
    articles = articles[articles["combined_text"].apply(lambda text: is_relevant_article(ticker, text, aliases))].copy()
    articles["category"] = articles["combined_text"].apply(classify_category)
    articles["sentiment"] = articles["combined_text"].apply(classify_sentiment)
    articles["risk_weight"] = articles["combined_text"].apply(risk_weight)
    articles["importance"] = articles["category"].map(
        {"earnings": 4, "acquisition": 4, "regulatory": 4, "market": 3, "financial": 3, "product": 3, "management": 3}
    ).fillna(2)
    return articles, raw_count


def classify_category(text: str) -> str:
    text_lower = text.lower()
    scores = {
        category: sum(1 for kw in keywords if kw in text_lower)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def classify_sentiment(text: str) -> str:
    text_lower = text.lower()
    positive = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    negative = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    if positive > negative + 1:
        return "positive"
    if negative > positive + 1:
        return "negative"
    return "neutral"


def risk_weight(text: str) -> float:
    text_lower = text.lower()
    return sum(weight for term, weight in RISK_TERMS.items() if term in text_lower)


def summarize_window(
    ticker: str,
    rebalance_date: pd.Timestamp,
    articles: pd.DataFrame,
    window_days: int,
    raw_news_count: int,
) -> dict:
    start = rebalance_date - timedelta(days=window_days)
    window = articles[(articles["published_ts"] <= rebalance_date) & (articles["published_ts"] >= start)].copy()
    if window.empty:
        return {
            "date": rebalance_date.date().isoformat(),
            "ticker": ticker,
            "news_window_days": window_days,
            "raw_news_count": raw_news_count,
            "filtered_news_count": 0,
            "news_count": 0,
            "positive_news": 0,
            "negative_news": 0,
            "neutral_news": 0,
            "high_impact_news": 0,
            "risk_news_count": 0,
            "risk_weighted_hits": 0.0,
            "historical_catalyst_score": 0.5,
            "historical_risk_score": 0.5,
            "news_start_date": start.date().isoformat(),
            "news_end_date": rebalance_date.date().isoformat(),
            "top_news_titles": "",
        }

    pos = int((window["sentiment"] == "positive").sum())
    neg = int((window["sentiment"] == "negative").sum())
    neu = int((window["sentiment"] == "neutral").sum())
    total = int(len(window))
    directional = (pos + 1) / (pos + neg + 2)
    coverage = min(total / 8.0, 1.0)
    catalyst = clip01(0.70 * directional + 0.30 * coverage)

    risk_hits = float(window["risk_weight"].sum())
    risk_count = int((window["risk_weight"] > 0).sum())
    risk_intensity = risk_hits / total
    risk_article_rate = risk_count / total
    intensity_penalty = clip01((risk_intensity - 0.05) / 0.30)
    breadth_penalty = clip01((risk_article_rate - 0.05) / 0.25)
    risk_penalty = 0.65 * intensity_penalty + 0.35 * breadth_penalty
    risk_score = clip01(1.0 - 0.75 * risk_penalty)
    top_titles = " | ".join(window.sort_values(["importance", "published_ts"], ascending=[False, False])["title"].head(3))
    return {
        "date": rebalance_date.date().isoformat(),
        "ticker": ticker,
        "news_window_days": window_days,
        "raw_news_count": raw_news_count,
        "filtered_news_count": total,
        "news_count": total,
        "positive_news": pos,
        "negative_news": neg,
        "neutral_news": neu,
        "high_impact_news": int((window["importance"] >= 4).sum()),
        "risk_news_count": risk_count,
        "risk_weighted_hits": round(risk_hits, 4),
        "historical_catalyst_score": round(catalyst, 4),
        "historical_risk_score": round(risk_score, 4),
        "news_start_date": start.date().isoformat(),
        "news_end_date": rebalance_date.date().isoformat(),
        "top_news_titles": top_titles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finrl-csv", default=str(DEFAULT_FINRL))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--company-map-csv", default=str(DEFAULT_COMPANY_MAP))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    finrl = pd.read_csv(args.finrl_csv)
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    selected["date"] = pd.to_datetime(selected["date"]).dt.tz_localize(None)
    tickers = sorted(selected["ticker"].unique())
    api_key = load_fmp_key(Path(args.config_file))
    aliases = load_company_aliases(Path(args.company_map_csv))

    rows = []
    for ticker in tickers:
        for date in sorted(selected.loc[selected["ticker"] == ticker, "date"].unique()):
            rebalance_date = pd.Timestamp(date)
            start = (rebalance_date - timedelta(days=args.window_days)).date().isoformat()
            end = rebalance_date.date().isoformat()
            print(f"Fetching historical news for {ticker}: {start} to {end}")
            raw = fetch_stock_news(ticker, start, end, args.limit, api_key)
            articles, raw_news_count = build_articles_frame(raw, ticker, aliases)
            rows.append(summarize_window(ticker, rebalance_date, articles, args.window_days, raw_news_count))

    output = pd.DataFrame(rows).sort_values(["date", "ticker"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    print(output[["date", "ticker", "raw_news_count", "filtered_news_count", "positive_news", "negative_news", "historical_catalyst_score", "historical_risk_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
