# portfolio_strategy/finrobot_score.py

import os
import json
import pandas as pd


def normalize_score(x):
    """
    Convert 0-10 score to 0-1 score.
    """
    try:
        x = float(x)
        return max(0.0, min(1.0, x / 10.0))
    except Exception:
        return None


def read_text_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_finrobot_text_for_ticker(output_root, ticker):
    """
    Load FinRobot-generated analysis text for one ticker.

    Expected folder examples:
    output/AAPL/hybrid_analysis/
    output/AAPL/gemini_analysis/
    output/AAPL/claude_analysis/
    output/AAPL/gpt41_analysis/
    output/AAPL/analysis/

    The function prioritizes hybrid_analysis because it contains the richest
    combined output from FinRobot, including catalysts, earnings, news,
    SEC filing analysis, valuation, risks, and summaries.
    """
    possible_dirs = [
        os.path.join(output_root, ticker, "hybrid_analysis"),
        os.path.join(output_root, ticker, "gemini_analysis"),
        os.path.join(output_root, ticker, "claude_analysis"),
        os.path.join(output_root, ticker, "gpt41_analysis"),
        os.path.join(output_root, ticker, "analysis"),
    ]

    sections = [
        "analysis_summary.json",
        "catalyst_analysis.json",
        "catalyst_summary.md",
        "company_news.json",
        "company_overview.txt",
        "competitor_analysis.txt",
        "earnings_analysis.json",
        "earnings_summary.md",
        "enhanced_news.json",
        "investment_overview.txt",
        "major_takeaways.txt",
        "news_summary.md",
        "news_summary.txt",
        "retail_sentiment.json",
        "revenue_driver_analysis.json",
        "risks.txt",
        "sec_filing_analysis.json",
        "sec_filing_summary.md",
        "sensitivity_analysis.json",
        "sensitivity_summary.md",
        "tagline.txt",
        "valuation_overview.txt",
    ]

    combined_text = ""

    for folder in possible_dirs:
        if not os.path.exists(folder):
            continue

        for section in sections:
            section_path = os.path.join(folder, section)
            text = read_text_file(section_path)
            if text:
                combined_text += f"\n\n## {os.path.basename(folder)}/{section}\n{text}"

    return combined_text.strip()


def simple_heuristic_finrobot_score(text):
    """
    Original simple heuristic FinRobot scoring fallback.

    This is kept for ablation comparison against the risk-aware FinRobot score.
    It rewards positive research-language terms and penalizes negative/risk terms.
    """
    text_lower = text.lower()

    positive_terms = [
        "strong", "growth", "margin", "competitive", "leadership",
        "cash flow", "profitability", "demand", "expansion",
        "resilient", "innovation", "market share"
    ]

    negative_terms = [
        "risk", "decline", "pressure", "competition", "valuation",
        "uncertainty", "slowdown", "regulatory", "margin pressure",
        "cyclical", "weakness"
    ]

    positive_count = sum(text_lower.count(term) for term in positive_terms)
    negative_count = sum(text_lower.count(term) for term in negative_terms)

    raw = 0.5 + 0.03 * positive_count - 0.02 * negative_count
    score = max(0.0, min(1.0, raw))

    return {
        "finrobot_score": score,
        "growth_score": score,
        "moat_score": score,
        "valuation_risk_score": 1.0 - score,
        "business_risk_score": 1.0 - score,
        "scoring_method": "simple_heuristic_text_score"
    }


def risk_aware_finrobot_score(text):
    """
    Risk-aware heuristic FinRobot scoring fallback.

    This converts FinRobot hybrid_analysis text into a normalized company score
    when an LLM-based structured scorer is not available yet.

    FinRobot Score =
        25% Growth Quality
      + 20% Competitive Position
      + 20% Earnings Quality
      + 15% Catalyst Strength
      + 10% Valuation Attractiveness
      - 10% Business Risk
      - 10% Valuation Risk

    All component scores are clipped to the 0-1 range.
    """
    text_lower = text.lower()

    def count_terms(terms):
        return sum(text_lower.count(term) for term in terms)

    def positive_component_score(positive_terms, negative_terms=None, base=0.50, pos_weight=0.035, neg_weight=0.025):
        if negative_terms is None:
            negative_terms = []
        raw = base + pos_weight * count_terms(positive_terms) - neg_weight * count_terms(negative_terms)
        return max(0.0, min(1.0, raw))

    def risk_component_score(risk_terms, offset_terms=None, base=0.35, risk_weight=0.035, offset_weight=0.020):
        if offset_terms is None:
            offset_terms = []
        raw = base + risk_weight * count_terms(risk_terms) - offset_weight * count_terms(offset_terms)
        return max(0.0, min(1.0, raw))

    growth_quality_score = positive_component_score(
        positive_terms=[
            "growth", "revenue growth", "earnings growth", "expansion",
            "demand", "secular", "accelerating", "market opportunity",
            "ai demand", "cloud growth", "subscription growth"
        ],
        negative_terms=[
            "slowdown", "decline", "weak demand", "deceleration",
            "revenue pressure", "growth pressure"
        ],
    )

    competitive_position_score = positive_component_score(
        positive_terms=[
            "competitive", "leadership", "market share", "moat",
            "scale", "ecosystem", "brand", "pricing power",
            "differentiated", "dominant", "leader"
        ],
        negative_terms=[
            "competition", "competitive pressure", "share loss",
            "commoditized", "disruption"
        ],
    )

    earnings_quality_score = positive_component_score(
        positive_terms=[
            "margin", "operating margin", "gross margin", "profitability",
            "cash flow", "free cash flow", "earnings quality", "recurring revenue",
            "operating leverage", "resilient margin"
        ],
        negative_terms=[
            "margin pressure", "earnings pressure", "profit decline",
            "cash burn", "losses", "negative free cash flow"
        ],
    )

    catalyst_strength_score = positive_component_score(
        positive_terms=[
            "catalyst", "upside", "product launch", "ai", "innovation",
            "earnings beat", "guidance raise", "new product", "partnership",
            "buyback", "restructuring", "monetization"
        ],
        negative_terms=[
            "lack of catalyst", "limited upside", "guidance cut",
            "negative catalyst", "execution risk"
        ],
    )

    valuation_risk_score = risk_component_score(
        risk_terms=[
            "valuation risk", "overvalued", "expensive", "premium valuation",
            "multiple compression", "high multiple", "rich valuation",
            "priced in", "downside risk"
        ],
        offset_terms=[
            "attractive valuation", "reasonable valuation", "undervalued",
            "discount", "valuation support", "free cash flow yield"
        ],
    )

    business_risk_score = risk_component_score(
        risk_terms=[
            "risk", "uncertainty", "regulatory", "cyclical", "weakness",
            "slowdown", "decline", "pressure", "execution risk",
            "supply chain", "geopolitical", "litigation", "competition risk"
        ],
        offset_terms=[
            "resilient", "diversified", "stable", "recurring", "defensive",
            "strong balance sheet", "cash flow", "durable"
        ],
    )

    valuation_attractiveness_score = 1.0 - valuation_risk_score

    finrobot_score = (
        0.25 * growth_quality_score
        + 0.20 * competitive_position_score
        + 0.20 * earnings_quality_score
        + 0.15 * catalyst_strength_score
        + 0.10 * valuation_attractiveness_score
        - 0.10 * business_risk_score
        - 0.10 * valuation_risk_score
    )
    finrobot_score = max(0.0, min(1.0, finrobot_score))

    return {
        "finrobot_score": finrobot_score,
        "growth_quality_score": growth_quality_score,
        "competitive_position_score": competitive_position_score,
        "earnings_quality_score": earnings_quality_score,
        "catalyst_strength_score": catalyst_strength_score,
        "valuation_attractiveness_score": valuation_attractiveness_score,
        "business_risk_score": business_risk_score,
        "valuation_risk_score": valuation_risk_score,
        "scoring_method": "risk_aware_heuristic_text_score"
    }


def heuristic_finrobot_score(text):
    """
    Backward-compatible default scorer.
    Uses the risk-aware FinRobot score as the default version.
    """
    return risk_aware_finrobot_score(text)


def build_finrobot_scores(universe, output_root="./output", output_csv="./output/TECH_STRATEGY/finrobot_scores.csv"):
    """
    Build FinRobot scores for all tickers with available FinRobot reports.
    """
    rows = []

    for ticker in universe:
        text = load_finrobot_text_for_ticker(output_root, ticker)

        if not text:
            print(f"No FinRobot text found for {ticker}; skipping.")
            continue

        simple_scores = simple_heuristic_finrobot_score(text)
        risk_aware_scores = risk_aware_finrobot_score(text)

        scores = {
            "finrobot_score": risk_aware_scores["finrobot_score"],
            "simple_finrobot_score": simple_scores["finrobot_score"],
            "risk_aware_finrobot_score": risk_aware_scores["finrobot_score"],
            "simple_growth_score": simple_scores.get("growth_score"),
            "simple_moat_score": simple_scores.get("moat_score"),
            "growth_quality_score": risk_aware_scores.get("growth_quality_score"),
            "competitive_position_score": risk_aware_scores.get("competitive_position_score"),
            "earnings_quality_score": risk_aware_scores.get("earnings_quality_score"),
            "catalyst_strength_score": risk_aware_scores.get("catalyst_strength_score"),
            "valuation_attractiveness_score": risk_aware_scores.get("valuation_attractiveness_score"),
            "business_risk_score": risk_aware_scores.get("business_risk_score"),
            "valuation_risk_score": risk_aware_scores.get("valuation_risk_score"),
            "scoring_method": "simple_and_risk_aware_heuristic_scores"
        }

        row = {
            "ticker": ticker,
            **scores
        }
        rows.append(row)

    scores_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scores_df.to_csv(output_csv, index=False)

    print(f"Saved FinRobot scores to {output_csv}")
    return scores_df


def load_finrobot_scores(path):
    """
    Load saved FinRobot scores.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df