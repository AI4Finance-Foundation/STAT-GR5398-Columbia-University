"""
agent1/schema.py — Pydantic schema for Agent 1 output.

Sentiment is now expressed as a 3-class label (POSITIVE / NEGATIVE / NEUTRAL)
derived deterministically from LLM-reported logits via softmax post-processing.
The NewsFingerprint also carries the full probability vector and calibration
temperature so downstream consumers can reproduce the probability calculation.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# 3-class sentiment aligned with the logits prompt order:
#   index 0 → POSITIVE,  index 1 → NEGATIVE,  index 2 → NEUTRAL
SentimentLabel = Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Numeric score derived from sentiment label (for backward compat with backtester).
_SENTIMENT_SCORE: dict[str, float] = {
    "POSITIVE": 1.0,
    "NEGATIVE": -1.0,
    "NEUTRAL": 0.0,
}


class NewsFingerprint(BaseModel):
    # -----------------------------------------------------------------------
    # Fact-extraction fields (unchanged from previous version)
    # -----------------------------------------------------------------------
    source: str
    published_at: str
    headline: str
    companies_named: list[str]
    event_keywords: list[str]

    # -----------------------------------------------------------------------
    # Sentiment fields — now populated from logits post-processing
    # -----------------------------------------------------------------------
    sentiment_label: SentimentLabel

    # Scalar score kept for backward compatibility with backtester metrics.
    # Derived from label: POSITIVE=1.0, NEUTRAL=0.0, NEGATIVE=-1.0.
    sentiment_score: float = Field(ge=-1.0, le=1.0)

    # max(softmax(logits / calibration_T)) — true calibrated confidence.
    sentiment_confidence: float = Field(ge=0.0, le=1.0)

    # Full calibrated probability vector, e.g.
    #   {"POSITIVE": 0.87, "NEGATIVE": 0.08, "NEUTRAL": 0.05}
    sentiment_probabilities: dict[str, float] = Field(default_factory=dict)

    # Temperature used for softmax calibration (from config.CALIBRATION_T).
    calibration_T: float = Field(default=1.0, ge=0.0)

    # Raw logits reported by the LLM (before softmax), stored for auditing.
    sentiment_logits: Optional[list[float]] = None

    # Original article text — stored so Agent 2 can include it in its prompt
    # without requiring a separate argument to generate_signal().
    article_text: str = ""

    @field_validator("companies_named")
    @classmethod
    def companies_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("no company identified - skip article")
        return value

    @field_validator("event_keywords")
    @classmethod
    def keywords_must_be_lowercase(cls, value: list[str]) -> list[str]:
        return [keyword.lower() for keyword in value]
