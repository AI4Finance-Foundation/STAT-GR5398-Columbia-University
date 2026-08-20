"""Deterministic scoring rules for FinRobot research flags."""

from __future__ import annotations

from dataclasses import dataclass


FLAG_MULTIPLIERS = {
    "risk_flag": {"low": 0.00, "medium": -0.05, "high": -0.20},
    "valuation_flag": {"cheap": 0.10, "fair": 0.00, "expensive": -0.10, "extreme": -0.15},
    "catalyst_flag": {"positive": 0.10, "neutral": 0.00, "negative": -0.10},
    "sentiment_flag": {"positive": 0.05, "mixed": 0.00, "negative": -0.10},
    "thesis_strength": {"strong": 0.15, "moderate": 0.00, "weak": -0.15},
}

MULTIPLIER_MIN = 0.40
MULTIPLIER_MAX = 1.25


def decision_from_multiplier(multiplier: float) -> str:
    if multiplier < 0.60:
        return "cut"
    if multiplier < 0.90:
        return "reduce"
    if multiplier <= 1.10:
        return "keep"
    return "boost"


def score_flags(flags: dict) -> tuple[float, dict]:
    multiplier = 1.0
    detail = {}
    for field, mapping in FLAG_MULTIPLIERS.items():
        value = str(flags.get(field, "")).strip().lower()
        delta = mapping.get(value)
        if delta is None:
            raise ValueError(f"Invalid {field}: {flags.get(field)!r}")
        multiplier += delta
        detail[field] = {"value": value, "delta": delta}

    multiplier = max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, multiplier))
    detail["multiplier"] = multiplier
    detail["decision"] = decision_from_multiplier(multiplier)
    return multiplier, detail

