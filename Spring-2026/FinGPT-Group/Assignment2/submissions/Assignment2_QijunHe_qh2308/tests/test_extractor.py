"""
tests/test_extractor.py — Unit tests for Agent 1 (FinGPT fact extractor).

The model is mocked so no GPU or model checkpoint is required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent1.schema import NewsFingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint_json(**overrides) -> str:
    """Return a valid NewsFingerprint JSON string, optionally overriding fields."""
    base = {
        "source": "Reuters",
        "published_at": "2026-04-17T09:00:00Z",
        "headline": "Apple Reports Record Q2 Revenue of $94B, Up 8% YoY",
        "companies_named": ["AAPL"],
        "figures_quoted": {"revenue": "$94B", "growth": "+8% YoY"},
        "event_keywords": ["earnings", "revenue", "quarterly results"],
    }
    base.update(overrides)
    return json.dumps(base)


def _mock_model_output(raw_json: str):
    """
    Patch agent1.extractor so _load_model() is a no-op and the HuggingFace
    generate + decode pipeline returns raw_json.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # tokenizer(...) returns a dict-like object with .to() that returns itself
    encoded = MagicMock()
    encoded.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])  # fake input_ids
    encoded.to = MagicMock(return_value=encoded)
    mock_tokenizer.return_value = encoded
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = raw_json

    mock_model.device = "cpu"
    mock_model.generate.return_value = MagicMock()  # output_ids stub

    return mock_model, mock_tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractFingerprint:
    def test_valid_article_returns_fingerprint(self):
        """A well-formed article should produce a populated NewsFingerprint."""
        raw = _make_fingerprint_json()
        mock_model, mock_tokenizer = _mock_model_output(raw)

        with (
            patch("agent1.extractor._load_model"),
            patch("agent1.extractor._model", mock_model),
            patch("agent1.extractor._tokenizer", mock_tokenizer),
        ):
            from agent1.extractor import extract_fingerprint

            result = extract_fingerprint("Apple Reports Record Q2 Revenue of $94B, Up 8% YoY.")

        assert result is not None
        assert isinstance(result, NewsFingerprint)
        assert result.source == "Reuters"
        assert "AAPL" in result.companies_named
        assert result.figures_quoted["revenue"] == "$94B"
        # validator must have lowercased keywords
        assert all(kw == kw.lower() for kw in result.event_keywords)

    def test_no_company_returns_none(self):
        """An article whose fingerprint has an empty companies_named must return None."""
        raw = _make_fingerprint_json(companies_named=[])
        mock_model, mock_tokenizer = _mock_model_output(raw)

        with (
            patch("agent1.extractor._load_model"),
            patch("agent1.extractor._model", mock_model),
            patch("agent1.extractor._tokenizer", mock_tokenizer),
        ):
            from agent1.extractor import extract_fingerprint

            result = extract_fingerprint("Some vague market commentary with no company.")

        assert result is None

    def test_invalid_json_returns_none(self):
        """Malformed JSON from the model must return None without raising."""
        mock_model, mock_tokenizer = _mock_model_output("NOT_VALID_JSON{{{{")

        with (
            patch("agent1.extractor._load_model"),
            patch("agent1.extractor._model", mock_model),
            patch("agent1.extractor._tokenizer", mock_tokenizer),
        ):
            from agent1.extractor import extract_fingerprint

            result = extract_fingerprint("Any article text.")

        assert result is None

    def test_figures_quoted_values_are_strings(self):
        """figures_quoted values must be strings; numeric values must be rejected."""
        raw = _make_fingerprint_json(figures_quoted={"revenue": 94_000_000_000})
        mock_model, mock_tokenizer = _mock_model_output(raw)

        with (
            patch("agent1.extractor._load_model"),
            patch("agent1.extractor._model", mock_model),
            patch("agent1.extractor._tokenizer", mock_tokenizer),
        ):
            from agent1.extractor import extract_fingerprint

            result = extract_fingerprint("Apple revenue article.")

        assert result is None

    def test_event_keywords_coerced_to_lowercase(self):
        """event_keywords containing mixed-case strings must be lowercased."""
        raw = _make_fingerprint_json(event_keywords=["Earnings", "MERGER", "Rate Hike"])
        mock_model, mock_tokenizer = _mock_model_output(raw)

        with (
            patch("agent1.extractor._load_model"),
            patch("agent1.extractor._model", mock_model),
            patch("agent1.extractor._tokenizer", mock_tokenizer),
        ):
            from agent1.extractor import extract_fingerprint

            result = extract_fingerprint("Mixed case keyword article.")

        assert result is not None
        assert result.event_keywords == ["earnings", "merger", "rate hike"]
