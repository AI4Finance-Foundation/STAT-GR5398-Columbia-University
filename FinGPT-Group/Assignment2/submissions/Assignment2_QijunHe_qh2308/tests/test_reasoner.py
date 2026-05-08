"""
tests/test_reasoner.py — Unit tests for Agent 2 (Claude hypothesis reasoner).

The Anthropic API is mocked; no real API calls are made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent1.schema import NewsFingerprint
from agent2.schema import TradingSignal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fingerprint() -> NewsFingerprint:
    return NewsFingerprint(
        source="Bloomberg",
        published_at="2026-04-17T10:00:00Z",
        headline="NVIDIA Announces H200 Shipment Ahead of Schedule",
        companies_named=["NVDA"],
        figures_quoted={"shipment_advance": "2 weeks ahead of schedule"},
        event_keywords=["product launch", "supply chain", "ai chips"],
    )


def _make_claude_response(signal_dict: dict) -> MagicMock:
    """Build a mock Anthropic Messages response with one thinking + one text block."""
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "Step-by-step reasoning about NVDA supply news..."

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(signal_dict)

    mock_response = MagicMock()
    mock_response.content = [thinking_block, text_block]
    return mock_response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

VALID_SIGNAL = {
    "ticker": "NVDA",
    "direction": "long",
    "strategy_tag": "event_driven",
    "cot": "Early shipment of H200 GPUs is a positive supply signal. Event-driven setup favours near-term upside.",
}


class TestGenerateSignal:
    def test_valid_fingerprint_returns_trading_signal(self, sample_fingerprint):
        """A valid fingerprint should yield a TradingSignal."""
        mock_response = _make_claude_response(VALID_SIGNAL)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is not None
        assert isinstance(result, TradingSignal)

    def test_direction_is_valid(self, sample_fingerprint):
        """direction must be one of long, short, neutral."""
        mock_response = _make_claude_response(VALID_SIGNAL)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is not None
        assert result.direction in {"long", "short", "neutral"}

    def test_strategy_tag_is_valid(self, sample_fingerprint):
        """strategy_tag must be one of the five allowed values."""
        mock_response = _make_claude_response(VALID_SIGNAL)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is not None
        assert result.strategy_tag in {
            "momentum", "mean_reversion", "event_driven", "macro", "none"
        }

    def test_invalid_direction_returns_none(self, sample_fingerprint):
        """A response with an illegal direction value must return None."""
        bad_signal = {**VALID_SIGNAL, "direction": "bullish"}
        mock_response = _make_claude_response(bad_signal)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is None

    def test_invalid_strategy_tag_returns_none(self, sample_fingerprint):
        """A response with an illegal strategy_tag value must return None."""
        bad_signal = {**VALID_SIGNAL, "strategy_tag": "arbitrage"}
        mock_response = _make_claude_response(bad_signal)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is None

    def test_api_error_returns_none(self, sample_fingerprint):
        """An Anthropic API error must return None without crashing."""
        import anthropic

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking"),
        ):
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = anthropic.APIStatusError(
                "rate limit",
                response=MagicMock(status_code=429),
                body={},
            )
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            result = generate_signal(sample_fingerprint)

        assert result is None

    def test_thinking_block_is_logged(self, sample_fingerprint, tmp_path, monkeypatch):
        """The thinking block must be persisted; _log_thinking must be called."""
        mock_response = _make_claude_response(VALID_SIGNAL)

        with (
            patch("agent2.reasoner._get_client") as mock_get_client,
            patch("agent2.reasoner._log_thinking") as mock_log,
        ):
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            from agent2.reasoner import generate_signal

            generate_signal(sample_fingerprint)

        mock_log.assert_called_once()
