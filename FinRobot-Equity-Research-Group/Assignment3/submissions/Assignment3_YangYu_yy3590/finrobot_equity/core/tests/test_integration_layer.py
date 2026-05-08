#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

CORE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CORE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from modules import market_data_api
from modules.market_data_api import _apply_as_of_filter
from modules.news_integrator import NewsIntegrator
from integrations.active_setup import load_active_research_setup
from integrations.flag_rules import decision_from_multiplier, score_flags
from integrations.research_flags import _parse_llm_json, _validate_payload
from integrations.research_automation import build_analysis_command
from integrations.weights import load_base_weights, apply_research_adjustment


def test_flag_scoring_rules():
    multiplier, detail = score_flags(
        {
            "risk_flag": "high",
            "valuation_flag": "expensive",
            "catalyst_flag": "negative",
            "sentiment_flag": "negative",
            "thesis_strength": "weak",
        }
    )
    assert round(multiplier, 2) == 0.40
    assert detail["decision"] == "cut"
    assert decision_from_multiplier(1.15) == "boost"


def test_validate_payload_schema():
    flags = _validate_payload(
        {
            "risk_flag": "medium",
            "valuation_flag": "fair",
            "catalyst_flag": "neutral",
            "sentiment_flag": "mixed",
            "thesis_strength": "moderate",
        },
        ticker="AAPL",
        as_of_date="2026-01-01",
    )
    assert flags.ticker == "AAPL"
    assert flags.risk_flag == "medium"


def test_parse_llm_json_accepts_markdown_fence():
    payload = _parse_llm_json(
        """```json
{
  "risk_flag": "high",
  "valuation_flag": "expensive",
  "catalyst_flag": "neutral",
  "sentiment_flag": "mixed",
  "thesis_strength": "moderate"
}
```"""
    )
    assert payload["risk_flag"] == "high"


def test_base_weights_and_adjustment(tmp_path: Path):
    csv_path = tmp_path / "weights.csv"
    pd.DataFrame(
        {
            "ticker": ["AAPL", "SPY", "CASH"],
            "weight": [0.50, 0.30, 0.20],
        }
    ).to_csv(csv_path, index=False)

    base = load_base_weights(csv_path)
    adjusted, audit = apply_research_adjustment(
        base,
        {
            "AAPL": {
                "risk_flag": "high",
                "valuation_flag": "expensive",
                "catalyst_flag": "negative",
                "sentiment_flag": "negative",
                "thesis_strength": "weak",
            }
        },
    )
    assert round(float(adjusted.loc[adjusted["ticker"] == "AAPL", "adjusted_weight"].iloc[0]), 4) == 0.20
    assert round(float(adjusted.loc[adjusted["ticker"] == "SPY", "adjusted_weight"].iloc[0]), 4) == 0.30
    assert round(float(adjusted.loc[adjusted["ticker"] == "CASH", "adjusted_weight"].iloc[0]), 4) == 0.50
    assert audit["AAPL"]["decision"] == "cut"


def test_load_finrl_wide_weights(tmp_path: Path):
    csv_path = tmp_path / "ars_portfolio_weights.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-09"],
            "AAPL": [0.40, 0.30],
            "SPY": [0.20, 0.30],
            "cash": [0.40, 0.40],
            "regime": ["risk_on", "neutral"],
        }
    ).to_csv(csv_path, index=False)

    base = load_base_weights(csv_path)
    assert set(base["ticker"]) == {"AAPL", "SPY", "CASH"}
    assert len(base) == 6
    assert round(float(base.loc[(base["date"] == pd.Timestamp("2026-01-02")) & (base["ticker"] == "AAPL"), "weight"].iloc[0]), 2) == 0.40
    assert base.loc[base["ticker"] == "CASH", "asset_type"].iloc[0] == "cash"
    assert "regime" not in base.columns


def test_as_of_filter_excludes_future_rows():
    df = pd.DataFrame(
        {
            "date": ["2025-12-31", "2026-03-31", "2026-06-30"],
            "value": [1, 2, 3],
        }
    )
    filtered = _apply_as_of_filter(df, "2026-04-30", period="annual")
    assert list(filtered["value"]) == [2, 1]


def test_fmp_reuse_until_forced_reuses_exact_key_stale_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(market_data_api._FMP_RUNTIME, "cache_dir", str(tmp_path))
    cache_path = market_data_api._cache_file_path(
        endpoint="profile",
        ticker="AAPL",
        params={"symbol": "AAPL"},
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "meta": {
                    "endpoint": "profile",
                    "ticker": "AAPL",
                    "params": {"symbol": "AAPL"},
                    "cache_date_local": "2000-01-01",
                    "origin": "fresh_fetch",
                },
                "data": [{"symbol": "AAPL"}],
            }
        ),
        encoding="utf-8",
    )

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be called")

    monkeypatch.setattr(market_data_api.requests, "get", fail_get)
    market_data_api.configure_fmp_runtime(
        cache_dir=str(tmp_path),
        cache_policy="reuse-until-forced",
        force_refresh=False,
    )
    data, meta = market_data_api._fmp_get_json("profile", "AAPL", "KEY", {"symbol": "AAPL"})
    assert data == [{"symbol": "AAPL"}]
    assert meta["origin"] == "cache_hit"
    assert market_data_api.summarize_fmp_events()["origin_counts"]["cache_hit"] == 1


def test_fmp_force_refresh_bypasses_existing_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(market_data_api._FMP_RUNTIME, "cache_dir", str(tmp_path))
    cache_path = market_data_api._cache_file_path("profile", "AAPL", {"symbol": "AAPL"})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"meta": {"cache_date_local": "2000-01-01"}, "data": [{"symbol": "OLD"}]}),
        encoding="utf-8",
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"symbol": "NEW"}]

    monkeypatch.setattr(market_data_api.requests, "get", lambda *args, **kwargs: Response())
    market_data_api.configure_fmp_runtime(
        cache_dir=str(tmp_path),
        cache_policy="reuse-until-forced",
        force_refresh=True,
    )
    data, meta = market_data_api._fmp_get_json("profile", "AAPL", "KEY", {"symbol": "AAPL"})
    assert data == [{"symbol": "NEW"}]
    assert meta["origin"] == "fresh_fetch"


def test_news_exact_windows_have_distinct_cache_keys(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(market_data_api._FMP_RUNTIME, "cache_dir", str(tmp_path))
    market_data_api.configure_fmp_runtime(cache_dir=str(tmp_path), cache_policy="reuse-until-forced")
    params_a = {"tickers": "AAPL", "from": "2026-01-01", "to": "2026-01-08", "limit": 50}
    params_b = {"tickers": "AAPL", "from": "2026-02-01", "to": "2026-02-08", "limit": 50}
    path_a = market_data_api._cache_file_path("news/stock-latest", "AAPL", params_a)
    path_b = market_data_api._cache_file_path("news/stock-latest", "AAPL", params_b)
    assert path_a != path_b


def test_company_news_as_of_date_includes_full_calendar_day(tmp_path: Path, monkeypatch):
    market_data_api.configure_fmp_runtime(
        cache_dir=str(tmp_path),
        cache_policy="reuse-until-forced",
        force_refresh=True,
    )

    def fake_fmp_get_json(endpoint, ticker, api_key, params=None, timeout=None):
        return [
            {
                "symbol": "AAPL",
                "title": "Apple event",
                "publishedDate": "2026-03-01 16:30:54",
                "text": "Apple announced product updates.",
                "site": "example.com",
                "url": "https://example.com/a",
            },
            {
                "symbol": "AAPL",
                "title": "Future Apple event",
                "publishedDate": "2026-03-02 00:00:01",
                "text": "Future article should be excluded.",
                "site": "example.com",
                "url": "https://example.com/b",
            },
        ], {"origin": "fresh_fetch"}

    monkeypatch.setattr(market_data_api, "_fmp_get_json", fake_fmp_get_json)

    news = market_data_api.get_company_news(
        "AAPL",
        "KEY",
        days_back=7,
        limit=100,
        as_of_date="2026-03-01",
    )

    assert [item["title"] for item in news] == ["Apple event"]


def test_news_integrator_historical_anchor_uses_as_of_date():
    integrator = NewsIntegrator("AAPL")
    integrator.set_news_data(
        [
            {
                "symbol": "AAPL",
                "title": "Apple plans product announcements",
                "publishedDate": "2026-03-01 16:30:54",
                "text": "Apple is expected to announce new products this week.",
                "site": "example.com",
                "url": "https://example.com/a",
            }
        ]
    )

    processed = integrator.process_news(days_window=7, anchor_date="2026-03-01")

    assert len(processed) == 1
    assert processed[0].title == "Apple plans product announcements"


def test_news_integrator_keeps_matching_fmp_symbol_without_ticker_text():
    integrator = NewsIntegrator("AAPL")
    integrator.set_news_data(
        [
            {
                "symbol": "AAPL",
                "title": "Product announcements planned",
                "publishedDate": "2026-03-01 16:30:54",
                "text": "The company is expected to debut several products this week.",
                "site": "example.com",
                "url": "https://example.com/a",
            }
        ]
    )

    processed = integrator.process_news(days_window=7, anchor_date="2026-03-01")

    assert len(processed) == 1
    assert processed[0].symbol == "AAPL"


def test_analysis_command_passes_news_controls(tmp_path: Path):
    command = build_analysis_command(
        python_cmd="python",
        src_dir=tmp_path / "src",
        ticker="aapl",
        company_name="Apple Inc.",
        analyst_config_path=tmp_path / "config.ini",
        as_of_date="2026-03-01",
        analysis_dir=tmp_path / "analysis",
        fmp_cache_policy="reuse-until-forced",
        enable_enhanced_news=True,
        enable_catalyst_analysis=True,
        news_days_back=30,
        news_limit=75,
    )

    assert "--enable-enhanced-news" in command
    assert "--enable-catalyst-analysis" in command
    assert command[command.index("--news-days-back") + 1] == "30"
    assert command[command.index("--news-limit") + 1] == "75"
    assert command[command.index("--company-ticker") + 1] == "AAPL"


def test_active_setup_validates_single_and_role_mixed(tmp_path: Path):
    single_path = tmp_path / "single.json"
    single_path.write_text(
        json.dumps(
            {
                "setup_type": "single_model",
                "selected_setup_id": "gemini_only",
                "analyst": {"provider": "gemini", "model": "gemini-2.5-flash"},
            }
        ),
        encoding="utf-8",
    )
    single = load_active_research_setup(single_path)
    assert single.setup_type == "single_model"
    assert single.critic is None

    role_path = tmp_path / "role.json"
    role_path.write_text(
        json.dumps(
            {
                "setup_type": "role_mixed",
                "selected_setup_id": "role_mixed_b",
                "analyst": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
                "critic": {"provider": "claude", "model": "claude-opus-4-6"},
                "critic_sections": ["risks"],
            }
        ),
        encoding="utf-8",
    )
    role = load_active_research_setup(role_path)
    assert role.is_role_mixed
    assert role.analyst.provider == "gemini"
    assert role.critic.provider == "claude"
    assert role.critic_sections == ["risks"]
