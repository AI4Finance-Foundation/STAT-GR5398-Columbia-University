"""Capture/replay real FMP inputs and execute deterministic upstream Step 1.

No narrative generation and no paid model call occurs here. HTTP records, exact
as-of processing clock, raw tables, forecast assumptions and fallback provenance
are retained. A snapshot is immutable after successful completion.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

from audit_io import SnapshotHTTP, credentials, redact, write_json, append_jsonl, sha256, utc_now
from independent_valuation import UPSTREAM_SHA, load_inputs, recompute, run_engine, reconciliation, assumptions_rows, write_csv, write_notebook


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def parse_as_of(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("--as-of must include an explicit timezone offset")
    return result.astimezone(timezone.utc)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_for(http_root: Path, url_path: str) -> list[dict]:
    result = []
    for meta_path in sorted(http_root.glob("*/response.json")):
        meta = read_json(meta_path)
        if url_path in meta["source_url"]:
            result.append(dict(meta, response_key=meta_path.parent.name))
    return result


def record_source_files(upstream: Path) -> dict:
    tracked = subprocess.run(["git", "ls-files", "finrobot_equity"], cwd=upstream,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    return {relative: sha256((upstream / relative).read_bytes()) for relative in tracked
            if (upstream / relative).is_file() and relative.endswith(".py")}


def load_raw_body(http_root: Path, url_path: str):
    entries = metadata_for(http_root, url_path)
    if not entries:
        return None
    return json.loads((http_root / entries[0]["response_key"] / "body.bin").read_bytes())


def execute(args, config: dict, output: Path, as_of: datetime, run_id: str):
    import requests
    import pandas as pd
    from modules.financial_data_processor import extract_historical_metrics_from_api_data, calculate_growth_and_forecasts, normalize_forecast_config
    from modules.market_data_api import get_comprehensive_financial_data, combine_peer_financial_data, project_ebitda_for_peers, get_comprehensive_company_metrics
    from modules.sensitivity_analyzer import SensitivityAnalyzer
    import modules.news_integrator as news_module
    import modules.catalyst_analyzer as catalyst_module

    key = config.get("fmp_api_key", "")  # Replay strips this empty parameter from cache keys; no fake key.
    http_root = args.snapshot / "http"
    if args.mode == 'replay' and not any(http_root.rglob('body.bin')):
        # Derived analysis retains its original capture rather than duplicating HTTP.
        # Resolve the sibling by name so relocation never follows a developer path.
        current = read_json(args.snapshot / 'manifest.json')
        derivation = current.get('derivation', {})
        original_name = Path(derivation.get('original_snapshot','').replace('\\','/')).name
        if not original_name:
            raise ValueError('Replay snapshot has no HTTP capture or declared parent')
        original = args.snapshot.parent / original_name
        original_manifest = original / 'manifest.json'
        if sha256(original_manifest.read_bytes()) != derivation['original_manifest_sha256']:
            raise ValueError('Derived snapshot parent manifest changed')
        for relative, digest in read_json(original_manifest)['file_hashes'].items():
            if sha256((original/relative).read_bytes()) != digest:
                raise ValueError('Derived snapshot parent input changed: '+relative)
        http_root = original / 'http'
    analysis_dir = output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fallbacks = []
    missing = []
    local_as_of = as_of.astimezone(ZoneInfo("America/New_York"))

    class FrozenProcessingDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return as_of.astimezone(tz) if tz else local_as_of.replace(tzinfo=None)

    with SnapshotHTTP(http_root, args.mode, [key]).installed():
        financial = get_comprehensive_financial_data(args.ticker, key, period="annual", limit=args.years_limit)
        for name, frame in financial.items():
            if frame is None or frame.empty:
                missing.append({"field": name, "status": "missing", "consequence": "required financial source unavailable"})
            else:
                frame.to_csv(analysis_dir / f"{name}_raw_data.csv", index=False)
        if missing:
            write_json(output / "missing_data.json", missing)
            raise ValueError("Required financial endpoint(s) unavailable; inspect sanitized HTTP records and missing_data.json")
        period_rows = []
        for name, frame in financial.items():
            for index, row in frame.iterrows():
                accepted = row.get("acceptedDate", row.get("fillingDate", row.get("filingDate")))
                if accepted is not None and pd.notna(accepted):
                    accepted_dt = pd.to_datetime(accepted, utc=True)
                    if accepted_dt > pd.Timestamp(as_of):
                        raise ValueError(f"{name} contains a filing accepted after as-of; historical reconstruction is not supported")
                period_rows.append({"ticker": args.ticker, "source_table": name, "source_row": int(index),
                    "financial_period_end": str(row.get("date", "")), "pipeline_calendar_end_year": row.get("year"),
                    "provider_calendarYear": row.get("calendarYear"), "provider_fiscalYear": row.get("fiscalYear"), "period": row.get("period"),
                    "reported_currency": row.get("reportedCurrency"), "filingDate": row.get("filingDate", row.get("fillingDate")),
                    "acceptedDate": row.get("acceptedDate"),
                    "publication_timestamp_available": accepted is not None and pd.notna(accepted),
                    "fiscal_calendar_caveat": "Pipeline year is period-end calendar year; issuer fiscal year requires filing reconciliation"})
        write_csv(analysis_dir / "financial_period_provenance.csv", clean(period_rows))
        income_currencies = financial["income_statement"].get("reportedCurrency", pd.Series(dtype=str)).dropna().unique()
        if len(income_currencies) != 1:
            raise ValueError("Need exactly one disclosed statement currency before valuation")
        historical = extract_historical_metrics_from_api_data(financial)
        if historical is None or historical.empty:
            raise ValueError("Upstream historical extraction returned no financial metrics")
        actual_years = sorted(column for column in historical if column.endswith("A"))
        legacy_config = {"revenue_base_year": actual_years[-1],
            "revenue_growth_assumptions": {"2025E": .05, "2026E": .06, "2027E": .04},
            "ebitda_growth_factor": 1.05,
            "margin_improvement": {"Contribution Margin": .01, "EBITDA Margin": .01}, "sga_margin_change": -.005}
        forecast_config = normalize_forecast_config(historical, legacy_config)
        fallbacks.extend([
            {"field": "forecast_config", "value": forecast_config, "status": "model_assumption",
             "reason": "Universal upstream CLI defaults, not company guidance or a fitted forecast"},
            {"field": "valuation.net_debt", "value": "10% of model enterprise value", "status": "model_assumption",
             "reason": "Baseline engine does not use actual balance-sheet net debt"},
            {"field": "dcf.defaults", "value": {"growth_years_1_5": .10, "growth_years_6_10": .05, "wacc": .10, "terminal_growth": .025},
             "status": "model_assumption", "reason": "Fixed upstream DCF inputs, not estimated company financing costs/growth"},
            {"field": "native_sensitivity.confidence_intervals", "value": "fixed 15% standard deviation with normal quantile", "status": "model_assumption",
             "reason": "Uncalibrated scenario band; no historical standard deviation estimated"},
        ])
        final = calculate_growth_and_forecasts(historical, forecast_config)
        final.to_csv(analysis_dir / "financial_metrics_and_forecasts.csv", index=False)
        write_json(analysis_dir / "forecast_config_original_labels.json", legacy_config)
        write_json(analysis_dir / "forecast_config.json", forecast_config)
        metric_lookup = final.set_index("metrics")
        forecast_rows = []
        for horizon, (year, growth) in enumerate(forecast_config["revenue_growth_assumptions"].items(), 1):
            forecast_rows.append({"ticker": args.ticker, "run_id": run_id, "base_period": actual_years[-1],
                "forecast_period": year, "horizon": horizon, "revenue_growth_assumption": growth,
                "contribution_margin_change_per_year_pp": 1, "sga_margin_change_per_year_pp": -.5,
                "nominal_ebitda_margin_input_change_per_year_pp": 1,
                "forecast_revenue": metric_lookup.at["Revenue", year],
                "forecast_contribution_margin": metric_lookup.at["Contribution Margin", year],
                "forecast_sga_margin": metric_lookup.at["SG&A Margin", year],
                "forecast_sga": metric_lookup.at["SG&A", year], "forecast_ebitda": metric_lookup.at["EBITDA", year],
                "actual_ebitda_margin": metric_lookup.at["EBITDA Margin", actual_years[-1]],
                "forecast_ebitda_margin": metric_lookup.at["EBITDA Margin", year],
                "forecast_eps": metric_lookup.at["EPS", year], "forecast_pe": metric_lookup.at["PE Ratio", year],
                "basis": "annual financial periods; labels follow calendar year of period end, not calendar-year economic totals",
                "formula_caveat": "Forecast EBITDA=gross contribution profit-SG&A; R&D/other expenses excluded. Baseline retained for audit."})
        write_csv(analysis_dir / "forecast_assumptions_audit.csv", clean(forecast_rows))
        for _, row in final.iterrows():
            for year in actual_years:
                if pd.isna(row[year]):
                    missing.append({"field": f"{row['metrics']}/{year}", "status": "missing_pipeline_metric", "consequence": "must be shown as unavailable, not zero"})
        all_tickers = list(dict.fromkeys([*args.peers, args.ticker]))
        peer_ebitda, peer_multiples = combine_peer_financial_data(all_tickers, key, years_limit=args.years_limit)
        if peer_ebitda.empty or peer_multiples.empty:
            raise ValueError("Peer comparison tables unavailable")
        if len([ticker for ticker in args.peers if ticker in peer_ebitda and ticker in peer_multiples]) < 2:
            raise ValueError("At least two peers must have observed EBITDA and EV/EBITDA tables")
        projected = project_ebitda_for_peers(peer_ebitda, num_projection_years=1)
        projected.to_csv(analysis_dir / "peer_ebitda_comparison.csv")
        peer_multiples.to_csv(analysis_dir / "peer_ev_ebitda_comparison.csv")

        news_url = "https://financialmodelingprep.com/stable/news/stock"
        response = requests.get(news_url, params={"symbols": args.ticker,
            "from": (local_as_of - timedelta(days=args.news_days_back)).strftime("%Y-%m-%d"),
            "to": local_as_of.strftime("%Y-%m-%d"), "limit": args.news_limit, "apikey": key})
        news_status = {"source": "FMP stable/news/stock", "http_status": response.status_code, "available": response.ok}
        if response.ok:
            raw_news = response.json()
        else:
            raw_news = []
            missing.append({"field": "news_endpoint", "status": f"HTTP_{response.status_code}",
                            "consequence": "FMP news entitlement unavailable; native news/catalyst processing cannot establish news-supported claims"})
        if not response.ok and args.news_snapshot:
            alternate = read_json(args.news_snapshot)
            raw_news = alternate.get("articles", []) if isinstance(alternate, dict) else alternate
            if not isinstance(raw_news, list):
                raise ValueError("Alternate news snapshot must be an article list or an object containing articles")
            write_json(analysis_dir / "alternate_news_input.json", alternate)
            news_status.update(alternate_source="SEC/issuer disclosures supplied by --news-snapshot",
                alternate_file_sha256=sha256(args.news_snapshot.read_bytes()), alternate_article_count=len(raw_news),
                interpretation="Alternative primary-source event evidence; does not imply FMP news entitlement or general press-news coverage")
            missing[-1]["consequence"] = "FMP news entitlement unavailable; explicitly labeled alternate SEC/issuer disclosures supplied"
        write_json(analysis_dir / "news_status.json", news_status)
        if not isinstance(raw_news, list):
            raise ValueError("News endpoint did not return an article list")
        admitted, excluded = [], []
        for index, article in enumerate(raw_news):
            text_date = str(article.get("publishedDate", ""))
            try:
                published = datetime.fromisoformat(text_date.replace("Z", "+00:00"))
                if published.tzinfo is None:
                    published = published.replace(tzinfo=ZoneInfo("America/New_York"))
                if not as_of - timedelta(days=args.news_days_back) <= published.astimezone(timezone.utc) <= as_of:
                    raise ValueError("outside exact frozen news window")
            except (ValueError, TypeError) as exc:
                excluded.append({"raw_article_index": index, "publishedDate": text_date, "reason": str(exc)})
                continue
            admitted.append(article)
        write_json(analysis_dir / "news_window_filter.json", {"as_of_utc": as_of.isoformat(), "days_back": args.news_days_back,
            "raw_count": len(raw_news), "admitted_count": len(admitted), "excluded": excluded,
            "naive_timestamp_timezone_assumption": "America/New_York; original raw publishedDate remains preserved",
            "compatibility_note": "Original includes unparseable dates and has no upper bound. Strict timestamp filter prevents unverifiable/future articles entering frozen evidence."})
        with patch.object(news_module, "datetime", FrozenProcessingDateTime), patch.object(catalyst_module, "datetime", FrozenProcessingDateTime):
            integrator = news_module.NewsIntegrator(args.ticker, key, company_name=args.company_name)
            integrator.set_news_data(admitted)
            integrator.process_news(args.news_days_back)
            categorized = integrator.get_news_by_category()
            articles = integrator.get_news_for_catalyst_analysis()
            source_lookup = {(item.get("title"), item.get("text")): item for item in admitted}
            for article in articles:
                source = source_lookup.get((article.get("title"), article.get("text")), {})
                for field in ("url", "source_type", "accession", "source_document_sha256", "excerpt_sha256", "source_path"):
                    if field in source:
                        article[field] = source[field]
            write_json(analysis_dir / "news_article_provenance.json", admitted)
            sentiments = [article.sentiment for article in integrator.articles]
            enhanced = {"ticker": args.ticker, "articles": articles, "summary": integrator.generate_news_summary(),
                "categorized": {category: [{"title": article.title, "sentiment": article.sentiment, "importance": article.importance} for article in values] for category, values in categorized.items()},
                "sentiment_overview": {label: sentiments.count(label) for label in ("positive", "negative", "neutral")},
                "high_impact": [{"title": article.title, "category": article.category, "sentiment": article.sentiment} for article in integrator.get_high_impact_news()]}
            analyzer = catalyst_module.CatalystAnalyzer(args.ticker, key, company_name=args.company_name)
            catalysts = analyzer.identify_catalysts(articles)
            catalyst_result = {"catalysts": [asdict(catalyst) for catalyst in catalysts],
                "categorized": {category: [{"description": value.description, "impact": value.impact_level} for value in values] for category, values in analyzer.categorize_catalysts().items()},
                "top_catalysts": analyzer.get_top_catalysts(5), "summary": analyzer.generate_catalyst_summary()}
        write_json(analysis_dir / "company_news.json", articles)
        write_json(analysis_dir / "enhanced_news.json", enhanced)
        (analysis_dir / "news_summary.md").write_text(enhanced["summary"], encoding="utf-8")
        write_json(analysis_dir / "catalyst_analysis.json", catalyst_result)
        (analysis_dir / "catalyst_summary.md").write_text(catalyst_result["summary"], encoding="utf-8")
        for index, catalyst in enumerate(catalyst_result["catalysts"]):
            if catalyst["source"] == "Estimated Calendar":
                fallbacks.append({"field": f"catalyst_analysis.catalysts[{index}]", "value": catalyst,
                    "status": "assumption", "reason": "Upstream default earnings event at as-of+90 days, 95% heuristic probability; not actual scheduled earnings"})
        if not articles:
            missing.append({"field": "company_news", "status": "no_eligible_articles", "consequence": "news analysis evidence unavailable; do not invent company news"})
        native = SensitivityAnalyzer(final)
        sensitivity = {"revenue_sensitivity": native.analyze_revenue_sensitivity().to_dict(),
                       "margin_sensitivity": native.analyze_margin_sensitivity().to_dict(),
                       "combined_sensitivity": native.generate_sensitivity_table().to_dict()}
        native.calculate_confidence_interval("Revenue")
        native.calculate_confidence_interval("EBITDA")
        sensitivity.update(confidence_intervals=native.confidence_intervals, summary=native.generate_sensitivity_summary())
        write_json(analysis_dir / "sensitivity_analysis.json", clean(sensitivity))
        (analysis_dir / "sensitivity_summary.md").write_text(sensitivity["summary"], encoding="utf-8")

        raw_metrics = clean(get_comprehensive_company_metrics(args.ticker, key))
        write_json(output / "market_snapshot_upstream.json", raw_metrics)
        metrics = dict(raw_metrics)
        profile_data = load_raw_body(http_root, f"/stable/profile?")
        # This snapshot's profile request belongs to the subject; peers use statement endpoints only.
        profile = profile_data[0] if isinstance(profile_data, list) and profile_data else {}
        if not profile.get("sector"):
            metrics["sector"] = None
            fallbacks.append({"field": "sector", "value": raw_metrics.get("sector"), "status": "assumption_removed_from_observed", "reason": "Upstream falls back to Technology"})
        write_json(output / "market_snapshot.json", {"ticker": args.ticker, "run_id": run_id, "as_of_utc": as_of.isoformat(),
            "metrics": metrics, "classification_observed": {"sector": profile.get("sector"), "industry": profile.get("industry")},
            "provenance_notes": {"free_float": "stable/shares-float observation replaces legacy hard-coded 95%; unavailable if endpoint fails",
                "fwd_pe": "historical ratios/key-metrics PE mislabeled as forward PE upstream; not analyst forecast EPS multiple",
                "net_debt_to_equity": "historical debtEquityRatio, not independently derived net-debt/equity",
                "target_price": "stable price-target-consensus targetConsensus, a disclosed semantic change from legacy single-analyst target; not engine target",
                "rating": "latest single newGrade or action, not LLM recommendation",
                "dividend_yield": "lastDiv/current price; upstream assumes lastDiv is annual"}})
        if not metrics.get("shares_outstanding") or not metrics.get("share_price"):
            raise ValueError("Observed shares and current market price are required; no one-share or zero-price substitution")
        profile_currency = profile.get("currency")
        if not profile_currency or profile_currency != income_currencies[0]:
            raise ValueError("Observed price currency must match financial statement currency; automatic FX conversion is not implemented")
        quote_sources = metadata_for(http_root, "/stable/shares-float?")
        latest_cash = financial["cash_flow"].sort_values("date", ascending=False).iloc[0]
        peer_data = {}
        for ticker in args.peers:
            observed = peer_multiples[ticker].dropna() if ticker in peer_multiples else pd.Series(dtype=float)
            if not observed.empty:
                peer_data[ticker] = {"ev_ebitda": float(observed.iloc[-1]), "period_end_year": str(observed.index[-1])}
        valuation_inputs = {"ticker": args.ticker, "run_id": run_id, "current_price": metrics["share_price"],
            "shares_outstanding": metrics["shares_outstanding"], "currency": profile_currency, "peer_data": peer_data,
            "source_metadata": {"shares_outstanding": {"sources": quote_sources, "json_location": "[0].outstandingShares", "unit": "shares"},
                "current_price": {"sources": metadata_for(http_root, "/stable/quote-short?"), "unit": profile_currency + "/share", "as_of_utc": as_of.isoformat()},
                "peer_data": {"path": "analysis/peer_ev_ebitda_comparison.csv", "selection": "latest nonmissing annual period per peer; may differ across companies"}}}
        if pd.notna(latest_cash.get("freeCashFlow")):
            valuation_inputs["free_cash_flow"] = float(latest_cash["freeCashFlow"])
            valuation_inputs["free_cash_flow_period"] = str(latest_cash["date"])
            valuation_inputs["source_metadata"]["free_cash_flow"] = {"path": "analysis/cash_flow_raw_data.csv", "row_selector": "latest date", "field": "freeCashFlow", "unit": profile_currency}
        write_json(output / "valuation_inputs.json", clean(valuation_inputs))
        history_response = requests.get("https://financialmodelingprep.com/stable/historical-price-eod/full",
            params={"symbol": args.ticker, "from": "2025-10-10", "to": "2026-10-09", "apikey": key})
        price_history_status = {"endpoint": "stable/historical-price-eod/full", "http_status": history_response.status_code,
            "requested_start": "2025-10-10", "requested_end": "2026-10-09",
            "return_type": "close-to-close price return; not dividend-reinvested total return; adjustment treatment must be checked against provider"}
        if history_response.ok:
            prices = history_response.json()
            if isinstance(prices, list) and prices:
                price_df = pd.DataFrame(prices).sort_values("date")
                price_df.to_csv(analysis_dir / "historical_prices.csv", index=False)
                price_history_status.update(actual_start=price_df.iloc[0]["date"], actual_end=price_df.iloc[-1]["date"],
                    start_close=float(price_df.iloc[0]["close"]), end_close=float(price_df.iloc[-1]["close"]),
                    price_return_pct=(float(price_df.iloc[-1]["close"])/float(price_df.iloc[0]["close"])-1)*100)
        write_json(analysis_dir / "price_performance.json", price_history_status)
        write_json(output / "fallback_register.json", clean(fallbacks))
        write_json(output / "missing_data.json", clean(missing))
        write_json(analysis_dir / "analysis_summary.json", {"company_ticker": args.ticker, "company_name": args.company_name,
            "run_id": run_id, "analysis_date": as_of.isoformat(), "data_source": "Financial Modeling Prep API frozen snapshot",
            "data_period": "annual", "years_analyzed": len(actual_years), "available_years": actual_years,
            "latest_year": actual_years[-1], "forecast_years": list(forecast_config["revenue_growth_assumptions"]),
            "forecast_config": forecast_config, "peer_tickers": args.peers, "news_days_back": args.news_days_back,
            "news_articles_fetched": len(raw_news), "news_articles_in_evidence": len(articles),
            "enhanced_news_enabled": True, "sensitivity_analysis_enabled": True, "catalyst_analysis_enabled": True,
            "text_sections": [], "text_generation_status": "not run by this deterministic capture stage"})
    return missing, fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--peers", nargs="+", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="ISO-8601 time with offset; freezes news/catalyst clock")
    parser.add_argument("--mode", choices=("record", "replay"), default="record")
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--derive-analysis", action="store_true", help="Replay original HTTP into a separately versioned analysis after a documented compatibility fix")
    parser.add_argument("--derivation-reason", help="Required explanation when --derive-analysis is used")
    parser.add_argument("--years-limit", type=int, default=5)
    parser.add_argument("--news-days-back", type=int, default=5)
    parser.add_argument("--news-limit", type=int, default=50)
    parser.add_argument("--news-snapshot", type=Path, help="Explicit alternate frozen SEC/issuer article JSON, used only when FMP news is denied")
    args = parser.parse_args()
    if args.derive_analysis and (args.mode != "replay" or not args.derivation_reason):
        parser.error("--derive-analysis requires --mode replay and --derivation-reason")
    args.snapshot = args.snapshot.resolve()
    args.upstream = args.upstream.resolve()
    if len(set(args.peers) - {args.ticker}) < 2 or args.ticker in args.peers:
        parser.error("Specify at least two distinct peers, excluding subject ticker")
    if min(args.years_limit, args.news_days_back, args.news_limit) <= 0:
        parser.error("History/news limits must be positive")
    as_of = parse_as_of(args.as_of)
    config = credentials(args.config)
    if args.mode == "record" and not config.get("fmp_api_key"):
        raise SystemExit("FMP credential missing. Set FMP_API_KEY locally or use a private config; no request was made.")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.upstream, capture_output=True, text=True, check=True).stdout.strip()
    if sha != UPSTREAM_SHA:
        raise SystemExit("Upstream HEAD differs from pinned revision")
    parameters = {"ticker": args.ticker, "company_name": args.company_name, "peers": args.peers,
                  "as_of_utc": as_of.isoformat(), "years_limit": args.years_limit,
                  "news_days_back": args.news_days_back, "news_limit": args.news_limit}
    parameters["alternate_news_sha256"] = sha256(args.news_snapshot.read_bytes()) if args.news_snapshot else None
    existing = read_json(args.snapshot / "manifest.json") if (args.snapshot / "manifest.json").exists() else None
    if existing and existing.get("parameters") != parameters:
        raise SystemExit("Snapshot parameters differ; use a new snapshot directory")
    if args.mode == "record":
        if existing and existing.get("status") == "frozen":
            raise SystemExit("Snapshot already frozen. Use replay into a separate output directory")
        if abs((datetime.now(timezone.utc) - as_of).total_seconds()) > 86400:
            raise SystemExit("Current FMP quote endpoints cannot reconstruct a historical as-of. Capture within 24h of as-of; replay older snapshots offline")
        output = args.snapshot
    else:
        if not existing or existing.get("status") != "frozen":
            raise SystemExit("Replay requires a successfully frozen snapshot manifest")
        if not args.replay_output or args.replay_output.resolve() == args.snapshot:
            raise SystemExit("Replay requires --replay-output distinct from immutable snapshot")
        output = args.replay_output.resolve()
        if output.exists() and any(output.iterdir()):
            raise SystemExit("Replay output must be empty/new")
        for relative, digest in existing["file_hashes"].items():
            if sha256((args.snapshot / relative).read_bytes()) != digest:
                raise SystemExit("Snapshot hash mismatch: " + relative)
    output.mkdir(parents=True, exist_ok=True)
    upstream_src = args.upstream / "finrobot_equity" / "core" / "src"
    sys.path.insert(0, str(upstream_src))
    run_id = existing["run_id"] if existing else f"{args.ticker}-{as_of.strftime('%Y%m%dT%H%M%SZ')}-snapshot"
    if args.derive_analysis:
        run_id += "-derived-" + sha256(Path(__file__).read_bytes())[:8]
    start = time.perf_counter()
    manifest = {"run_id": run_id, "status": "running", "mode": args.mode, "parameters": parameters,
                "upstream_sha": sha, "source_sha256": record_source_files(args.upstream), "started_at_utc": utc_now(),
                "model_calls": 0, "model_cost_usd": 0, "provider_data_fees": "existing account entitlement; no purchase or upgrade performed"}
    if args.derive_analysis:
        manifest["derivation"] = {"original_snapshot": str(args.snapshot), "original_run_id": existing["run_id"],
            "original_manifest_sha256": sha256((args.snapshot / "manifest.json").read_bytes()), "reason": args.derivation_reason,
            "interpretation": "New deterministic analysis version from unchanged raw responses; not an exact analysis replay"}
    write_json(output / "manifest.json", manifest)
    log = io.StringIO()
    try:
        with redirect_stdout(log), redirect_stderr(log):
            missing, fallbacks = execute(args, config, output, as_of, run_id)
            context = load_inputs(output / "analysis", output / "valuation_inputs.json")
            independent = recompute(context)
            engine = run_engine(context, upstream_src)
            rows = reconciliation(context, independent, engine)
            valuation_output = output / "valuation"
            valuation_output.mkdir(parents=True, exist_ok=True)
            write_json(valuation_output / "engine_valuation.json", clean(engine))
            write_json(valuation_output / "independent_valuation.json", {"input_sources": context["sources"], "independent_results": independent})
            write_csv(valuation_output / "valuation_reconciliation.csv", rows)
            write_csv(valuation_output / "valuation_assumptions.csv", assumptions_rows(context, independent))
            write_csv(valuation_output / "dcf_cash_flows.csv", independent["DCF"]["cash_flows"])
            write_notebook(valuation_output / "independent_valuation.ipynb", context)
            if any(row["verdict"] != "MATCH" for row in rows):
                raise ValueError("Independent valuation reconciliation mismatch")
            if args.mode == "replay":
                compared = []
                for source in sorted((args.snapshot / "analysis").glob("*")):
                    if source.is_file() and (output / "analysis" / source.name).exists():
                        reproduced = output / "analysis" / source.name
                        compared.append({"file": "analysis/" + source.name, "original_sha256": sha256(source.read_bytes()),
                            "replayed_sha256": sha256(reproduced.read_bytes()), "identical": source.read_bytes() == reproduced.read_bytes()})
                write_json(output / "replay_comparison.json", compared)
                if any(not item["identical"] for item in compared) and not args.derive_analysis:
                    raise ValueError("Offline replay changed deterministic analysis artifacts; inspect replay_comparison.json")
        manifest.update(status="frozen", coverage_status="incomplete_missing_source_data" if missing else "deterministic_sources_available",
                        missing_data_count=len(missing), disclosed_fallback_count=len(fallbacks))
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=redact(exc, [config.get("fmp_api_key")]))
    finally:
        (output / "pipeline.log").write_text(redact(log.getvalue(), [config.get("fmp_api_key")]), encoding="utf-8")
        manifest.update(completed_at_utc=utc_now(), runtime_seconds=time.perf_counter() - start)
        append_jsonl(output / "attempts.jsonl", {key: manifest.get(key) for key in ("run_id", "status", "mode", "started_at_utc", "completed_at_utc", "runtime_seconds", "error_type", "error")})
        manifest["file_hashes"] = {str(path.relative_to(output)).replace("\\", "/"): sha256(path.read_bytes())
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    if manifest["status"] != "frozen":
        raise SystemExit("Snapshot failed; see sanitized manifest and pipeline.log: " + str(output))
    print(f"Frozen deterministic snapshot: {output}; no model call was made.")


if __name__ == "__main__":
    main()
