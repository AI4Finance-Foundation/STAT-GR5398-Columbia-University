#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import json 
from datetime import datetime, timezone

from modules.common_utils import load_config, get_api_key
from modules.financial_data_processor import calculate_growth_and_forecasts, extract_historical_metrics_from_api_data
from modules.market_data_api import (
    get_comprehensive_financial_data,
    combine_peer_financial_data,
    project_ebitda_for_peers,
    get_company_news,
    configure_fmp_runtime,
    get_fmp_event_cursor,
    summarize_fmp_events,
)
from modules.text_generator_agents import generate_text_section

# 新增模块导入
from modules.sensitivity_analyzer import SensitivityAnalyzer
from modules.catalyst_analyzer import CatalystAnalyzer
from modules.news_integrator import NewsIntegrator, get_enhanced_company_news
from modules.retail_sentiment_client import RetailSentimentClient
from modules.llm_gateway import load_llm_settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _collect_evaluation_summary(output_dir: str) -> dict:
    """Collect evaluator outputs if report_evaluate.py has been run."""
    candidates = [
        "report_evaluation.json",
        "evaluation_summary.json",
        "evaluation_results.json",
    ]

    for filename in candidates:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            score = None
            if isinstance(data, dict):
                score = (
                    data.get("overall_score")
                    or data.get("total_score")
                    or data.get("score")
                )

            return {
                "detected": True,
                "file": path,
                "score": score,
            }
        except Exception as e:
            return {
                "detected": True,
                "file": path,
                "error": str(e),
            }

    return {"detected": False}


def _build_preflight_report(args, fmp_api_key, llm_settings, adanos_api_key: str) -> tuple:
    required = [
        {
            "name": "company_ticker",
            "status": "ok" if bool(args.company_ticker) else "missing",
            "required": True,
            "detail": args.company_ticker,
        },
        {
            "name": "company_name",
            "status": "ok" if bool(args.company_name) else "missing",
            "required": True,
            "detail": args.company_name,
        },
        {
            "name": "fmp_api_key",
            "status": "ok" if bool(fmp_api_key) else "missing",
            "required": True,
            "detail": "configured" if fmp_api_key else "missing",
        },
    ]

    optional = [
        {
            "name": "llm_for_text_generation",
            "status": (
                "ok" if (args.generate_text_sections and llm_settings and llm_settings.api_key)
                else "skipped" if not args.generate_text_sections
                else "missing"
            ),
            "required": False,
            "detail": (
                f"{llm_settings.provider}/{llm_settings.model}" if (llm_settings and llm_settings.api_key) else "fallback_only"
            ),
        },
        {
            "name": "adanos_api_key",
            "status": "ok" if bool(adanos_api_key) else "missing",
            "required": False,
            "detail": "configured" if adanos_api_key else "retail_sentiment_disabled",
        },
    ]

    can_continue = all(item["status"] == "ok" for item in required)
    return {"required": required, "optional": optional}, can_continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate financial analysis data using FMP API instead of PDF extraction.")
    
    # Company Identifiers
    parser.add_argument("--company-ticker", type=str, required=True, help="Stock ticker (e.g., AAPL).")
    parser.add_argument("--company-name", type=str, required=True, help="Full company name (e.g., Apple Inc.).")
    
    # API Configuration
    parser.add_argument("--config-file", type=str, default=None, help="Path to the configuration file (e.g., config.ini).")
    parser.add_argument("--years-limit", type=int, default=5, help="Number of years of historical data to fetch.")
    
    # Output Configuration
    parser.add_argument("--output-dir", type=str, help="Directory to save all outputs. Default: ./output/[TICKER]/analysis/")
    parser.add_argument("--output-csv-name", type=str, default="financial_metrics_and_forecasts.csv", help="Name for the output CSV file.")

    # Peer Analysis
    parser.add_argument("--peer-tickers", type=str, nargs="*", default=[], help="List of peer tickers for comparative analysis (e.g., GOOG MSFT).")

    # Text Generation
    parser.add_argument("--generate-text-sections", action="store_true", help="Enable generation of text sections using the configured LLM provider.")
    parser.add_argument("--text-output-dir", type=str, default=None, help="Directory to save generated text files.")

    # NEWS PARAMETERS
    parser.add_argument("--news-days-back", type=int, default=5, help="Number of days to look back for company news (default: 5)")
    parser.add_argument("--news-limit", type=int, default=50, help="Maximum number of news articles to fetch (default: 50)")
    parser.add_argument("--retail-sentiment-days-back", type=int, default=7, help="Number of days to look back for retail sentiment insights (default: 7)")

    # 新增分析选项
    parser.add_argument("--enable-sensitivity-analysis", action="store_true", help="Enable sensitivity analysis for forecasts")
    parser.add_argument("--enable-catalyst-analysis", action="store_true", help="Enable catalyst identification and analysis")
    parser.add_argument("--enable-enhanced-news", action="store_true", help="Enable enhanced news integration with categorization")

    # Forecast Configuration
    parser.add_argument("--forecast-horizon-years", type=int, default=3, help="Number of forecast years to generate.")
    parser.add_argument("--revenue-growth-values", type=float, nargs="*", default=None, help="Optional sequence of growth assumptions (e.g. 0.08 0.06 0.05).")
    parser.add_argument("--revenue-growth-default", type=float, default=0.05, help="Fallback growth if neither CLI sequence nor historical signal is available.")
    parser.add_argument("--margin-improvement", type=float, default=0.01, help="Annual margin improvement assumption (default: 1%)")
    parser.add_argument("--sga-margin-improvement", type=float, default=-0.005, help="SG&A margin change assumption (default: -0.5% efficiency gain)")

    # API Options
    parser.add_argument("--period", type=str, default="annual", choices=["annual", "quarterly"], help="Data period (annual or quarterly)")
    parser.add_argument("--force-refresh-fmp", action="store_true", help="Ignore same-day FMP cache and refresh from API.")
    parser.add_argument("--fmp-cache-dir", type=str, default=None, help="Optional directory for FMP cache files.")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(".", "output", args.company_ticker, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # Text output directory
    text_output_dir = args.text_output_dir if args.text_output_dir else output_dir
    if args.generate_text_sections:
        os.makedirs(text_output_dir, exist_ok=True)
        print(f"Text outputs will be saved to: {text_output_dir}")

    # Load configuration and API keys
    llm_settings = None
    fmp_api_key = None
    adanos_api_key = os.getenv("ADANOS_API_KEY")
    adanos_base_url = os.getenv("ADANOS_BASE_URL", "https://api.adanos.org")

    try:
        config = load_config(args.config_file)
        fmp_api_key = get_api_key(config, section="API_KEYS", key="fmp_api_key")
        adanos_api_key = config.get("API_KEYS", "adanos_api_key", fallback=adanos_api_key)
        adanos_base_url = config.get("API_KEYS", "adanos_base_url", fallback=adanos_base_url)
        if args.generate_text_sections:
            llm_settings = load_llm_settings(config)
            print(f"Using LLM provider/model: {llm_settings.provider}/{llm_settings.model}")
            if llm_settings.base_url:
                print(f"Using LLM base URL: {llm_settings.base_url}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Please ensure config.ini exists with valid API keys:")
        print("[API_KEYS]")
        print("fmp_api_key = YOUR_FMP_API_KEY")
        print("llm_provider = openai  # or claude or gemini")
        print("openai_api_key = YOUR_OPENAI_API_KEY")
        print("claude_api_key = YOUR_CLAUDE_API_KEY")
        print("gemini_api_key = YOUR_GEMINI_API_KEY")
        return 1

    preflight_report, can_continue = _build_preflight_report(
        args=args,
        fmp_api_key=fmp_api_key,
        llm_settings=llm_settings,
        adanos_api_key=adanos_api_key,
    )

    print("\nPreflight checks:")
    for item in preflight_report["required"] + preflight_report["optional"]:
        level = "REQUIRED" if item["required"] else "OPTIONAL"
        print(f"- [{level}] {item['name']}: {item['status']} ({item['detail']})")

    if not can_continue:
        print("\nError: required inputs missing. Abort.")
        return 1

    configure_fmp_runtime(
        force_refresh=args.force_refresh_fmp,
        cache_dir=args.fmp_cache_dir,
        allow_stale_fallback=True,
        clear_events=True,
    )

    text_generation_summary = {}
    generated_text_files = {}

    source_status = {
        "fmp_financial_data": {"provider": "Financial Modeling Prep", "status": "unknown", "origin": "no_data", "detail": ""},
        "peer_comparison": {"provider": "Financial Modeling Prep", "status": "unknown", "origin": "no_data", "detail": ""},
        "company_news": {"provider": "Financial Modeling Prep", "status": "unknown", "origin": "no_data", "detail": ""},
        "retail_sentiment": {"provider": "Adanos", "status": "unknown", "origin": "no_data", "detail": ""},
        "valuation_inputs": {"provider": "Internal Forecast Engine", "status": "unknown", "origin": "no_data", "detail": ""},
    }

    run_manifest = {
        "created_at_utc": _utc_now_iso(),
        "ticker": args.company_ticker,
        "company_name": args.company_name,
        "inputs": {
            "company_ticker": args.company_ticker,
            "company_name": args.company_name,
            "period": args.period,
            "years_limit": args.years_limit,
            "peer_tickers": args.peer_tickers,
            "generate_text_sections": args.generate_text_sections,
            "forecast_horizon_years": args.forecast_horizon_years,
            "enable_sensitivity_analysis": args.enable_sensitivity_analysis,
            "enable_catalyst_analysis": args.enable_catalyst_analysis,
            "enable_enhanced_news": args.enable_enhanced_news,
        },
        "llm": {
            "provider": llm_settings.provider if llm_settings else None,
            "model": llm_settings.model if llm_settings else None,
            "base_url": llm_settings.base_url if llm_settings else None,
        },
        "preflight": preflight_report,
        "paths": {
            "output_dir": output_dir,
            "text_output_dir": text_output_dir,
        },
        "files": {},
        "evaluation": {},
    }

    print(f"Starting FMP API-based financial analysis for {args.company_name} ({args.company_ticker})")

    # 1. Fetch Financial Data from FMP API
    print(f"Fetching financial data from FMP API...")
    fmp_cursor_financial = get_fmp_event_cursor()
    financial_data = get_comprehensive_financial_data(
        ticker=args.company_ticker, 
        api_key=fmp_api_key, 
        period=args.period, 
        limit=args.years_limit
    )
    financial_fmp_summary = summarize_fmp_events(fmp_cursor_financial)

    # Check if we got the required data
    if financial_data.get('income_statement') is None or financial_data['income_statement'].empty:
        source_status["fmp_financial_data"].update({
            "status": "failed",
            "origin": financial_fmp_summary.get("primary_origin", "no_data"),
            "detail": "income_statement missing or empty",
        })
        print("Error: Could not fetch income statement data from FMP API. Exiting.")
        print("Please check:")
        print("1. FMP API key is valid and has remaining quota")
        print("2. Ticker symbol is correct")
        print("3. Internet connection is working")
        return 1

    source_status["fmp_financial_data"].update({
        "status": "ok",
        "origin": financial_fmp_summary.get("primary_origin", "no_data"),
        "detail": (
            f"income_statement_rows={len(financial_data['income_statement'])}; "
            f"origin_counts={financial_fmp_summary.get('origin_counts', {})}"
        ),
    })

    print("Successfully fetched financial data from FMP API")
    income_df = financial_data['income_statement']
    print(f"Retrieved {len(income_df)} years of income statement data")
    
    # Display available years for confirmation
    available_years = sorted(income_df['year'].tolist(), reverse=True)
    print(f"Available years: {available_years}")

    # 2. Process Historical Metrics
    print("Processing historical financial metrics...")
    historical_metrics_df = extract_historical_metrics_from_api_data(financial_data)

    if historical_metrics_df is None or historical_metrics_df.empty:
        print("Error: Failed to process historical metrics from API data. Exiting.")
        return 1
    
    print("\nHistorical Metrics Extracted from API:")
    print(historical_metrics_df.to_string())

    # 3. Forecasting
    print("Generating forecasts...")
    
    # Determine the latest actual year for base year
    actual_years = [col for col in historical_metrics_df.columns if col.endswith("A") and col != "metrics"]
    latest_year = max(actual_years) if actual_years else f"{pd.Timestamp.now().year}A"
    print(f"Using {latest_year} as base year for forecasts")

    base_year_num = int(latest_year[:4])
    forecast_horizon = max(1, args.forecast_horizon_years)
    forecast_year_labels = [f"{base_year_num + i}E" for i in range(1, forecast_horizon+1)]

    if args.revenue_growth_values:
        growth_inputs = list(args.revenue_growth_values)[:forecast_horizon]
        if len(growth_inputs) < forecast_horizon:
            growth_inputs += [growth_inputs[-1]] * (forecast_horizon - len(growth_inputs)) # use the latest provided value
    else:
        growth_inputs = [args.revenue_growth_default] * forecast_horizon

    forecast_config = {
        "revenue_base_year": latest_year,
        "revenue_growth_assumptions": dict(zip(forecast_year_labels, growth_inputs)), 
        "ebitda_growth_factor": 1.05,
        "margin_improvement": {
            "Contribution Margin": args.margin_improvement, 
            "EBITDA Margin": args.margin_improvement
        },
        "sga_margin_change": args.sga_margin_improvement
    }
    
    print(f"Forecast assumptions:")
    for index, y in enumerate(forecast_year_labels):
        print(f"  Revenue Growth {y}: {growth_inputs[index]*100:.1f}%")

    print(f"  Margin Improvement: {args.margin_improvement*100:.1f}% annually")
    print(f"  SG&A Efficiency: {args.sga_margin_improvement*100:.1f}% annually")
    
    final_data_df = calculate_growth_and_forecasts(historical_metrics_df, forecast_config)
    print("\nFinal Metrics with Forecasts:")
    print(final_data_df.to_string())
    source_status["valuation_inputs"].update({
        "status": "ok" if final_data_df is not None and not final_data_df.empty else "failed",
        "origin": "computed",
        "detail": f"forecast_rows={0 if final_data_df is None else len(final_data_df)}",
    })

    # Save the main analysis file
    output_csv_path = os.path.join(output_dir, args.output_csv_name)
    final_data_df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved financial analysis to: {output_csv_path}")

    # 4. Peer Comparison Analysis
    projected_peer_ebitda = None
    df_ev_ebitda_peers = None
    ebitda_peers_avail = False
    ev_ebitda_peers_avail = False
    if args.peer_tickers and fmp_api_key:
        print(f"\nFetching data for peer comparison: {args.peer_tickers}")
        all_tickers = args.peer_tickers + [args.company_ticker]

        try:
            fmp_cursor_peers = get_fmp_event_cursor()
            df_ebitda_peers, df_ev_ebitda_peers = combine_peer_financial_data(
                all_tickers, fmp_api_key, years_limit=args.years_limit
            )
            peers_fmp_summary = summarize_fmp_events(fmp_cursor_peers)

            ebitda_peers_avail = df_ebitda_peers is not None and not df_ebitda_peers.empty
            ev_ebitda_peers_avail = df_ev_ebitda_peers is not None and not df_ev_ebitda_peers.empty

            if ebitda_peers_avail:
                projected_peer_ebitda = project_ebitda_for_peers(df_ebitda_peers, num_projection_years=1)
                peer_ebitda_path = os.path.join(output_dir, "peer_ebitda_comparison.csv")
                projected_peer_ebitda.to_csv(peer_ebitda_path)
                print(f"Saved peer EBITDA comparison to: {peer_ebitda_path}")
            else:
                print("Warning: No peer EBITDA data could be retrieved")

            if ev_ebitda_peers_avail:
                peer_ev_ebitda_path = os.path.join(output_dir, "peer_ev_ebitda_comparison.csv")
                df_ev_ebitda_peers.to_csv(peer_ev_ebitda_path)
                print(f"Saved peer EV/EBITDA comparison to: {peer_ev_ebitda_path}")
            else:
                print("Warning: No peer EV/EBITDA data could be retrieved")

            if ebitda_peers_avail or ev_ebitda_peers_avail:
                source_status["peer_comparison"].update({
                    "status": "ok",
                    "origin": peers_fmp_summary.get("primary_origin", "no_data"),
                    "detail": (
                        f"peer_count={len(args.peer_tickers)}, "
                        f"ebitda={ebitda_peers_avail}, ev_ebitda={ev_ebitda_peers_avail}, "
                        f"origin_counts={peers_fmp_summary.get('origin_counts', {})}"
                    ),
                })
            else:
                source_status["peer_comparison"].update({
                    "status": "degraded",
                    "origin": peers_fmp_summary.get("primary_origin", "no_data"),
                    "detail": (
                        "peer data fetch returned empty frames; "
                        f"origin_counts={peers_fmp_summary.get('origin_counts', {})}"
                    ),
                })

        except Exception as e:
            print(f"Error fetching peer data: {e}")
            print("Continuing without peer comparison...")
            ebitda_peers_avail = False
            ev_ebitda_peers_avail = False
            source_status["peer_comparison"].update({
                "status": "failed",
                "origin": "no_data",
                "detail": str(e),
            })
    else:
        print("Skipping peer comparison (no peer tickers provided)")
        source_status["peer_comparison"].update({
            "status": "skipped",
            "origin": "skipped",
            "detail": "no peer tickers provided",
        })

    # 4.5 Fetch Company News
    company_news = None
    enhanced_news_data = None
    if fmp_api_key:
        print(f"\nFetching company news for {args.company_ticker}...")
        try:
            fmp_cursor_news = get_fmp_event_cursor()
            if args.enable_enhanced_news:
                # 使用增强版新闻获取
                enhanced_news_data = get_enhanced_company_news(
                    ticker=args.company_ticker,
                    api_key=fmp_api_key,
                    days_back=args.news_days_back,
                    limit=args.news_limit
                )
                company_news = enhanced_news_data.get('articles', [])

                # 保存增强版新闻数据
                enhanced_news_path = os.path.join(output_dir, "enhanced_news.json")
                with open(enhanced_news_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_news_data, f, indent=2, ensure_ascii=False)
                print(f"Saved enhanced news data to: {enhanced_news_path}")

                # 保存新闻摘要
                news_summary_path = os.path.join(output_dir, "news_summary.md")
                with open(news_summary_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_news_data.get('summary', ''))
                print(f"Saved news summary to: {news_summary_path}")
            else:
                # 使用原始新闻获取
                company_news = get_company_news(
                    ticker=args.company_ticker,
                    api_key=fmp_api_key,
                    days_back=args.news_days_back,
                    limit=args.news_limit
                )
            news_fmp_summary = summarize_fmp_events(fmp_cursor_news)

            if company_news:
                news_output_path = os.path.join(output_dir, "company_news.json")
                with open(news_output_path, 'w', encoding='utf-8') as f:
                    json.dump(company_news, f, indent=2, ensure_ascii=False)
                print(f"Saved company news to: {news_output_path}")
                source_status["company_news"].update({
                    "status": "ok",
                    "origin": news_fmp_summary.get("primary_origin", "no_data"),
                    "detail": (
                        f"articles={len(company_news)}; "
                        f"origin_counts={news_fmp_summary.get('origin_counts', {})}"
                    ),
                })
            else:
                print("Warning: No news data could be retrieved")
                source_status["company_news"].update({
                    "status": "degraded",
                    "origin": news_fmp_summary.get("primary_origin", "no_data"),
                    "detail": (
                        "empty news payload; "
                        f"origin_counts={news_fmp_summary.get('origin_counts', {})}"
                    ),
                })
        except Exception as e:
            print(f"Error fetching company news: {e}")
            print("Continuing without news data...")
            source_status["company_news"].update({
                "status": "failed",
                "origin": "no_data",
                "detail": str(e),
            })
    else:
        print("Skipping news fetch (no FMP API key)")
        source_status["company_news"].update({
            "status": "skipped",
            "origin": "skipped",
            "detail": "no fmp api key",
        })

    # 4.55 Fetch Retail Sentiment Insights
    retail_sentiment_data = None
    if adanos_api_key:
        print(f"\nFetching retail sentiment insights for {args.company_ticker}...")
        try:
            retail_client = RetailSentimentClient(
                api_key=adanos_api_key,
                base_url=adanos_base_url,
            )
            retail_sentiment_data = retail_client.get_snapshot(
                args.company_ticker,
                days_back=args.retail_sentiment_days_back,
            )

            retail_sentiment_path = os.path.join(output_dir, "retail_sentiment.json")
            with open(retail_sentiment_path, "w", encoding="utf-8") as f:
                json.dump(retail_sentiment_data, f, indent=2, ensure_ascii=False)
            print(f"Saved retail sentiment insights to: {retail_sentiment_path}")
            source_status["retail_sentiment"].update({
                "status": "ok",
                "origin": "fresh_fetch",
                "detail": f"days_back={args.retail_sentiment_days_back}",
            })
        except Exception as e:
            print(f"Error fetching retail sentiment insights: {e}")
            print("Continuing without retail sentiment insights...")
            source_status["retail_sentiment"].update({
                "status": "failed",
                "origin": "no_data",
                "detail": str(e),
            })
    else:
        print("Skipping retail sentiment insights (no ADANOS_API_KEY / adanos_api_key configured)")
        source_status["retail_sentiment"].update({
            "status": "skipped",
            "origin": "skipped",
            "detail": "no adanos api key",
        })

    # 4.6 敏感性分析
    sensitivity_results = None
    if args.enable_sensitivity_analysis:
        print(f"\nPerforming sensitivity analysis...")
        try:
            sensitivity_analyzer = SensitivityAnalyzer(final_data_df)
            
            # 收入敏感性分析
            revenue_sensitivity = sensitivity_analyzer.analyze_revenue_sensitivity()
            
            # 利润率敏感性分析
            margin_sensitivity = sensitivity_analyzer.analyze_margin_sensitivity()
            
            # 综合敏感性表格
            combined_sensitivity = sensitivity_analyzer.generate_sensitivity_table()
            
            # 置信区间
            revenue_ci = sensitivity_analyzer.calculate_confidence_interval('Revenue')
            ebitda_ci = sensitivity_analyzer.calculate_confidence_interval('EBITDA')
            
            sensitivity_results = {
                'revenue_sensitivity': revenue_sensitivity.to_dict() if not revenue_sensitivity.empty else {},
                'margin_sensitivity': margin_sensitivity.to_dict() if not margin_sensitivity.empty else {},
                'combined_sensitivity': combined_sensitivity.to_dict() if not combined_sensitivity.empty else {},
                'confidence_intervals': sensitivity_analyzer.confidence_intervals,
                'summary': sensitivity_analyzer.generate_sensitivity_summary()
            }
            
            # 保存敏感性分析结果
            sensitivity_path = os.path.join(output_dir, "sensitivity_analysis.json")
            with open(sensitivity_path, 'w', encoding='utf-8') as f:
                json.dump(sensitivity_results, f, indent=2, default=str)
            print(f"Saved sensitivity analysis to: {sensitivity_path}")
            
            # 保存敏感性摘要
            sensitivity_summary_path = os.path.join(output_dir, "sensitivity_summary.md")
            with open(sensitivity_summary_path, 'w', encoding='utf-8') as f:
                f.write(sensitivity_results['summary'])
            print(f"Saved sensitivity summary to: {sensitivity_summary_path}")
            
        except Exception as e:
            print(f"Error performing sensitivity analysis: {e}")
            print("Continuing without sensitivity analysis...")

    # 4.7 催化剂分析
    catalyst_results = None
    if args.enable_catalyst_analysis and company_news:
        print(f"\nPerforming catalyst analysis...")
        try:
            catalyst_analyzer = CatalystAnalyzer(args.company_ticker, fmp_api_key, company_name=args.company_name)
            
            # 识别催化剂
            catalysts = catalyst_analyzer.identify_catalysts(company_news)
            
            # 分类催化剂
            categorized_catalysts = catalyst_analyzer.categorize_catalysts()
            
            # 获取顶级催化剂
            top_catalysts = catalyst_analyzer.get_top_catalysts(5)
            
            # 生成摘要
            catalyst_summary = catalyst_analyzer.generate_catalyst_summary()
            
            catalyst_results = {
                'catalysts': [
                    {
                        'event_type': c.event_type,
                        'description': c.description,
                        'expected_date': c.expected_date,
                        'impact_level': c.impact_level,
                        'probability': c.probability,
                        'sentiment': c.sentiment
                    }
                    for c in catalysts
                ],
                'categorized': {
                    k: [{'description': c.description, 'impact': c.impact_level} for c in v]
                    for k, v in categorized_catalysts.items()
                },
                'top_catalysts': top_catalysts,
                'summary': catalyst_summary
            }
            
            # 保存催化剂分析结果
            catalyst_path = os.path.join(output_dir, "catalyst_analysis.json")
            with open(catalyst_path, 'w', encoding='utf-8') as f:
                json.dump(catalyst_results, f, indent=2, default=str)
            print(f"Saved catalyst analysis to: {catalyst_path}")
            
            # 保存催化剂摘要
            catalyst_summary_path = os.path.join(output_dir, "catalyst_summary.md")
            with open(catalyst_summary_path, 'w', encoding='utf-8') as f:
                f.write(catalyst_summary)
            print(f"Saved catalyst summary to: {catalyst_summary_path}")
            
        except Exception as e:
            print(f"Error performing catalyst analysis: {e}")
            print("Continuing without catalyst analysis...")

    # 5. Text Generation (Unified Logic)
    if args.generate_text_sections:
        print("\nGenerating AI-powered text sections...")

        if not llm_settings or not llm_settings.api_key:
            print("Error: LLM settings not loaded. Skipping text generation.")
        else:
            data_for_text_gen = {
                "financial_metrics": final_data_df,
                "peer_ebitda": projected_peer_ebitda,
                "peer_ev_ebitda": df_ev_ebitda_peers,
                "company_news": company_news,
                "enhanced_news": enhanced_news_data,
                "retail_sentiment": retail_sentiment_data,
                "sensitivity_analysis": sensitivity_results,
                "catalyst_analysis": catalyst_results
            }

            # A single list for all text types to be generated (including news_summary)
            all_text_types = [
                "tagline", "company_overview", "investment_overview",
                "valuation_overview", "risks", "competitor_analysis",
                "major_takeaways", "news_summary"
            ]

            # A single loop calling the unified generation function
            for text_type in all_text_types:
                # Skip news_summary if no news data available
                if text_type == "news_summary" and not company_news:
                    print(f"Skipping 'news_summary' - no news data available")
                    fallback_text = f"No recent news available for {args.company_name} ({args.company_ticker})."
                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(fallback_text)

                    generated_text_files[text_type] = file_path
                    text_generation_summary[text_type] = {
                        "section": text_type,
                        "source": "fallback",
                        "error": "missing_news_data",
                        "chars": len(fallback_text),
                        "latency_ms": 0,
                        "tokens": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                        "provider": llm_settings.provider,
                        "model": llm_settings.model,
                    }
                    print(f"Created placeholder for '{text_type}' at {file_path}")
                    continue

                print(f"Generating '{text_type}' for {args.company_name} ({args.company_ticker})...")
                try:
                    generated_text, generation_meta = generate_text_section(
                        data_for_text_gen,
                        text_type,
                        llm_settings.api_key,
                        args.company_name,
                        args.company_ticker,
                        base_url=llm_settings.base_url,
                        model=llm_settings.model,
                        provider=llm_settings.provider,
                        return_metadata=True
                    )

                    # Fallback validation can remain here as a safety net
                    if text_type == "competitor_analysis" and (not generated_text or len(generated_text.split('.')) < 3):
                        print(f"⚠️ Warning: Competitor analysis seems too short, using fallback.")
                        generated_text = f"{args.company_name} demonstrates competitive positioning within its industry sector through consistent financial performance and strategic market positioning relative to key competitors."
                        generation_meta["source"] = "fallback"
                        generation_meta["error"] = "post_validation_too_short"
                        generation_meta["chars"] = len(generated_text)

                    elif text_type == "major_takeaways":
                        generated_text_norm = (generated_text or "").lower()
                        required_headings = ["revenue growth", "gross profit margin", "sg&a", "ebitda margin"]
                        required_hits = sum(h in generated_text_norm for h in required_headings)
                        if required_hits < 3:
                            print(f"⚠️ Warning: Major takeaways missing required sections, using fallback.")
                            generated_text = (
                                f"Revenue Growth: {args.company_name}'s revenue growth shows consistent performance trends.\n\n"
                                f"Gross Profit Margin: {args.company_name}'s gross profit margins demonstrate operational effectiveness.\n\n"
                                f"SG&A Expense Margin: {args.company_name}'s SG&A expense management shows disciplined cost control.\n\n"
                                f"EBITDA Margin Stability: {args.company_name}'s EBITDA margin stability reflects strong underlying fundamentals."
                            )
                            generation_meta["source"] = "fallback"
                            generation_meta["error"] = "post_validation_missing_headings"
                            generation_meta["chars"] = len(generated_text)

                    elif text_type == "news_summary" and (not generated_text or len(generated_text.split()) < 50):
                        print(f"⚠️ Warning: News summary seems too short, using fallback.")
                        generated_text = f"Recent news coverage for {args.company_name} reflects ongoing market interest and developments in the company's operations and strategic initiatives."
                        generation_meta["source"] = "fallback"
                        generation_meta["error"] = "post_validation_too_short"
                        generation_meta["chars"] = len(generated_text)

                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(generated_text)
                    generated_text_files[text_type] = file_path
                    text_generation_summary[text_type] = generation_meta
                    print(f"✅ Saved '{text_type}' to {file_path} ({len(generated_text or '')} chars)")

                except Exception as e:
                    print(f"Error generating text for '{text_type}': {e}")
                    fallback_text = f"{args.company_name} ({args.company_ticker}) {text_type.replace('_', ' ')} analysis not available."
                    file_path = os.path.join(text_output_dir, f"{text_type}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(fallback_text)

                    generated_text_files[text_type] = file_path
                    text_generation_summary[text_type] = {
                        "section": text_type,
                        "source": "fallback",
                        "error": str(e),
                        "chars": len(fallback_text),
                        "latency_ms": 0,
                        "tokens": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                        "provider": llm_settings.provider if llm_settings else None,
                        "model": llm_settings.model if llm_settings else None,
                    }
                    print(f"Created fallback text for '{text_type}' at {file_path}")
    else:
        print("Skipping text generation (not enabled)")

    # 6. Save additional financial statement data for reference
    print("\nSaving additional financial statement data...")
    
    for statement_name, df in financial_data.items():
        if df is not None and not df.empty:
            statement_path = os.path.join(output_dir, f"{statement_name}_raw_data.csv")
            df.to_csv(statement_path, index=False)
            print(f"Saved {statement_name} to: {statement_path} ({len(df)} rows)")

    # 7. Create summary report
    all_text_types = [
        "tagline", "company_overview", "investment_overview", "valuation_overview",
        "risks", "competitor_analysis", "major_takeaways", "news_summary"
    ]

    sources_registry = {
        "generated_at_utc": _utc_now_iso(),
        "company_ticker": args.company_ticker,
        "company_name": args.company_name,
        "sources": [
            {
                "name": name,
                "provider": payload["provider"],
                "status": payload["status"],
                "origin": payload.get("origin", "no_data"),
                "detail": payload["detail"],
                "timestamp_utc": _utc_now_iso(),
            }
            for name, payload in source_status.items()
        ],
    }

    sources_path = os.path.join(output_dir, "sources.json")
    _write_json(sources_path, sources_registry)
    print(f"Saved sources registry to: {sources_path}")

    summary_data = {
        "company_ticker": args.company_ticker,
        "company_name": args.company_name,
        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "Financial Modeling Prep API",
        "data_period": args.period,
        "years_analyzed": len(actual_years),
        "available_years": available_years,
        "latest_year": latest_year,
        "forecast_years": list(forecast_config["revenue_growth_assumptions"].keys()),
        "peer_tickers": args.peer_tickers,
        "news_days_back": args.news_days_back,
        "news_articles_fetched": len(company_news) if company_news else 0,
        "enhanced_news_enabled": args.enable_enhanced_news,
        "sensitivity_analysis_enabled": args.enable_sensitivity_analysis,
        "catalyst_analysis_enabled": args.enable_catalyst_analysis,
        "forecast_config": forecast_config,
        "preflight": preflight_report,
        "text_generation": text_generation_summary,
        "files_generated": {
            "main_analysis": args.output_csv_name,
            "peer_ebitda": "peer_ebitda_comparison.csv" if projected_peer_ebitda is not None else None,
            "peer_ev_ebitda": "peer_ev_ebitda_comparison.csv" if ev_ebitda_peers_avail else None,
            "company_news": "company_news.json" if company_news else None,
            "enhanced_news": "enhanced_news.json" if enhanced_news_data else None,
            "sensitivity_analysis": "sensitivity_analysis.json" if sensitivity_results else None,
            "catalyst_analysis": "catalyst_analysis.json" if catalyst_results else None,
            "sources": "sources.json",
            "run_manifest": "run_manifest.json",
            "text_sections": generated_text_files if args.generate_text_sections else {}
        }
    }
    
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    _write_json(summary_path, summary_data)
    print(f"Saved analysis summary to: {summary_path}")

    run_manifest["finished_at_utc"] = _utc_now_iso()
    run_manifest["files"] = {
        "analysis_summary": summary_path,
        "sources": sources_path,
        "main_analysis_csv": output_csv_path,
        "text_sections": generated_text_files,
    }
    run_manifest["text_generation"] = text_generation_summary
    run_manifest["evaluation"] = _collect_evaluation_summary(output_dir)
    run_manifest["fmp_cache"] = summarize_fmp_events(0)

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    _write_json(manifest_path, run_manifest)
    print(f"Saved run manifest to: {manifest_path}")

    # 8. Print final summary and next steps
    print(f"\n" + "="*60)
    print(f"✅ Financial analysis completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Main analysis file: {output_csv_path}")
    if args.generate_text_sections:
        print(f"📝 AI text sections generated and saved in: {text_output_dir}")
    
    print(f"\n🚀 Ready to create equity report using:")
    print(f"python create_equity_report.py \\")
    print(f"  --company-ticker {args.company_ticker} \\")
    print(f"  --company-name \"{args.company_name}\" \\")
    print(f"  --analysis-csv {output_csv_path} \\")
    print(f"  --ratios-csv {os.path.join(output_dir, 'ratios_raw_data.csv')} \\")
    
    if args.peer_tickers and projected_peer_ebitda is not None:
        print(f"  --peer-ebitda-csv {os.path.join(output_dir, 'peer_ebitda_comparison.csv')} \\")
    if args.peer_tickers and ev_ebitda_peers_avail:
        print(f"  --peer-ev-ebitda-csv {os.path.join(output_dir, 'peer_ev_ebitda_comparison.csv')} \\")
    
    if args.generate_text_sections:
        for i, text_type in enumerate(all_text_types):
            param_name = text_type.replace('_', '-')
            file_path = os.path.join(text_output_dir, f'{text_type}.txt')
            # Add backslash for line continuation, except for the last item
            continuation = " \\" if i < len(all_text_types) - 1 else ""
            print(f"  --{param_name}-file {file_path}{continuation}")
    
    print(f"\n🎯 All files ready for report generation!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
