#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
from datetime import datetime
import pytz
import json
import re

EASTERN_TZ = pytz.timezone('America/New_York')

from modules.common_utils import load_config, get_api_key
from modules.report_data_loader import load_analysis_csv, load_text_from_file
from modules.html_renderer import render_html_report, render_combined_html_report, HTML_TEMPLATE_PAGE_1, HTML_TEMPLATE_PAGE_2_FINANCIAL_SUMMARY, HTML_TEMPLATE_PAGE_3_PEER_COMPARISON, HTML_TEMPLATE_PAGE_4_SENSITIVITY_CATALYST, HTML_TEMPLATE_PAGE_5_NEWS_CHARTS, HTML_TEMPLATE_COMBINED, format_dataframe_to_html_table
from modules.html_template_professional import render_professional_html_report
from modules.chart_generator import (
    generate_revenue_ebitda_chart, 
    generate_ev_ebitda_peer_chart, 
    generate_eps_pe_chart,
    # 高级图表函数
    generate_stock_price_chart,
    generate_financial_radar_chart,
    generate_time_series_chart,
    generate_sensitivity_heatmap,
    generate_technical_indicators_chart,
    generate_valuation_waterfall_chart,
    generate_quarterly_comparison_chart,
    generate_cash_flow_chart
)
from modules.market_data_api import get_comprehensive_company_metrics, get_technical_indicators

# Import the single, unified text generation function
from modules.text_generator_agents import generate_text_section

# 新增模块导入
from modules.enhanced_chart_generator import EnhancedChartGenerator, ChartConfig
from modules.valuation_engine import ValuationEngine
from modules.report_structure import ReportStructureManager
from modules.enhanced_text_generator import EnhancedTextGenerator
from modules.llm_gateway import load_llm_settings

def load_credit_cashflow_metrics_from_csv(file_path: str) -> pd.DataFrame:
    """Load credit and cashflow metrics from a pre-computed CSV file."""
    if not file_path or not os.path.exists(file_path):
        print(f"Warning: Ratios CSV file not found at {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        # Define the mapping from CSV columns to the desired metric names
        year_colname = 'fiscalYear'
        metrics_config = [
            {'key': 'debtToEquityRatio', 'name': 'Debt/Equity', 'fmt': lambda x: f"{x:.2f}"},
            {'key': 'debtToAssetsRatio', 'name': 'Debt/Assets', 'fmt': lambda x: f"{x:.2f}"},
            {'key': 'interestCoverageRatio', 'name': 'EBITDA/Int Exp', 'fmt': lambda x: f"{x:.1f}x"},
            {'key': 'netProfitMargin', 'name': 'Net Margin', 'fmt': lambda x: f"{x*100:.1f}%"},
            {'key': 'currentRatio', 'name': 'Current Ratio', 'fmt': lambda x: f"{x:.1f}"},
            {'key': 'operatingCashFlowCoverageRatio', 'name': 'Cash Flow Coverage Ratio', 'fmt': lambda x: f"{x:.2f}"},
        ]
        required_col = [m['key'] for m in metrics_config]
        
        # Check if necessary columns exist
        if year_colname not in df.columns or not all(key in df.columns for key in required_col):
            print("Warning: The ratios CSV file is missing required columns.")
            return pd.DataFrame()

        # Reverse the DataFrame to have the latest year last
        df = df.sort_values(by=year_colname).reset_index(drop=True)

        # Initialize the dictionary to hold the formatted data
        credit_metrics_data = {'metrics': [m['name'] for m in metrics_config]}

        year_cols = sorted(df[year_colname].unique())

        for year in year_cols:
            year_str = f"{year}A" # Append 'A' to match existing format
            credit_metrics_data[year_str] = []
            year_data = df[df[year_colname] == year].iloc[0]

            # Populate the data for the year based on the mapping
            for metric in metrics_config:
                val = year_data.get(metric['key'])
                if pd.isna(val):
                    formatted_val = "N/A"
                else:
                    formatted_val = metric['fmt'](val)
                credit_metrics_data[year_str].append(formatted_val)

        return pd.DataFrame(credit_metrics_data)

    except Exception as e:
        print(f"An error occurred while loading credit metrics from CSV: {e}")
        return pd.DataFrame()


def filter_actual_years_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to include only actual years (ending with 'A'), excluding estimates."""
    if df is None or df.empty:
        return df

    # Get columns that are actual years (ending with 'A') and the metrics column
    actual_year_cols = [col for col in df.columns if isinstance(col, str) and col.endswith('A')]
    if 'metrics' in df.columns:
        cols_to_keep = ['metrics'] + actual_year_cols
    else:
        cols_to_keep = actual_year_cols

    # Filter to only include these columns
    return df[cols_to_keep]


def generate_major_takeaways(analysis_df: pd.DataFrame, company_ticker: str) -> dict:
    """Generate major takeaways from the financial analysis data."""
    takeaways = {}

    try:
        # Get recent years data (only actual years, no estimates)
        year_cols = [col for col in analysis_df.columns if col.endswith('A') and col != 'metrics']
        if len(year_cols) < 2:
            # Return default takeaways if not enough data
            return {
                "revenue_growth_takeaway": f"{company_ticker}'s revenue growth data requires additional analysis.",
                "gross_margin_takeaway": f"{company_ticker}'s gross profit margin trends require further evaluation.",
                "sga_margin_takeaway": f"{company_ticker}'s SG&A expense management efficiency needs assessment.",
                "ebitda_margin_takeaway": f"{company_ticker}'s EBITDA margin stability shows consistent performance."
            }

        # Revenue Growth analysis
        revenue_growth_rows = analysis_df[analysis_df['metrics'] == 'Revenue Growth']
        if not revenue_growth_rows.empty:
            latest_growth = str(revenue_growth_rows[year_cols[-1]].iloc[0])
            prev_growth = str(revenue_growth_rows[year_cols[-2]].iloc[0]) if len(year_cols) > 1 else "N/A"
            takeaways["revenue_growth_takeaway"] = f"{company_ticker}'s revenue growth of {latest_growth} in {year_cols[-1]} shows solid momentum compared to {prev_growth} in {year_cols[-2]}."

        # Contribution/Gross Margin analysis
        margin_rows = analysis_df[analysis_df['metrics'] == 'Contribution Margin']
        if not margin_rows.empty:
            latest_margin = str(margin_rows[year_cols[-1]].iloc[0])
            prev_margin = str(margin_rows[year_cols[-2]].iloc[0]) if len(year_cols) > 1 else "N/A"
            takeaways["gross_margin_takeaway"] = f"{company_ticker}'s contribution margin improved to {latest_margin} in {year_cols[-1]} from {prev_margin} in {year_cols[-2]}, indicating operational efficiency gains."

        # SG&A Margin analysis
        sga_rows = analysis_df[analysis_df['metrics'] == 'SG&A Margin']
        if not sga_rows.empty:
            latest_sga = str(sga_rows[year_cols[-1]].iloc[0])
            prev_sga = str(sga_rows[year_cols[-2]].iloc[0]) if len(year_cols) > 1 else "N/A"
            takeaways["sga_margin_takeaway"] = f"{company_ticker}'s SG&A margin of {latest_sga} in {year_cols[-1]} compared to {prev_sga} in {year_cols[-2]} demonstrates expense management focus."

        # EBITDA Margin analysis
        ebitda_rows = analysis_df[analysis_df['metrics'] == 'EBITDA Margin']
        if not ebitda_rows.empty:
            latest_ebitda = str(ebitda_rows[year_cols[-1]].iloc[0])
            prev_ebitda = str(ebitda_rows[year_cols[-2]].iloc[0]) if len(year_cols) > 1 else "N/A"
            takeaways["ebitda_margin_takeaway"] = f"{company_ticker}'s EBITDA margin of {latest_ebitda} in {year_cols[-1]} vs {prev_ebitda} in {year_cols[-2]} shows stable profitability."

    except Exception as e:
        print(f"Warning: Error generating takeaways: {e}")
        # Return default takeaways
        takeaways = {
            "revenue_growth_takeaway": f"{company_ticker}'s revenue growth shows consistent performance trends.",
            "gross_margin_takeaway": f"{company_ticker}'s gross profit margins demonstrate operational effectiveness.",
            "sga_margin_takeaway": f"{company_ticker}'s SG&A expense management shows disciplined cost control.",
            "ebitda_margin_takeaway": f"{company_ticker}'s EBITDA margin stability reflects strong underlying fundamentals."
        }

    return takeaways


def validate_and_fix_text_content(text_content: str, text_type: str, company_name: str, company_ticker: str) -> str:
    if not text_content or text_content.strip() == "":
        print(f"⚠️ Warning: {text_type} is empty")
        return ""

    content = text_content.strip()

    # Check if content looks like CSV data
    strict_csv_types = {"major_takeaways"}
    if text_type in strict_csv_types:
        content_lower = content.lower()
        first_line = content_lower.split('\n')[0] if '\n' in content_lower else content_lower

        csv_header_like = (
            content_lower.startswith("year,")
            or content_lower.startswith("ticker,")
            or content_lower.startswith("date,")
        )

        # Use line-level checks instead of global comma count.
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        long_comma_lines = [ln for ln in lines if ln.count(',') >= 8]
        csv_block_like = len(long_comma_lines) >= 2

        first_line_header_like = (
            first_line.count(",") >= 3
            and not any(word in first_line for word in ["the", "and", "is", "are", "has", "have"])
        )

        is_csv_like = csv_header_like or csv_block_like or first_line_header_like
        if is_csv_like:
            print(f"⚠️ Warning: {text_type} contains CSV-like data, marking for regeneration")
            return ""

    print(f"✅ {text_type} validation passed ({len(content)} chars)")
    return content


def regenerate_text_if_needed(text_content: str, text_type: str, company_name: str, company_ticker: str, 
                              analysis_df: pd.DataFrame, peer_ebitda_df: pd.DataFrame, peer_ev_ebitda_df: pd.DataFrame, 
                              api_key: str = None, base_url: str = None, model: str = None, provider: str = None) -> str:
    """Generate text content using AI, calling the single unified function."""
    
    # This function handles all text types through the same logic if regeneration is enabled.
    if api_key:
        try:
            # Prepare data for generation
            data_for_generation = {
                "financial_metrics": analysis_df,
                "peer_ebitda": peer_ebitda_df,
                "peer_ev_ebitda": peer_ev_ebitda_df,
            }
            
            print(f"🤖 Regenerating '{text_type}' using AI...")
            # Call the single, unified text generation function
            generated_text = generate_text_section(
                data_for_generation, 
                text_type, 
                api_key, 
                company_name, 
                company_ticker,
                base_url=base_url,
                model=model,
                provider=provider,
            )
            
            # Basic validation of the generated content
            if generated_text and len(generated_text.strip()) > 50:
                print(f"✅ Successfully regenerated '{text_type}' using AI")
                return generated_text
            else:
                print(f"❌ AI regenerated '{text_type}' is insufficient, using original content from file.")
                return validate_and_fix_text_content(text_content, text_type, company_name, company_ticker)
            
        except Exception as e:
            print(f"❌ Failed to regenerate '{text_type}' with AI: {e}")
            return validate_and_fix_text_content(text_content, text_type, company_name, company_ticker)
    
    # For other text types or if regeneration is off, just validate the content from the file.
    return validate_and_fix_text_content(text_content, text_type, company_name, company_ticker)


def process_text_content(args, analysis_df, peer_ebitda_df, peer_ev_ebitda_df, llm_api_key, llm_base_url=None, 
                         llm_model=None, llm_provider=None):
    """Process all text content with enhanced AI generation for competitor analysis and takeaways."""
    
    print("📖 Loading and processing text content...")
    
    # Load raw text content
    raw_texts = {
        "tagline": load_text_from_file(args.tagline_file),
        "company_overview": load_text_from_file(args.company_overview_file),
        "investment_overview": load_text_from_file(args.investment_overview_file),
        "valuation_overview": load_text_from_file(args.valuation_overview_file),
        "risks": load_text_from_file(args.risks_file),
        "competitor_analysis": load_text_from_file(args.competitor_analysis_file),
        "major_takeaways": load_text_from_file(args.major_takeaways_file),
        "news_summary": load_text_from_file(args.news_summary_file) if args.news_summary_file else ""
    }
    
    # Process text content
    processed_texts = {}
    for text_type, raw_content in raw_texts.items():
        print(f"📝 Processing {text_type}...")
        
        if text_type in ["competitor_analysis", "major_takeaways"]:
            validated = validate_and_fix_text_content(
                raw_content or "", text_type, args.company_name, args.company_ticker
            )

            if validated:
                processed_texts[text_type] = validated
            elif llm_api_key and args.enable_text_regeneration:
                print(f"⚠️ {text_type} validation failed, triggering AI regeneration...")
                processed_texts[text_type] = regenerate_text_if_needed(
                    raw_content or "",
                    text_type,
                    args.company_name,
                    args.company_ticker,
                    analysis_df,
                    peer_ebitda_df,
                    peer_ev_ebitda_df,
                    llm_api_key,
                    base_url=llm_base_url,
                    model=llm_model,
                    provider=llm_provider,
                )
            else:
                if text_type == "competitor_analysis":
                    processed_texts[text_type] = (
                        f"{args.company_name} demonstrates competitive positioning within its industry sector "
                        "through consistent financial performance and strategic market positioning relative to key competitors."
                    )
                else:
                    processed_texts[text_type] = (
                        f"Revenue Growth: {args.company_name}'s revenue growth shows consistent performance trends.\n\n"
                        f"Gross Profit Margin: {args.company_name}'s gross profit margins demonstrate operational effectiveness.\n\n"
                        f"SG&A Expense Margin: {args.company_name}'s SG&A expense management shows disciplined cost control.\n\n"
                        f"EBITDA Margin Stability: {args.company_name}'s EBITDA margin stability reflects strong underlying fundamentals."
                    )
        else:
            # Regular flow for other text types:
            # validate first; regenerate only when validation fails.
            validated = validate_and_fix_text_content(
                raw_content or "", text_type, args.company_name, args.company_ticker
            )
            if validated:
                processed_texts[text_type] = validated
            elif args.enable_text_regeneration and llm_api_key:
                print(f"⚠️ {text_type} validation failed, triggering AI regeneration...")
                processed_texts[text_type] = regenerate_text_if_needed(
                    raw_content or "", text_type, args.company_name, args.company_ticker,
                    analysis_df, peer_ebitda_df, peer_ev_ebitda_df, llm_api_key,
                    base_url=llm_base_url, model=llm_model, provider=llm_provider
                )
            else:
                processed_texts[text_type] = ""

    print(f"✅ All text content processed")
    return processed_texts


def _coerce_numeric(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = value.replace(',', '').replace('$', '').replace('%', '').replace('x', '').strip()
            if cleaned == "":
                return None
            return float(cleaned)
        return float(value)
    except Exception:
        return None


def _extract_metric_time_series(analysis_df: pd.DataFrame, metric_names):
    """
    Return per-metric numeric series across all year columns (A/E), and baseline pinned to latest actual year (A).
    """
    if analysis_df is None or analysis_df.empty:
        return {}

    def _year_col_sort_key(col: str):
        m = re.match(r"^(\d{4})([AE])$", str(col))
        if not m:
            return (0, 2)
        year = int(m.group(1))
        suffix = m.group(2)
        suffix_rank = 0 if suffix == "A" else 1
        return (year, suffix_rank)

    year_cols = [
        c for c in analysis_df.columns
        if isinstance(c, str) and re.match(r"^\d{4}[AE]$", c)
    ]
    year_cols = sorted(year_cols, key=_year_col_sort_key)

    actual_year_cols = [c for c in year_cols if c.endswith("A")]

    out = {}
    for metric in metric_names:
        row = analysis_df[analysis_df['metrics'] == metric]
        if row.empty:
            out[metric] = {
                "baseline_column": actual_year_cols[-1] if actual_year_cols else (year_cols[-1] if year_cols else None),
                "baseline_value": None,
                "values_by_column": {},
            }
            continue

        values_by_column = {}
        for col in year_cols:
            values_by_column[col] = _coerce_numeric(row[col].iloc[0])

        values_by_column = {k: v for k, v in values_by_column.items() if v is not None}

        # Baseline: latest actual year if available; fallback to latest available year
        baseline_column = next((c for c in reversed(actual_year_cols) if c in values_by_column), None)
        if baseline_column is None:
            baseline_column = next((c for c in reversed(year_cols) if c in values_by_column), None)
        baseline_value = values_by_column.get(baseline_column)

        out[metric] = {
            "baseline_column": baseline_column,
            "baseline_value": baseline_value,
            "values_by_column": values_by_column,
        }
    return out


def _extract_numeric_mentions(text: str):
    if not text:
        return []
    tokens = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    nums = []
    for token in tokens:
        n = _coerce_numeric(token)
        if n is not None:
            nums.append(n)
    return nums


def _parse_scaled_number(expr: str):
    if not expr:
        return None

    s = expr.strip().lower().replace('$', '').replace(',', '')
    m = re.match(r"^([-+]?\d+(?:\.\d+)?)\s*(billion|bn|b|million|mn|m|thousand|k)?$", s)
    if not m:
        return None

    base = float(m.group(1))
    unit = (m.group(2) or "").lower()
    factor_map = {
        "": 1.0,
        "b": 1e9,
        "bn": 1e9,
        "billion": 1e9,
        "m": 1e6,
        "mn": 1e6,
        "million": 1e6,
        "k": 1e3,
        "thousand": 1e3,
    }
    return base * factor_map.get(unit, 1.0)


def _extract_scaled_numeric_mentions(text: str):
    if not text:
        return []

    pattern = re.compile(
        r"\$?\s*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|bn|b|million|mn|m|thousand|k)?",
        re.IGNORECASE,
    )
    out = []
    for m in pattern.finditer(text):
        val = _parse_scaled_number(m.group(0))
        if val is not None:
            out.append(val)
    return out


def _match_metric_mentions_to_series(metric: str, numeric_mentions, values_by_column: dict):
    """Return matched columns when mention is close to any year-point in metric series."""
    matched_cols = set()
    for n in numeric_mentions:
        for col, metric_value in values_by_column.items():
            tol = max(abs(metric_value) * 0.08, 0.15 if metric == "EPS" else 1.0)
            if abs(n - metric_value) <= tol:
                matched_cols.add(col)
    return sorted(matched_cols)


def run_numeric_consistency_checks(analysis_df: pd.DataFrame, section_texts: dict) -> dict:
    """
    P1 numeric consistency checks for Revenue / EPS / EBITDA.

    Improvement:
    - Match against full metric series (all A/E columns), not only latest year.
    - Keep strict checking only on sections expected to carry concrete numeric claims.
    """
    metrics = ["Revenue", "EPS", "EBITDA"]
    keyword_map = {
        "Revenue": ["revenue"],
        "EPS": ["eps", "earnings per share"],
        "EBITDA": ["ebitda"],
    }
    strict_sections = {
        "Revenue": {"valuation_overview", "major_takeaways", "investment_overview"},
        "EPS": {"valuation_overview", "major_takeaways", "investment_overview"},
        "EBITDA": {"valuation_overview", "major_takeaways", "competitor_analysis", "investment_overview"},
    }

    metric_series = _extract_metric_time_series(analysis_df, metrics)
    checks = {}
    pass_count = 0
    warning_count = 0
    skip_count = 0

    for metric in metrics:
        series_info = metric_series.get(metric, {})
        source_col = series_info.get("baseline_column")
        expected_latest = series_info.get("baseline_value")
        values_by_column = series_info.get("values_by_column", {})

        if not values_by_column:
            checks[metric] = {
                "status": "skipped",
                "reason": "metric_not_available",
                "source_column": source_col,
                "expected_value": None,
                "matched_sections": [],
            }
            skip_count += 1
            continue

        mentioned_sections = []
        strict_mentioned_sections = []
        matched_sections = []
        matched_columns = set()

        for section_name, section_text in section_texts.items():
            text = section_text or ""
            text_lower = text.lower()
            if not any(k in text_lower for k in keyword_map[metric]):
                continue

            mentioned_sections.append(section_name)
            if section_name in strict_sections.get(metric, set()):
                strict_mentioned_sections.append(section_name)

                scaled_nums = _extract_scaled_numeric_mentions(text)
                plain_nums = _extract_numeric_mentions(text)
                numeric_mentions = scaled_nums + plain_nums

                matched_cols = _match_metric_mentions_to_series(metric, numeric_mentions, values_by_column)
                if matched_cols:
                    matched_sections.append(section_name)
                    matched_columns.update(matched_cols)

        if not mentioned_sections:
            checks[metric] = {
                "status": "info",
                "reason": "metric_not_explicitly_mentioned_in_text",
                "source_column": source_col,
                "expected_value": expected_latest,
                "mentioned_sections": [],
                "strict_mentioned_sections": [],
                "matched_sections": [],
                "matched_columns": [],
            }
            skip_count += 1
            continue

        if strict_mentioned_sections:
            if matched_sections:
                checks[metric] = {
                    "status": "pass",
                    "source_column": source_col,
                    "expected_value": expected_latest,
                    "mentioned_sections": mentioned_sections,
                    "strict_mentioned_sections": strict_mentioned_sections,
                    "matched_sections": matched_sections,
                    "matched_columns": sorted(matched_columns),
                }
                pass_count += 1
            else:
                checks[metric] = {
                    "status": "warning",
                    "reason": "numeric_mismatch_or_missing_exact_value",
                    "source_column": source_col,
                    "expected_value": expected_latest,
                    "mentioned_sections": mentioned_sections,
                    "strict_mentioned_sections": strict_mentioned_sections,
                    "matched_sections": [],
                    "matched_columns": [],
                }
                warning_count += 1
        else:
            checks[metric] = {
                "status": "info",
                "reason": "only_general_mention_no_strict_numeric_claim",
                "source_column": source_col,
                "expected_value": expected_latest,
                "mentioned_sections": mentioned_sections,
                "strict_mentioned_sections": [],
                "matched_sections": [],
                "matched_columns": [],
            }
            skip_count += 1

    return {
        "generated_at": datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "summary": {
            "pass": pass_count,
            "warning": warning_count,
            "skipped_or_info": skip_count,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Create an equity research report in HTML format with auto-fetched market data.")

    # --- Command-line arguments ---
    # Required
    parser.add_argument("--company-ticker", type=str, required=True, help="Stock ticker.")
    parser.add_argument("--company-name", type=str, required=True, help="Full company name.")
    parser.add_argument("--analysis-csv", type=str, required=True, help="Path to the financial_metrics_and_forecasts.csv file.")
    parser.add_argument("--ratios-csv", type=str, required=True, help="Path to the ratios_raw_data.csv file.")
    parser.add_argument("--tagline-file", type=str, required=True, help="Path to a text file for the report tagline.")
    parser.add_argument("--company-overview-file", type=str, required=True, help="Path to a text file for the company overview.")
    parser.add_argument("--investment-overview-file", type=str, required=True, help="Path to a text file for the investment overview.")
    parser.add_argument("--valuation-overview-file", type=str, required=True, help="Path to a text file for the valuation section.")
    parser.add_argument("--risks-file", type=str, required=True, help="Path to a text file for the risks section.")
    parser.add_argument("--competitor-analysis-file", type=str, required=True, help="Path to a text file for the competitor analysis section.")
    parser.add_argument("--major-takeaways-file", type=str, required=True, help="Path to a text file for the major takeaways section.")
    parser.add_argument("--news-summary-file", type=str, required=False, default=None, help="Path to a text file for the news summary section (optional).")

    # Optional with defaults
    parser.add_argument("--report-date", type=str, default=datetime.now(EASTERN_TZ).strftime("%B %d, %Y"), help="Date for the report (Eastern Time).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save HTML reports. Default: ./output/[TICKER]/report/")
    parser.add_argument("--html-report-prefix", type=str, default="Equity_Report", help="Prefix for output HTML filenames.")
    parser.add_argument("--analyst-names", type=str, nargs="*", default=["Analyst Name"], help="List of analyst names.")
    parser.add_argument("--analyst-emails", type=str, nargs="*", default=["analyst@example.com"], help="List of analyst emails.")
    parser.add_argument("--research-source", type=str, default="AI4Finance Foundation FinRobot Equity Research", help="Source of the research.")
    parser.add_argument("--data-source-text", type=str, default="Company Filings, FMP, Yahoo Finance, AI4Finance Estimates", help="Text for data sources.")
    parser.add_argument("--disclaimer-text", type=str, default="Disclaimer: The information contained in this document is intended only for use by the person to whom it has been delivered and should not be disseminated or distributed to third parties without our prior written consent. Our firm accepts no liability whatsoever with respect to the use of this document or its contents.", help="Disclaimer text.")
    parser.add_argument("--closing-price-date", type=str, default=datetime.now(EASTERN_TZ).strftime("%B %d, %Y"), help="Date of the share price (Eastern Time).")

    # Market data - Optional (will be auto-fetched if not provided)
    parser.add_argument("--share-price", type=float, default=None, help="Current share price (will be auto-fetched if not provided).")
    parser.add_argument("--target-price", type=float, default=None, help="12-month target price (will be auto-fetched if not provided).")
    parser.add_argument("--rating", type=str, default=None, help="Analyst rating (will be auto-fetched if not provided).")
    parser.add_argument("--market-cap", type=float, default=None, help="Market cap in billions (will be auto-fetched if not provided).")
    parser.add_argument("--volume", type=float, default=None, help="Average daily volume in millions (will be auto-fetched if not provided).")
    parser.add_argument("--fwd-pe", type=float, default=None, help="Forward P/E ratio (will be auto-fetched if not provided).")
    parser.add_argument("--pb-ratio", type=float, default=None, help="Price to Book ratio (will be auto-fetched if not provided).")
    parser.add_argument("--dividend-yield", type=str, default=None, help="Dividend yield (will be auto-fetched if not provided).")
    parser.add_argument("--free-float", type=str, default=None, help="Free float percentage (will be auto-fetched if not provided).")
    parser.add_argument("--roe", type=str, default=None, help="Return on Equity (will be auto-fetched if not provided).")
    parser.add_argument("--net-debt-to-equity", type=str, default=None, help="Net Debt to Equity ratio (will be auto-fetched if not provided).")
    parser.add_argument("--sector", type=str, default=None, help="Company sector (will be auto-fetched if not provided).")

    # Configuration and paths
    parser.add_argument("--config-file", type=str, default=None, help="Path to config.ini file.")
    parser.add_argument("--logo-image-path", type=str, default="./assets/piclogo.png", help="Path to the logo image.")
    parser.add_argument("--revenue-chart-path", type=str, default=None, help="Path to a pre-generated revenue/EBITDA chart.")
    parser.add_argument("--ev-ebitda-chart-path", type=str, default=None, help="Path to a pre-generated EV/EBITDA peer comparison chart.")
    parser.add_argument("--peer-ebitda-csv", type=str, help="Path to peer_ebitda_comparison.csv.")
    parser.add_argument("--peer-ev-ebitda-csv", type=str, help="Path to peer_ev_ebitda_comparison.csv.")

    # Auto-fetch control
    parser.add_argument("--skip-auto-fetch", action="store_true", help="Skip automatic fetching of market data from FMP API.")
    
    # Text regeneration option
    parser.add_argument("--enable-text-regeneration", action="store_true", help="Enable LLM text regeneration if content quality is poor.")
    
    # 新增增强功能选项
    parser.add_argument("--enable-enhanced-charts", action="store_true", help="Enable enhanced chart generation with 11 professional chart types.")
    parser.add_argument("--enable-valuation-analysis", action="store_true", help="Enable multi-method valuation analysis.")
    parser.add_argument("--sensitivity-analysis-file", type=str, default=None, help="Path to sensitivity analysis JSON file.")
    parser.add_argument("--catalyst-analysis-file", type=str, default=None, help="Path to catalyst analysis JSON file.")
    parser.add_argument("--enhanced-news-file", type=str, default=None, help="Path to enhanced news JSON file.")
    parser.add_argument("--retail-sentiment-file", type=str, default=None, help="Path to retail sentiment JSON file.")

    args = parser.parse_args()

    # --- Setup directories ---
    output_dir = args.output_dir or os.path.join(".", "output", args.company_ticker, "report")
    os.makedirs(output_dir, exist_ok=True)
    print(f"HTML reports will be saved to: {output_dir}")

    # --- Load configuration and API key ---
    llm_api_key = None
    llm_base_url = None
    llm_model = None
    llm_provider = None
    try:
        config = load_config(args.config_file)
        fmp_api_key = get_api_key(config, "API_KEYS", "fmp_api_key")
        if args.enable_text_regeneration:
            try:
                llm_settings = load_llm_settings(config)
                llm_api_key = llm_settings.api_key
                llm_base_url = llm_settings.base_url
                llm_model = llm_settings.model
                llm_provider = llm_settings.provider
                print(f"✅ LLM loaded for text regeneration: {llm_provider}/{llm_model}")
                if llm_base_url:
                    print(f"📡 Using LLM base URL for regeneration: {llm_base_url}")
            except Exception as e:
                print(f"⚠️ Warning: LLM configuration not available: {e}")
                print("Text regeneration will be disabled")
    except Exception as e:
        print(f"Warning: Could not load FMP API key: {e}")
        fmp_api_key = None

    # --- Auto-fetch market data if not provided and API key available ---
    auto_fetched_metrics = {}
    market_data_origin = "provided"
    if not args.skip_auto_fetch and fmp_api_key:
        print(f"Auto-fetching market data for {args.company_ticker}...")
        try:
            auto_fetched_metrics = get_comprehensive_company_metrics(args.company_ticker, fmp_api_key)
            market_data_origin = auto_fetched_metrics.get("_fmp_cache", {}).get("primary_origin", "fresh_fetch")
            print("✅ Successfully auto-fetched market data")
            # Validate market cap vs share price consistency
            try:
                _price = auto_fetched_metrics.get('share_price')
                _mktcap_b = auto_fetched_metrics.get('market_cap')
                if _price and _mktcap_b and _price > 0 and _mktcap_b > 0:
                    _implied_shares = (_mktcap_b * 1e9) / _price
                    if _implied_shares < 1e7 or _implied_shares > 1e11:
                        print(f"  ⚠️  Market cap (${_mktcap_b:.2f}B) may be inconsistent with share price (${_price:.2f})")
                    else:
                        auto_fetched_metrics.setdefault('shares_outstanding', _implied_shares)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️  Warning: Auto-fetch failed: {e}")
            print("Will use provided values or defaults")
            market_data_origin = "no_data"
    elif args.skip_auto_fetch:
        print("Skipping auto-fetch as requested")
        market_data_origin = "skipped"
    else:
        print("⚠️  No FMP API key found, skipping auto-fetch")
        market_data_origin = "no_data"

    # --- Determine final values (command line args override auto-fetched) ---
    def get_value(arg_value, auto_key, default_value, format_func=None):
        """Get the final value, prioritizing: command line arg > auto-fetched > default"""
        if arg_value is not None:
            return format_func(arg_value) if format_func else arg_value
        elif auto_key in auto_fetched_metrics and auto_fetched_metrics[auto_key] is not None:
            value = auto_fetched_metrics[auto_key]
            return format_func(value) if format_func else value
        else:
            return default_value

    # Apply the logic for each metric
    share_price = get_value(args.share_price, 'share_price', 0.0, lambda x: f"${x:.2f}")
    target_price = get_value(args.target_price, 'target_price', 0.0, lambda x: f"${x:.2f}")
    rating = get_value(args.rating, 'rating', "N/A")
    market_cap = get_value(args.market_cap, 'market_cap', 0.0, lambda x: f"${x:,.2f}B")
    volume = get_value(args.volume, 'volume', 0.0, lambda x: f"{x:.2f}M")
    fwd_pe = get_value(args.fwd_pe, 'fwd_pe', 0.0, lambda x: f"{x:.1f}x")
    pb_ratio = get_value(args.pb_ratio, 'pb_ratio', 0.0, lambda x: f"{x:.2f}x")
    dividend_yield = get_value(args.dividend_yield, 'dividend_yield', "N/A", lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
    free_float = get_value(args.free_float, 'free_float', "N/A", lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x))
    roe = get_value(args.roe, 'roe', "N/A", lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x))
    net_debt_to_equity = get_value(args.net_debt_to_equity, 'net_debt_to_equity', "N/A", lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
    sector = get_value(args.sector, 'sector', "Industrials")
    week_52_range = get_value(None, '52w_range', 'N/A')

    # Print summary of what was auto-fetched vs. provided
    print(f"\n📊 Market Data Summary for {args.company_ticker}:")
    print(f"Note: auto-fetched now distinguishes FMP source origin: fresh_fetch / cache_hit / fallback_stale_cache.")
    print(f"  Share Price: {share_price} {(f'(auto-fetched:{market_data_origin})' if args.share_price is None and 'share_price' in auto_fetched_metrics else '(provided)')}")
    print(f"  Target Price: {target_price} {(f'(auto-fetched:{market_data_origin})' if args.target_price is None and 'target_price' in auto_fetched_metrics else '(provided)')}")
    print(f"  Rating: {rating} {(f'(auto-fetched:{market_data_origin})' if args.rating is None and 'rating' in auto_fetched_metrics else '(provided)')}")
    print(f"  Market Cap: {market_cap} {(f'(auto-fetched:{market_data_origin})' if args.market_cap is None and 'market_cap' in auto_fetched_metrics else '(provided)')}")
    print(f"  Sector: {sector} {(f'(auto-fetched:{market_data_origin})' if args.sector is None and 'sector' in auto_fetched_metrics else '(provided)')}")

    # --- Load data ---
    analysis_df = load_analysis_csv(args.analysis_csv)
    if analysis_df is None:
        print("Error: Could not load analysis CSV file")
        return

    peer_ebitda_df = load_analysis_csv(args.peer_ebitda_csv) if args.peer_ebitda_csv else pd.DataFrame()
    peer_ev_ebitda_df = load_analysis_csv(args.peer_ev_ebitda_csv) if args.peer_ev_ebitda_csv else pd.DataFrame()

    # Process text content with AI enhancement
    processed_texts = process_text_content(
        args,
        analysis_df,
        peer_ebitda_df,
        peer_ev_ebitda_df,
        llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_provider=llm_provider,
    )
    
    # Fix stale dates in cached text (e.g. "June 2024" → actual report date)
    import re as _re
    _report_date_label = args.report_date  # e.g. "March 2026"
    _stale_date_pattern = _re.compile(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}')
    def _fix_stale_dates(text: str) -> str:
        """Replace old month-year dates in title lines with current report date."""
        if not text:
            return text
        lines = text.split('\n')
        # Only fix the first 3 lines (titles/headers)
        for i in range(min(3, len(lines))):
            if _stale_date_pattern.search(lines[i]):
                lines[i] = _stale_date_pattern.sub(_report_date_label, lines[i])
        return '\n'.join(lines)

    # Assign processed texts
    tagline_text = processed_texts["tagline"]
    company_overview_text = _fix_stale_dates(processed_texts["company_overview"])
    investment_overview_text = _fix_stale_dates(processed_texts["investment_overview"])
    valuation_overview_text = _fix_stale_dates(processed_texts["valuation_overview"])
    risks_text = processed_texts["risks"]
    competitor_analysis_text = processed_texts["competitor_analysis"]
    major_takeaways_text = processed_texts["major_takeaways"]
    news_summary_text = processed_texts["news_summary"]

    numeric_consistency = run_numeric_consistency_checks(
        analysis_df=analysis_df,
        section_texts={
            "tagline": tagline_text,
            "company_overview": company_overview_text,
            "investment_overview": investment_overview_text,
            "valuation_overview": valuation_overview_text,
            "competitor_analysis": competitor_analysis_text,
            "major_takeaways": major_takeaways_text,
            "news_summary": news_summary_text,
        },
    )
    numeric_consistency_path = os.path.join(output_dir, "numeric_consistency.json")
    with open(numeric_consistency_path, "w", encoding="utf-8") as f:
        json.dump(numeric_consistency, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved numeric consistency report to: {numeric_consistency_path}")

    # --- Compute technical indicators ---
    technical_indicators = {}
    if fmp_api_key:
        try:
            technical_indicators = get_technical_indicators(args.company_ticker, fmp_api_key)
        except Exception as e:
            print(f"⚠️ Could not compute technical indicators: {e}")

    # --- Prepare report data ---
    report_data = {
        "company_ticker": args.company_ticker,
        "company_name_full": args.company_name,
        "company_name_ticker": f"{args.company_name} ({args.company_ticker})",
        "report_date": args.report_date,
        "sector": sector,
        "share_price": share_price,
        "target_price": target_price,
        "rating": rating,
        "market_cap": market_cap,
        "volume": volume,
        "fwd_pe": fwd_pe,
        "pb_ratio": pb_ratio,
        "roe": roe,
        "free_float": free_float,
        "dividend_yield": dividend_yield,
        "net_debt_to_equity": net_debt_to_equity,
        "52w_range": week_52_range,
        "tagline": tagline_text,
        "company_overview": company_overview_text,
        "investment_overview": investment_overview_text,
        "valuation_overview": valuation_overview_text,
        "risks": risks_text,
        "competitor_analysis": competitor_analysis_text,
        "major_takeaways": major_takeaways_text,
        "news_summary": news_summary_text,  # NEW
        "research_source": args.research_source,
        "data_source_text": args.data_source_text,
        "disclaimer_text": args.disclaimer_text,
        "logo_image_path": args.logo_image_path,
        "analyst_names": args.analyst_names,
        "analyst_emails": args.analyst_emails,
        "closing_price_date": args.closing_price_date,
        "technical_indicators": technical_indicators,
        "retail_sentiment": {},
        "numeric_consistency": numeric_consistency,
    }

    # --- Generate or load charts ---
    print("Handling charts...")
    if args.revenue_chart_path and os.path.exists(args.revenue_chart_path):
        report_data['revenue_chart_path'] = args.revenue_chart_path
    else:
        chart_path = os.path.join(output_dir, f"{args.company_ticker}_revenue_ebitda_chart.png")
        chart_result = generate_revenue_ebitda_chart(analysis_df, chart_path, args.company_ticker)
        report_data['revenue_chart_path'] = chart_result or ""

    # Generate EPS × PE chart
    eps_pe_chart_path = os.path.join(output_dir, f"{args.company_ticker}_eps_pe_chart.png")
    eps_pe_chart_result = generate_eps_pe_chart(analysis_df, eps_pe_chart_path, args.company_ticker)
    report_data['eps_pe_chart_path'] = eps_pe_chart_result or ""

    if args.ev_ebitda_chart_path and os.path.exists(args.ev_ebitda_chart_path):
        report_data['ev_ebitda_chart_path'] = args.ev_ebitda_chart_path
    elif peer_ev_ebitda_df is not None and not peer_ev_ebitda_df.empty:
        if 'year' in peer_ev_ebitda_df.columns:
            peer_ev_ebitda_df.set_index('year', inplace=True)
        chart_path = os.path.join(output_dir, f"{args.company_ticker}_peer_ev_ebitda_chart.png")
        chart_result = generate_ev_ebitda_peer_chart(peer_ev_ebitda_df, chart_path, args.company_ticker)
        report_data['ev_ebitda_chart_path'] = chart_result or ""
    else:
        report_data['ev_ebitda_chart_path'] = ""

    # --- Enhanced Charts Generation ---
    if args.enable_enhanced_charts:
        print("Generating enhanced charts...")
        try:
            chart_config = ChartConfig()
            enhanced_chart_gen = EnhancedChartGenerator(chart_config)
            enhanced_charts = {}
            
            # Prepare financial data dict for generate_all_charts
            # Load income statement data if available
            income_csv_path = os.path.join(os.path.dirname(args.analysis_csv), "income_statement_raw_data.csv")
            income_df = pd.DataFrame()
            if os.path.exists(income_csv_path):
                income_df = pd.read_csv(income_csv_path)
                print(f"✅ Loaded income statement data from {income_csv_path}")
            
            # Load price data if available
            price_csv_path = os.path.join(os.path.dirname(args.analysis_csv), "historical_price_full.csv")
            price_df = pd.DataFrame()
            if os.path.exists(price_csv_path):
                price_df = pd.read_csv(price_csv_path)
                print(f"✅ Loaded price data from {price_csv_path}")
            
            financial_data_for_charts = {
                'analysis': analysis_df,
                'income_statement': income_df,
                'peer_data': {},
                'valuation_data': {}
            }
            
            # Generate EPS × PE chart (works with analysis_df)
            eps_pe_result = enhanced_chart_gen.generate_eps_pe_chart(
                analysis_df, args.company_ticker, output_dir
            )
            if eps_pe_result:
                enhanced_charts['eps_pe'] = eps_pe_result
                print(f"✅ Generated EPS × PE chart")
            
            # Generate charts that work with income_df if available
            if not income_df.empty:
                # Revenue YoY chart
                revenue_yoy_result = enhanced_chart_gen.generate_revenue_yoy_chart(
                    income_df, args.company_ticker, output_dir
                )
                if revenue_yoy_result:
                    enhanced_charts['revenue_yoy'] = revenue_yoy_result
                    print(f"✅ Generated Revenue YoY chart")
                
                # EBITDA Margin chart
                ebitda_margin_result = enhanced_chart_gen.generate_ebitda_margin_chart(
                    income_df, args.company_ticker, output_dir
                )
                if ebitda_margin_result:
                    enhanced_charts['ebitda_margin'] = ebitda_margin_result
                    print(f"✅ Generated EBITDA Margin chart")
            
            # ========== 高级图表生成 ==========
            print("Generating advanced charts...")
            
            # 1. 股价走势图（含移动平均线和成交量）
            if not price_df.empty:
                stock_price_path = os.path.join(output_dir, f"{args.company_ticker}_stock_price_chart.png")
                stock_price_result = generate_stock_price_chart(
                    price_df, stock_price_path, args.company_ticker, "1Y"
                )
                if stock_price_result:
                    enhanced_charts['stock_price'] = stock_price_result
                    print(f"✅ Generated Stock Price chart")
                
                # 2. 技术指标图（RSI, MACD）
                tech_indicators_path = os.path.join(output_dir, f"{args.company_ticker}_technical_indicators.png")
                tech_result = generate_technical_indicators_chart(
                    price_df, tech_indicators_path, args.company_ticker
                )
                if tech_result:
                    enhanced_charts['technical_indicators'] = tech_result
                    print(f"✅ Generated Technical Indicators chart")
            
            # 3. 财务比率雷达图
            financial_ratios = {}
            if analysis_df is not None and not analysis_df.empty:
                # 从分析数据中提取财务比率
                ratio_metrics = ['EBITDA Margin', 'Contribution Margin', 'SG&A Margin', 'Revenue Growth']
                year_cols = [col for col in analysis_df.columns if col.endswith('A')]
                if year_cols:
                    latest_year = sorted(year_cols)[-1]
                    for metric in ratio_metrics:
                        row = analysis_df[analysis_df['metrics'] == metric]
                        if not row.empty:
                            val = row[latest_year].iloc[0]
                            if isinstance(val, str):
                                val = val.replace('%', '')
                            try:
                                financial_ratios[metric] = float(val)
                            except:
                                pass
                
                # 添加其他比率
                if roe and roe != 'N/A':
                    try:
                        financial_ratios['ROE'] = float(str(roe).replace('%', ''))
                    except:
                        pass
                if fwd_pe and fwd_pe != 'N/A':
                    try:
                        financial_ratios['P/E Ratio'] = float(str(fwd_pe).replace('x', ''))
                    except:
                        pass
            
            if financial_ratios:
                radar_path = os.path.join(output_dir, f"{args.company_ticker}_financial_radar.png")
                radar_result = generate_financial_radar_chart(
                    financial_ratios, radar_path, args.company_ticker
                )
                if radar_result:
                    enhanced_charts['financial_radar'] = radar_result
                    print(f"✅ Generated Financial Radar chart")
            
            # 4. 敏感性热力图（如果有敏感性分析数据）
            if report_data.get('sensitivity_analysis'):
                sensitivity_data = report_data['sensitivity_analysis']
                if 'matrix' in sensitivity_data:
                    sensitivity_df = pd.DataFrame(sensitivity_data['matrix'])
                    heatmap_path = os.path.join(output_dir, f"{args.company_ticker}_sensitivity_heatmap.png")
                    heatmap_result = generate_sensitivity_heatmap(
                        sensitivity_df, heatmap_path, args.company_ticker
                    )
                    if heatmap_result:
                        enhanced_charts['sensitivity_heatmap'] = heatmap_result
                        print(f"✅ Generated Sensitivity Heatmap")
            
            # 5. 估值瀑布图（如果有估值分析数据）
            if report_data.get('valuation_analysis'):
                valuation_data = report_data['valuation_analysis']
                waterfall_data = {}

                methods = valuation_data.get('methods', []) if isinstance(valuation_data, dict) else []
                if isinstance(methods, list):
                    for item in methods:
                        if not isinstance(item, dict):
                            continue
                        method_name = str(item.get('method', '')).strip()
                        target_val = _coerce_numeric(item.get('target_price'))
                        if method_name and target_val is not None and target_val > 0:
                            waterfall_data[method_name] = target_val

                # Backward compatibility
                legacy_map = {
                    'ev_ebitda': 'EV/EBITDA',
                    'dcf': 'DCF',
                    'peer_comparison': 'Peer Comp',
                }
                if isinstance(valuation_data, dict):
                    for old_key, display_name in legacy_map.items():
                        old_payload = valuation_data.get(old_key)
                        if isinstance(old_payload, dict):
                            target_val = _coerce_numeric(old_payload.get('target_price'))
                            if target_val is not None and target_val > 0:
                                waterfall_data.setdefault(display_name, target_val)

                if waterfall_data:
                    waterfall_path = os.path.join(output_dir, f"{args.company_ticker}_valuation_waterfall.png")
                    waterfall_result = generate_valuation_waterfall_chart(
                        waterfall_data, waterfall_path, args.company_ticker
                    )
                    if waterfall_result:
                        enhanced_charts['valuation_waterfall'] = waterfall_result
                        print(f"✅ Generated Valuation Waterfall chart")
            
            # 6. 现金流分析图（如果有现金流数据）
            cashflow_candidates = [
                os.path.join(os.path.dirname(args.analysis_csv), "cash_flow_raw_data.csv"),
                os.path.join(os.path.dirname(args.analysis_csv), "cash_flow_statement_raw_data.csv"),
            ]
            cashflow_csv_path = next((p for p in cashflow_candidates if os.path.exists(p)), None)
            if cashflow_csv_path:
                try:
                    cashflow_df = pd.read_csv(cashflow_csv_path)
                    if not cashflow_df.empty:
                        # 提取现金流数据
                        cf_data = {
                            'periods': cashflow_df['calendarYear'].tolist() if 'calendarYear' in cashflow_df.columns else [],
                            'Operating': cashflow_df['operatingCashFlow'].tolist() if 'operatingCashFlow' in cashflow_df.columns else [],
                            'Investing': cashflow_df['netCashUsedForInvestingActivites'].tolist() if 'netCashUsedForInvestingActivites' in cashflow_df.columns else [],
                            'Financing': cashflow_df['netCashUsedProvidedByFinancingActivities'].tolist() if 'netCashUsedProvidedByFinancingActivities' in cashflow_df.columns else []
                        }
                        
                        if cf_data['Operating']:
                            cashflow_path = os.path.join(output_dir, f"{args.company_ticker}_cash_flow_chart.png")
                            cf_result = generate_cash_flow_chart(
                                cf_data, cashflow_path, args.company_ticker
                            )
                            if cf_result:
                                enhanced_charts['cash_flow'] = cf_result
                                print(f"✅ Generated Cash Flow chart")
                except Exception as e:
                    print(f"⚠️ Error generating cash flow chart: {e}")
            
            report_data['enhanced_charts'] = enhanced_charts
            print(f"✅ Generated {len(enhanced_charts)} enhanced charts total")
        except Exception as e:
            print(f"⚠️ Error generating enhanced charts: {e}")
            import traceback
            traceback.print_exc()
            report_data['enhanced_charts'] = {}

    # --- Valuation Analysis ---
    if args.enable_valuation_analysis:
        print("Performing valuation analysis...")
        try:
            shares_outstanding = auto_fetched_metrics.get('shares_outstanding')
            if not shares_outstanding:
                try:
                    share_price_num = float(str(share_price).replace('$', '').replace(',', '')) if isinstance(share_price, str) else float(share_price)
                    market_cap_num = auto_fetched_metrics.get('market_cap')
                    if market_cap_num and share_price_num and share_price_num > 0:
                        # market_cap assumed to be in billions
                        inferred_shares = (float(market_cap_num) * 1e9) / share_price_num
                        if 1e6 <= inferred_shares <= 2e11:
                            shares_outstanding = inferred_shares
                except Exception:
                    pass

            if not shares_outstanding:
                shares_outstanding = 1e9

            # Add key_metrics from analysis folder for EV/EBITDA calibration if available
            key_metrics_path = os.path.join(os.path.dirname(args.analysis_csv), "key_metrics_raw_data.csv")
            key_metrics_df = pd.DataFrame()
            if os.path.exists(key_metrics_path):
                try:
                    key_metrics_df = pd.read_csv(key_metrics_path)
                except Exception as e:
                    print(f"⚠️ Warning: failed to load key metrics CSV: {e}")

            # Prepare financial data for valuation engine
            financial_data_for_valuation = {
                'analysis': analysis_df,
                'current_price': float(str(share_price).replace('$', '').replace(',', '')) if isinstance(share_price, str) else share_price,
                'shares_outstanding': shares_outstanding,
                'key_metrics': key_metrics_df,
            }

            # Prepare peer data if available
            peer_data_for_valuation = pd.DataFrame(columns=['year', 'ticker', 'ev_ebitda'])
            if peer_ev_ebitda_df is not None and not peer_ev_ebitda_df.empty:
                _peer = peer_ev_ebitda_df.copy()
                if 'year' in _peer.columns:
                    _peer = _peer.set_index('year')
                _peer_long = _peer.stack(dropna=True).reset_index(name='ev_ebitda')
                _peer_long.columns = ["year", "ticker", "ev_ebitda"]
                peer_data_for_valuation = _peer_long

            valuation_engine = ValuationEngine(financial_data_for_valuation, peer_data_for_valuation)

            # Run all three valuation methods
            valuation_engine.calculate_ev_ebitda_valuation()
            valuation_engine.calculate_dcf_valuation()
            valuation_engine.calculate_peer_comparison_valuation()

            # Synthesize weighted average
            synthesis = valuation_engine.synthesize_valuation()

            # Build structured results for HTML rendering
            valuation_results = {
                'synthesis': synthesis,
                'methods': []
            }
            for r in valuation_engine.valuation_results:
                if r.target_price > 0:
                    valuation_results['methods'].append({
                        'method': r.method,
                        'target_price': r.target_price,
                        'low_estimate': r.low_estimate,
                        'high_estimate': r.high_estimate,
                        'assumptions': r.assumptions,
                        'confidence': r.confidence,
                        'description': r.description,
                    })

            report_data['valuation_analysis'] = valuation_results
            print(f"✅ Valuation analysis completed with {len(valuation_results['methods'])} methods")
            if synthesis.get('target_price') and synthesis['target_price'] > 0:
                print(f"  Synthesized target: ${synthesis['target_price']:.2f} "
                      f"(range ${synthesis['range'][0]:.2f}-${synthesis['range'][1]:.2f}, "
                      f"upside {synthesis['upside']:.1f}%)")

            if args.enable_enhanced_charts:
                try:
                    methods = valuation_results.get('methods', [])
                    waterfall_data = {}
                    for item in methods:
                        if not isinstance(item, dict):
                            continue
                        method_name = str(item.get('method', '')).strip()
                        target_val = _coerce_numeric(item.get('target_price'))
                        if method_name and target_val is not None and target_val > 0:
                            waterfall_data[method_name] = target_val

                    if waterfall_data:
                        waterfall_path = os.path.join(output_dir, f"{args.company_ticker}_valuation_waterfall.png")
                        waterfall_result = generate_valuation_waterfall_chart(
                            waterfall_data, waterfall_path, args.company_ticker
                        )
                        if waterfall_result:
                            report_data.setdefault('enhanced_charts', {})
                            report_data['enhanced_charts']['valuation_waterfall'] = waterfall_result
                            print("✅ Generated Valuation Waterfall chart (post valuation step)")
                except Exception as e:
                    print(f"⚠️ Error generating valuation waterfall chart (post valuation): {e}")
    
        except Exception as e:
            print(f"⚠️ Error performing valuation analysis: {e}")
            import traceback
            traceback.print_exc()
            report_data['valuation_analysis'] = {}

    # --- Load Enhanced Analysis Files ---
    # Load sensitivity analysis
    if args.sensitivity_analysis_file and os.path.exists(args.sensitivity_analysis_file):
        print(f"Loading sensitivity analysis from {args.sensitivity_analysis_file}...")
        try:
            with open(args.sensitivity_analysis_file, 'r', encoding='utf-8') as f:
                report_data['sensitivity_analysis'] = json.load(f)
            print("✅ Sensitivity analysis loaded")
        except Exception as e:
            print(f"⚠️ Error loading sensitivity analysis: {e}")
            report_data['sensitivity_analysis'] = {}
    
    # Load catalyst analysis
    if args.catalyst_analysis_file and os.path.exists(args.catalyst_analysis_file):
        print(f"Loading catalyst analysis from {args.catalyst_analysis_file}...")
        try:
            with open(args.catalyst_analysis_file, 'r', encoding='utf-8') as f:
                catalyst_data = json.load(f)
            # Re-classify sentiments for cached data (fix analyst action misclassification)
            _pos_patterns = ['acquires new holdings', 'initiates coverage', 'initiate coverage',
                             'starts coverage', 'begins coverage', 'upgrades to', 'builds position',
                             'overweight', 'buy rating', 'outperform', 'top pick', 'new position']
            def _fix_sentiment(desc):
                dl = desc.lower()
                for p in _pos_patterns:
                    if p in dl:
                        return 'positive'
                return None
            # Fix catalysts list
            if 'catalysts' in catalyst_data:
                for cat in catalyst_data['catalysts']:
                    fix = _fix_sentiment(cat.get('description', ''))
                    if fix:
                        cat['sentiment'] = fix
            # Fix categorized lists
            if 'categorized' in catalyst_data:
                moved = []
                for cat_item in catalyst_data['categorized'].get('negative', []):
                    fix = _fix_sentiment(cat_item.get('description', ''))
                    if fix:
                        cat_item['sentiment'] = fix
                        moved.append(cat_item)
                for m in moved:
                    catalyst_data['categorized']['negative'].remove(m)
                    catalyst_data['categorized'].setdefault('positive', []).append(m)
            # Fix top_catalysts
            if 'top_catalysts' in catalyst_data:
                for tc in catalyst_data['top_catalysts']:
                    fix = _fix_sentiment(tc.get('catalyst', ''))
                    if fix:
                        tc['sentiment'] = fix
                        tc['weighted_impact'] = abs(tc.get('weighted_impact', 0))
            report_data['catalyst_analysis'] = catalyst_data
            print("✅ Catalyst analysis loaded (sentiments re-validated)")
        except Exception as e:
            print(f"⚠️ Error loading catalyst analysis: {e}")
            report_data['catalyst_analysis'] = {}
    
    # Load enhanced news
    if args.enhanced_news_file and os.path.exists(args.enhanced_news_file):
        print(f"Loading enhanced news from {args.enhanced_news_file}...")
        try:
            with open(args.enhanced_news_file, 'r', encoding='utf-8') as f:
                report_data['enhanced_news'] = json.load(f)
            print("✅ Enhanced news loaded")
        except Exception as e:
            print(f"⚠️ Error loading enhanced news: {e}")
            report_data['enhanced_news'] = {}

    retail_sentiment_path = args.retail_sentiment_file
    if not retail_sentiment_path:
        candidate_path = os.path.join(os.path.dirname(args.analysis_csv), "retail_sentiment.json")
        if os.path.exists(candidate_path):
            retail_sentiment_path = candidate_path

    if retail_sentiment_path and os.path.exists(retail_sentiment_path):
        print(f"Loading retail sentiment insights from {retail_sentiment_path}...")
        try:
            with open(retail_sentiment_path, "r", encoding="utf-8") as f:
                report_data["retail_sentiment"] = json.load(f)
            print("✅ Retail sentiment insights loaded")
        except Exception as e:
            print(f"⚠️ Error loading retail sentiment insights: {e}")
            report_data["retail_sentiment"] = {}

    # --- Format tables for HTML (EXCLUDE ESTIMATES FOR PAGE 3 TABLES) ---
    print("Formatting tables for HTML...")

    # For Page 3 tables, filter to only include actual years (no estimates)
    analysis_actual_only = filter_actual_years_only(analysis_df)

    summary_metrics = ["Revenue", "Revenue Growth", "EBITDA", "EBITDA Margin", "Contribution Profit", "Contribution Margin", "SG&A", "SG&A Margin"]
    financial_summary_df = analysis_actual_only[analysis_actual_only["metrics"].isin(summary_metrics)].set_index("metrics")
    report_data["financial_summary_table_html"] = format_dataframe_to_html_table(financial_summary_df, table_id="financial-summary")

    valuation_metrics = ["EBITDA Margin", "Contribution Margin", "SG&A Margin", "Revenue Growth"]
    valuation_metrics_df = analysis_actual_only[analysis_actual_only["metrics"].isin(valuation_metrics)].set_index("metrics")
    report_data["valuation_metrics_table_html"] = format_dataframe_to_html_table(valuation_metrics_df, table_id="valuation-metrics")

    # Load and format Credit & Cashflow metrics from the provided CSV
    print("Loading Credit & Cashflow metrics from CSV...")
    credit_cashflow_df = load_credit_cashflow_metrics_from_csv(args.ratios_csv)
    if not credit_cashflow_df.empty:
        credit_cashflow_actual = filter_actual_years_only(credit_cashflow_df)
        credit_cashflow_formatted = credit_cashflow_actual.set_index("metrics")
        report_data["credit_cashflow_table_html"] = format_dataframe_to_html_table(credit_cashflow_formatted, table_id="credit-cashflow")
        print("✅ Successfully loaded and formatted Credit & Cashflow metrics from CSV")
    else:
        report_data["credit_cashflow_table_html"] = "<p>Credit & Cashflow metrics not available.</p>"
        print("❌ Failed to load Credit & Cashflow metrics from CSV")


    # Handle peer data for tables - fix the filtering issue
    if peer_ebitda_df is not None and not peer_ebitda_df.empty:
        print(f"Processing peer EBITDA data with shape: {peer_ebitda_df.shape}")
        print(f"Peer EBITDA columns: {peer_ebitda_df.columns.tolist()}")

        if 'year' in peer_ebitda_df.columns:
            peer_ebitda_df.set_index('year', inplace=True)

        # Don't filter out estimates - show all available data
        print(f"Peer EBITDA index: {peer_ebitda_df.index.tolist()}")

        if not peer_ebitda_df.empty:
            report_data["peer_ebitda_table_html"] = format_dataframe_to_html_table(peer_ebitda_df.T, table_id="peer-ebitda-summary")
            print("✅ Successfully formatted peer EBITDA table")
        else:
            report_data["peer_ebitda_table_html"] = "<p>Peer EBITDA data not available.</p>"
            print("❌ Peer EBITDA DataFrame is empty")
    else:
        report_data["peer_ebitda_table_html"] = "<p>Peer EBITDA data not available.</p>"
        print("❌ No peer EBITDA data provided")

    if peer_ev_ebitda_df is not None and not peer_ev_ebitda_df.empty:
        print(f"Processing peer EV/EBITDA data with shape: {peer_ev_ebitda_df.shape}")
        print(f"Peer EV/EBITDA columns: {peer_ev_ebitda_df.columns.tolist()}")

        if 'year' in peer_ev_ebitda_df.columns:
            peer_ev_ebitda_df.set_index('year', inplace=True)

        # Don't filter out estimates - show all available data
        print(f"Peer EV/EBITDA index: {peer_ev_ebitda_df.index.tolist()}")

        if not peer_ev_ebitda_df.empty:
            # Replace negative EV/EBITDA with "N/M" (not meaningful — negative EBITDA)
            peer_ev_ebitda_display = peer_ev_ebitda_df.copy()
            _nm_func = lambda x: "N/M" if isinstance(x, (int, float)) and x < 0 else x
            peer_ev_ebitda_display = peer_ev_ebitda_display.apply(lambda col: col.map(_nm_func))
            report_data["peer_ev_ebitda_table_html"] = format_dataframe_to_html_table(peer_ev_ebitda_display.T, table_id="peer-ev-ebitda-summary")
            print("✅ Successfully formatted peer EV/EBITDA table")
        else:
            report_data["peer_ev_ebitda_table_html"] = "<p>Peer EV/EBITDA data not available.</p>"
            print("❌ Peer EV/EBITDA DataFrame is empty")
    else:
        report_data["peer_ev_ebitda_table_html"] = "<p>Peer EV/EBITDA data not available.</p>"
        print("❌ No peer EV/EBITDA data provided")

    # --- Generate Professional HTML Report (matching PDF structure) ---
    print("Generating professional HTML report (matching PDF structure)...")
    
    # Add additional data needed for professional template
    report_data['revenue_analysis_text'] = f"{report_data.get('company_name_full', 'The company')} has demonstrated consistent revenue performance over the analysis period. Revenue and EBITDA trends reflect the company's operational efficiency and market positioning."
    report_data['eps_analysis_text'] = f"{report_data.get('company_name_full', 'The company')}'s earnings trajectory reflects the company's profitability trends, while valuation multiples indicate market expectations for future growth."
    
    # Extract key figures from analysis_df
    if analysis_df is not None and not analysis_df.empty:
        years = [col for col in analysis_df.columns if col.endswith('A')]
        latest_year = years[-1] if years else None
        
        revenue_figures = {}
        eps_figures = {}
        
        if latest_year:
            for metric in ['Revenue', 'EBITDA', 'Revenue Growth']:
                row = analysis_df[analysis_df['metrics'] == metric]
                if not row.empty:
                    val = row[latest_year].values[0]
                    revenue_figures[f"{metric} ({latest_year})"] = str(val)
            
            for metric in ['EPS', 'PE Ratio']:
                row = analysis_df[analysis_df['metrics'] == metric]
                if not row.empty:
                    val = row[latest_year].values[0]
                    eps_figures[f"{metric} ({latest_year})"] = str(val)
        
        report_data['revenue_key_figures'] = revenue_figures
        report_data['eps_key_figures'] = eps_figures
    
    # Generate professional HTML report
    professional_html_path = os.path.join(output_dir, f"Professional_Equity_Report_{args.company_ticker}.html")
    professional_html_content = render_professional_html_report(report_data)
    with open(professional_html_path, "w", encoding="utf-8") as f:
        f.write(professional_html_content)
    print(f"✅ Generated Professional HTML Report: {professional_html_path}")

    # --- Generate Combined HTML Report (all sections in one file) ---
    print("Generating combined HTML report (all sections in one file)...")
    combined_html_path = os.path.join(output_dir, f"Combined_Equity_Report_{args.company_ticker}.html")
    combined_html_content = render_combined_html_report(report_data)
    with open(combined_html_path, "w", encoding="utf-8") as f:
        f.write(combined_html_content)
    print(f"✅ Generated Combined HTML Report: {combined_html_path}")

    # --- Also render legacy HTML pages for backward compatibility ---
    print("Rendering legacy HTML pages...")
    templates = [HTML_TEMPLATE_PAGE_1, HTML_TEMPLATE_PAGE_2_FINANCIAL_SUMMARY, HTML_TEMPLATE_PAGE_3_PEER_COMPARISON]
    
    # Add Page 4 (Sensitivity & Catalyst) if enabled
    has_sensitivity_catalyst = (
        report_data.get('sensitivity_analysis') or 
        report_data.get('catalyst_analysis')
    )
    if has_sensitivity_catalyst:
        templates.append(HTML_TEMPLATE_PAGE_4_SENSITIVITY_CATALYST)
        print("✅ Sensitivity/Catalyst content detected, adding Page 4")
    
    # Add Page 5 (News & Charts) if enabled
    has_news_charts = (
        report_data.get('enhanced_news') or 
        report_data.get('enhanced_charts')
    )
    if has_news_charts:
        templates.append(HTML_TEMPLATE_PAGE_5_NEWS_CHARTS)
        print("✅ News/Charts content detected, adding Page 5")
    
    for page_num, template in enumerate(templates, 1):
        page_path = os.path.join(output_dir, f"{args.html_report_prefix}_Page{page_num}_{args.company_ticker}.html")
        html_content = render_html_report(template, report_data)
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Generated Page {page_num}: {page_path}")

    print(f"\n✅ Equity report generation complete!")
    print(f"📁 Reports saved to: {output_dir}")
    if auto_fetched_metrics:
        print(f"🤖 Market data automatically fetched from FMP API")


if __name__ == "__main__":
    main()
