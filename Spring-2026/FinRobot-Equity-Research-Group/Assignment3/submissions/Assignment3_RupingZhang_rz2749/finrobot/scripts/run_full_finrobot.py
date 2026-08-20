#!/usr/bin/env python
"""Run real FinRobot equity analysis for all FinRL selected tickers.

This wrapper keeps course integration code outside the vendored
external/FinRobot repository.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINRL = ROOT / "finrl" / "finrl_stock_selection.csv"
DEFAULT_CONFIG = ROOT / "external" / "FinRobot" / "finrobot_equity" / "core" / "config" / "config.ini"
GENERATE_SCRIPT = ROOT / "external" / "FinRobot" / "finrobot_equity" / "core" / "src" / "generate_financial_analysis.py"
REPORT_SCRIPT = ROOT / "external" / "FinRobot" / "finrobot_equity" / "core" / "src" / "create_equity_report.py"
DEFAULT_PEER_MAP = ROOT / "finrobot_tracka" / "company_peer_map.csv"
DEFAULT_OUTPUT = ROOT / "finrobot_outputs"


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def existing_analysis_complete(analysis_dir: Path, require_text: bool) -> bool:
    required = [
        "financial_metrics_and_forecasts.csv",
        "ratios_raw_data.csv",
        "peer_ebitda_comparison.csv",
        "peer_ev_ebitda_comparison.csv",
        "company_news.json",
        "enhanced_news.json",
        "catalyst_analysis.json",
        "sensitivity_analysis.json",
        "retail_sentiment.json",
    ]
    if require_text:
        required += [
            "tagline.txt",
            "company_overview.txt",
            "investment_overview.txt",
            "valuation_overview.txt",
            "risks.txt",
            "competitor_analysis.txt",
            "major_takeaways.txt",
            "news_summary.txt",
        ]
    return all((analysis_dir / name).exists() for name in required)


def existing_report_complete(report_dir: Path, ticker: str) -> bool:
    return (
        (report_dir / f"Professional_Equity_Report_{ticker}.html").exists()
        and (report_dir / f"Combined_Equity_Report_{ticker}.html").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finrl-csv", default=str(DEFAULT_FINRL))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--peer-map", default=str(DEFAULT_PEER_MAP))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tickers", nargs="*", help="Optional subset to run.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-text", action="store_true", help="Skip OpenAI text generation.")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finrl = pd.read_csv(args.finrl_csv)
    selected = finrl[finrl["selected"] == 1].copy() if "selected" in finrl.columns else finrl.copy()
    tickers = sorted(selected["ticker"].unique())
    if args.tickers:
        requested = set(args.tickers)
        tickers = [ticker for ticker in tickers if ticker in requested]

    peer_map = pd.read_csv(args.peer_map).set_index("ticker")
    output_root = Path(args.output_root)

    print(f"FinRL input: {args.finrl_csv}")
    print(f"Selected tickers ({len(tickers)}): {', '.join(tickers)}")

    for idx, ticker in enumerate(tickers, start=1):
        if ticker not in peer_map.index:
            raise ValueError(f"Missing {ticker} in {args.peer_map}")

        company_name = str(peer_map.loc[ticker, "company_name"])
        peers = str(peer_map.loc[ticker, "peer_tickers"]).split()
        ticker_dir = output_root / ticker
        analysis_dir = ticker_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{idx}/{len(tickers)}] {ticker} - {company_name} ===", flush=True)

        require_text = not args.no_text
        if args.skip_existing and existing_analysis_complete(analysis_dir, require_text):
            print(f"Skipping analysis for {ticker}; existing outputs look complete.")
        else:
            cmd = [
                args.python,
                str(GENERATE_SCRIPT),
                "--company-ticker",
                ticker,
                "--company-name",
                company_name,
                "--config-file",
                args.config_file,
                "--peer-tickers",
                *peers,
                "--enable-enhanced-news",
                "--enable-catalyst-analysis",
                "--enable-sensitivity-analysis",
                "--output-dir",
                str(analysis_dir),
            ]
            if not args.no_text:
                cmd.append("--generate-text-sections")
            run(cmd, dry_run=args.dry_run)

        if args.no_report:
            continue

        if args.skip_existing and existing_report_complete(ticker_dir, ticker):
            print(f"Skipping report for {ticker}; existing HTML reports look complete.")
            continue

        report_cmd = [
            args.python,
            str(REPORT_SCRIPT),
            "--company-ticker",
            ticker,
            "--company-name",
            company_name,
            "--analysis-csv",
            str(analysis_dir / "financial_metrics_and_forecasts.csv"),
            "--ratios-csv",
            str(analysis_dir / "ratios_raw_data.csv"),
            "--peer-ebitda-csv",
            str(analysis_dir / "peer_ebitda_comparison.csv"),
            "--peer-ev-ebitda-csv",
            str(analysis_dir / "peer_ev_ebitda_comparison.csv"),
            "--tagline-file",
            str(analysis_dir / "tagline.txt"),
            "--company-overview-file",
            str(analysis_dir / "company_overview.txt"),
            "--investment-overview-file",
            str(analysis_dir / "investment_overview.txt"),
            "--valuation-overview-file",
            str(analysis_dir / "valuation_overview.txt"),
            "--risks-file",
            str(analysis_dir / "risks.txt"),
            "--competitor-analysis-file",
            str(analysis_dir / "competitor_analysis.txt"),
            "--major-takeaways-file",
            str(analysis_dir / "major_takeaways.txt"),
            "--news-summary-file",
            str(analysis_dir / "news_summary.txt"),
            "--sensitivity-analysis-file",
            str(analysis_dir / "sensitivity_analysis.json"),
            "--catalyst-analysis-file",
            str(analysis_dir / "catalyst_analysis.json"),
            "--enhanced-news-file",
            str(analysis_dir / "enhanced_news.json"),
            "--retail-sentiment-file",
            str(analysis_dir / "retail_sentiment.json"),
            "--config-file",
            args.config_file,
            "--output-dir",
            str(ticker_dir),
            "--html-report-prefix",
            f"{ticker}_Equity_Research_Report",
        ]
        run(report_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
