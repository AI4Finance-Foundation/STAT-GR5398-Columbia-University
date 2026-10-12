"""Runs the finrobot_equity pipeline (Step 1 + Step 2) for one company.

Usage (from the "Bruce Yang MR" folder, with the .venv interpreter):
    python scripts/run_pipeline.py NVDA "NVIDIA Corporation" --peers AMD INTC
    python scripts/run_pipeline.py NVDA "NVIDIA Corporation" --peers AMD INTC --tag run2

Outputs go to FinRobot/output/<TICKER>[/<tag>]/{analysis,report}, with the full
console output of each step saved next to them as step1.log / step2.log.
Any extra arguments after "--" are passed to Step 1 (e.g. forecast assumptions):
    python scripts/run_pipeline.py AMD "Advanced Micro Devices, Inc." --peers NVDA INTC -- --revenue-growth-2025 0.2
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "FinRobot"
SRC = REPO / "finrobot_equity" / "core" / "src"
CONFIG = REPO / "finrobot_equity" / "core" / "config" / "config.ini"


def run_step(name: str, cmd: list[str], log_path: Path) -> float:
    print(f"[{name}] running... (log: {log_path.relative_to(ROOT)})")
    start = time.time()
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        sys.exit(f"[{name}] failed with exit code {result.returncode}; see {log_path}")
    print(f"[{name}] done in {elapsed:.0f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("name")
    parser.add_argument("--peers", nargs="+", required=True)
    parser.add_argument("--tag", help="sub-folder for repeated runs, e.g. run2")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]

    run_dir = REPO / "output" / args.ticker
    if args.tag:
        run_dir = run_dir / args.tag
    analysis, report = run_dir / "analysis", run_dir / "report"
    analysis.mkdir(parents=True, exist_ok=True)

    step1 = [
        sys.executable, str(SRC / "generate_financial_analysis.py"),
        "--company-ticker", args.ticker, "--company-name", args.name,
        "--config-file", str(CONFIG),
        "--peer-tickers", *args.peers,
        "--output-dir", str(analysis),
        "--generate-text-sections",
        "--enable-sensitivity-analysis", "--enable-catalyst-analysis", "--enable-enhanced-news",
        *extra,
    ]
    t1 = run_step("step1", step1, run_dir / "step1.log")

    a = lambda f: str(analysis / f)
    step2 = [
        sys.executable, str(SRC / "create_equity_report.py"),
        "--company-ticker", args.ticker, "--company-name", args.name,
        "--analysis-csv", a("financial_metrics_and_forecasts.csv"),
        "--ratios-csv", a("ratios_raw_data.csv"),
        "--tagline-file", a("tagline.txt"),
        "--company-overview-file", a("company_overview.txt"),
        "--investment-overview-file", a("investment_overview.txt"),
        "--valuation-overview-file", a("valuation_overview.txt"),
        "--risks-file", a("risks.txt"),
        "--competitor-analysis-file", a("competitor_analysis.txt"),
        "--major-takeaways-file", a("major_takeaways.txt"),
        "--news-summary-file", a("news_summary.txt"),
        "--peer-ebitda-csv", a("peer_ebitda_comparison.csv"),
        "--peer-ev-ebitda-csv", a("peer_ev_ebitda_comparison.csv"),
        "--sensitivity-analysis-file", a("sensitivity_analysis.json"),
        "--enable-enhanced-charts", "--enable-valuation-analysis",
        "--enable-text-regeneration",
        "--output-dir", str(report),
        "--config-file", str(CONFIG),
    ]
    t2 = run_step("step2", step2, run_dir / "step2.log")
    print(f"{args.ticker}: total {t1 + t2:.0f}s -> {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
