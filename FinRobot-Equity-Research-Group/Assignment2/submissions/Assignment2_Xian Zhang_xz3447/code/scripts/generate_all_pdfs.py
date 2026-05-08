#!/usr/bin/env python
# coding: utf-8
"""Generate Professional PDF reports for every (profile, ticker) cell.

Runs as a third pass after `run_experiments.py` has produced the analysis +
HTML reports. Reuses the cached FMP data and section .txt files so no LLM
calls are needed.

Usage:
    cd finrobot_equity
    python scripts/generate_all_pdfs.py                  # all 20 cells
    python scripts/generate_all_pdfs.py --skip-existing  # skip cells that already have a PDF
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EQUITY_ROOT = THIS_DIR.parent
CORE_ROOT = EQUITY_ROOT / "core"
SRC_DIR = CORE_ROOT / "src"
PDF_SCRIPT = SRC_DIR / "generate_pdf_report.py"
CONFIG_PATH = CORE_ROOT / "config" / "config.ini"
EXPERIMENTS_DIR = CORE_ROOT / "output" / "experiments"

PROFILES = ["gpt_baseline", "claude_all", "gemini_all", "mixed_critic"]
COMPANIES = [
    ("NVDA",  "NVIDIA Corporation"),
    ("AMD",   "Advanced Micro Devices, Inc."),
    ("INTC",  "Intel Corporation"),
    ("AAPL",  "Apple Inc."),
    ("GOOGL", "Alphabet Inc."),
]


def run_pdf(profile: str, ticker: str, name: str, skip_existing: bool) -> bool:
    cell = EXPERIMENTS_DIR / profile / ticker
    analysis_dir = cell / "analysis"
    report_dir = cell / "report"
    pdf_path = report_dir / f"Professional_Equity_Report_{ticker}.pdf"

    if not analysis_dir.exists():
        print(f"  [skip] {profile}/{ticker}: no analysis dir")
        return False

    if skip_existing and pdf_path.exists():
        size = pdf_path.stat().st_size
        print(f"  [skip] {profile}/{ticker}: PDF exists ({size:,} bytes)")
        return True

    cmd = [
        sys.executable, str(PDF_SCRIPT),
        "--company-ticker", ticker,
        "--company-name", name,
        "--analysis-dir", str(analysis_dir),
        "--output-dir", str(report_dir),
        "--config-file", str(CONFIG_PATH),
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(SRC_DIR), capture_output=True, text=True, timeout=600)
    duration = time.time() - started

    if proc.returncode != 0 or not pdf_path.exists():
        sys.stdout.write(proc.stdout[-2000:])
        sys.stdout.write(proc.stderr[-1500:])
        print(f"  [FAIL] {profile}/{ticker}: returncode={proc.returncode}")
        return False
    size = pdf_path.stat().st_size
    print(f"  [OK] {profile}/{ticker}: {size:,} bytes in {duration:.1f}s")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="append", choices=PROFILES, help="Limit to profile(s).")
    p.add_argument("--ticker", action="append", help="Limit to ticker(s).")
    p.add_argument("--skip-existing", action="store_true", help="Skip cells with existing PDF.")
    args = p.parse_args()

    profiles = args.profile or PROFILES
    tickers = set(args.ticker) if args.ticker else None

    total, ok = 0, 0
    for prof in profiles:
        print(f"\n=== {prof} ===")
        for ticker, name in COMPANIES:
            if tickers and ticker not in tickers:
                continue
            total += 1
            if run_pdf(prof, ticker, name, args.skip_existing):
                ok += 1

    print(f"\n{'='*50}\nDone: {ok}/{total} PDFs generated")


if __name__ == "__main__":
    main()
