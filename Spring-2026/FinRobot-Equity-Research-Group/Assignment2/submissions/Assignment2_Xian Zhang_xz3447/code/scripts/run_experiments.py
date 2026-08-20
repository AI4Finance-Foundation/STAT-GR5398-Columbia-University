#!/usr/bin/env python
# coding: utf-8
"""Batch experiment runner for the FinRobot Assignment 2 model matrix.

For each (profile, ticker) cell, this script runs the existing two-step
pipeline (`generate_financial_analysis.py` → `create_equity_report.py`) with:

  - FINROBOT_PROFILE  set so text_generator_agents picks the right routing
  - OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY exported from config.ini
  - FINROBOT_AUDIT_DIR set so each section writes a {section}.audit.json file
    capturing draft / critique / verdict / final text

Output layout:

  output/experiments/
    <profile>/
      <TICKER>/
        analysis/      raw FMP CSVs + .txt section files (post-revise)
        audit/         per-section audit JSON (draft, critique, final)
        report/        Equity_Report.html

Usage:
    cd finrobot_equity
    python scripts/run_experiments.py                      # all profiles × all 5 companies
    python scripts/run_experiments.py --profile claude_all # one profile, all 5 companies
    python scripts/run_experiments.py --ticker NVDA        # all profiles, just NVDA
    python scripts/run_experiments.py --profile mixed_critic --ticker NVDA
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
EQUITY_ROOT = THIS_DIR.parent                           # finrobot_equity/
CORE_ROOT = EQUITY_ROOT / "core"                        # finrobot_equity/core/
SRC_DIR = CORE_ROOT / "src"
CONFIG_PATH = CORE_ROOT / "config" / "config.ini"
EXPERIMENTS_DIR = CORE_ROOT / "output" / "experiments"

GEN_FIN_SCRIPT = SRC_DIR / "generate_financial_analysis.py"
CREATE_REPORT_SCRIPT = SRC_DIR / "create_equity_report.py"

# Required-companies list from the assignment + their peer baskets
COMPANIES: List[Tuple[str, str, List[str]]] = [
    ("NVDA",  "NVIDIA Corporation",          ["AMD", "INTC"]),
    ("AMD",   "Advanced Micro Devices, Inc.", ["NVDA", "INTC"]),
    ("INTC",  "Intel Corporation",            ["NVDA", "AMD"]),
    ("AAPL",  "Apple Inc.",                   ["GOOGL", "MSFT"]),
    ("GOOGL", "Alphabet Inc.",                ["AAPL", "MSFT"]),
]

# Profiles available (must match keys in modules.model_routing.PROFILES)
ALL_PROFILES: List[str] = [
    "gpt_baseline",
    "claude_all",
    "gemini_all",
    "mixed_critic",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_keys() -> Dict[str, str]:
    cp = configparser.ConfigParser()
    if not CONFIG_PATH.exists():
        sys.exit(f"config.ini not found at {CONFIG_PATH}")
    cp.read(CONFIG_PATH)
    return {
        "FMP_API_KEY": cp.get("API_KEYS", "fmp_api_key", fallback=""),
        "OPENAI_API_KEY": cp.get("API_KEYS", "openai_api_key", fallback=""),
        "ANTHROPIC_API_KEY": cp.get("API_KEYS", "claude_api_key", fallback=""),
        "GEMINI_API_KEY": cp.get("API_KEYS", "gemini_api_key", fallback=""),
    }


def make_env(profile: str, audit_dir: Path, api_keys: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env["FINROBOT_PROFILE"] = profile
    env["FINROBOT_AUDIT_DIR"] = str(audit_dir)
    for k, v in api_keys.items():
        if v:
            env[k] = v
    return env


@dataclass
class CellResult:
    profile: str
    ticker: str
    analysis_ok: bool = False
    report_ok: bool = False
    error: Optional[str] = None
    duration_sec: float = 0.0
    sections_generated: List[str] = field(default_factory=list)
    html_report_path: Optional[str] = None


def section_files_present(analysis_dir: Path) -> List[str]:
    return [
        s for s in [
            "tagline", "company_overview", "investment_overview",
            "valuation_overview", "risks", "competitor_analysis",
            "major_takeaways", "news_summary",
        ] if (analysis_dir / f"{s}.txt").exists()
    ]


def run_step1(profile: str, ticker: str, name: str, peers: List[str],
              analysis_dir: Path, audit_dir: Path, api_keys: Dict[str, str]) -> Tuple[bool, str]:
    cmd = [
        sys.executable, str(GEN_FIN_SCRIPT),
        "--company-ticker", ticker,
        "--company-name", name,
        "--config-file", str(CONFIG_PATH),
        "--output-dir", str(analysis_dir),
        "--peer-tickers", *peers,
        "--generate-text-sections",
    ]
    env = make_env(profile, audit_dir, api_keys)
    print(f"\n=== STEP1 [{profile}/{ticker}] ===")
    print(" ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(SRC_DIR), capture_output=True, text=True, timeout=2400,
        )
    except subprocess.TimeoutExpired:
        return False, "Step1 timeout (40 min)"
    sys.stdout.write(proc.stdout[-4000:])
    sys.stdout.write(proc.stderr[-2000:])
    if proc.returncode != 0:
        return False, f"Step1 returncode={proc.returncode}"
    return True, ""


def run_step2(profile: str, ticker: str, name: str, analysis_dir: Path,
              report_dir: Path, audit_dir: Path, api_keys: Dict[str, str]) -> Tuple[bool, str]:
    files = {
        "tagline-file": analysis_dir / "tagline.txt",
        "company-overview-file": analysis_dir / "company_overview.txt",
        "investment-overview-file": analysis_dir / "investment_overview.txt",
        "valuation-overview-file": analysis_dir / "valuation_overview.txt",
        "risks-file": analysis_dir / "risks.txt",
        "competitor-analysis-file": analysis_dir / "competitor_analysis.txt",
        "major-takeaways-file": analysis_dir / "major_takeaways.txt",
        "news-summary-file": analysis_dir / "news_summary.txt",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        return False, f"Missing section files: {missing}"

    cmd = [
        sys.executable, str(CREATE_REPORT_SCRIPT),
        "--company-ticker", ticker,
        "--company-name", name,
        "--analysis-csv", str(analysis_dir / "financial_metrics_and_forecasts.csv"),
        "--ratios-csv", str(analysis_dir / "ratios_raw_data.csv"),
        "--config-file", str(CONFIG_PATH),
        "--output-dir", str(report_dir),
    ]
    for k, p in files.items():
        cmd += [f"--{k}", str(p)]
    if (analysis_dir / "peer_ebitda_comparison.csv").exists():
        cmd += ["--peer-ebitda-csv", str(analysis_dir / "peer_ebitda_comparison.csv")]
    if (analysis_dir / "peer_ev_ebitda_comparison.csv").exists():
        cmd += ["--peer-ev-ebitda-csv", str(analysis_dir / "peer_ev_ebitda_comparison.csv")]

    env = make_env(profile, audit_dir, api_keys)
    print(f"\n=== STEP2 [{profile}/{ticker}] ===")
    print(" ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(SRC_DIR), capture_output=True, text=True, timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return False, "Step2 timeout (20 min)"
    sys.stdout.write(proc.stdout[-2000:])
    sys.stdout.write(proc.stderr[-1500:])
    if proc.returncode != 0:
        return False, f"Step2 returncode={proc.returncode}"
    return True, ""


def find_html_report(report_dir: Path, ticker: str) -> Optional[Path]:
    if not report_dir.exists():
        return None
    candidates = sorted(report_dir.glob(f"{ticker}*.html"))
    if not candidates:
        candidates = sorted(report_dir.glob("*.html"))
    return candidates[0] if candidates else None


def run_cell(profile: str, ticker: str, name: str, peers: List[str],
             api_keys: Dict[str, str], skip_existing: bool) -> CellResult:
    cell_root = EXPERIMENTS_DIR / profile / ticker
    analysis_dir = cell_root / "analysis"
    audit_dir = cell_root / "audit"
    report_dir = cell_root / "report"
    for d in (analysis_dir, audit_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = CellResult(profile=profile, ticker=ticker)
    started = time.time()

    # Skip step1 if all section files already exist (resume mode)
    existing_sections = section_files_present(analysis_dir)
    has_csvs = (analysis_dir / "financial_metrics_and_forecasts.csv").exists()
    if skip_existing and has_csvs and len(existing_sections) >= 7:
        print(f"\n[skip] {profile}/{ticker}: section files already present ({len(existing_sections)}/8)")
        result.analysis_ok = True
        result.sections_generated = existing_sections
    else:
        ok, err = run_step1(profile, ticker, name, peers, analysis_dir, audit_dir, api_keys)
        result.analysis_ok = ok
        if not ok:
            result.error = err
            result.duration_sec = time.time() - started
            return result
        result.sections_generated = section_files_present(analysis_dir)

    # Step 2: render HTML report
    existing_html = find_html_report(report_dir, ticker)
    if skip_existing and existing_html:
        print(f"[skip] {profile}/{ticker}: HTML already exists at {existing_html}")
        result.report_ok = True
        result.html_report_path = str(existing_html)
    else:
        ok, err = run_step2(profile, ticker, name, analysis_dir, report_dir, audit_dir, api_keys)
        result.report_ok = ok
        if not ok:
            result.error = err
        else:
            html = find_html_report(report_dir, ticker)
            result.html_report_path = str(html) if html else None

    result.duration_sec = time.time() - started
    return result


def write_results_summary(results: List[CellResult]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EXPERIMENTS_DIR / "results.json"
    payload = {
        "results": [r.__dict__ for r in results],
        "matrix": {},
    }
    for r in results:
        payload["matrix"].setdefault(r.profile, {})[r.ticker] = {
            "analysis_ok": r.analysis_ok,
            "report_ok": r.report_ok,
            "sections": len(r.sections_generated),
            "duration_sec": round(r.duration_sec, 1),
            "error": r.error,
        }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults summary → {summary_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="append", choices=ALL_PROFILES,
                   help="Run only this profile (can be repeated). Default: all four.")
    p.add_argument("--ticker", action="append",
                   help="Run only this ticker (can be repeated). Default: NVDA AMD INTC AAPL GOOGL.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells whose outputs already exist (resume mode).")
    p.add_argument("--clean", action="store_true",
                   help="Wipe output/experiments/<profile>/ before running.")
    args = p.parse_args()

    profiles = args.profile or ALL_PROFILES
    tickers = set(args.ticker) if args.ticker else None

    cells = [
        (prof, t, n, peers)
        for prof in profiles
        for (t, n, peers) in COMPANIES
        if tickers is None or t in tickers
    ]
    if not cells:
        sys.exit("No cells to run. Check --profile / --ticker filters.")

    api_keys = load_api_keys()
    if not api_keys.get("FMP_API_KEY"):
        sys.exit("FMP_API_KEY not configured in config.ini")

    if args.clean:
        for prof in profiles:
            d = EXPERIMENTS_DIR / prof
            if d.exists():
                print(f"[clean] removing {d}")
                shutil.rmtree(d)

    print(f"Running {len(cells)} cells: profiles={profiles} tickers={[c[1] for c in cells]}")
    results: List[CellResult] = []
    for prof, ticker, name, peers in cells:
        r = run_cell(prof, ticker, name, peers, api_keys, skip_existing=args.skip_existing)
        results.append(r)
        write_results_summary(results)
        status = "OK" if (r.analysis_ok and r.report_ok) else "FAIL"
        print(f"\n[{status}] {prof}/{ticker}  duration={r.duration_sec:.1f}s  sections={len(r.sections_generated)}  error={r.error}")

    n_ok = sum(1 for r in results if r.analysis_ok and r.report_ok)
    print(f"\n{'='*60}\nDone: {n_ok}/{len(results)} cells succeeded")


if __name__ == "__main__":
    main()
