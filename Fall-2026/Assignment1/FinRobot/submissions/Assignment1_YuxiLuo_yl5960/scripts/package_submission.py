"""Assembles submissions/Assignment1_YuxiLuo_yl5960 from the working folders, then scans it for API keys.

    python scripts/package_submission.py

Re-runnable: the target folder is rebuilt from scratch each time (README.md and research_report.pdf, which are
written by hand, are kept).
"""
import configparser
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "FinRobot" / "output"
DEST = ROOT / "submissions" / "Assignment1_YuxiLuo_yl5960"
KEEP = {"README.md", "research_report.pdf", "research_report.html", "report_figures"}
TICKERS = ["NVDA", "AMD", "INTC", "AAPL", "GOOGL", "PFE", "ABBV", "JNJ", "UNH", "MRNA"]
EXTRA_RUNS = {"NVDA/stab1": "NVDA_stability_run1", "NVDA/stab2": "NVDA_stability_run2",
              "NVDA/stab3": "NVDA_stability_run3", "NVDA/part3": "NVDA_part3_verifier_run"}
AUDIT_FILES = ["2_1_data_audit.ipynb", "2_2_forecast_valuation_audit.ipynb", "2_3_grounding_audit.ipynb",
               "2_3b_stability.ipynb", "part3_evaluation.ipynb", "grounding.py", "part3_experiment.py"]


def copy_run(src: Path, dst: Path):
    shutil.copytree(src / "analysis", dst / "analysis")
    for log in src.glob("*.log"):
        shutil.copy2(log, dst / log.name)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for p in DEST.iterdir():
        if p.name not in KEEP:
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    # 1. reports (self-contained HTML, charts embedded)
    (DEST / "reports").mkdir()
    for t in TICKERS:
        shutil.copy2(OUT / t / "report" / f"Professional_Equity_Report_{t}.html",
                     DEST / "reports" / f"{t}_Equity_Research_Report.html")

    # 2. artifacts: Step 1 analysis folders + logs of every run
    for t in TICKERS:
        copy_run(OUT / t, DEST / "artifacts" / t)
    for src, name in EXTRA_RUNS.items():
        copy_run(OUT / src, DEST / "artifacts" / name)

    # 3. audit: notebooks, tables, scorer, experiment, notebook builders, cached API responses
    audit = DEST / "audit"
    audit.mkdir()
    for f in AUDIT_FILES:
        shutil.copy2(ROOT / "audit" / f, audit / f)
    for csv in (ROOT / "audit").glob("*.csv"):
        shutil.copy2(csv, audit / csv.name)
    shutil.copy2(ROOT / "audit" / "part3_texts.jsonl", audit / "part3_texts.jsonl")
    shutil.copytree(ROOT / "audit" / "notebook_builders", audit / "notebook_builders",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "audit" / "data", audit / "data")
    shutil.copy2(ROOT / "notes" / "run_summary.csv", audit / "run_summary.csv")

    # 4. improvement: the two commits as patches against upstream master (2717499b)
    imp = DEST / "improvement"
    imp.mkdir()
    subprocess.run(["git", "format-patch", "2717499b..feat/verified-sections", "-o", str(imp)],
                   cwd=ROOT / "FinRobot-part3", check=True, capture_output=True)

    # 5. scripts
    shutil.copytree(ROOT / "scripts", DEST / "scripts", ignore=shutil.ignore_patterns("__pycache__"))

    # 6. no API key anywhere
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "FinRobot" / "finrobot_equity" / "core" / "config" / "config.ini", encoding="utf-8")
    secrets = [v.strip() for k, v in cfg["API_KEYS"].items()
               if k.endswith("api_key") and v.strip() and not v.strip().startswith("YOUR_")]
    leaks = [p for p in DEST.rglob("*") if p.is_file()
             and any(s.encode() in p.read_bytes() for s in secrets)]
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / 1e6
    print(f"packaged {sum(1 for p in DEST.rglob('*') if p.is_file())} files, {size:.1f} MB -> {DEST}")
    print("API keys found in:", [str(p.relative_to(DEST)) for p in leaks] or "none")
    if leaks:
        raise SystemExit("key leak - do not submit")


if __name__ == "__main__":
    main()
