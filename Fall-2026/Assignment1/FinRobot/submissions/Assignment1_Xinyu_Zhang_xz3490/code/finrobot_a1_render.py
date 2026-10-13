#!/usr/bin/env python3

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from urllib.parse import urlsplit

from finrobot_a1_run import Recorder, SafeTee, git_output, utc_now


def file_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description="FinRobot report renderer.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analyst-name", default="Xinyu Zhang")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    analysis = run_dir / "analysis"
    target = repo / "finrobot_equity/core/src/create_equity_report.py"
    config_path = repo / "finrobot_equity/core/config/config.ini"
    if not target.is_file() or not config_path.is_file():
        raise SystemExit("STOP: renderer or configuration not found.")
    try:
        baseline = json.loads((run_dir / "audit/run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("STOP: analysis manifest could not be read.") from None
    ticker, company_name = baseline["company_ticker"], baseline["company_name"]
    inputs = {
        "--analysis-csv": "financial_metrics_and_forecasts.csv",
        "--ratios-csv": "ratios_raw_data.csv",
        "--peer-ebitda-csv": "peer_ebitda_comparison.csv",
        "--peer-ev-ebitda-csv": "peer_ev_ebitda_comparison.csv",
        "--tagline-file": "tagline.txt",
        "--company-overview-file": "company_overview.txt",
        "--investment-overview-file": "investment_overview.txt",
        "--valuation-overview-file": "valuation_overview.txt",
        "--risks-file": "risks.txt",
        "--competitor-analysis-file": "competitor_analysis.txt",
        "--major-takeaways-file": "major_takeaways.txt",
        "--news-summary-file": "news_summary.txt",
    }
    missing = [name for name in inputs.values() if not (analysis / name).is_file()]
    if missing:
        raise SystemExit("STOP: missing analysis files: " + ", ".join(missing))
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_path, encoding="utf-8")
    except (OSError, configparser.Error):
        raise SystemExit("STOP: configuration could not be parsed.") from None
    if not config.get("API_KEYS", "fmp_api_key", fallback="").strip():
        raise SystemExit("STOP: FMP key not configured.")
    ignored = subprocess.run(["git", "-C", str(repo), "check-ignore", "--quiet", "--", str(config_path)])
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", str(config_path)], capture_output=True)
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise SystemExit("STOP: config.ini must be untracked and ignored by Git.")
    try:
        help_result = subprocess.run([sys.executable, str(target), "--help"], cwd=repo,
                                     capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise SystemExit("STOP: renderer CLI check failed.") from None
    if help_result.returncode or not all(flag in help_result.stdout for flag in (*inputs, "--output-dir", "--config-file")):
        raise SystemExit("STOP: renderer CLI does not match the required inputs.")
    secrets, hosts = [], {"api.openai.com"}
    for section in config.sections():
        for key, value in config.items(section):
            if any(word in key.lower() for word in ("api_key", "apikey", "password", "secret", "token")):
                secrets.append(value.strip())
            if key.lower() == "openai_base_url":
                hosts.add((urlsplit(value).hostname or "").lower())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    render_dir = run_dir / f"render_{stamp}"
    report_dir = render_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(render_dir / "audit", secrets, hosts)
    argv = [str(target), "--company-ticker", ticker, "--company-name", company_name,
            "--config-file", str(config_path), "--output-dir", str(report_dir),
            "--analyst-names", args.analyst_name, "--analyst-emails", ""]
    for flag, name in inputs.items():
        argv.extend([flag, str(analysis / name)])
    for flag, name in (("--sensitivity-analysis-file", "sensitivity_analysis.json"),
                       ("--retail-sentiment-file", "retail_sentiment.json")):
        if (analysis / name).is_file() and flag in help_result.stdout:
            argv.extend([flag, str(analysis / name)])
    for logo in (repo / "finrobot_equity/core/assets/piclogo.png",
                 repo / "finrobot_equity/core/src/assets/piclogo.png", repo / "assets/piclogo.png"):
        if logo.is_file():
            argv.extend(["--logo-image-path", str(logo)])
            break
    manifest = {
        "stage": "report", "started_at_utc": utc_now(), "source_run": str(run_dir),
        "company_ticker": ticker, "company_name": company_name,
        "git_commit": git_output(repo, "rev-parse", "HEAD"), "pipeline_argv": argv,
        "text_regeneration": False, "analysis_hashes_before": file_hashes(analysis),
        "pipeline_source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    recorder.save("render_manifest.json", manifest)
    (recorder.directory / "pipeline_help.txt").write_text(recorder.redact(help_result.stdout + help_result.stderr), encoding="utf-8")
    old_stdout, old_stderr, old_argv, old_cwd = sys.stdout, sys.stderr, sys.argv, Path.cwd()
    old_path = list(sys.path)
    exit_code = 0
    print("Rendering:", ticker)
    with (render_dir / "render.log").open("w", encoding="utf-8") as log:
        out, err = SafeTee(old_stdout, log, recorder), SafeTee(old_stderr, log, recorder)
        sys.stdout, sys.stderr = out, err
        try:
            recorder.install()
            os.chdir(repo)
            sys.path.insert(0, str(target.parent))
            sys.argv = argv
            runpy.run_path(str(target), run_name="__main__")
        except SystemExit as error:
            exit_code = error.code if isinstance(error.code, int) else (1 if error.code else 0)
            if exit_code:
                print("Renderer stopped:", error)
        except KeyboardInterrupt:
            exit_code = 130
            print("Run interrupted.")
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            recorder.restore()
            out.finish()
            err.finish()
            sys.stdout, sys.stderr, sys.argv = old_stdout, old_stderr, old_argv
            sys.path[:] = old_path
            os.chdir(old_cwd)
    html_files = sorted(report_dir.glob("*.html"))
    if exit_code == 0 and not (report_dir / f"Combined_Equity_Report_{ticker}.html").is_file():
        exit_code = 1
    manifest.update(finished_at_utc=utc_now(), exit_code=exit_code,
                    captured_fmp_http_attempts=recorder.fmp_count, captured_llm_http_attempts=recorder.llm_count,
                    analysis_unchanged=manifest["analysis_hashes_before"] == file_hashes(analysis),
                    report_hashes=file_hashes(report_dir))
    recorder.save("render_manifest.json", manifest)
    print("\nRENDER SUMMARY")
    print("Renderer exit code:", exit_code)
    print("HTML reports:", len(html_files))
    print("Captured FMP attempts:", recorder.fmp_count, "LLM attempts:", recorder.llm_count)
    print("Analysis unchanged:", manifest["analysis_unchanged"])
    print("Report folder:", report_dir)
    for path in html_files:
        print(" ", path.name)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
