#!/usr/bin/env python3

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import finrobot_a1_batch as batch
from finrobot_a1_run import Recorder, utc_now


COMPANIES = {
    "AAPL": ("Apple Inc.", ["CSCO", "NVDA"]),
    "CSCO": ("Cisco Systems, Inc.", ["AAPL", "NVDA"]),
}
ENDPOINTS = ["profile", "income-statement", "balance-sheet-statement",
             "cash-flow-statement", "ratios", "key-metrics", "enterprise-values"]


class CoverageRecorder(Recorder):
    def save(self, name, value):
        if name == "batch_summary.json":
            value = dict(value, sector="Technology",
                         coverage_tickers=["NVDA", "AMD", "INTC", "AAPL", "CSCO"])
        super().save(name, value)


def find_baseline(repo, ticker):
    root = repo / "output/a1_runs"
    for path in sorted(root.glob(f"{ticker}_baseline_*/audit/run_manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            run_dir = path.parent.parent
            if manifest.get("exit_code") != 0 or not manifest.get("captured_llm_http_attempts"):
                continue
            files = manifest.get("analysis_files", [])
            if not files or any(hashlib.sha256((run_dir / item["path"]).read_bytes()).hexdigest()
                                != item["sha256"] for item in files):
                continue
            for render in sorted(run_dir.glob("render_*/audit/render_manifest.json")):
                metadata = json.loads(render.read_text(encoding="utf-8"))
                report = render.parent.parent / "report" / f"Combined_Equity_Report_{ticker}.html"
                if metadata.get("exit_code") == 0 and metadata.get("analysis_unchanged") and report.is_file():
                    return run_dir
        except (OSError, ValueError, KeyError):
            continue
    raise SystemExit(f"STOP: a complete, unchanged {ticker} baseline was not found.")


def main():
    parser = argparse.ArgumentParser(description="Complete the five-company technology baseline.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--zip-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    zip_dir = args.zip_dir.expanduser().resolve()
    helpers = Path(__file__).resolve().parent
    baseline_runs = [find_baseline(repo, ticker) for ticker in ["NVDA", "AMD", "INTC"]]
    check = subprocess.run([sys.executable, str(helpers / "finrobot_a1_run.py"),
                            "--repo", str(repo), "--company-ticker", "AAPL",
                            "--company-name", COMPANIES["AAPL"][0],
                            "--peer-tickers", *COMPANIES["AAPL"][1], "--plan-only"], cwd=repo)
    if check.returncode:
        return check.returncode
    print("Coverage: NVDA, AMD, INTC, AAPL, CSCO; Sector: Technology", flush=True)
    if args.plan_only:
        return 0
    config = configparser.ConfigParser(interpolation=None)
    config.read(repo / "finrobot_equity/core/config/config.ini", encoding="utf-8")
    key = config.get("API_KEYS", "fmp_api_key").strip()
    secrets = [value.strip() for section in config.sections()
               for name, value in config.items(section)
               if any(word in name.lower() for word in ("api_key", "apikey", "password", "secret", "token"))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    check_dir = repo / "output/a1_entitlement_checks" / f"technology_{stamp}"
    recorder = Recorder(check_dir / "audit", secrets, {"api.openai.com"})
    results = {"started_at_utc": utc_now(), "sector": "Technology", "checks": []}
    exit_code = 0
    import requests
    try:
        recorder.install()
        for ticker in COMPANIES:
            for endpoint in ENDPOINTS:
                parameters = {"symbol": ticker, "apikey": key}
                if endpoint != "profile":
                    parameters.update(period="annual", limit=5)
                response = requests.get("https://financialmodelingprep.com/stable/" + endpoint,
                                        params=parameters, timeout=20)
                try:
                    data = response.json()
                except ValueError:
                    data = None
                valid = (response.status_code == 200 and isinstance(data, list) and bool(data)
                         and all(isinstance(row, dict) for row in data))
                detail = ""
                if valid and endpoint == "profile":
                    detail = str(data[0].get("sector", ""))
                    valid = detail.strip().lower() == "technology"
                elif valid and endpoint != "profile":
                    valid = len(data) >= 5 and all(isinstance(row, dict) for row in data)
                record = {"ticker": ticker, "endpoint": endpoint,
                          "http_status": response.status_code, "passed": valid,
                          "rows": len(data) if isinstance(data, list) else 0,
                          "sector": detail or None}
                if not valid:
                    record["error"] = recorder.redact(response.text[:600])
                    exit_code = 1
                results["checks"].append(record)
                print(ticker, endpoint, "PASS" if valid else "FAILED",
                      f"HTTP {response.status_code}", detail, flush=True)
                if exit_code:
                    break
            if exit_code:
                break
    except requests.exceptions.RequestException as error:
        results["error"] = recorder.redact(str(error))
        exit_code = 1
        print("STOP:", results["error"], flush=True)
    finally:
        recorder.restore()
    results.update(finished_at_utc=utc_now(), exit_code=exit_code,
                   captured_fmp_http_attempts=recorder.fmp_count,
                   captured_llm_http_attempts=recorder.llm_count)
    recorder.save("availability_summary.json", results)
    zip_dir.mkdir(parents=True, exist_ok=True)
    if exit_code:
        zip_path = zip_dir / f"FinRobot_A1_availability_{stamp}.zip"
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(check_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path("availability_checks") / path.relative_to(check_dir)))
        print("STOP: company data access did not pass; no new model calls were made.")
        print("ZIP:", zip_path)
        return exit_code
    before = set(zip_dir.glob("FinRobot_A1_baselines_*.zip"))
    previous_argv, previous_recorder = sys.argv, batch.Recorder
    batch.COMPANIES.update(COMPANIES)
    batch.Recorder = CoverageRecorder
    sys.argv = [str(helpers / "finrobot_a1_batch.py"), "--repo", str(repo),
                "--tickers", *COMPANIES, "--zip-dir", str(zip_dir)]
    for run in baseline_runs:
        sys.argv.extend(["--include-run", str(run)])
    try:
        exit_code = batch.main()
    finally:
        sys.argv, batch.Recorder = previous_argv, previous_recorder
    new_archives = set(zip_dir.glob("FinRobot_A1_baselines_*.zip")) - before
    if len(new_archives) != 1:
        raise SystemExit("STOP: the batch archive could not be identified.")
    zip_path = new_archives.pop()
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(check_dir.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path("availability_checks") / path.relative_to(check_dir)))
        archive.writestr("code/finrobot_a1_complete.py", recorder.redact(Path(__file__).read_text(encoding="utf-8")))
    print("Final ZIP:", zip_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
