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

from finrobot_a1_run import Recorder, utc_now


COMPANIES = {
    "AMD": ("Advanced Micro Devices, Inc.", ["NVDA", "INTC"]),
    "INTC": ("Intel Corporation", ["AMD", "NVDA"]),
    "QCOM": ("QUALCOMM Incorporated", ["AVGO", "NVDA"]),
    "AVGO": ("Broadcom Inc.", ["QCOM", "NVDA"]),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifests(directory, pattern):
    return set(directory.glob(pattern))


def command_for(runner, repo, ticker):
    name, peers = COMPANIES[ticker]
    return [sys.executable, str(runner), "--repo", str(repo),
            "--company-ticker", ticker, "--company-name", name,
            "--peer-tickers", *peers]


def main():
    parser = argparse.ArgumentParser(description="FinRobot semiconductor baseline batch.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tickers", nargs="+", choices=COMPANIES, default=list(COMPANIES))
    parser.add_argument("--include-run", type=Path, action="append", default=[])
    parser.add_argument("--analyst-name", default="Xinyu Zhang")
    parser.add_argument("--zip-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    helper_dir = Path(__file__).resolve().parent
    runner, renderer = helper_dir / "finrobot_a1_run.py", helper_dir / "finrobot_a1_render.py"
    if not all(path.is_file() for path in (runner, renderer)):
        raise SystemExit("STOP: finrobot_a1_run.py and finrobot_a1_render.py must be beside this file.")
    if len(set(args.tickers)) != len(args.tickers):
        raise SystemExit("STOP: duplicate tickers.")
    included = [path.expanduser().resolve() for path in args.include_run]
    try:
        for path in included:
            if read_json(path / "audit/run_manifest.json").get("exit_code") != 0:
                raise ValueError("incomplete analysis")
    except (OSError, ValueError):
        raise SystemExit("STOP: an included run is missing or incomplete.") from None
    print("BASELINE PLAN", flush=True)
    for ticker in args.tickers:
        print(ticker, "Peers:", ", ".join(COMPANIES[ticker][1]), flush=True)
    check = subprocess.run(command_for(runner, repo, args.tickers[0]) + ["--plan-only"], cwd=repo)
    if check.returncode:
        return check.returncode
    if args.plan_only:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    batch_dir = repo / "output/a1_batches" / f"baseline_{stamp}"
    config = configparser.ConfigParser(interpolation=None)
    config.read(repo / "finrobot_equity/core/config/config.ini", encoding="utf-8")
    secrets = [value.strip() for section in config.sections()
               for key, value in config.items(section)
               if any(word in key.lower() for word in ("api_key", "apikey", "password", "secret", "token"))]
    recorder = Recorder(batch_dir / "audit", secrets, {"api.openai.com"})
    source_dir = repo / "finrobot_equity/core/src"
    source_hashes = {}
    for path in sorted(source_dir.rglob("*.py")):
        relative = path.relative_to(source_dir)
        output = batch_dir / "source_snapshot" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        output.write_text(recorder.redact(data.decode("utf-8")), encoding="utf-8")
        source_hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    summary = {"started_at_utc": utc_now(), "sector": "Semiconductors",
               "requested_tickers": args.tickers, "included_runs": [str(path) for path in included],
               "source_sha256": source_hashes, "runs": []}
    run_root = repo / "output/a1_runs"
    selected_runs = list(dict.fromkeys(included))
    exit_code = 0
    for ticker in args.tickers:
        before = manifests(run_root, f"{ticker}_baseline_*/audit/run_manifest.json")
        result = subprocess.run(command_for(runner, repo, ticker), cwd=repo)
        new = manifests(run_root, f"{ticker}_baseline_*/audit/run_manifest.json") - before
        record = {"ticker": ticker, "analysis_exit_code": result.returncode}
        summary["runs"].append(record)
        if len(new) != 1:
            record["error"] = "Expected one new analysis manifest."
            exit_code = result.returncode or 1
        else:
            run_dir = new.pop().parent.parent
            selected_runs.append(run_dir)
            record["run_dir"] = str(run_dir)
            manifest = read_json(run_dir / "audit/run_manifest.json")
            record["llm_attempts"] = manifest.get("captured_llm_http_attempts", 0)
            record["fmp_attempts"] = manifest.get("captured_fmp_http_attempts", 0)
            if result.returncode or manifest.get("exit_code") != 0 or not record["llm_attempts"]:
                record["error"] = "Analysis failed or no LLM request was captured."
                exit_code = result.returncode or 1
            else:
                before_render = manifests(run_dir, "render_*/audit/render_manifest.json")
                render = subprocess.run([sys.executable, str(renderer), "--repo", str(repo),
                                         "--run-dir", str(run_dir), "--analyst-name", args.analyst_name], cwd=repo)
                record["render_exit_code"] = render.returncode
                new_render = manifests(run_dir, "render_*/audit/render_manifest.json") - before_render
                if render.returncode or len(new_render) != 1:
                    record["error"] = "Report rendering failed."
                    exit_code = render.returncode or 1
                else:
                    rendered = read_json(new_render.pop())
                    record["analysis_unchanged"] = rendered.get("analysis_unchanged", False)
                    if rendered.get("exit_code") != 0 or not record["analysis_unchanged"]:
                        record["error"] = "Report manifest failed validation."
                        exit_code = 1
        recorder.save("batch_summary.json", summary)
        if exit_code:
            print("STOP:", ticker, record.get("error", "Batch failed."), flush=True)
            break
    summary.update(finished_at_utc=utc_now(), exit_code=exit_code)
    recorder.save("batch_summary.json", summary)
    zip_dir = args.zip_dir.expanduser().resolve()
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"FinRobot_A1_baselines_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(batch_dir.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path("batch") / path.relative_to(batch_dir)))
        for path in (runner, renderer, Path(__file__).resolve()):
            archive.writestr(str(Path("code") / path.name), recorder.redact(path.read_text(encoding="utf-8")))
        for directory in dict.fromkeys(selected_runs):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path("runs") / directory.name / path.relative_to(directory)))
    print("\nBATCH SUMMARY")
    for record in summary["runs"]:
        print(record["ticker"], "PASS" if not record.get("error") else "FAILED")
    print("Exit code:", exit_code)
    print("ZIP:", zip_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
