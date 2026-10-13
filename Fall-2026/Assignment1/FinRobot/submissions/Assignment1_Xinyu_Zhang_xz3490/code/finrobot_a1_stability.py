#!/usr/bin/env python3

import argparse
import configparser
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlsplit

import finrobot_a1_run as runner
import finrobot_a1_render as renderer
from finrobot_a1_run import Recorder, utc_now


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def url_key(method, url):
    parsed = urlsplit(str(url))
    query = tuple(sorted((key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                         if key.lower() not in {"apikey", "api_key", "key", "token"}))
    return method.upper(), (parsed.hostname or "").lower(), parsed.path, query


def fmp_cache(audit):
    cache = {}
    for request_path in sorted(audit.glob("*_fmp_request.json")):
        request = read_json(request_path)
        response_path = request_path.with_name(request_path.name.replace("_request.json", "_response.json"))
        response = read_json(response_path)
        if "http_status" not in response or "body" not in response:
            raise SystemExit("STOP: baseline FMP responses are incomplete.")
        key = url_key(request["method"], request["url"])
        cache.setdefault(key, []).append(response)
    if not cache:
        raise SystemExit("STOP: no baseline FMP responses were found.")
    return cache


def llm_inputs(audit):
    return [read_json(path)["body"] for path in sorted(audit.glob("*_llm_request.json"))]


def llm_results(audit):
    return [read_json(path) for path in sorted(audit.glob("*_llm_response.json"))]


def model_ids(results):
    return {result["body"].get("model") if isinstance(result.get("body"), dict) else None for result in results}


def csv_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.csv"))}


def recorder_class(cache, source, instances, expected_inputs):
    class ReplayRecorder(Recorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replay_hits = 0
            self.replay_misses = []
            self.positions = {}
            self.input_position = 0
            self.window_overrides = []
            instances.append(self)

        def begin(self, method, url, body):
            if self.category(url) == "llm":
                if self.replay_misses:
                    raise ValueError("A frozen FMP response is missing.")
                try:
                    incoming = json.loads(body)
                except (TypeError, ValueError):
                    raise ValueError("Model request is not valid JSON.") from None
                if self.input_position >= len(expected_inputs) or incoming != expected_inputs[self.input_position]:
                    raise ValueError("Model input differs from the frozen baseline.")
                self.input_position += 1
            return super().begin(method, url, body)

        def install(self):
            import requests
            original = requests.sessions.Session.send
            current = self

            def replay_send(client, request, **kwargs):
                if current.category(request.url) != "fmp":
                    return original(client, request, **kwargs)
                key = url_key(request.method, request.url)
                if key not in cache and key[2] in {"/stable/news/stock", "/stable/historical-price-eod/full"}:
                    fixed = tuple(pair for pair in key[3] if pair[0] not in {"from", "to"})
                    matches = [candidate for candidate in cache if candidate[:3] == key[:3]
                               and tuple(pair for pair in candidate[3] if pair[0] not in {"from", "to"}) == fixed]
                    if len(matches) == 1:
                        self.window_overrides.append({"requested_url": current.safe_url(request.url),
                                                      "frozen_query": matches[0][3]})
                        key = matches[0]
                if key not in cache:
                    safe = current.safe_url(request.url)
                    current.replay_misses.append(safe)
                    raise requests.exceptions.RequestException("Frozen FMP response missing: " + safe)
                position = current.positions.get(key, 0)
                saved = cache[key][min(position, len(cache[key]) - 1)]
                current.positions[key] = position + 1
                current.replay_hits += 1
                response = requests.Response()
                response.status_code = saved["http_status"]
                response.url = request.url
                response.request = request
                response.encoding = "utf-8"
                body = saved["body"]
                if isinstance(body, (dict, list)):
                    response._content = json.dumps(body, ensure_ascii=False).encode("utf-8")
                    response.headers["Content-Type"] = "application/json"
                else:
                    response._content = str(body).encode("utf-8")
                    response.headers["Content-Type"] = "text/plain"
                return response

            requests.sessions.Session.send = replay_send
            try:
                super().install()
            except Exception:
                requests.sessions.Session.send = original
                raise
            self.restorers[0] = (requests.sessions.Session, original)

        def save(self, name, value):
            if name.endswith("_response.json") and "_fmp_" in name:
                value = dict(value, response_origin="baseline_replay", replay_source=str(source))
            if name in {"run_manifest.json", "render_manifest.json"}:
                value = dict(value, study="same_input_stability", response_origin="baseline_replay",
                             replay_source=str(source), fmp_network_calls=0,
                             fmp_replay_hits=self.replay_hits, fmp_replay_misses=self.replay_misses,
                             frozen_date_windows=self.window_overrides)
            super().save(name, value)

    return ReplayRecorder


def main():
    parser = argparse.ArgumentParser(description="NVDA stability test with frozen FMP inputs.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--zip-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    baseline = args.baseline_run.expanduser().resolve()
    config_path = repo / "finrobot_equity/core/config/config.ini"
    try:
        manifest = read_json(baseline / "audit/run_manifest.json")
        if manifest.get("exit_code") != 0 or manifest.get("company_ticker") != "NVDA":
            raise ValueError("Baseline is not a successful NVDA run.")
        for item in manifest["analysis_files"]:
            if hashlib.sha256((baseline / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Baseline analysis changed.")
        renders = sorted(baseline.glob("render_*/audit/render_manifest.json"))
        render_path = next(path for path in renders if read_json(path).get("exit_code") == 0
                           and read_json(path).get("analysis_unchanged"))
        render_manifest = read_json(render_path)
        for relative, digest in render_manifest["report_hashes"].items():
            if hashlib.sha256((render_path.parent.parent / "report" / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Baseline report changed.")
        for name, recorded in [("generate_financial_analysis.py", manifest),
                               ("create_equity_report.py", render_manifest)]:
            source = repo / "finrobot_equity/core/src" / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != recorded["pipeline_source_sha256"]:
                raise ValueError("Pipeline source changed: " + name)
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path, encoding="utf-8")
        for setting, expected in manifest.get("settings", {}).items():
            if config.get("API_KEYS", setting, fallback="").strip() != expected.strip():
                raise ValueError("Model configuration changed: " + setting)
        original_inputs = llm_inputs(baseline / "audit")
        if not original_inputs:
            raise ValueError("Baseline model requests are missing.")
        baseline_models = model_ids(llm_results(baseline / "audit"))
        analysis_cache = fmp_cache(baseline / "audit")
        render_cache = fmp_cache(render_path.parent)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        raise SystemExit("STOP: " + str(error)) from None
    analysis_argv = [str(Path(runner.__file__).resolve()), "--repo", str(repo),
                     "--company-ticker", "NVDA", "--company-name", manifest["company_name"],
                     "--peer-tickers", *manifest["peer_tickers"], "--years-limit", "5"]
    old_argv = sys.argv
    try:
        sys.argv = analysis_argv + ["--plan-only"]
        check = runner.main()
    finally:
        sys.argv = old_argv
    if check or args.plan_only:
        return check
    secrets = [value.strip() for section in config.sections()
               for key, value in config.items(section)
               if any(word in key.lower() for word in ("api_key", "apikey", "password", "secret", "token"))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    study_dir = repo / "output/a1_stability" / f"NVDA_{stamp}"
    recorder = Recorder(study_dir / "audit", secrets, {"api.openai.com"})
    summary = {"started_at_utc": utc_now(), "ticker": "NVDA", "baseline_run": str(baseline),
               "source_repo_origin": recorder.safe_url(runner.git_output(repo, "config", "--get", "remote.origin.url")),
               "baseline_csv_sha256": csv_hashes(baseline / "analysis"),
               "baseline_llm_input_sha256": hashlib.sha256(json.dumps(original_inputs, sort_keys=True).encode()).hexdigest(),
               "repeats": [], "fmp_network_calls": 0}
    recorder.save("stability_manifest.json", summary)
    selected_runs = [baseline]
    run_root = repo / "output/a1_runs"
    old_runner, old_renderer = runner.Recorder, renderer.Recorder
    exit_code = 0
    try:
        for repeat in [2, 3]:
            record = {"repeat": repeat}
            summary["repeats"].append(record)
            instances = []
            runner.Recorder = recorder_class(analysis_cache, baseline / "audit", instances, original_inputs)
            before = set(run_root.glob("NVDA_baseline_*/audit/run_manifest.json"))
            sys.argv = analysis_argv
            code = runner.main()
            new = set(run_root.glob("NVDA_baseline_*/audit/run_manifest.json")) - before
            record["analysis_exit_code"] = code
            if len(new) != 1:
                raise ValueError("Expected one new NVDA analysis run.")
            run_dir = new.pop().parent.parent
            selected_runs.append(run_dir)
            record["run_dir"] = str(run_dir)
            record["csv_inputs_identical"] = csv_hashes(run_dir / "analysis") == summary["baseline_csv_sha256"]
            inputs = llm_inputs(run_dir / "audit")
            record["model_requests_identical"] = inputs == original_inputs
            record["live_llm_http_attempts"] = read_json(run_dir / "audit/run_manifest.json").get("captured_llm_http_attempts", 0)
            responses = llm_results(run_dir / "audit")
            record["successful_llm_responses"] = sum(result.get("http_status") == 200
                                                    and isinstance(result.get("body"), dict)
                                                    and bool(result["body"].get("choices"))
                                                    for result in responses)
            record["actual_model_identical"] = model_ids(responses) == baseline_models
            record["analysis_fmp_replay_hits"] = sum(instance.replay_hits for instance in instances)
            record["analysis_fmp_replay_misses"] = [url for instance in instances for url in instance.replay_misses]
            record["analysis_frozen_date_windows"] = [item for instance in instances for item in instance.window_overrides]
            if (code or not record["csv_inputs_identical"] or not record["model_requests_identical"]
                    or record["analysis_fmp_replay_misses"] or not record["actual_model_identical"]
                    or record["successful_llm_responses"] != len(original_inputs)):
                raise ValueError("The analysis repeat failed or its inputs differed.")
            instances = []
            renderer.Recorder = recorder_class(render_cache, render_path.parent, instances, [])
            sys.argv = [str(Path(renderer.__file__).resolve()), "--repo", str(repo), "--run-dir", str(run_dir)]
            code = renderer.main()
            record["render_exit_code"] = code
            record["render_fmp_replay_hits"] = sum(instance.replay_hits for instance in instances)
            record["render_fmp_replay_misses"] = [url for instance in instances for url in instance.replay_misses]
            record["render_frozen_date_windows"] = [item for instance in instances for item in instance.window_overrides]
            if code or record["render_fmp_replay_misses"]:
                raise ValueError("The report repeat failed or requested uncached market data.")
            recorder.save("stability_manifest.json", summary)
    except (OSError, ValueError, KeyError) as error:
        exit_code = 1
        summary["error"] = recorder.redact(str(error))
        print("STOP:", summary["error"])
    finally:
        runner.Recorder, renderer.Recorder, sys.argv = old_runner, old_renderer, old_argv
    summary.update(finished_at_utc=utc_now(), exit_code=exit_code,
                   additional_live_llm_http_attempts=sum(item.get("live_llm_http_attempts", 0) for item in summary["repeats"]))
    recorder.save("stability_manifest.json", summary)
    zip_dir = args.zip_dir.expanduser().resolve()
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"FinRobot_A1_stability_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(study_dir.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path("study") / path.relative_to(study_dir)))
        for directory in selected_runs:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path("runs") / directory.name / path.relative_to(directory)))
        for path in [Path(__file__), Path(runner.__file__), Path(renderer.__file__)]:
            archive.writestr(str(Path("code") / path.name), recorder.redact(path.read_text(encoding="utf-8")))
    print("\nSTABILITY SUMMARY")
    for item in summary["repeats"]:
        print("Repeat", item["repeat"], "CSV inputs identical:", item.get("csv_inputs_identical"),
              "Model requests identical:", item.get("model_requests_identical"))
    print("New live LLM attempts:", summary["additional_live_llm_http_attempts"])
    print("FMP network calls:", summary["fmp_network_calls"])
    print("Exit code:", exit_code)
    print("ZIP:", zip_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
