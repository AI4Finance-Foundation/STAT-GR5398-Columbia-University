#!/usr/bin/env python3

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    def __init__(self, directory, secrets, llm_hosts):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.secrets = sorted({s for s in secrets if len(s) >= 6}, key=len, reverse=True)
        self.llm_hosts = set(llm_hosts)
        self.lock = threading.Lock()
        self.count = 0
        self.llm_count = 0
        self.fmp_count = 0
        self.restorers = []

    def redact(self, text):
        text = str(text)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}", "[REDACTED]", text)
        return text

    def scrub(self, value):
        if isinstance(value, dict):
            sensitive = {"apikey", "api_key", "authorization", "password", "secret", "access_token", "refresh_token"}
            return {str(k): "[REDACTED]" if str(k).lower() in sensitive else self.scrub(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.scrub(v) for v in value]
        if isinstance(value, str):
            return self.redact(value)
        return value

    def safe_url(self, url):
        parsed = urlsplit(str(url))
        hidden = {"apikey", "api_key", "key", "token", "access_token", "password"}
        query = [(k, "[REDACTED]" if k.lower() in hidden else self.redact(v))
                 for k, v in parse_qsl(parsed.query, keep_blank_values=True)]
        return urlunsplit((parsed.scheme, parsed.netloc.split("@")[-1], parsed.path,
                           urlencode(query), ""))

    def category(self, url):
        parsed = urlsplit(str(url))
        host = (parsed.hostname or "").lower()
        if host == "financialmodelingprep.com" or host.endswith(".financialmodelingprep.com"):
            return "fmp"
        if host in self.llm_hosts and parsed.path.rstrip("/").endswith(("/chat/completions", "/responses")):
            return "llm"
        return None

    def save(self, name, value):
        data = json.dumps(self.scrub(value), ensure_ascii=False, indent=2, default=str)
        (self.directory / name).write_text(self.redact(data) + "\n", encoding="utf-8")

    def begin(self, method, url, body):
        category = self.category(url)
        if category is None:
            return None
        with self.lock:
            self.count += 1
            self.llm_count += category == "llm"
            self.fmp_count += category == "fmp"
            prefix = f"{self.count:04d}_{category}"
        try:
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            body = json.loads(body) if body else None
        except (ValueError, UnicodeError, TypeError):
            body = {"body_unavailable_or_not_json": True}
        self.save(prefix + "_request.json", {
            "recorded_at_utc": utc_now(), "category": category,
            "method": method, "url": self.safe_url(url), "body": body,
        })
        return prefix, time.monotonic()

    def finish(self, token, response=None, error=None, streamed=False):
        if token is None:
            return
        prefix, started = token
        record = {"recorded_at_utc": utc_now(), "elapsed_seconds": round(time.monotonic() - started, 4)}
        if error is not None:
            record.update(error_type=type(error).__name__, error=self.redact(error))
        else:
            record["http_status"] = response.status_code
            if streamed:
                record["body_not_captured"] = "streaming"
            else:
                try:
                    record["body"] = response.json()
                except (ValueError, UnicodeError):
                    record["body"] = self.redact(response.text)
        self.save(prefix + "_response.json", record)

    def install(self):
        import httpx
        import requests
        original_requests = requests.sessions.Session.send
        original_sync = httpx.Client.send
        original_async = httpx.AsyncClient.send
        recorder = self

        def requests_send(client, request, **kwargs):
            token = recorder.begin(request.method, request.url, request.body)
            try:
                response = original_requests(client, request, **kwargs)
            except Exception as error:
                recorder.finish(token, error=error)
                raise
            recorder.finish(token, response=response, streamed=kwargs.get("stream", False))
            return response

        def sync_send(client, request, *args, **kwargs):
            try:
                body = request.content
            except httpx.RequestNotRead:
                body = None
            token = recorder.begin(request.method, str(request.url), body)
            try:
                response = original_sync(client, request, *args, **kwargs)
            except Exception as error:
                recorder.finish(token, error=error)
                raise
            recorder.finish(token, response=response, streamed=kwargs.get("stream", False))
            return response

        async def async_send(client, request, *args, **kwargs):
            try:
                body = request.content
            except httpx.RequestNotRead:
                body = None
            token = recorder.begin(request.method, str(request.url), body)
            try:
                response = await original_async(client, request, *args, **kwargs)
            except Exception as error:
                recorder.finish(token, error=error)
                raise
            recorder.finish(token, response=response, streamed=kwargs.get("stream", False))
            return response

        requests.sessions.Session.send = requests_send
        httpx.Client.send = sync_send
        httpx.AsyncClient.send = async_send
        self.restorers = [
            (requests.sessions.Session, original_requests),
            (httpx.Client, original_sync), (httpx.AsyncClient, original_async),
        ]

    def restore(self):
        for cls, method in self.restorers:
            cls.send = method
        self.restorers = []


class SafeTee:
    def __init__(self, console, log, recorder):
        self.console, self.log, self.recorder = console, log, recorder
        self.pending = ""

    def write(self, text):
        self.pending += str(text)
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            cleaned = self.recorder.redact(line) + "\n"
            self.console.write(cleaned)
            self.log.write(cleaned)
            self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def finish(self):
        if self.pending:
            cleaned = self.recorder.redact(self.pending)
            self.console.write(cleaned)
            self.log.write(cleaned)
            self.pending = ""
        self.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.console, name)


def git_output(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description="FinRobot analysis runner.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--company-ticker", default="NVDA")
    parser.add_argument("--company-name", default="NVIDIA Corporation")
    parser.add_argument("--peer-tickers", nargs="+", default=["AMD", "INTC"])
    parser.add_argument("--years-limit", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    ticker = args.company_ticker.upper()
    if not re.fullmatch(r"[A-Z0-9.-]+", ticker):
        raise SystemExit("STOP: invalid ticker.")
    target = repo / "finrobot_equity/core/src/generate_financial_analysis.py"
    config_path = repo / "finrobot_equity/core/config/config.ini"
    if not target.is_file() or not config_path.is_file():
        raise SystemExit("STOP: analysis script or config.ini was not found in the selected repository.")
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read(config_path, encoding="utf-8")
    except (OSError, configparser.Error):
        raise SystemExit("STOP: config.ini could not be parsed.") from None
    if not config.has_section("API_KEYS") or not all(config.get("API_KEYS", key, fallback="").strip()
                                                     for key in ("fmp_api_key", "openai_api_key")):
        raise SystemExit("STOP: one or both required API keys are missing.")
    ignored = subprocess.run(["git", "-C", str(repo), "check-ignore", "--quiet", "--", str(config_path)])
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", str(config_path)],
                             capture_output=True)
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise SystemExit("STOP: config.ini must be untracked and ignored by Git.")
    source = target.read_text(encoding="utf-8")
    flags = ("--output-dir", "--text-output-dir", "--generate-text-sections", "--enable-sensitivity-analysis")
    try:
        help_result = subprocess.run([sys.executable, str(target), "--help"], cwd=repo,
                                     capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise SystemExit("STOP: the pipeline CLI check could not finish.") from None
    if help_result.returncode != 0 or not all(flag in help_result.stdout for flag in flags):
        raise SystemExit("STOP: this pipeline CLI does not contain the expected flags.")
    import httpx, requests
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run_dir = repo / "output/a1_runs" / f"{ticker}_baseline_{stamp}"
    analysis = run_dir / "analysis"
    argv = [str(target), "--company-ticker", ticker, "--company-name", args.company_name,
            "--config-file", str(config_path), "--years-limit", str(args.years_limit),
            "--output-dir", str(analysis), "--text-output-dir", str(analysis),
            "--peer-tickers", *args.peer_tickers, "--period", "annual",
            "--generate-text-sections", "--enable-sensitivity-analysis"]
    if args.plan_only:
        print("PLAN ONLY")
        print("Repository:", repo)
        print("Company:", ticker, "Peers:", ", ".join(args.peer_tickers))
        print("Pipeline arguments:", json.dumps(argv[1:], ensure_ascii=False))
        return 0
    secrets, hosts, settings = [], {"api.openai.com"}, {}
    for section in config.sections():
        for key, value in config.items(section):
            if any(word in key.lower() for word in ("api_key", "apikey", "password", "secret", "token")):
                secrets.append(value.strip())
            if key.lower() in ("openai_model", "openai_base_url"):
                settings[key] = value
                if key.lower() == "openai_base_url":
                    hosts.add((urlsplit(value).hostname or "").lower())
    if os.environ.get("OPENAI_API_KEY"):
        secrets.append(os.environ["OPENAI_API_KEY"])
    analysis.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(run_dir / "audit", secrets, hosts)
    manifest = {
        "started_at_utc": utc_now(), "company_ticker": ticker, "company_name": args.company_name,
        "peer_tickers": args.peer_tickers, "git_commit": git_output(repo, "rev-parse", "HEAD"),
        "python_version": sys.version, "pipeline_argv": argv, "settings": settings,
        "pipeline_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "stage": "analysis_only", "api_cost_usd": None,
        "capture_transports": ["requests", "httpx"],
        "capture_headers": False,
        "capture_streaming_responses": False,
    }
    recorder.save("run_manifest.json", manifest)
    (recorder.directory / "pipeline_help.txt").write_text(recorder.redact(help_result.stdout + help_result.stderr), encoding="utf-8")
    recorder.save("source_diff.json", {"git_diff": git_output(repo, "diff", "HEAD", "--", "finrobot_equity/core/src")})
    old_stdout, old_stderr, old_argv, old_cwd = sys.stdout, sys.stderr, sys.argv, Path.cwd()
    old_path = list(sys.path)
    exit_code = 0
    print("Run folder:", run_dir)
    print("Starting analysis:", ticker)
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
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
                print("Pipeline stopped:", error)
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
    files = [{"path": str(p.relative_to(run_dir)), "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(analysis.rglob("*")) if p.is_file()]
    manifest.update(finished_at_utc=utc_now(), exit_code=exit_code,
                    captured_llm_http_attempts=recorder.llm_count,
                    captured_fmp_http_attempts=recorder.fmp_count, analysis_files=files)
    recorder.save("run_manifest.json", manifest)
    print("\nRUN SUMMARY")
    print("Pipeline exit code:", exit_code)
    print("Captured LLM attempts:", recorder.llm_count, "FMP attempts:", recorder.fmp_count)
    if recorder.llm_count == 0:
        print("WARNING: no LLM requests captured.")
    print("Analysis files:", len(files))
    for item in files:
        print(" ", item["path"], f"({item['bytes']} bytes)")
    print("Saved run folder:", run_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
