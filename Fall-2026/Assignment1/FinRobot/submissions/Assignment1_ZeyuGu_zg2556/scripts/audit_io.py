"""Evidence and budget primitives. No network request is made on import.

Credentials are read at runtime; they never belong in a submission artifact.
All timestamps are UTC, independent of the host's local timezone.
"""
from __future__ import annotations

import configparser
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def credentials(path=None):
    result = {}
    if path:
        cfg = configparser.ConfigParser(interpolation=None)
        with Path(path).open(encoding="utf-8-sig") as stream:
            cfg.read_file(stream)
        result.update(dict(cfg["API_KEYS"]))
    for key in ("fmp_api_key", "openai_api_key", "openai_base_url", "openai_model"):
        result[key] = os.environ.get(key.upper(), result.get(key, ""))
    for key in ("fmp_api_key", "openai_api_key"):
        if result.get(key, "").upper().startswith(("YOUR_", "TEST_", "REPLACE")):
            result[key] = ""
    result["openai_base_url"] = result.get("openai_base_url") or "https://api.openai.com/v1"
    return result


def redact(text, secrets=()):
    import re
    from urllib.parse import quote
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
            text = text.replace(quote(secret, safe=""), "[REDACTED]")
    names = r'(?:api[-_]?key|access_token|token|authorization|password|secret)'
    text = re.sub(r'(?i)(["\']?' + names + r'["\']?\s*[:=]\s*)(["\'])(.*?)\2',
                  lambda m: m.group(1) + m.group(2) + '[REDACTED]' + m.group(2), text)
    text = re.sub(r'(?i)(\b' + names + r'\s*=\s*)[^&\s,"\'}]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)\bBearer\s+[^\s,"\'}]+', 'Bearer [REDACTED]', text)
    return text


def public_url(url):
    parts = urlsplit(url)
    if parts.username or parts.password:
        raise ValueError("Credentials in base URL are not allowed")
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower().replace("-", "_") not in {"apikey", "api_key", "token", "key", "access_token", "authorization", "password", "secret"}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sorted(pairs)), ""))


class SnapshotHTTP:
    """Record immutable GET responses or replay them without a network fallback.

    No request headers are persisted. Raw response bytes are retained unless a
    provider echoes a secret, in which case only that secret is redacted and the
    redaction is explicitly recorded. Non-GET and other hosts are rejected.
    """
    def __init__(self, root, mode="replay", secrets=()):
        if mode not in {"record", "replay"}:
            raise ValueError(mode)
        self.root, self.mode, self.secrets = Path(root), mode, secrets
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def installed(self):
        import requests
        original = requests.sessions.Session.send
        store = self

        def send(session, request, **kwargs):
            if request.method != "GET":
                raise RuntimeError("Snapshot transport only permits GET")
            safe_url = public_url(request.url)
            if urlsplit(safe_url).hostname != "financialmodelingprep.com":
                raise RuntimeError("Unrecorded network host is forbidden in the data stage")
            key = sha256(safe_url)
            directory = store.root / key
            meta_path = directory / "response.json"
            body_path = directory / "body.bin"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                body = body_path.read_bytes()
                if sha256(body) != meta["body_sha256"]:
                    raise RuntimeError("Snapshot integrity failure: " + key)
            else:
                if store.mode == "replay":
                    raise RuntimeError("Frozen snapshot missing request: " + safe_url)
                started = time.perf_counter()
                try:
                    kwargs["timeout"] = (10, 45)
                    response = original(session, request, **kwargs)
                except Exception as exc:
                    append_jsonl(store.root / "errors.jsonl", {
                        "timestamp_utc": utc_now(), "source_url": safe_url,
                        "error_type": type(exc).__name__, "message": redact(exc, store.secrets)})
                    raise RuntimeError("Data request failed; see sanitized errors.jsonl") from None
                original_body = response.content
                body = original_body
                for secret in store.secrets:
                    if secret:
                        body = body.replace(secret.encode(), b"[REDACTED]")
                directory.mkdir(parents=True, exist_ok=True)
                body_path.write_bytes(body)
                meta = {"source_url": safe_url, "fetched_at_utc": utc_now(),
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type"),
                        "body_sha256": sha256(body), "secret_echo_redacted": body != original_body,
                        "runtime_seconds": time.perf_counter()-started}
                write_json(meta_path, meta)
                append_jsonl(store.root / "requests.jsonl", dict(meta, snapshot_key=key))
            replay = requests.Response()
            replay.status_code = meta["status_code"]
            replay._content = body
            replay.url = safe_url
            replay.request = requests.Request("GET", safe_url).prepare()
            replay.encoding = "utf-8"
            if meta.get("content_type"):
                replay.headers["content-type"] = meta["content_type"]
            return replay

        requests.sessions.Session.send = send
        try:
            yield self
        finally:
            requests.sessions.Session.send = original


class Budget:
    """Serial paid calls only; pessimistic reservations survive unknown outcomes.

    Policy is user-supplied JSON, never inferred from a subscription. A failed
    request retains its reservation until billing is reconciled by the user.
    Prices are explicit USD per million tokens for the selected provider/model.
    """
    def __init__(self, policy_path, ledger_path):
        self.policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        self.path = Path(ledger_path)
        for field in ("approved_cap_usd", "input_usd_per_million", "output_usd_per_million", "model", "price_source"):
            if self.policy.get(field) is None:
                raise ValueError("Missing approved budget/pricing field: " + field)
        for field in ("approved_cap_usd", "input_usd_per_million", "output_usd_per_million"):
            if isinstance(self.policy[field], bool) or not isinstance(self.policy[field], (float, int)) or not math.isfinite(self.policy[field]):
                raise ValueError("Budget and prices must be finite numbers")
        if self.policy["approved_cap_usd"] <= 0:
            raise ValueError("No positive spending authorization")
        if any(self.policy[field] < 0 for field in ("input_usd_per_million", "output_usd_per_million")):
            raise ValueError("Token prices cannot be negative")
        for field in ("model", "price_source"):
            if not isinstance(self.policy[field], str) or not self.policy[field].strip():
                raise ValueError("Budget model and price source must be explicit")
        import atexit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        try:
            self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("Cost ledger locked. Confirm the owning process has ended before removing its lock.") from None
        os.write(self.lock_fd, json.dumps({"pid": os.getpid(), "created_at_utc": utc_now()}).encode())
        atexit.register(self.close)

    def close(self):
        if getattr(self, "lock_fd", None) is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
            self.lock_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def rows(self):
        return [json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []

    def reserve(self, call_id, input_text, max_output_tokens):
        if self.lock_fd is None:
            raise RuntimeError("Budget ledger is closed")
        if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
            raise ValueError("Output token limit must be a positive integer")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Call ID is required")
        rows = self.rows()
        if any(r["event"] == "bound_exceeded" for r in rows):
            raise RuntimeError("Provider usage exceeded a reservation; reconcile the ledger before further calls")
        reservations = {r["call_id"]: r["reserved_usd"] for r in rows if r["event"] == "reserve"}
        settled = {r["call_id"]: r["cost_usd"] for r in rows if r["event"] == "settle"}
        if call_id in reservations:
            raise ValueError("Call ID already used; do not overwrite independent generations")
        # UTF-8 bytes plus a generous schema/message overhead bounds usual tokenization.
        upper_input = 2 * len(input_text.encode("utf-8")) + 8192
        price = self.policy
        reserved = (upper_input * price["input_usd_per_million"] + max_output_tokens * price["output_usd_per_million"]) / 1e6
        committed = sum(settled.get(k, v) for k, v in reservations.items())
        if committed + reserved > price["approved_cap_usd"]:
            raise RuntimeError(f"Budget guard: {committed + reserved:.4f} exceeds approved cap")
        reservation = {"event": "reserve", "call_id": call_id,
                     "timestamp_utc": utc_now(), "reserved_usd": reserved,
                     "input_token_upper_bound": upper_input, "max_output_tokens": max_output_tokens}
        append_jsonl(self.path, reservation)
        return reservation

    def settle(self, call_id, input_tokens, output_tokens, runtime_seconds):
        if self.lock_fd is None:
            raise RuntimeError("Budget ledger is closed")
        for value in (input_tokens, output_tokens):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Usage must contain nonnegative integer token counts")
        if not isinstance(runtime_seconds, (float, int)) or not math.isfinite(runtime_seconds) or runtime_seconds < 0:
            raise ValueError("Runtime must be finite and nonnegative")
        rows = self.rows()
        matches = [r for r in rows if r["event"] == "reserve" and r["call_id"] == call_id]
        if len(matches) != 1 or any(r["event"] == "settle" and r["call_id"] == call_id for r in rows):
            raise ValueError("Settlement needs exactly one unsettled reservation")
        reservation = matches[0]
        cost = (input_tokens*self.policy["input_usd_per_million"] + output_tokens*self.policy["output_usd_per_million"])/1e6
        append_jsonl(self.path, {"event": "settle", "call_id": call_id, "timestamp_utc": utc_now(),
                     "input_tokens": input_tokens, "output_tokens": output_tokens,
                     "cost_usd": cost, "runtime_seconds": runtime_seconds,
                     "cost_basis": "configured list prices; cached input conservatively charged full price"})
        if input_tokens > reservation["input_token_upper_bound"] or output_tokens > reservation["max_output_tokens"]:
            append_jsonl(self.path, {"event": "bound_exceeded", "call_id": call_id,
                         "timestamp_utc": utc_now(), "cost_usd": cost})
            raise RuntimeError("Provider reported usage beyond the authorized reservation bounds")
        return cost


async def call_agent(agent, prompt, *, config, budget, directory, call_id, temperature=0.7, max_tokens=4000):
    """Run the real upstream typed agent, record full wire request and response."""
    import httpx
    from openai import AsyncOpenAI
    from agents import Runner, RunConfig, ModelSettings, OpenAIChatCompletionsModel
    from openai.types.shared import Reasoning
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model = config["openai_model"]
    if not model or model != budget.policy["model"]:
        raise ValueError("Model must equal the approved budget model")
    if not config.get("openai_api_key"):
        raise ValueError("No model credential configured")
    instructions = agent.instructions
    if not isinstance(instructions, str):
        raise ValueError("Audit requires explicit, stable instructions")
    if agent.tools or agent.handoffs:
        raise ValueError("Audited calls permit one text completion without tools or handoffs")
    reasoning_effort=config.get('reasoning_effort')
    if reasoning_effort:
        if reasoning_effort not in {'low','medium','high'}:
            raise ValueError('Unsupported explicit reasoning setting')
        temperature=None
    request_record = {"call_id": call_id, "agent_name": agent.name, "model": model,
        "provider_base_url": public_url(config["openai_base_url"]), "instructions": instructions,
        "evidence_prompt": prompt, "evidence_sha256": sha256(prompt),
        "instructions_sha256": sha256(instructions), "temperature": temperature,
        "max_output_tokens": max_tokens, "tracing_disabled": True,
        "sdk_retries": 0, "timestamp_utc": utc_now(),
        "output_schema": agent.output_type.model_json_schema()}
    if reasoning_effort:
        request_record.update(reasoning_effort=reasoning_effort,
            output_limit_parameter='max_completion_tokens',temperature_omitted=True)
    secrets = [config.get(key, '') for key in ('openai_api_key', 'fmp_api_key')]
    if any(secret and secret in json.dumps(request_record, ensure_ascii=False) for secret in secrets):
        raise ValueError("A credential appeared in evidence or instructions; refusing to persist or submit it")
    write_json(directory / "request.json", request_record)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    (directory / "instructions.txt").write_text(instructions, encoding="utf-8")
    wire_count = 0

    async def request_hook(request):
        nonlocal wire_count
        if wire_count:
            raise RuntimeError("Unexpected extra SDK request; budget authorizes one completion per call")
        wire_count += 1
        body = await request.aread()
        if len(body) > reservation["input_token_upper_bound"]:
            raise RuntimeError("Actual wire input exceeds the pessimistic reservation bound")
        # Never save headers; neither SDK request bodies nor responses need the API key.
        (directory / "wire_request.json").write_text(redact(body.decode(), secrets), encoding="utf-8")

    async def response_hook(response):
        body = await response.aread()
        (directory / "wire_response.json").write_text(redact(body.decode(), secrets), encoding="utf-8")
        write_json(directory / "http_response.json", {"status_code": response.status_code,
            "timestamp_utc": utc_now(), "request_id": redact(response.headers.get("x-request-id"), secrets)})

    reservation = budget.reserve(call_id, json.dumps(request_record), max_tokens)
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(event_hooks={"request": [request_hook], "response": [response_hook]}, timeout=httpx.Timeout(900, connect=30)) as http:
            client = AsyncOpenAI(api_key=config["openai_api_key"], base_url=config["openai_base_url"], max_retries=0, http_client=http)
            run_model = OpenAIChatCompletionsModel(model=model, openai_client=client)
            settings=ModelSettings(temperature=temperature,max_tokens=max_tokens,include_usage=True)
            if reasoning_effort:
                # Preserve the historical writer transport exactly. Reasoning
                # candidates explicitly opt in to the documented completion
                # limit, including hidden reasoning tokens, without temperature.
                settings=ModelSettings(temperature=None,max_tokens=None,include_usage=True,
                    reasoning=Reasoning(effort=reasoning_effort),
                    extra_args={'max_completion_tokens':max_tokens})
            result = await Runner.run(agent, prompt, max_turns=1,
                run_config=RunConfig(model=run_model,model_settings=settings,tracing_disabled=True))
        runtime = time.perf_counter()-start
        usage = result.context_wrapper.usage
        if not result.raw_responses or not usage.total_tokens or usage.requests != 1:
            raise RuntimeError("Provider did not report token usage; cannot certify cost")
        if usage.total_tokens != usage.input_tokens + usage.output_tokens:
            raise RuntimeError("Provider usage totals do not reconcile")
        cost = budget.settle(call_id, usage.input_tokens, usage.output_tokens, runtime)
        output = result.final_output.model_dump()
        usage_record = asdict(usage)
        for key, value in list(usage_record.items()):
            if hasattr(value, 'model_dump'):
                usage_record[key] = value.model_dump()
        write_json(directory / "response.json", {"call_id": call_id, "structured_output": output,
            "usage": usage_record, "runtime_seconds": runtime, "cost_usd": cost,
            "completed_at_utc": utc_now(), "attempts": 1})
        return result.final_output
    except Exception as exc:
        write_json(directory / "error.json", {"call_id": call_id, "error_type": type(exc).__name__,
            "message": redact(exc, [config["openai_api_key"]]), "runtime_seconds": time.perf_counter()-start,
            "timestamp_utc": utc_now(), "billing_status": "see_cost_ledger; unsettled_reservation_retained_pending_reconciliation"})
        raise RuntimeError("Model call failed; see sanitized error record at " + str(directory)) from None
