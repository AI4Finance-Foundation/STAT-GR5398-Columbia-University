"""Inventory credentials, endpoint entitlements, model schema support and render tools.

Default is local-only. --probe-data makes real GETs; --probe-model requires an
explicit approved budget and issues one tiny real typed completion.
"""
import argparse
import asyncio
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from audit_io import credentials, public_url, utc_now, write_json, SnapshotHTTP, Budget, call_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-data", action="store_true")
    parser.add_argument("--probe-model", action="store_true")
    parser.add_argument("--budget")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = credentials(args.config)
    packages = {}
    for name in ("pandas", "numpy", "openai", "openai-agents", "pydantic", "matplotlib", "reportlab", "playwright", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"created_at_utc": utc_now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "credentials_present": {k: bool(cfg[k]) for k in ("fmp_api_key", "openai_api_key")},
        "model": cfg["openai_model"] or None, "provider_base_url": public_url(cfg["openai_base_url"]),
        "budget_policy_provided": bool(args.budget), "tools": {name: shutil.which(name) for name in ("git", "node", "pdftoppm", "pdfinfo")},
        "data_entitlements": [], "model_schema_probe": "not_run", "blockers": []}
    for name in ("fmp_api_key", "openai_api_key"):
        if not cfg[name]:
            result["blockers"].append("missing_" + name)
    if not args.budget:
        result["blockers"].append("missing_approved_budget_and_verified_pricing")
    if args.probe_data and cfg["fmp_api_key"]:
        import requests
        endpoints = {
            "income": "/api/v3/income-statement/NVDA?period=annual&limit=1",
            "balance": "/api/v3/balance-sheet-statement/NVDA?period=annual&limit=1",
            "cash_flow": "/api/v3/cash-flow-statement/NVDA?period=annual&limit=1",
            "ratios": "/api/v3/ratios/NVDA?period=annual&limit=1",
            "key_metrics": "/api/v3/key-metrics/NVDA?period=annual&limit=1",
            "quote": "/api/v3/quote/NVDA",
            "profile": "/api/v3/profile/NVDA",
            "rating": "/api/v4/upgrades-downgrades?symbol=NVDA",
            "target": "/api/v4/price-target?symbol=NVDA",
            "news": "/api/v3/stock_news?tickers=NVDA&limit=1",
            "stable_income": "/stable/income-statement?symbol=NVDA&period=annual&limit=1"}
        with SnapshotHTTP(args.output / "http", "record", [cfg["fmp_api_key"]]).installed():
            for name, path in endpoints.items():
                response = requests.get("https://financialmodelingprep.com" + path, params={"apikey": cfg["fmp_api_key"]})
                try:
                    body = response.json()
                    valid = response.ok and isinstance(body, list) and bool(body) and isinstance(body[0], dict)
                except ValueError:
                    valid = False
                result["data_entitlements"].append({"endpoint": name, "url": public_url(response.url),
                    "http_status": response.status_code, "nonempty_list_response": valid})
    elif args.probe_data:
        result["data_entitlements_status"] = "not_testable_without_credential"
    if args.probe_model:
        if not cfg["openai_api_key"] or not args.budget or not args.ledger:
            result["model_schema_probe"] = "blocked_missing_key_budget_or_ledger"
        else:
            from agents import Agent
            from pydantic import BaseModel
            class ProbeResponse(BaseModel):
                ok: bool
            probe = Agent(name="StructuredOutputProbe", instructions="Return ok true.", output_type=ProbeResponse)
            try:
                with Budget(args.budget, args.ledger) as budget:
                    cfg["openai_model"] = cfg["openai_model"] or budget.policy["model"]
                    result['model'] = cfg['openai_model']
                    answer = asyncio.run(call_agent(probe, "Confirm structured output.", config=cfg, budget=budget,
                        directory=args.output / "model_probe", call_id="preflight_" + utc_now(), max_tokens=32, temperature=0))
                    result["model_schema_probe"] = "passed" if answer.ok else "failed"
            except Exception as exc:
                result["model_schema_probe"] = "failed"
                result['model_probe_error_type'] = type(exc).__name__
                result['blockers'].append('model_schema_probe_failed; inspect sanitized probe record and ledger')
    write_json(args.output / "preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
