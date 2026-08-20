#!/usr/bin/env python
# coding: utf-8

import argparse
from collections import defaultdict
import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_COMPANIES: Dict[str, str] = {
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
}

DEFAULT_PEERS: Dict[str, List[str]] = {
    "NVDA": ["AMD", "INTC"],
    "AMD": ["NVDA", "INTC"],
    "INTC": ["NVDA", "AMD"],
    "AAPL": ["MSFT", "GOOGL"],
    "GOOGL": ["MSFT", "META"],
}

DEFAULT_MODEL_SPECS: List[str] = [
    "openai::gpt-4o-mini",
    "openai::gpt-5-nano",
    "claude::[AWS-按量计费]claude-haiku-4-5-20251001",
]

REQUIRED_ANALYSIS_FILES: List[str] = [
    "financial_metrics_and_forecasts.csv",
    "ratios_raw_data.csv",
    "peer_ebitda_comparison.csv",
    "peer_ev_ebitda_comparison.csv",
    "tagline.txt",
    "company_overview.txt",
    "investment_overview.txt",
    "valuation_overview.txt",
    "risks.txt",
    "competitor_analysis.txt",
    "major_takeaways.txt",
    "news_summary.txt",
    "run_manifest.json",
]


@dataclass
class ModelSpec:
    provider: str
    model: str

    @property
    def tag(self) -> str:
        return f"{self.provider}_{_sanitize_tag(self.model)}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sanitize_tag(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    cleaned = cleaned.strip("-_.")
    return cleaned.lower() or "unknown"


def _safe_print_text(text: str) -> str:
    try:
        return text.encode("ascii", errors="ignore").decode("ascii")
    except Exception:
        return text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_model_specs(model_specs: Sequence[str]) -> List[ModelSpec]:
    parsed: List[ModelSpec] = []
    for item in model_specs:
        if "::" not in item:
            raise ValueError(
                f"Invalid --model-spec '{item}'. Expected format: provider::model"
            )
        provider_raw, model_raw = item.split("::", 1)
        provider = provider_raw.strip().lower()
        model = model_raw.strip()
        if provider not in {"openai", "claude", "gemini"}:
            raise ValueError(
                f"Unsupported provider '{provider}' in --model-spec '{item}'. "
                "Supported providers: openai, claude, gemini."
            )
        if not model:
            raise ValueError(f"Empty model name in --model-spec '{item}'.")
        parsed.append(ModelSpec(provider=provider, model=model))
    if not parsed:
        raise ValueError("No model specs resolved.")
    return parsed


def _build_task_config(
    base_config_path: Path,
    provider: str,
    model: str,
    temp_dir: Path,
) -> Path:
    cfg = configparser.ConfigParser()
    try:
        with base_config_path.open("r", encoding="utf-8") as f:
            cfg.read_file(f)
    except configparser.Error as e:
        raise ValueError(
            f"Invalid INI config file: {base_config_path}. "
            "Please provide a real config with [API_KEYS] (not placeholder text). "
            f"Parser error: {e}"
        ) from e

    if not cfg.has_section("API_KEYS"):
        raise ValueError(
            f"Config file missing [API_KEYS] section: {base_config_path}"
        )

    cfg.set("API_KEYS", "llm_provider", provider)
    cfg.set("API_KEYS", "llm_model", model)
    if provider == "openai":
        cfg.set("API_KEYS", "openai_model", model)
    elif provider == "claude":
        cfg.set("API_KEYS", "claude_model", model)
    elif provider == "gemini":
        cfg.set("API_KEYS", "gemini_model", model)

    out_path = temp_dir / f"config_{provider}_{_sanitize_tag(model)}.ini"
    with out_path.open("w", encoding="utf-8") as f:
        cfg.write(f)
    return out_path


def _run_command(command: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as f:
        f.write(result.stdout or "")
    with stderr_path.open("w", encoding="utf-8") as f:
        f.write(result.stderr or "")
    return result.returncode


def _missing_files(base_dir: Path, expected_files: Sequence[str]) -> List[str]:
    missing = []
    for rel in expected_files:
        if not (base_dir / rel).exists():
            missing.append(rel)
    return missing


def _move_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _run_two_stage_evaluation(
    *,
    args,
    project_root: Path,
    config_path: Path,
    evaluate_script: Path,
    reports_index: List[dict],
    storage_root: Path,
) -> dict:
    eval_root = storage_root / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)

    stage1_records: List[dict] = []
    stage2_records: List[dict] = []
    stage1_ranked_by_ticker: Dict[str, List[dict]] = defaultdict(list)

    print("")
    print("Starting two-stage evaluation...")
    print(
        "Stage1 evaluator: "
        f"{args.eval_stage1_provider}/{_safe_print_text(args.eval_stage1_model)} | "
        "Stage2 evaluator: "
        f"{args.eval_stage2_provider}/{_safe_print_text(args.eval_stage2_model)}"
    )

    # Stage 1: evaluate every successful report independently.
    for report_meta in reports_index:
        ticker = report_meta["ticker"]
        model_tag = report_meta["model_tag"]
        report_path = Path(report_meta["report_path"])
        logs_dir = Path(report_meta.get("logs_dir", eval_root / "logs" / ticker / model_tag))
        logs_dir.mkdir(parents=True, exist_ok=True)

        stage1_output = eval_root / "stage1" / ticker / f"{model_tag}.json"
        stage1_output.parent.mkdir(parents=True, exist_ok=True)
        stage1_stdout = logs_dir / "eval_stage1_stdout.log"
        stage1_stderr = logs_dir / "eval_stage1_stderr.log"

        stage1_cmd = [
            args.python_executable,
            str(evaluate_script),
            "--report-a",
            str(report_path),
            "--ticker",
            ticker,
            "--config-file",
            str(config_path),
            "--llm-provider",
            args.eval_stage1_provider,
            "--llm-model",
            args.eval_stage1_model,
            "--evaluated-model-provider",
            report_meta["model_provider"],
            "--evaluated-model-name",
            report_meta["model_name"],
            "--evaluated-model-tag",
            report_meta["model_tag"],
            "--max-chars",
            str(args.eval_max_chars),
            "--output-file",
            str(stage1_output),
        ]
        rc = _run_command(stage1_cmd, cwd=project_root, stdout_path=stage1_stdout, stderr_path=stage1_stderr)
        payload = _read_json(stage1_output)
        score = None
        if payload:
            score = _to_float(payload.get("combined_score"))
            if score is None:
                score = _to_float((payload.get("llm_evaluation") or {}).get("overall_score"))

        record = {
            "ticker": ticker,
            "company_name": report_meta["company_name"],
            "model_provider": report_meta["model_provider"],
            "model_name": report_meta["model_name"],
            "model_tag": model_tag,
            "judge_provider": args.eval_stage1_provider,
            "judge_model": args.eval_stage1_model,
            "report_path": str(report_path),
            "stage1_output": str(stage1_output),
            "status": "success" if (rc == 0 and payload is not None) else "failed",
            "return_code": rc,
            "stage1_score": score,
            "judge_origin": "stage1",
        }
        stage1_records.append(record)

        if record["status"] == "success" and score is not None:
            stage1_ranked_by_ticker[ticker].append(record)

    # Stage 2: pairwise review only for boundary cases within each ticker.
    stage2_pairs_total = 0
    for ticker, ranked in stage1_ranked_by_ticker.items():
        ranked_sorted = sorted(ranked, key=lambda x: x["stage1_score"], reverse=True)
        if len(ranked_sorted) < 2:
            continue

        candidate_pairs = []
        for idx in range(len(ranked_sorted) - 1):
            left = ranked_sorted[idx]
            right = ranked_sorted[idx + 1]
            margin = (left["stage1_score"] or 0.0) - (right["stage1_score"] or 0.0)
            if margin <= args.eval_stage2_margin_threshold:
                candidate_pairs.append((left, right, margin))
            if len(candidate_pairs) >= args.eval_stage2_max_pairs_per_ticker:
                break

        for pair_idx, (left, right, margin) in enumerate(candidate_pairs, start=1):
            stage2_pairs_total += 1
            stage2_output = eval_root / "stage2" / ticker / f"pair_{pair_idx}_{left['model_tag']}_vs_{right['model_tag']}.json"
            stage2_output.parent.mkdir(parents=True, exist_ok=True)
            logs_dir = eval_root / "logs" / ticker
            logs_dir.mkdir(parents=True, exist_ok=True)
            stage2_stdout = logs_dir / f"eval_stage2_pair{pair_idx}_stdout.log"
            stage2_stderr = logs_dir / f"eval_stage2_pair{pair_idx}_stderr.log"

            stage2_cmd = [
                args.python_executable,
                str(evaluate_script),
                "--report-a",
                left["report_path"],
                "--report-b",
                right["report_path"],
                "--ticker",
                ticker,
                "--config-file",
                str(config_path),
                "--llm-provider",
                args.eval_stage2_provider,
                "--llm-model",
                args.eval_stage2_model,
                "--evaluated-model-a-provider",
                left["model_provider"],
                "--evaluated-model-a-name",
                left["model_name"],
                "--evaluated-model-a-tag",
                left["model_tag"],
                "--evaluated-model-b-provider",
                right["model_provider"],
                "--evaluated-model-b-name",
                right["model_name"],
                "--evaluated-model-b-tag",
                right["model_tag"],
                "--max-chars",
                str(args.eval_max_chars),
                "--output-file",
                str(stage2_output),
            ]
            rc = _run_command(stage2_cmd, cwd=project_root, stdout_path=stage2_stdout, stderr_path=stage2_stderr)
            payload = _read_json(stage2_output)

            winner = None
            score_a = None
            score_b = None
            if payload:
                comparison = payload.get("llm_comparison") if isinstance(payload, dict) else None
                winner = payload.get("winner")
                score_a = _to_float(payload.get("score_a"))
                score_b = _to_float(payload.get("score_b"))
                if isinstance(comparison, dict):
                    if winner is None:
                        winner = comparison.get("winner")
                    if score_a is None:
                        score_a = _to_float(comparison.get("score_a"))
                    if score_b is None:
                        score_b = _to_float(comparison.get("score_b"))

            stage2_records.append(
                {
                    "ticker": ticker,
                    "pair_index": pair_idx,
                    "margin_stage1": round(margin, 4),
                    "left_model_tag": left["model_tag"],
                    "left_model_provider": left["model_provider"],
                    "left_model_name": left["model_name"],
                    "right_model_tag": right["model_tag"],
                    "right_model_provider": right["model_provider"],
                    "right_model_name": right["model_name"],
                    "left_report_path": left["report_path"],
                    "right_report_path": right["report_path"],
                    "stage2_output": str(stage2_output),
                    "status": "success" if (rc == 0 and payload is not None) else "failed",
                    "return_code": rc,
                    "judge_provider": args.eval_stage2_provider,
                    "judge_model": args.eval_stage2_model,
                    "winner": winner,
                    "score_a": score_a,
                    "score_b": score_b,
                }
            )

    # Merge stage1 + stage2 into final ranking.
    final_map: dict[tuple[str, str], dict] = {}
    for row in stage1_records:
        key = (row["ticker"], row["model_tag"])
        final_map[key] = {
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "model_provider": row["model_provider"],
            "model_name": row["model_name"],
            "model_tag": row["model_tag"],
            "report_path": row["report_path"],
            "stage1_score": row["stage1_score"],
            "final_score": row["stage1_score"],
            "judge_origin": "stage1",
            "stage2_pairs": [],
        }

    for row in stage2_records:
        if row["status"] != "success":
            continue
        key_a = (row["ticker"], row["left_model_tag"])
        key_b = (row["ticker"], row["right_model_tag"])
        left_item = final_map.get(key_a)
        right_item = final_map.get(key_b)
        if not left_item or not right_item:
            continue

        score_a = row.get("score_a")
        score_b = row.get("score_b")
        if left_item["stage1_score"] is not None and score_a is not None:
            blended = round(left_item["stage1_score"] * 0.6 + score_a * 0.4, 2)
            left_item["final_score"] = blended
        if right_item["stage1_score"] is not None and score_b is not None:
            blended = round(right_item["stage1_score"] * 0.6 + score_b * 0.4, 2)
            right_item["final_score"] = blended

        # Tie-break when stage2 returns winner but no numeric scores.
        winner = (row.get("winner") or "").upper()
        if winner == "A" and (score_a is None or score_b is None):
            if left_item["final_score"] is not None:
                left_item["final_score"] = round(left_item["final_score"] + 0.01, 4)
        elif winner == "B" and (score_a is None or score_b is None):
            if right_item["final_score"] is not None:
                right_item["final_score"] = round(right_item["final_score"] + 0.01, 4)

        left_item["judge_origin"] = "merged"
        right_item["judge_origin"] = "merged"
        left_item["stage2_pairs"].append(row)
        right_item["stage2_pairs"].append(row)

    final_records = list(final_map.values())
    final_by_ticker: Dict[str, List[dict]] = defaultdict(list)
    for row in final_records:
        final_by_ticker[row["ticker"]].append(row)

    per_ticker_winner: Dict[str, dict] = {}
    for ticker, rows in final_by_ticker.items():
        rows_sorted = sorted(
            rows,
            key=lambda x: (x["final_score"] is not None, x["final_score"]),
            reverse=True,
        )
        for rank, row in enumerate(rows_sorted, start=1):
            row["final_rank"] = rank
        if rows_sorted:
            top = rows_sorted[0]
            per_ticker_winner[ticker] = {
                "model_tag": top["model_tag"],
                "model_provider": top["model_provider"],
                "model_name": top["model_name"],
                "final_score": top["final_score"],
                "judge_origin": top["judge_origin"],
            }

    summary = {
        "enabled": True,
        "generated_at_utc": _utc_now_iso(),
        "stage1": {
            "provider": args.eval_stage1_provider,
            "model": args.eval_stage1_model,
            "records": stage1_records,
            "success_count": sum(1 for row in stage1_records if row["status"] == "success"),
            "failed_count": sum(1 for row in stage1_records if row["status"] == "failed"),
        },
        "stage2": {
            "provider": args.eval_stage2_provider,
            "model": args.eval_stage2_model,
            "margin_threshold": args.eval_stage2_margin_threshold,
            "max_pairs_per_ticker": args.eval_stage2_max_pairs_per_ticker,
            "executed_pairs": stage2_pairs_total,
            "records": stage2_records,
            "success_count": sum(1 for row in stage2_records if row["status"] == "success"),
            "failed_count": sum(1 for row in stage2_records if row["status"] == "failed"),
        },
        "final": {
            "records": sorted(
                final_records,
                key=lambda x: (
                    x["ticker"],
                    -(x["final_score"] if x["final_score"] is not None else -1e9),
                ),
            ),
            "winners_by_ticker": per_ticker_winner,
        },
    }
    _write_json(storage_root / "evaluation_summary.json", summary)
    return summary


def _aggregate_fmp_cache_stats(tasks: List[dict]) -> dict:
    totals = {
        "fresh_fetch": 0,
        "cache_hit": 0,
        "fallback_stale_cache": 0,
        "no_data": 0,
    }
    task_count_with_manifest = 0

    for task in tasks:
        manifest_path_str = task.get("paths", {}).get("analysis_dir")
        if not manifest_path_str:
            continue
        manifest_path = Path(manifest_path_str) / "run_manifest.json"
        payload = _read_json(manifest_path)
        if not payload:
            continue
        fmp_cache = payload.get("fmp_cache") or {}
        origin_counts = fmp_cache.get("origin_counts") or {}
        task_count_with_manifest += 1
        for key in totals.keys():
            totals[key] += int(origin_counts.get(key, 0) or 0)

    total_tracked_calls = sum(totals.values())
    cache_like = totals["cache_hit"] + totals["fallback_stale_cache"]
    cache_hit_ratio = round((cache_like / total_tracked_calls), 4) if total_tracked_calls > 0 else None

    return {
        "task_count_with_manifest": task_count_with_manifest,
        "origin_counts": totals,
        "total_tracked_calls": total_tracked_calls,
        "cache_hit_ratio": cache_hit_ratio,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FinRobot equity workflow in batch (multi-model, multi-company) with storage archiving."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke: NVDA x models only; full: all selected tickers x models.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="finrobot_equity/core/config/config.ini",
        help="Base config path. Per-task model overrides are generated automatically.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=list(DEFAULT_COMPANIES.keys()),
        help="Tickers for full mode. Ignored in smoke mode.",
    )
    parser.add_argument(
        "--smoke-ticker",
        type=str,
        default="NVDA",
        help="Ticker used in smoke mode (default: NVDA).",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Model mapping in format provider::model. Repeat this flag to add multiple models.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default UTC timestamp, e.g. 20260430_174500).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Base output root under FinRobot project.",
    )
    parser.add_argument(
        "--storage-subdir",
        type=str,
        default="storage",
        help="Storage sub-directory under output root.",
    )
    parser.add_argument(
        "--tmp-subdir",
        type=str,
        default="_batch_tmp",
        help="Temporary workspace sub-directory under output root.",
    )
    parser.add_argument("--years-limit", type=int, default=5)
    parser.add_argument("--news-days-back", type=int, default=5)
    parser.add_argument("--forecast-horizon-years", type=int, default=3)
    parser.add_argument(
        "--revenue-growth-values",
        type=float,
        nargs="*",
        default=[0.05, 0.05, 0.05],
    )
    parser.add_argument("--revenue-growth-default", type=float, default=0.05)
    parser.add_argument("--margin-improvement", type=float, default=0.01)
    parser.add_argument("--sga-margin-improvement", type=float, default=-0.005)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch immediately when one task fails. Default behavior is continue-on-error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks and commands without execution.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run child scripts.",
    )
    parser.add_argument(
        "--enable-two-stage-eval",
        action="store_true",
        help="Run two-stage evaluation across successful reports after batch generation.",
    )
    parser.add_argument(
        "--eval-stage1-provider",
        type=str,
        default="openai",
        help="Stage1 evaluator provider (openai/claude/gemini).",
    )
    parser.add_argument(
        "--eval-stage1-model",
        type=str,
        default="gpt-4o-mini",
        help="Stage1 evaluator model.",
    )
    parser.add_argument(
        "--eval-stage2-provider",
        type=str,
        default="openai",
        help="Stage2 evaluator provider (openai/claude/gemini).",
    )
    parser.add_argument(
        "--eval-stage2-model",
        type=str,
        default="gpt-5-nano",
        help="Stage2 evaluator model.",
    )
    parser.add_argument(
        "--eval-stage2-margin-threshold",
        type=float,
        default=4.0,
        help="Run stage2 pairwise review when adjacent stage1 scores differ by <= this margin.",
    )
    parser.add_argument(
        "--eval-stage2-max-pairs-per-ticker",
        type=int,
        default=1,
        help="Maximum number of stage2 pairs reviewed per ticker.",
    )
    parser.add_argument(
        "--eval-max-chars",
        type=int,
        default=12000,
        help="Max report chars sent to evaluator model.",
    )
    parser.add_argument(
        "--force-refresh-fmp",
        action="store_true",
        help="Pass through to analysis script: force refresh FMP data and bypass same-day cache.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = (project_root / args.config_file).resolve()
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    run_id = args.run_id or _utc_now_compact()
    tickers = (
        [args.smoke_ticker.strip().upper()]
        if args.mode == "smoke"
        else [t.strip().upper() for t in args.tickers if t.strip()]
    )
    if not tickers:
        print("ERROR: No tickers selected.")
        return 1

    model_specs_raw = args.model_spec if args.model_spec else DEFAULT_MODEL_SPECS
    try:
        models = _parse_model_specs(model_specs_raw)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    output_root = (project_root / args.output_root).resolve()
    storage_root = output_root / args.storage_subdir / run_id
    tmp_root = output_root / args.tmp_subdir / run_id
    storage_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    analysis_script = (project_root / "finrobot_equity" / "core" / "src" / "generate_financial_analysis.py").resolve()
    report_script = (project_root / "finrobot_equity" / "core" / "src" / "create_equity_report.py").resolve()
    evaluate_script = (project_root / "finrobot_equity" / "core" / "src" / "report_evaluate.py").resolve()
    if not analysis_script.exists() or not report_script.exists():
        print("ERROR: Workflow scripts not found under finrobot_equity/core/src.")
        return 1

    print(f"Run ID: {run_id}")
    print(f"Mode: {args.mode}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Models: {_safe_print_text(', '.join([f'{m.provider}/{m.model}' for m in models]))}")
    print(f"Storage root: {storage_root}")
    print("")

    temp_cfg_dir = Path(tempfile.mkdtemp(prefix=f"finrobot_batch_cfg_{run_id}_"))
    tasks = []
    reports_index = []
    eval_summary = {"enabled": False}
    failed_any = False

    try:
        total = len(tickers) * len(models)
        seq = 0
        for ticker in tickers:
            company_name = DEFAULT_COMPANIES.get(ticker, ticker)
            peer_tickers = DEFAULT_PEERS.get(ticker, [])
            for model_spec in models:
                seq += 1
                task_id = f"{seq:02d}_{ticker}_{model_spec.tag}"
                started = time.time()
                started_iso = _utc_now_iso()

                tmp_task_root = tmp_root / model_spec.tag / ticker
                final_task_root = storage_root / model_spec.tag / ticker
                logs_dir = tmp_task_root / "logs"
                analysis_dir = tmp_task_root / "analysis"
                report_dir = tmp_task_root / "report"
                logs_dir.mkdir(parents=True, exist_ok=True)
                analysis_dir.mkdir(parents=True, exist_ok=True)
                report_dir.mkdir(parents=True, exist_ok=True)

                print(f"[{seq}/{total}] {ticker} | {_safe_print_text(f'{model_spec.provider}/{model_spec.model}')}")
                task_status = "success"
                failure_reason = None
                analysis_rc = None
                report_rc = None
                missing_analysis = []
                missing_report = []

                try:
                    task_cfg_path = _build_task_config(
                        base_config_path=config_path,
                        provider=model_spec.provider,
                        model=model_spec.model,
                        temp_dir=temp_cfg_dir,
                    )
                except Exception as e:
                    print(f"  FAILED: {e}")
                    if args.dry_run:
                        task_status = "dry_run"
                    else:
                        task_status = "failed"
                    failure_reason = "invalid_config"
                    ended_iso = _utc_now_iso()
                    duration_sec = round(time.time() - started, 2)
                    task_meta = {
                        "task_id": task_id,
                        "run_id": run_id,
                        "sequence": seq,
                        "mode": args.mode,
                        "ticker": ticker,
                        "company_name": company_name,
                        "peer_tickers": peer_tickers,
                        "model": {
                            "provider": model_spec.provider,
                            "name": model_spec.model,
                            "tag": model_spec.tag,
                        },
                        "status": task_status,
                        "failure_reason": f"{failure_reason}: {e}",
                        "timing": {
                            "started_at_utc": started_iso,
                            "finished_at_utc": ended_iso,
                            "duration_sec": duration_sec,
                        },
                        "return_codes": {
                            "generate_financial_analysis": None,
                            "create_equity_report": None,
                        },
                        "missing_files": {
                            "analysis": [],
                            "report": [],
                        },
                        "paths": {
                            "analysis_dir": str(analysis_dir),
                            "report_dir": str(report_dir),
                            "logs_dir": str(logs_dir),
                        },
                    }
                    _write_json(tmp_task_root / "run_meta.json", task_meta)
                    _move_tree(tmp_task_root, final_task_root)
                    task_meta["paths"]["storage_root"] = str(final_task_root)
                    task_meta["paths"]["analysis_dir"] = str(final_task_root / "analysis")
                    task_meta["paths"]["report_dir"] = str(final_task_root / "report")
                    task_meta["paths"]["logs_dir"] = str(final_task_root / "logs")
                    task_meta["paths"]["run_meta"] = str(final_task_root / "run_meta.json")
                    tasks.append(task_meta)
                    if task_status == "failed":
                        failed_any = True
                    if args.stop_on_error:
                        print("  stop-on-error enabled; terminating batch.")
                        break
                    continue

                analysis_cmd = [
                    args.python_executable,
                    str(analysis_script),
                    "--company-ticker", ticker,
                    "--company-name", company_name,
                    "--config-file", str(task_cfg_path),
                    "--years-limit", str(args.years_limit),
                    "--generate-text-sections",
                    "--output-dir", str(analysis_dir),
                    "--news-days-back", str(args.news_days_back),
                    "--forecast-horizon-years", str(args.forecast_horizon_years),
                    "--revenue-growth-default", str(args.revenue_growth_default),
                    "--margin-improvement", str(args.margin_improvement),
                    "--sga-margin-improvement", str(args.sga_margin_improvement),
                ]
                if args.force_refresh_fmp:
                    analysis_cmd.append("--force-refresh-fmp")
                if args.revenue_growth_values:
                    analysis_cmd += ["--revenue-growth-values"] + [str(v) for v in args.revenue_growth_values]
                if peer_tickers:
                    analysis_cmd += ["--peer-tickers"] + peer_tickers

                report_cmd = [
                    args.python_executable,
                    str(report_script),
                    "--company-ticker", ticker,
                    "--company-name", company_name,
                    "--analysis-csv", str(analysis_dir / "financial_metrics_and_forecasts.csv"),
                    "--tagline-file", str(analysis_dir / "tagline.txt"),
                    "--company-overview-file", str(analysis_dir / "company_overview.txt"),
                    "--investment-overview-file", str(analysis_dir / "investment_overview.txt"),
                    "--valuation-overview-file", str(analysis_dir / "valuation_overview.txt"),
                    "--risks-file", str(analysis_dir / "risks.txt"),
                    "--competitor-analysis-file", str(analysis_dir / "competitor_analysis.txt"),
                    "--major-takeaways-file", str(analysis_dir / "major_takeaways.txt"),
                    "--news-summary-file", str(analysis_dir / "news_summary.txt"),
                    "--peer-ev-ebitda-csv", str(analysis_dir / "peer_ev_ebitda_comparison.csv"),
                    "--ratios-csv", str(analysis_dir / "ratios_raw_data.csv"),
                    "--peer-ebitda-csv", str(analysis_dir / "peer_ebitda_comparison.csv"),
                    "--enable-text-regeneration",
                    "--enable-valuation-analysis",
                    "--output-dir", str(report_dir),
                    "--config-file", str(task_cfg_path),
                ]

                if args.dry_run:
                    print(f"  dry-run analysis: {' '.join(analysis_cmd)}")
                    print(f"  dry-run report:   {' '.join(report_cmd)}")
                    task_status = "dry_run"
                else:
                    analysis_rc = _run_command(
                        command=analysis_cmd,
                        cwd=project_root,
                        stdout_path=logs_dir / "generate_stdout.log",
                        stderr_path=logs_dir / "generate_stderr.log",
                    )
                    if analysis_rc != 0:
                        task_status = "failed"
                        failure_reason = f"generate_financial_analysis_return_code={analysis_rc}"
                    else:
                        missing_analysis = _missing_files(analysis_dir, REQUIRED_ANALYSIS_FILES)
                        if missing_analysis:
                            task_status = "failed"
                            failure_reason = "analysis_outputs_missing"

                    if task_status == "success":
                        report_rc = _run_command(
                            command=report_cmd,
                            cwd=project_root,
                            stdout_path=logs_dir / "report_stdout.log",
                            stderr_path=logs_dir / "report_stderr.log",
                        )
                        if report_rc != 0:
                            task_status = "failed"
                            failure_reason = f"create_equity_report_return_code={report_rc}"
                        else:
                            expected_report = [
                                f"Professional_Equity_Report_{ticker}.html",
                                "numeric_consistency.json",
                            ]
                            missing_report = _missing_files(report_dir, expected_report)
                            if missing_report:
                                task_status = "failed"
                                failure_reason = "report_outputs_missing"

                ended_iso = _utc_now_iso()
                duration_sec = round(time.time() - started, 2)

                task_meta = {
                    "task_id": task_id,
                    "run_id": run_id,
                    "sequence": seq,
                    "mode": args.mode,
                    "ticker": ticker,
                    "company_name": company_name,
                    "peer_tickers": peer_tickers,
                    "model": {
                        "provider": model_spec.provider,
                        "name": model_spec.model,
                        "tag": model_spec.tag,
                    },
                    "status": task_status,
                    "failure_reason": failure_reason,
                    "timing": {
                        "started_at_utc": started_iso,
                        "finished_at_utc": ended_iso,
                        "duration_sec": duration_sec,
                    },
                    "return_codes": {
                        "generate_financial_analysis": analysis_rc,
                        "create_equity_report": report_rc,
                    },
                    "missing_files": {
                        "analysis": missing_analysis,
                        "report": missing_report,
                    },
                    "paths": {
                        "analysis_dir": str(analysis_dir),
                        "report_dir": str(report_dir),
                        "logs_dir": str(logs_dir),
                    },
                }
                _write_json(tmp_task_root / "run_meta.json", task_meta)
                _move_tree(tmp_task_root, final_task_root)

                task_meta["paths"]["storage_root"] = str(final_task_root)
                task_meta["paths"]["analysis_dir"] = str(final_task_root / "analysis")
                task_meta["paths"]["report_dir"] = str(final_task_root / "report")
                task_meta["paths"]["logs_dir"] = str(final_task_root / "logs")
                task_meta["paths"]["run_meta"] = str(final_task_root / "run_meta.json")
                tasks.append(task_meta)

                if task_status == "success":
                    reports_index.append(
                        {
                            "run_id": run_id,
                            "ticker": ticker,
                            "company_name": company_name,
                            "model_provider": model_spec.provider,
                            "model_name": model_spec.model,
                            "model_tag": model_spec.tag,
                            "report_path": str(final_task_root / "report" / f"Professional_Equity_Report_{ticker}.html"),
                            "numeric_consistency_path": str(final_task_root / "report" / "numeric_consistency.json"),
                            "run_manifest_path": str(final_task_root / "analysis" / "run_manifest.json"),
                            "logs_dir": str(final_task_root / "logs"),
                        }
                    )
                    print("  OK: success")
                elif task_status == "dry_run":
                    print("  DRYRUN: planned")
                else:
                    failed_any = True
                    print(f"  FAILED: {failure_reason}")
                    if args.stop_on_error:
                        print("  stop-on-error enabled; terminating batch.")
                        break

            if args.stop_on_error and failed_any:
                break

    finally:
        shutil.rmtree(temp_cfg_dir, ignore_errors=True)
        shutil.rmtree(tmp_root, ignore_errors=True)

    if (
        args.enable_two_stage_eval
        and not args.dry_run
        and reports_index
        and evaluate_script.exists()
    ):
        try:
            eval_summary = _run_two_stage_evaluation(
                args=args,
                project_root=project_root,
                config_path=config_path,
                evaluate_script=evaluate_script,
                reports_index=reports_index,
                storage_root=storage_root,
            )
            print("Two-stage evaluation complete.")
            print(f"Evaluation file: {storage_root / 'evaluation_summary.json'}")
        except Exception as e:
            eval_summary = {"enabled": True, "error": str(e)}
            print(f"WARNING: two-stage evaluation failed: {e}")

    success_count = sum(1 for t in tasks if t["status"] == "success")
    failed_count = sum(1 for t in tasks if t["status"] == "failed")
    dry_run_count = sum(1 for t in tasks if t["status"] == "dry_run")
    batch_summary = {
        "run_id": run_id,
        "generated_at_utc": _utc_now_iso(),
        "mode": args.mode,
        "tickers": tickers,
        "models": [{"provider": m.provider, "name": m.model, "tag": m.tag} for m in models],
        "counts": {
            "total_tasks": len(tasks),
            "success": success_count,
            "failed": failed_count,
            "dry_run": dry_run_count,
        },
        "storage_root": str(storage_root),
        "fmp_cache": _aggregate_fmp_cache_stats(tasks),
        "evaluation": eval_summary,
        "tasks": tasks,
    }
    _write_json(storage_root / "batch_summary.json", batch_summary)
    _write_json(storage_root / "reports_index.json", {"run_id": run_id, "reports": reports_index})

    print("")
    print("Batch complete.")
    print(f"Summary: total={len(tasks)}, success={success_count}, failed={failed_count}, dry_run={dry_run_count}")
    print(f"Summary file: {storage_root / 'batch_summary.json'}")
    print(f"Reports index: {storage_root / 'reports_index.json'}")

    if failed_count > 0 and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
