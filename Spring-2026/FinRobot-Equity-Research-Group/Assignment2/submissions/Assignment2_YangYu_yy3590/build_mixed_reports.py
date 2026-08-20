#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SECTIONS: List[str] = [
    "tagline",
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
    "major_takeaways",
    "news_summary",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mixed-section reports from model outputs using a global section mapping."
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run id under output/storage, e.g. 20260430_220342")
    parser.add_argument("--output-root", type=str, default="./output", help="Output root relative to FinRobot project.")
    parser.add_argument("--storage-subdir", type=str, default="storage", help="Storage subdir under output root.")
    parser.add_argument("--mapping-file", type=str, default="global_section_mapping.json", help="Mapping JSON filename under run dir.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="finrobot_equity/core/config/config.ini",
        help="Config INI passed to create_equity_report.py and report_evaluate.py",
    )
    parser.add_argument("--python-executable", type=str, default=sys.executable, help="Python executable for child scripts.")
    parser.add_argument("--skip-report-build", action="store_true", help="Only build mixed analysis files/manifests, skip HTML build.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and print actions without copying/building/evaluating.")
    parser.add_argument("--mixed-analysis-dirname", type=str, default="mixed_analysis")
    parser.add_argument("--mixed-report-dirname", type=str, default="mixed_reports")
    parser.add_argument("--manifest-json-name", type=str, default="mixed_build_manifest.json")
    parser.add_argument("--summary-md-name", type=str, default="summary_overview.md")
    parser.add_argument("--summary-csv-name", type=str, default="summary_overview.csv")
    parser.add_argument("--sections", type=str, nargs="*", default=DEFAULT_SECTIONS, help="Sections to mix.")
    parser.add_argument(
        "--skip-auto-fetch",
        action="store_true",
        help="Pass through to create_equity_report.py to skip FMP market-data auto-fetch during mixed report build.",
    )
    parser.add_argument(
        "--evaluate-mixed",
        action="store_true",
        help="Evaluate mixed reports after build and generate comparable mixed scoreboard.",
    )
    parser.add_argument(
        "--mixed-eval-stage2",
        action="store_true",
        help="Enable stage2 pairwise review for mixed report against baseline ticker winner.",
    )
    parser.add_argument("--eval-stage1-provider", type=str, default="openai", help="Mixed stage1 evaluator provider.")
    parser.add_argument("--eval-stage1-model", type=str, default="gpt-4o-mini", help="Mixed stage1 evaluator model.")
    parser.add_argument("--eval-stage2-provider", type=str, default="openai", help="Mixed stage2 evaluator provider.")
    parser.add_argument("--eval-stage2-model", type=str, default="gpt-5-nano", help="Mixed stage2 evaluator model.")
    parser.add_argument("--eval-max-chars", type=int, default=12000, help="Max report chars sent to evaluator model.")
    parser.add_argument("--eval-stage2-blend-alpha", type=float, default=0.6, help="Final blend: alpha*stage1 + (1-alpha)*stage2.")
    parser.add_argument("--mixed-eval-json-name", type=str, default="evaluation_mixed_summary.json")
    parser.add_argument("--scoreboard-csv-name", type=str, default="final_scoreboard.csv")
    parser.add_argument("--scoreboard-md-name", type=str, default="final_scoreboard.md")
    return parser.parse_args()


def _load_reports(run_dir: Path) -> tuple[List[dict], Dict[tuple, dict]]:
    reports_index_path = run_dir / "reports_index.json"
    payload = _read_json(reports_index_path)
    reports = payload.get("reports") or []
    by_ticker_model = {}
    for row in reports:
        by_ticker_model[(row["ticker"], row["model_tag"])] = row
    return reports, by_ticker_model


def _get_analysis_dir(report: dict, run_dir: Path) -> Path:
    rel_candidate = run_dir / report["model_tag"] / report["ticker"] / "analysis"
    if rel_candidate.exists():
        return rel_candidate

    manifest_path = report.get("run_manifest_path")
    if manifest_path:
        p = Path(manifest_path)
        if p.exists():
            return p.parent
    report_path = report.get("report_path")
    if report_path:
        p = Path(report_path)
        candidate = p.parent.parent / "analysis"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve analysis dir for report: {report}")


def _get_report_path(report: dict, run_dir: Path) -> Path:
    rel_candidate = run_dir / report["model_tag"] / report["ticker"] / "report" / f"Professional_Equity_Report_{report['ticker']}.html"
    if rel_candidate.exists():
        return rel_candidate
    report_path = report.get("report_path")
    if report_path:
        p = Path(report_path)
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot resolve report path for report: {report}")


def _build_summary_markdown(
    *,
    run_id: str,
    run_dir: Path,
    mapping: dict,
    winners_by_ticker: dict,
    mixed_rows: List[dict],
) -> str:
    lines = []
    lines.append(f"# Mixed Report Summary ({run_id})")
    lines.append("")
    lines.append(f"- Generated at UTC: {_utc_now_iso()}")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append("")
    lines.append("## Per-Ticker Winner (from existing evaluation)")
    lines.append("| Ticker | Winner Model Tag | Final Score |")
    lines.append("|---|---|---:|")
    for ticker in sorted(winners_by_ticker.keys()):
        row = winners_by_ticker[ticker]
        lines.append(f"| {ticker} | {row.get('model_tag')} | {row.get('final_score')} |")
    lines.append("")
    lines.append("## Global Section Mapping")
    lines.append("| Section | Selected Model Tag | Mean Score | Stddev | Sample Count |")
    lines.append("|---|---|---:|---:|---:|")
    global_map = mapping.get("global_section_mapping") or {}
    for section in sorted(global_map.keys()):
        row = global_map[section]
        lines.append(
            f"| {section} | {row.get('selected_model_tag')} | {row.get('mean_score')} | "
            f"{row.get('stddev_score')} | {row.get('sample_count')} |"
        )
    lines.append("")
    lines.append("## Mixed Section Sources")
    lines.append("| Ticker | Section | Selected Model Tag | Source File |")
    lines.append("|---|---|---|---|")
    for row in sorted(mixed_rows, key=lambda x: (x["ticker"], x["section"])):
        lines.append(
            f"| {row['ticker']} | {row['section']} | {row['selected_model_tag']} | `{row['source_path']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _evaluate_mixed_reports(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mixed_report_root: Path,
    reports: List[dict],
    evaluation_summary: dict,
) -> dict:
    eval_script = (PROJECT_ROOT / "finrobot_equity" / "core" / "src" / "report_evaluate.py").resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"report_evaluate.py not found: {eval_script}")

    baseline_by_ticker = {}
    for report in reports:
        baseline_by_ticker.setdefault(report["ticker"], []).append(report)

    winners_by_ticker = ((evaluation_summary.get("final") or {}).get("winners_by_ticker") or {})
    stage1_records: List[dict] = []
    stage2_records: List[dict] = []
    final_records: List[dict] = []

    for ticker in sorted({r["ticker"] for r in reports}):
        mixed_report_path = mixed_report_root / ticker / f"Professional_Equity_Report_{ticker}.html"
        company_name = next((r["company_name"] for r in reports if r["ticker"] == ticker), ticker)
        stage1_out = run_dir / "evaluation_mixed" / "stage1" / ticker / "mixed.json"
        stage1_out.parent.mkdir(parents=True, exist_ok=True)
        stage1_log_out = run_dir / "evaluation_mixed" / "logs" / ticker / "mixed_stage1_stdout.log"
        stage1_log_err = run_dir / "evaluation_mixed" / "logs" / ticker / "mixed_stage1_stderr.log"

        stage1_row = {
            "ticker": ticker,
            "company_name": company_name,
            "model_provider": "mixed",
            "model_name": "mixed_by_section_mapping",
            "model_tag": "mixed",
            "report_path": str(mixed_report_path),
            "judge_provider": args.eval_stage1_provider,
            "judge_model": args.eval_stage1_model,
            "stage1_output": str(stage1_out),
            "status": "failed",
            "return_code": None,
            "stage1_score": None,
            "judge_origin": "stage1",
            "error": None,
        }

        if not mixed_report_path.exists():
            stage1_row["error"] = f"missing_mixed_report:{mixed_report_path}"
            stage1_records.append(stage1_row)
            final_records.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "model_provider": "mixed",
                    "model_name": "mixed_by_section_mapping",
                    "model_tag": "mixed",
                    "report_path": str(mixed_report_path),
                    "stage1_score": None,
                    "final_score": None,
                    "judge_origin": "failed",
                    "stage2_pairs": [],
                    "final_rank": 1,
                    "error": stage1_row["error"],
                }
            )
            continue

        stage1_cmd = [
            args.python_executable,
            str(eval_script),
            "--report-a",
            str(mixed_report_path),
            "--ticker",
            ticker,
            "--config-file",
            str((PROJECT_ROOT / args.config_file).resolve()),
            "--llm-provider",
            args.eval_stage1_provider,
            "--llm-model",
            args.eval_stage1_model,
            "--evaluated-model-provider",
            "mixed",
            "--evaluated-model-name",
            "mixed_by_section_mapping",
            "--evaluated-model-tag",
            "mixed",
            "--max-chars",
            str(args.eval_max_chars),
            "--output-file",
            str(stage1_out),
        ]
        rc1 = _run_command(stage1_cmd, cwd=PROJECT_ROOT, stdout_path=stage1_log_out, stderr_path=stage1_log_err)
        stage1_row["return_code"] = rc1
        stage1_payload = _read_json_if_exists(stage1_out)
        stage1_score = None
        if stage1_payload:
            stage1_score = _to_float(stage1_payload.get("combined_score"))
            if stage1_score is None:
                stage1_score = _to_float((stage1_payload.get("llm_evaluation") or {}).get("overall_score"))
        stage1_row["stage1_score"] = stage1_score
        stage1_row["status"] = "success" if (rc1 == 0 and stage1_payload is not None and stage1_score is not None) else "failed"
        if stage1_row["status"] == "failed" and stage1_row["error"] is None:
            stage1_row["error"] = "stage1_eval_failed"
        stage1_records.append(stage1_row)

        final_row = {
            "ticker": ticker,
            "company_name": company_name,
            "model_provider": "mixed",
            "model_name": "mixed_by_section_mapping",
            "model_tag": "mixed",
            "report_path": str(mixed_report_path),
            "stage1_score": stage1_score,
            "final_score": stage1_score,
            "judge_origin": "stage1" if stage1_row["status"] == "success" else "failed",
            "stage2_pairs": [],
            "final_rank": 1,
            "error": stage1_row["error"],
        }

        if args.mixed_eval_stage2 and stage1_row["status"] == "success":
            winner = winners_by_ticker.get(ticker)
            winner_tag = winner.get("model_tag") if isinstance(winner, dict) else None
            winner_report_row = None
            if winner_tag:
                winner_report_row = next((r for r in baseline_by_ticker.get(ticker, []) if r["model_tag"] == winner_tag), None)

            if winner_report_row:
                try:
                    winner_report_path = _get_report_path(winner_report_row, run_dir)
                    stage2_out = run_dir / "evaluation_mixed" / "stage2" / ticker / f"mixed_vs_{winner_tag}.json"
                    stage2_out.parent.mkdir(parents=True, exist_ok=True)
                    stage2_log_out = run_dir / "evaluation_mixed" / "logs" / ticker / "mixed_stage2_stdout.log"
                    stage2_log_err = run_dir / "evaluation_mixed" / "logs" / ticker / "mixed_stage2_stderr.log"
                    stage2_cmd = [
                        args.python_executable,
                        str(eval_script),
                        "--report-a",
                        str(mixed_report_path),
                        "--report-b",
                        str(winner_report_path),
                        "--ticker",
                        ticker,
                        "--config-file",
                        str((PROJECT_ROOT / args.config_file).resolve()),
                        "--llm-provider",
                        args.eval_stage2_provider,
                        "--llm-model",
                        args.eval_stage2_model,
                        "--evaluated-model-a-provider",
                        "mixed",
                        "--evaluated-model-a-name",
                        "mixed_by_section_mapping",
                        "--evaluated-model-a-tag",
                        "mixed",
                        "--evaluated-model-b-provider",
                        winner_report_row["model_provider"],
                        "--evaluated-model-b-name",
                        winner_report_row["model_name"],
                        "--evaluated-model-b-tag",
                        winner_report_row["model_tag"],
                        "--max-chars",
                        str(args.eval_max_chars),
                        "--output-file",
                        str(stage2_out),
                    ]
                    rc2 = _run_command(stage2_cmd, cwd=PROJECT_ROOT, stdout_path=stage2_log_out, stderr_path=stage2_log_err)
                    stage2_payload = _read_json_if_exists(stage2_out)
                    winner_flag = None
                    score_a = None
                    score_b = None
                    if stage2_payload:
                        cmp_obj = stage2_payload.get("llm_comparison") or {}
                        winner_flag = cmp_obj.get("winner")
                        score_a = _to_float(cmp_obj.get("score_a"))
                        score_b = _to_float(cmp_obj.get("score_b"))

                    stage2_row = {
                        "ticker": ticker,
                        "pair_index": 1,
                        "left_model_tag": "mixed",
                        "left_model_provider": "mixed",
                        "left_model_name": "mixed_by_section_mapping",
                        "right_model_tag": winner_report_row["model_tag"],
                        "right_model_provider": winner_report_row["model_provider"],
                        "right_model_name": winner_report_row["model_name"],
                        "left_report_path": str(mixed_report_path),
                        "right_report_path": str(winner_report_path),
                        "stage2_output": str(stage2_out),
                        "status": "success" if (rc2 == 0 and stage2_payload is not None) else "failed",
                        "return_code": rc2,
                        "judge_provider": args.eval_stage2_provider,
                        "judge_model": args.eval_stage2_model,
                        "winner": winner_flag,
                        "score_a": score_a,
                        "score_b": score_b,
                    }
                    stage2_records.append(stage2_row)

                    if stage2_row["status"] == "success":
                        alpha = args.eval_stage2_blend_alpha
                        beta = 1.0 - alpha
                        if stage1_score is not None and score_a is not None:
                            final_row["final_score"] = round(stage1_score * alpha + score_a * beta, 2)
                            final_row["judge_origin"] = "merged"
                        elif stage1_score is not None and (winner_flag or "").upper() == "A":
                            final_row["final_score"] = round(stage1_score + 0.01, 4)
                            final_row["judge_origin"] = "merged"
                        final_row["stage2_pairs"].append(stage2_row)
                except Exception as e:
                    stage2_records.append(
                        {
                            "ticker": ticker,
                            "pair_index": 1,
                            "left_model_tag": "mixed",
                            "left_model_provider": "mixed",
                            "left_model_name": "mixed_by_section_mapping",
                            "right_model_tag": winner_tag,
                            "right_model_provider": None,
                            "right_model_name": None,
                            "left_report_path": str(mixed_report_path),
                            "right_report_path": None,
                            "stage2_output": None,
                            "status": "failed",
                            "return_code": None,
                            "judge_provider": args.eval_stage2_provider,
                            "judge_model": args.eval_stage2_model,
                            "winner": None,
                            "score_a": None,
                            "score_b": None,
                            "error": f"stage2_exception:{e}",
                        }
                    )
            else:
                stage2_records.append(
                    {
                        "ticker": ticker,
                        "pair_index": 1,
                        "left_model_tag": "mixed",
                        "left_model_provider": "mixed",
                        "left_model_name": "mixed_by_section_mapping",
                        "right_model_tag": winner_tag,
                        "right_model_provider": None,
                        "right_model_name": None,
                        "left_report_path": str(mixed_report_path),
                        "right_report_path": None,
                        "stage2_output": None,
                        "status": "skipped",
                        "return_code": None,
                        "judge_provider": args.eval_stage2_provider,
                        "judge_model": args.eval_stage2_model,
                        "winner": None,
                        "score_a": None,
                        "score_b": None,
                        "reason": "baseline_winner_report_not_found",
                    }
                )

        final_records.append(final_row)

    winners = {}
    for row in final_records:
        winners[row["ticker"]] = {
            "model_tag": "mixed",
            "model_provider": "mixed",
            "model_name": "mixed_by_section_mapping",
            "final_score": row.get("final_score"),
            "judge_origin": row.get("judge_origin"),
        }

    return {
        "enabled": True,
        "generated_at_utc": _utc_now_iso(),
        "stage1": {
            "provider": args.eval_stage1_provider,
            "model": args.eval_stage1_model,
            "records": stage1_records,
            "success_count": sum(1 for r in stage1_records if r.get("status") == "success"),
            "failed_count": sum(1 for r in stage1_records if r.get("status") == "failed"),
        },
        "stage2": {
            "enabled": bool(args.mixed_eval_stage2),
            "provider": args.eval_stage2_provider if args.mixed_eval_stage2 else None,
            "model": args.eval_stage2_model if args.mixed_eval_stage2 else None,
            "blend_alpha": args.eval_stage2_blend_alpha if args.mixed_eval_stage2 else None,
            "records": stage2_records,
            "success_count": sum(1 for r in stage2_records if r.get("status") == "success"),
            "failed_count": sum(1 for r in stage2_records if r.get("status") == "failed"),
            "skipped_count": sum(1 for r in stage2_records if r.get("status") == "skipped"),
        },
        "final": {
            "records": sorted(final_records, key=lambda x: x["ticker"]),
            "winners_by_ticker": winners,
        },
    }


def _build_final_scoreboard_rows(
    *,
    reports: List[dict],
    base_eval_summary: dict,
    mixed_eval_summary: Optional[dict],
) -> List[dict]:
    model_order = []
    seen = set()
    for row in reports:
        tag = row["model_tag"]
        if tag not in seen:
            seen.add(tag)
            model_order.append(tag)

    base_score_index = {}
    for row in ((base_eval_summary.get("final") or {}).get("records") or []):
        base_score_index[(row.get("ticker"), row.get("model_tag"))] = row.get("final_score")

    base_origin_index = {}
    for row in ((base_eval_summary.get("final") or {}).get("records") or []):
        base_origin_index[(row.get("ticker"), row.get("model_tag"))] = row.get("judge_origin")

    mixed_score_index = {}
    mixed_origin_index = {}
    if mixed_eval_summary:
        for row in ((mixed_eval_summary.get("final") or {}).get("records") or []):
            mixed_score_index[row.get("ticker")] = row.get("final_score")
            mixed_origin_index[row.get("ticker")] = row.get("judge_origin")

    tickers = sorted({r["ticker"] for r in reports})
    rows = []
    for ticker in tickers:
        row = {"ticker": ticker}
        best_tag = None
        best_score = None
        for tag in model_order:
            score = _to_float(base_score_index.get((ticker, tag)))
            row[tag] = score
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_tag = tag
        if mixed_eval_summary:
            row["mixed"] = _to_float(mixed_score_index.get(ticker))
            row["mixed_judge_origin"] = mixed_origin_index.get(ticker)
        row["best_baseline_model_tag"] = best_tag
        row["best_baseline_score"] = best_score
        rows.append(row)
    return rows


def _build_scoreboard_markdown(rows: List[dict], model_columns: List[str], mixed_enabled: bool) -> str:
    headers = ["Ticker"] + model_columns
    if mixed_enabled:
        headers += ["mixed", "mixed_judge_origin"]
    headers += ["best_baseline_model_tag", "best_baseline_score"]

    lines = []
    lines.append("# Final Scoreboard")
    lines.append("")
    lines.append(f"- Generated at UTC: {_utc_now_iso()}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [row.get("ticker")]
        for col in model_columns:
            v = row.get(col)
            cells.append("" if v is None else str(v))
        if mixed_enabled:
            v = row.get("mixed")
            cells.append("" if v is None else str(v))
            cells.append(row.get("mixed_judge_origin") or "")
        cells.append(row.get("best_baseline_model_tag") or "")
        v = row.get("best_baseline_score")
        cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    run_dir = (output_root / args.storage_subdir / args.run_id).resolve()
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    mapping_path = run_dir / args.mapping_file
    if not mapping_path.exists():
        print(f"ERROR: Mapping file not found: {mapping_path}")
        return 1
    reports, by_ticker_model = _load_reports(run_dir)
    if not reports:
        print("ERROR: reports_index.json has no reports.")
        return 1
    model_info_by_tag: Dict[str, dict] = {}
    for row in reports:
        model_info_by_tag.setdefault(
            row["model_tag"],
            {
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
            },
        )

    mapping_payload = _read_json(mapping_path)
    global_map = mapping_payload.get("global_section_mapping") or {}
    missing_sections = [sec for sec in args.sections if sec not in global_map]
    if missing_sections:
        if args.dry_run:
            fallback_tag = sorted(model_info_by_tag.keys())[0]
            fallback_info = model_info_by_tag[fallback_tag]
            print(
                "WARNING: mapping missing sections in dry-run mode. "
                f"Using fallback model tag '{fallback_tag}' for validation only: {missing_sections}"
            )
            for sec in missing_sections:
                global_map[sec] = {
                    "selected_model_tag": fallback_tag,
                    "selected_model_provider": fallback_info["model_provider"],
                    "selected_model_name": fallback_info["model_name"],
                    "mean_score": None,
                    "stddev_score": None,
                    "sample_count": 0,
                    "selection_rule": "dry_run_fallback_first_model_tag",
                }
        else:
            print(
                "ERROR: Mapping file is incomplete. "
                f"Missing sections: {missing_sections}. "
                "Run evaluate_sections.py without --dry-run to generate real mapping."
            )
            return 1

    tickers = sorted({row["ticker"] for row in reports})
    mixed_analysis_root = run_dir / args.mixed_analysis_dirname
    mixed_report_root = run_dir / args.mixed_report_dirname
    logs_root = mixed_report_root / "logs"
    summary_csv_path = run_dir / args.summary_csv_name
    summary_md_path = run_dir / args.summary_md_name
    manifest_path = run_dir / args.manifest_json_name
    mixed_eval_path = run_dir / args.mixed_eval_json_name
    scoreboard_csv_path = run_dir / args.scoreboard_csv_name
    scoreboard_md_path = run_dir / args.scoreboard_md_name

    evaluation_summary = {}
    evaluation_summary_path = run_dir / "evaluation_summary.json"
    winners_by_ticker = {}
    if evaluation_summary_path.exists():
        evaluation_summary = _read_json(evaluation_summary_path)
        winners_by_ticker = ((evaluation_summary.get("final") or {}).get("winners_by_ticker") or {})

    manifest_rows = []
    summary_rows = []
    build_failures = []

    print(f"Run dir: {run_dir}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Sections: {', '.join(args.sections)}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Skip report build: {args.skip_report_build}")
    print(f"Evaluate mixed: {args.evaluate_mixed}")
    print(f"Mixed eval stage2: {args.mixed_eval_stage2}")

    for ticker in tickers:
        ticker_analysis_dir = mixed_analysis_root / ticker
        ticker_report_dir = mixed_report_root / ticker
        ticker_analysis_dir.mkdir(parents=True, exist_ok=True)
        ticker_report_dir.mkdir(parents=True, exist_ok=True)

        first_selected_row = None
        for section in args.sections:
            selected = global_map[section]
            model_tag = selected["selected_model_tag"]
            source_row = by_ticker_model.get((ticker, model_tag))
            if source_row is None:
                build_failures.append(
                    {"ticker": ticker, "section": section, "error": f"missing_report_for_model_tag:{model_tag}"}
                )
                continue

            if first_selected_row is None:
                first_selected_row = source_row

            src_analysis_dir = _get_analysis_dir(source_row, run_dir=run_dir)
            src_text = src_analysis_dir / f"{section}.txt"
            dst_text = ticker_analysis_dir / f"{section}.txt"
            if not src_text.exists():
                build_failures.append(
                    {"ticker": ticker, "section": section, "error": f"missing_section_file:{src_text}"}
                )
                continue
            if not args.dry_run:
                shutil.copy2(src_text, dst_text)

            summary_rows.append(
                {
                    "run_id": args.run_id,
                    "ticker": ticker,
                    "section": section,
                    "selected_model_tag": model_tag,
                    "selected_model_provider": selected.get("selected_model_provider"),
                    "selected_model_name": selected.get("selected_model_name"),
                    "mean_score": selected.get("mean_score"),
                    "stddev_score": selected.get("stddev_score"),
                    "sample_count": selected.get("sample_count"),
                    "source_path": str(src_text),
                    "mixed_path": str(dst_text),
                }
            )

        if first_selected_row is None:
            build_failures.append({"ticker": ticker, "section": "*", "error": "no_selected_rows_for_ticker"})
            continue

        src_analysis_ref = _get_analysis_dir(first_selected_row, run_dir=run_dir)
        static_files = [
            "financial_metrics_and_forecasts.csv",
            "ratios_raw_data.csv",
            "peer_ebitda_comparison.csv",
            "peer_ev_ebitda_comparison.csv",
            "retail_sentiment.json",
            "sensitivity_analysis.json",
            "catalyst_analysis.json",
            "enhanced_news.json",
        ]
        copied_static = []
        missing_static = []
        for name in static_files:
            src = src_analysis_ref / name
            dst = ticker_analysis_dir / name
            if src.exists():
                if not args.dry_run:
                    shutil.copy2(src, dst)
                copied_static.append(name)
            else:
                missing_static.append(name)

        report_cmd = [
            args.python_executable,
            str((PROJECT_ROOT / "finrobot_equity" / "core" / "src" / "create_equity_report.py").resolve()),
            "--company-ticker",
            ticker,
            "--company-name",
            first_selected_row["company_name"],
            "--analysis-csv",
            str(ticker_analysis_dir / "financial_metrics_and_forecasts.csv"),
            "--ratios-csv",
            str(ticker_analysis_dir / "ratios_raw_data.csv"),
            "--tagline-file",
            str(ticker_analysis_dir / "tagline.txt"),
            "--company-overview-file",
            str(ticker_analysis_dir / "company_overview.txt"),
            "--investment-overview-file",
            str(ticker_analysis_dir / "investment_overview.txt"),
            "--valuation-overview-file",
            str(ticker_analysis_dir / "valuation_overview.txt"),
            "--risks-file",
            str(ticker_analysis_dir / "risks.txt"),
            "--competitor-analysis-file",
            str(ticker_analysis_dir / "competitor_analysis.txt"),
            "--major-takeaways-file",
            str(ticker_analysis_dir / "major_takeaways.txt"),
            "--news-summary-file",
            str(ticker_analysis_dir / "news_summary.txt"),
            "--peer-ev-ebitda-csv",
            str(ticker_analysis_dir / "peer_ev_ebitda_comparison.csv"),
            "--peer-ebitda-csv",
            str(ticker_analysis_dir / "peer_ebitda_comparison.csv"),
            "--output-dir",
            str(ticker_report_dir),
            "--config-file",
            str((PROJECT_ROOT / args.config_file).resolve()),
        ]
        report_cmd.append("--enable-valuation-analysis")
        if args.skip_auto_fetch:
            report_cmd.append("--skip-auto-fetch")

        rc = None
        if not args.dry_run and not args.skip_report_build:
            rc = _run_command(
                command=report_cmd,
                cwd=PROJECT_ROOT,
                stdout_path=logs_root / ticker / "mixed_report_stdout.log",
                stderr_path=logs_root / ticker / "mixed_report_stderr.log",
            )
            if rc != 0:
                build_failures.append({"ticker": ticker, "section": "*", "error": f"create_equity_report_return_code={rc}"})

        manifest_rows.append(
            {
                "ticker": ticker,
                "company_name": first_selected_row["company_name"],
                "sections": [row for row in summary_rows if row["ticker"] == ticker],
                "static_reference_model_tag": first_selected_row["model_tag"],
                "copied_static_files": copied_static,
                "missing_static_files": missing_static,
                "mixed_analysis_dir": str(ticker_analysis_dir),
                "mixed_report_dir": str(ticker_report_dir),
                "report_build_return_code": rc,
                "report_build_command": report_cmd,
            }
        )

    manifest_payload = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "mapping_file": str(mapping_path),
        "sections": args.sections,
        "dry_run": args.dry_run,
        "skip_report_build": args.skip_report_build,
        "evaluate_mixed": args.evaluate_mixed,
        "mixed_eval_stage2": args.mixed_eval_stage2,
        "rows": manifest_rows,
        "failures": build_failures,
    }

    mixed_eval_summary = None
    mixed_eval_error = None
    if not args.dry_run and args.evaluate_mixed:
        try:
            mixed_eval_summary = _evaluate_mixed_reports(
                args=args,
                run_dir=run_dir,
                mixed_report_root=mixed_report_root,
                reports=reports,
                evaluation_summary=evaluation_summary,
            )
            _write_json(mixed_eval_path, mixed_eval_summary)
        except Exception as e:
            mixed_eval_error = str(e)
            mixed_eval_summary = {
                "enabled": True,
                "generated_at_utc": _utc_now_iso(),
                "error": mixed_eval_error,
            }
            _write_json(mixed_eval_path, mixed_eval_summary)

    if not args.dry_run:
        _write_json(manifest_path, manifest_payload)
        fields = [
            "run_id",
            "ticker",
            "section",
            "selected_model_tag",
            "selected_model_provider",
            "selected_model_name",
            "mean_score",
            "stddev_score",
            "sample_count",
            "source_path",
            "mixed_path",
        ]
        _write_csv(summary_csv_path, summary_rows, fields)
        summary_md = _build_summary_markdown(
            run_id=args.run_id,
            run_dir=run_dir,
            mapping=mapping_payload,
            winners_by_ticker=winners_by_ticker,
            mixed_rows=summary_rows,
        )
        summary_md_path.write_text(summary_md, encoding="utf-8")

        scoreboard_rows = _build_final_scoreboard_rows(
            reports=reports,
            base_eval_summary=evaluation_summary,
            mixed_eval_summary=mixed_eval_summary if args.evaluate_mixed else None,
        )
        model_columns = []
        seen_cols = set()
        for report in reports:
            tag = report["model_tag"]
            if tag not in seen_cols:
                seen_cols.add(tag)
                model_columns.append(tag)
        scoreboard_fields = ["ticker"] + model_columns
        if args.evaluate_mixed:
            scoreboard_fields += ["mixed", "mixed_judge_origin"]
        scoreboard_fields += ["best_baseline_model_tag", "best_baseline_score"]
        _write_csv(scoreboard_csv_path, scoreboard_rows, scoreboard_fields)
        scoreboard_md = _build_scoreboard_markdown(
            rows=scoreboard_rows,
            model_columns=model_columns,
            mixed_enabled=bool(args.evaluate_mixed),
        )
        scoreboard_md_path.write_text(scoreboard_md, encoding="utf-8")

    print("")
    print("Mixed report build complete.")
    print(f"Tickers processed: {len(tickers)}")
    print(f"Section rows: {len(summary_rows)}")
    print(f"Build failures: {len(build_failures)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary MD:  {summary_md_path}")
    if not args.dry_run:
        print(f"Scoreboard CSV: {scoreboard_csv_path}")
        print(f"Scoreboard MD:  {scoreboard_md_path}")
    if args.evaluate_mixed and not args.dry_run:
        print(f"Mixed eval JSON: {mixed_eval_path}")
        if mixed_eval_error:
            print(f"Mixed eval warning: {mixed_eval_error}")

    if build_failures and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
