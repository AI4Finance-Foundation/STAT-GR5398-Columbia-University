#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
CORE_SRC = PROJECT_ROOT / "finrobot_equity" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from modules.common_utils import load_config
from modules.llm_gateway import call_llm, load_llm_settings


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

DEFAULT_STATIC_FILES: List[str] = [
    "financial_metrics_and_forecasts.csv",
    "ratios_raw_data.csv",
    "peer_ebitda_comparison.csv",
    "peer_ev_ebitda_comparison.csv",
    "retail_sentiment.json",
    "sensitivity_analysis.json",
    "catalyst_analysis.json",
    "enhanced_news.json",
]


ROLE_CRITIC_PROMPT = """You are a senior equity research critic and reviser.

Task:
Revise ONE section from an analyst draft report for {ticker} ({company_name}).
Section name: {section_name}

Requirements:
1. Keep factual consistency with the provided draft; do not invent financial figures.
2. Improve investment-thesis coherence and decision usefulness.
3. Strengthen risk/catalyst clarity where relevant.
4. Keep the section concise and analyst-facing.
5. Return plain text only. No markdown headers, no bullets unless the draft already uses them naturally.

Context (for consistency):
{context_text}

Current section draft:
{section_text}
"""


@dataclass(frozen=True)
class RoleCombo:
    key: str
    analyst_tag: str
    critic_tag: str


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


def _empty_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.has_section("API_KEYS"):
        cfg.add_section("API_KEYS")
    return cfg


def _load_config_lenient(config_path: Path) -> tuple[configparser.ConfigParser, Optional[str]]:
    if not config_path.exists():
        return _empty_config(), f"config_not_found:{config_path}"

    try:
        cfg = load_config(str(config_path))
        if not cfg.has_section("API_KEYS"):
            cfg.add_section("API_KEYS")
        return cfg, None
    except configparser.Error as e:
        return _empty_config(), f"config_parse_error:{type(e).__name__}:{e}"


def _prepare_runtime_config(
    *,
    run_dir: Path,
    cfg: configparser.ConfigParser,
    dry_run: bool,
) -> Path:
    runtime_dir = run_dir / "_runtime"
    runtime_path = runtime_dir / "role_mixed_runtime_config.ini"
    if dry_run:
        return runtime_path
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("w", encoding="utf-8") as f:
        cfg.write(f)
    return runtime_path


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
        description="Run role-mixed analyst+critic experiments and produce unified scoreboard."
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="./output")
    parser.add_argument("--storage-subdir", type=str, default="storage")
    parser.add_argument("--config-file", type=str, default="finrobot_equity/core/config/config.ini")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--skip-auto-fetch", action="store_true")
    parser.add_argument("--skip-role-report-build", action="store_true")
    parser.add_argument("--skip-role-eval", action="store_true")
    parser.add_argument("--disable-role-eval-stage2", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--role-mixed-dirname", type=str, default="role_mixed")
    parser.add_argument("--role-mixed-manifest-name", type=str, default="role_mixed_manifest.json")
    parser.add_argument("--role-mixed-eval-json-name", type=str, default="evaluation_role_mixed_summary.json")
    parser.add_argument("--scoreboard-csv-name", type=str, default="final_scoreboard.csv")
    parser.add_argument("--scoreboard-md-name", type=str, default="final_scoreboard.md")
    parser.add_argument("--eval-stage1-provider", type=str, default="openai")
    parser.add_argument("--eval-stage1-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval-stage2-provider", type=str, default="openai")
    parser.add_argument("--eval-stage2-model", type=str, default="gpt-5-nano")
    parser.add_argument("--eval-stage2-margin-threshold", type=float, default=4.0)
    parser.add_argument("--eval-stage2-blend-alpha", type=float, default=0.6)
    parser.add_argument("--eval-max-chars", type=int, default=12000)
    parser.add_argument("--critic-sections", nargs="*", default=["investment_overview", "risks", "major_takeaways"])
    parser.add_argument("--role-mixed-a-analyst-tag", type=str, default="claude_claude-opus-4-6")
    parser.add_argument("--role-mixed-a-critic-tag", type=str, default="gemini_gemini-3.1-pro-preview")
    parser.add_argument("--role-mixed-b-analyst-tag", type=str, default="gemini_gemini-3.1-pro-preview")
    parser.add_argument("--role-mixed-b-critic-tag", type=str, default="claude_claude-opus-4-6")
    return parser.parse_args()


def _load_reports(run_dir: Path) -> tuple[List[dict], Dict[tuple, dict], Dict[str, dict], List[str]]:
    reports_index_path = run_dir / "reports_index.json"
    payload = _read_json(reports_index_path)
    reports = payload.get("reports") or []
    by_ticker_model: Dict[tuple, dict] = {}
    model_info_by_tag: Dict[str, dict] = {}
    tickers = sorted({row["ticker"] for row in reports})
    for row in reports:
        by_ticker_model[(row["ticker"], row["model_tag"])] = row
        model_info_by_tag.setdefault(
            row["model_tag"],
            {
                "model_provider": row["model_provider"],
                "model_name": row["model_name"],
            },
        )
    return reports, by_ticker_model, model_info_by_tag, tickers


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_role_inputs(
    *,
    src_analysis_dir: Path,
    dst_analysis_dir: Path,
    sections: List[str],
    static_files: List[str],
    dry_run: bool,
) -> tuple[List[str], List[str]]:
    missing_sections: List[str] = []
    missing_static: List[str] = []

    for section in sections:
        src = src_analysis_dir / f"{section}.txt"
        dst = dst_analysis_dir / f"{section}.txt"
        if not src.exists():
            missing_sections.append(section)
            continue
        if not dry_run:
            dst_analysis_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for name in static_files:
        src = src_analysis_dir / name
        dst = dst_analysis_dir / name
        if not src.exists():
            missing_static.append(name)
            continue
        if not dry_run:
            dst_analysis_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return missing_sections, missing_static


def _make_critic_context(analysis_dir: Path) -> str:
    context_sections = ["tagline", "investment_overview", "valuation_overview", "risks", "major_takeaways"]
    parts = []
    for name in context_sections:
        path = analysis_dir / f"{name}.txt"
        if not path.exists():
            continue
        text = _read_text(path).strip()
        if not text:
            continue
        if len(text) > 1800:
            text = text[:1800] + "\n[TRUNCATED]"
        parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts) if parts else "No additional section context available."


def _critic_revise_section(
    *,
    llm_settings,
    ticker: str,
    company_name: str,
    section_name: str,
    section_text: str,
    context_text: str,
) -> str:
    prompt = ROLE_CRITIC_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        section_name=section_name,
        context_text=context_text,
        section_text=section_text,
    )
    revised = call_llm(
        settings=llm_settings,
        instructions="Return plain text only.",
        prompt=prompt,
        max_output_tokens=1600,
        temperature=0.2,
    )
    out = (revised or "").strip()
    if not out:
        return section_text
    return out


def _build_role_mixed_reports(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    runtime_config: configparser.ConfigParser,
    runtime_config_path: Path,
    combos: List[RoleCombo],
    by_ticker_model: Dict[tuple, dict],
    model_info_by_tag: Dict[str, dict],
    tickers: List[str],
) -> tuple[dict, List[dict]]:
    role_root = run_dir / args.role_mixed_dirname
    logs_root = role_root / "logs"

    critic_settings_by_tag = {}
    critic_settings_errors = {}
    if not args.dry_run:
        for combo in combos:
            critic_info = model_info_by_tag[combo.critic_tag]
            try:
                critic_settings_by_tag[combo.critic_tag] = load_llm_settings(
                    runtime_config,
                    provider=critic_info["model_provider"],
                    model=critic_info["model_name"],
                )
            except Exception as e:
                critic_settings_errors[combo.critic_tag] = str(e)

    build_rows = []
    failures = []
    role_reports = []

    for combo in combos:
        for ticker in tickers:
            base_row = by_ticker_model.get((ticker, combo.analyst_tag))
            if base_row is None:
                failures.append(
                    {
                        "combo": combo.key,
                        "ticker": ticker,
                        "error": f"analyst_model_not_found:{combo.analyst_tag}",
                    }
                )
                continue

            combo_root = role_root / combo.key / ticker
            analysis_dir = combo_root / "analysis"
            report_dir = combo_root / "report"
            log_dir = combo_root / "logs"
            if not args.dry_run:
                analysis_dir.mkdir(parents=True, exist_ok=True)
                report_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)

            src_analysis_dir = _get_analysis_dir(base_row, run_dir=run_dir)
            missing_sections, missing_static = _copy_role_inputs(
                src_analysis_dir=src_analysis_dir,
                dst_analysis_dir=analysis_dir,
                sections=DEFAULT_SECTIONS,
                static_files=DEFAULT_STATIC_FILES,
                dry_run=args.dry_run,
            )

            critic_diff = {
                "generated_at_utc": _utc_now_iso(),
                "combo": combo.key,
                "ticker": ticker,
                "analyst_model_tag": combo.analyst_tag,
                "critic_model_tag": combo.critic_tag,
                "sections": [],
                "errors": [],
            }

            critic_sections = [s for s in args.critic_sections if s in DEFAULT_SECTIONS]
            context_text = ""
            if not args.dry_run:
                context_text = _make_critic_context(analysis_dir)

            for section in critic_sections:
                dst_section_path = analysis_dir / f"{section}.txt"
                if not dst_section_path.exists():
                    critic_diff["errors"].append(f"missing_section_for_critic:{section}")
                    continue
                critic_settings = critic_settings_by_tag.get(combo.critic_tag)
                if not args.dry_run and critic_settings is None:
                    critic_diff["errors"].append(
                        f"critic_settings_unavailable:{combo.critic_tag}:{critic_settings_errors.get(combo.critic_tag, 'unknown_error')}"
                    )
                    continue
                if args.dry_run:
                    critic_diff["sections"].append(
                        {
                            "section": section,
                            "status": "dry_run",
                        }
                    )
                    continue

                original_text = _read_text(dst_section_path)
                try:
                    revised_text = _critic_revise_section(
                        llm_settings=critic_settings,
                        ticker=ticker,
                        company_name=base_row["company_name"],
                        section_name=section,
                        section_text=original_text,
                        context_text=context_text,
                    )
                    _write_text(dst_section_path, revised_text)
                    critic_diff["sections"].append(
                        {
                            "section": section,
                            "status": "success",
                            "original_chars": len(original_text),
                            "revised_chars": len(revised_text),
                        }
                    )
                except Exception as e:
                    critic_diff["sections"].append(
                        {
                            "section": section,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    critic_diff["errors"].append(f"critic_failed:{section}:{e}")

            if not args.dry_run:
                _write_json(analysis_dir / "critic_diff_summary.json", critic_diff)

            report_cmd = [
                args.python_executable,
                str((PROJECT_ROOT / "finrobot_equity" / "core" / "src" / "create_equity_report.py").resolve()),
                "--company-ticker",
                ticker,
                "--company-name",
                base_row["company_name"],
                "--analysis-csv",
                str(analysis_dir / "financial_metrics_and_forecasts.csv"),
                "--ratios-csv",
                str(analysis_dir / "ratios_raw_data.csv"),
                "--tagline-file",
                str(analysis_dir / "tagline.txt"),
                "--company-overview-file",
                str(analysis_dir / "company_overview.txt"),
                "--investment-overview-file",
                str(analysis_dir / "investment_overview.txt"),
                "--valuation-overview-file",
                str(analysis_dir / "valuation_overview.txt"),
                "--risks-file",
                str(analysis_dir / "risks.txt"),
                "--competitor-analysis-file",
                str(analysis_dir / "competitor_analysis.txt"),
                "--major-takeaways-file",
                str(analysis_dir / "major_takeaways.txt"),
                "--news-summary-file",
                str(analysis_dir / "news_summary.txt"),
                "--peer-ev-ebitda-csv",
                str(analysis_dir / "peer_ev_ebitda_comparison.csv"),
                "--peer-ebitda-csv",
                str(analysis_dir / "peer_ebitda_comparison.csv"),
                "--output-dir",
                str(report_dir),
                "--config-file",
                str(runtime_config_path),
                "--enable-valuation-analysis",
            ]
            if args.skip_auto_fetch:
                report_cmd.append("--skip-auto-fetch")

            rc = None
            if not args.dry_run and not args.skip_role_report_build:
                rc = _run_command(
                    command=report_cmd,
                    cwd=PROJECT_ROOT,
                    stdout_path=logs_root / combo.key / ticker / "role_report_stdout.log",
                    stderr_path=logs_root / combo.key / ticker / "role_report_stderr.log",
                )

            report_path = report_dir / f"Professional_Equity_Report_{ticker}.html"
            role_reports.append(
                {
                    "combo": combo.key,
                    "ticker": ticker,
                    "company_name": base_row["company_name"],
                    "analyst_model_tag": combo.analyst_tag,
                    "critic_model_tag": combo.critic_tag,
                    "report_path": str(report_path),
                    "status": "success" if (args.dry_run or rc == 0) else "failed",
                }
            )

            row = {
                "combo": combo.key,
                "ticker": ticker,
                "company_name": base_row["company_name"],
                "analyst_model_tag": combo.analyst_tag,
                "critic_model_tag": combo.critic_tag,
                "analysis_dir": str(analysis_dir),
                "report_dir": str(report_dir),
                "report_path": str(report_path),
                "report_build_return_code": rc,
                "missing_sections": missing_sections,
                "missing_static_files": missing_static,
                "critic_errors": critic_diff["errors"],
                "report_build_command": report_cmd,
            }
            build_rows.append(row)

            if not args.dry_run and rc not in (None, 0):
                failures.append(
                    {
                        "combo": combo.key,
                        "ticker": ticker,
                        "error": f"create_equity_report_return_code={rc}",
                    }
                )

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "role_mixed_dir": str(role_root),
        "runtime_config_path": str(runtime_config_path),
        "dry_run": args.dry_run,
        "skip_role_report_build": args.skip_role_report_build,
        "combos": [combo.__dict__ for combo in combos],
        "rows": build_rows,
        "failures": failures,
        "critic_settings_errors": critic_settings_errors,
    }
    return manifest, role_reports


def _evaluate_role_mixed(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    runtime_config_path: Path,
    role_reports: List[dict],
    combos: List[RoleCombo],
) -> dict:
    eval_script = (PROJECT_ROOT / "finrobot_equity" / "core" / "src" / "report_evaluate.py").resolve()
    stage1_records = []
    stage2_records = []
    final_map: Dict[tuple, dict] = {}

    combo_by_key = {c.key: c for c in combos}
    by_ticker_combo = {}
    for row in role_reports:
        by_ticker_combo[(row["ticker"], row["combo"])] = row

    tickers = sorted({row["ticker"] for row in role_reports})

    # Stage1
    for row in role_reports:
        ticker = row["ticker"]
        combo_key = row["combo"]
        stage1_out = run_dir / "evaluation_role_mixed" / "stage1" / ticker / f"{combo_key}.json"
        stage1_out.parent.mkdir(parents=True, exist_ok=True)
        stage1_stdout = run_dir / "evaluation_role_mixed" / "logs" / ticker / f"{combo_key}_stage1_stdout.log"
        stage1_stderr = run_dir / "evaluation_role_mixed" / "logs" / ticker / f"{combo_key}_stage1_stderr.log"

        stage1_cmd = [
            args.python_executable,
            str(eval_script),
            "--report-a",
            row["report_path"],
            "--ticker",
            ticker,
            "--config-file",
            str(runtime_config_path),
            "--llm-provider",
            args.eval_stage1_provider,
            "--llm-model",
            args.eval_stage1_model,
            "--evaluated-model-provider",
            "role_mixed",
            "--evaluated-model-name",
            combo_key,
            "--evaluated-model-tag",
            combo_key,
            "--max-chars",
            str(args.eval_max_chars),
            "--output-file",
            str(stage1_out),
        ]

        rc = _run_command(stage1_cmd, cwd=PROJECT_ROOT, stdout_path=stage1_stdout, stderr_path=stage1_stderr)
        payload = _read_json_if_exists(stage1_out)
        score = None
        if payload:
            score = _to_float(payload.get("combined_score"))
            if score is None:
                score = _to_float((payload.get("llm_evaluation") or {}).get("overall_score"))

        rec = {
            "ticker": ticker,
            "company_name": row["company_name"],
            "model_provider": "role_mixed",
            "model_name": combo_key,
            "model_tag": combo_key,
            "judge_provider": args.eval_stage1_provider,
            "judge_model": args.eval_stage1_model,
            "report_path": row["report_path"],
            "stage1_output": str(stage1_out),
            "status": "success" if (rc == 0 and payload is not None and score is not None) else "failed",
            "return_code": rc,
            "stage1_score": score,
            "judge_origin": "stage1",
        }
        stage1_records.append(rec)
        final_map[(ticker, combo_key)] = {
            "ticker": ticker,
            "company_name": row["company_name"],
            "model_provider": "role_mixed",
            "model_name": combo_key,
            "model_tag": combo_key,
            "report_path": row["report_path"],
            "stage1_score": score,
            "final_score": score,
            "judge_origin": "stage1" if rec["status"] == "success" else "failed",
            "stage2_pairs": [],
            "error": None if rec["status"] == "success" else "stage1_failed",
        }

    # Stage2
    if not args.disable_role_eval_stage2:
        if len(combos) >= 2:
            left_key = combos[0].key
            right_key = combos[1].key
            for ticker in tickers:
                left_stage1 = next((r for r in stage1_records if r["ticker"] == ticker and r["model_tag"] == left_key), None)
                right_stage1 = next((r for r in stage1_records if r["ticker"] == ticker and r["model_tag"] == right_key), None)
                left_report = by_ticker_combo.get((ticker, left_key))
                right_report = by_ticker_combo.get((ticker, right_key))
                if left_stage1 is None or right_stage1 is None or left_report is None or right_report is None:
                    stage2_records.append(
                        {
                            "ticker": ticker,
                            "pair_index": 1,
                            "left_model_tag": left_key,
                            "right_model_tag": right_key,
                            "status": "skipped",
                            "reason": "missing_stage1_or_report",
                        }
                    )
                    continue
                if left_stage1["status"] != "success" or right_stage1["status"] != "success":
                    stage2_records.append(
                        {
                            "ticker": ticker,
                            "pair_index": 1,
                            "left_model_tag": left_key,
                            "right_model_tag": right_key,
                            "status": "skipped",
                            "reason": "stage1_failed",
                        }
                    )
                    continue

                margin = abs((left_stage1["stage1_score"] or 0.0) - (right_stage1["stage1_score"] or 0.0))
                if margin > args.eval_stage2_margin_threshold:
                    stage2_records.append(
                        {
                            "ticker": ticker,
                            "pair_index": 1,
                            "left_model_tag": left_key,
                            "right_model_tag": right_key,
                            "status": "skipped",
                            "reason": "margin_above_threshold",
                            "margin_stage1": round(margin, 4),
                        }
                    )
                    continue

                stage2_out = run_dir / "evaluation_role_mixed" / "stage2" / ticker / f"{left_key}_vs_{right_key}.json"
                stage2_out.parent.mkdir(parents=True, exist_ok=True)
                stage2_stdout = run_dir / "evaluation_role_mixed" / "logs" / ticker / f"{left_key}_vs_{right_key}_stage2_stdout.log"
                stage2_stderr = run_dir / "evaluation_role_mixed" / "logs" / ticker / f"{left_key}_vs_{right_key}_stage2_stderr.log"
                stage2_cmd = [
                    args.python_executable,
                    str(eval_script),
                    "--report-a",
                    left_report["report_path"],
                    "--report-b",
                    right_report["report_path"],
                    "--ticker",
                    ticker,
                    "--config-file",
                    str(runtime_config_path),
                    "--llm-provider",
                    args.eval_stage2_provider,
                    "--llm-model",
                    args.eval_stage2_model,
                    "--evaluated-model-a-provider",
                    "role_mixed",
                    "--evaluated-model-a-name",
                    left_key,
                    "--evaluated-model-a-tag",
                    left_key,
                    "--evaluated-model-b-provider",
                    "role_mixed",
                    "--evaluated-model-b-name",
                    right_key,
                    "--evaluated-model-b-tag",
                    right_key,
                    "--max-chars",
                    str(args.eval_max_chars),
                    "--output-file",
                    str(stage2_out),
                ]
                rc = _run_command(stage2_cmd, cwd=PROJECT_ROOT, stdout_path=stage2_stdout, stderr_path=stage2_stderr)
                payload = _read_json_if_exists(stage2_out)
                winner = None
                score_a = None
                score_b = None
                if payload:
                    comp = payload.get("llm_comparison") or {}
                    winner = comp.get("winner")
                    score_a = _to_float(comp.get("score_a"))
                    score_b = _to_float(comp.get("score_b"))

                row_stage2 = {
                    "ticker": ticker,
                    "pair_index": 1,
                    "margin_stage1": round(margin, 4),
                    "left_model_tag": left_key,
                    "left_model_provider": "role_mixed",
                    "left_model_name": left_key,
                    "right_model_tag": right_key,
                    "right_model_provider": "role_mixed",
                    "right_model_name": right_key,
                    "left_report_path": left_report["report_path"],
                    "right_report_path": right_report["report_path"],
                    "stage2_output": str(stage2_out),
                    "status": "success" if (rc == 0 and payload is not None) else "failed",
                    "return_code": rc,
                    "judge_provider": args.eval_stage2_provider,
                    "judge_model": args.eval_stage2_model,
                    "winner": winner,
                    "score_a": score_a,
                    "score_b": score_b,
                }
                stage2_records.append(row_stage2)

                if row_stage2["status"] == "success":
                    alpha = args.eval_stage2_blend_alpha
                    beta = 1.0 - alpha
                    left_final = final_map.get((ticker, left_key))
                    right_final = final_map.get((ticker, right_key))
                    if left_final and left_final["stage1_score"] is not None and score_a is not None:
                        left_final["final_score"] = round(left_final["stage1_score"] * alpha + score_a * beta, 2)
                        left_final["judge_origin"] = "merged"
                    if right_final and right_final["stage1_score"] is not None and score_b is not None:
                        right_final["final_score"] = round(right_final["stage1_score"] * alpha + score_b * beta, 2)
                        right_final["judge_origin"] = "merged"
                    if left_final:
                        left_final["stage2_pairs"].append(row_stage2)
                    if right_final:
                        right_final["stage2_pairs"].append(row_stage2)

    final_records = list(final_map.values())
    winners_by_ticker = {}
    for ticker in sorted({row["ticker"] for row in final_records}):
        rows = [r for r in final_records if r["ticker"] == ticker]
        rows_sorted = sorted(rows, key=lambda x: (x["final_score"] is not None, x["final_score"]), reverse=True)
        for i, row in enumerate(rows_sorted, start=1):
            row["final_rank"] = i
        if rows_sorted:
            top = rows_sorted[0]
            winners_by_ticker[ticker] = {
                "model_tag": top["model_tag"],
                "model_provider": top["model_provider"],
                "model_name": top["model_name"],
                "final_score": top["final_score"],
                "judge_origin": top["judge_origin"],
            }

    return {
        "enabled": True,
        "generated_at_utc": _utc_now_iso(),
        "stage1": {
            "provider": args.eval_stage1_provider,
            "model": args.eval_stage1_model,
            "records": stage1_records,
            "success_count": sum(1 for r in stage1_records if r["status"] == "success"),
            "failed_count": sum(1 for r in stage1_records if r["status"] == "failed"),
        },
        "stage2": {
            "enabled": not args.disable_role_eval_stage2,
            "provider": args.eval_stage2_provider if not args.disable_role_eval_stage2 else None,
            "model": args.eval_stage2_model if not args.disable_role_eval_stage2 else None,
            "margin_threshold": args.eval_stage2_margin_threshold if not args.disable_role_eval_stage2 else None,
            "blend_alpha": args.eval_stage2_blend_alpha if not args.disable_role_eval_stage2 else None,
            "records": stage2_records,
            "success_count": sum(1 for r in stage2_records if r.get("status") == "success"),
            "failed_count": sum(1 for r in stage2_records if r.get("status") == "failed"),
            "skipped_count": sum(1 for r in stage2_records if r.get("status") == "skipped"),
        },
        "final": {
            "records": sorted(
                final_records,
                key=lambda x: (x["ticker"], x["model_tag"]),
            ),
            "winners_by_ticker": winners_by_ticker,
        },
    }


def _build_scoreboard_rows(
    *,
    reports: List[dict],
    baseline_eval: dict,
    mixed_eval: Optional[dict],
    role_eval: Optional[dict],
) -> tuple[List[dict], List[str]]:
    model_columns = []
    seen_cols = set()
    for report in reports:
        tag = report["model_tag"]
        if tag not in seen_cols:
            seen_cols.add(tag)
            model_columns.append(tag)

    baseline_scores = {}
    for row in ((baseline_eval.get("final") or {}).get("records") or []):
        baseline_scores[(row.get("ticker"), row.get("model_tag"))] = _to_float(row.get("final_score"))

    mixed_scores = {}
    mixed_origins = {}
    if mixed_eval:
        for row in ((mixed_eval.get("final") or {}).get("records") or []):
            mixed_scores[row.get("ticker")] = _to_float(row.get("final_score"))
            mixed_origins[row.get("ticker")] = row.get("judge_origin")

    role_scores = {}
    role_origins = {}
    if role_eval:
        for row in ((role_eval.get("final") or {}).get("records") or []):
            role_scores[(row.get("ticker"), row.get("model_tag"))] = _to_float(row.get("final_score"))
            role_origins[(row.get("ticker"), row.get("model_tag"))] = row.get("judge_origin")

    tickers = sorted({r["ticker"] for r in reports})
    rows = []
    for ticker in tickers:
        row = {"ticker": ticker}
        best_base_tag = None
        best_base_score = None
        for col in model_columns:
            v = baseline_scores.get((ticker, col))
            row[col] = v
            if v is not None and (best_base_score is None or v > best_base_score):
                best_base_score = v
                best_base_tag = col

        if mixed_eval:
            row["mixed"] = mixed_scores.get(ticker)
            row["mixed_judge_origin"] = mixed_origins.get(ticker)

        if role_eval:
            a_score = role_scores.get((ticker, "role_mixed_a"))
            b_score = role_scores.get((ticker, "role_mixed_b"))
            row["role_mixed_a_score"] = a_score
            row["role_mixed_a_judge_origin"] = role_origins.get((ticker, "role_mixed_a"))
            row["role_mixed_b_score"] = b_score
            row["role_mixed_b_judge_origin"] = role_origins.get((ticker, "role_mixed_b"))
            if a_score is None and b_score is None:
                row["role_mixed_best_tag"] = None
                row["role_mixed_best_score"] = None
            elif b_score is None or (a_score is not None and a_score >= b_score):
                row["role_mixed_best_tag"] = "role_mixed_a"
                row["role_mixed_best_score"] = a_score
            else:
                row["role_mixed_best_tag"] = "role_mixed_b"
                row["role_mixed_best_score"] = b_score

        row["best_baseline_model_tag"] = best_base_tag
        row["best_baseline_score"] = best_base_score
        rows.append(row)
    return rows, model_columns


def _build_scoreboard_markdown(rows: List[dict], headers: List[str]) -> str:
    lines = []
    lines.append("# Final Scoreboard")
    lines.append("")
    lines.append(f"- Generated at UTC: {_utc_now_iso()}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h)
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

    reports, by_ticker_model, model_info_by_tag, tickers = _load_reports(run_dir)
    if not reports:
        print("ERROR: No reports found in reports_index.json")
        return 1

    config_path = (PROJECT_ROOT / args.config_file).resolve()
    runtime_config, config_warning = _load_config_lenient(config_path)
    runtime_config_path = _prepare_runtime_config(
        run_dir=run_dir,
        cfg=runtime_config,
        dry_run=args.dry_run,
    )
    if config_warning:
        print(f"WARNING: {config_warning}")
        print("WARNING: Falling back to runtime config with [API_KEYS] only; API keys will be read from environment if available.")

    combos = [
        RoleCombo(
            key="role_mixed_a",
            analyst_tag=args.role_mixed_a_analyst_tag,
            critic_tag=args.role_mixed_a_critic_tag,
        ),
        RoleCombo(
            key="role_mixed_b",
            analyst_tag=args.role_mixed_b_analyst_tag,
            critic_tag=args.role_mixed_b_critic_tag,
        ),
    ]

    for combo in combos:
        if combo.analyst_tag not in model_info_by_tag:
            print(f"ERROR: analyst model tag not found in run outputs: {combo.analyst_tag}")
            return 1
        if combo.critic_tag not in model_info_by_tag:
            print(f"ERROR: critic model tag not found in run outputs: {combo.critic_tag}")
            return 1

    print(f"Run dir: {run_dir}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Skip role report build: {args.skip_role_report_build}")
    print(f"Skip role eval: {args.skip_role_eval}")
    print(f"Role eval stage2 enabled: {not args.disable_role_eval_stage2}")
    print("Role combos:")
    for combo in combos:
        print(f"  - {combo.key}: analyst={combo.analyst_tag}, critic={combo.critic_tag}")

    role_manifest, role_reports = _build_role_mixed_reports(
        args=args,
        run_dir=run_dir,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        combos=combos,
        by_ticker_model=by_ticker_model,
        model_info_by_tag=model_info_by_tag,
        tickers=tickers,
    )

    role_manifest_path = run_dir / args.role_mixed_manifest_name
    if not args.dry_run:
        _write_json(role_manifest_path, role_manifest)

    role_eval_summary = None
    role_eval_path = run_dir / args.role_mixed_eval_json_name
    if not args.dry_run and not args.skip_role_eval:
        role_eval_summary = _evaluate_role_mixed(
            args=args,
            run_dir=run_dir,
            runtime_config_path=runtime_config_path,
            role_reports=role_reports,
            combos=combos,
        )
        _write_json(role_eval_path, role_eval_summary)

    baseline_eval = _read_json_if_exists(run_dir / "evaluation_summary.json") or {}
    mixed_eval = _read_json_if_exists(run_dir / "evaluation_mixed_summary.json")

    scoreboard_rows, model_columns = _build_scoreboard_rows(
        reports=reports,
        baseline_eval=baseline_eval,
        mixed_eval=mixed_eval,
        role_eval=role_eval_summary,
    )

    scoreboard_fields = ["ticker"] + model_columns
    if mixed_eval is not None:
        scoreboard_fields += ["mixed", "mixed_judge_origin"]
    if role_eval_summary is not None:
        scoreboard_fields += [
            "role_mixed_a_score",
            "role_mixed_a_judge_origin",
            "role_mixed_b_score",
            "role_mixed_b_judge_origin",
            "role_mixed_best_tag",
            "role_mixed_best_score",
        ]
    scoreboard_fields += ["best_baseline_model_tag", "best_baseline_score"]

    scoreboard_csv = run_dir / args.scoreboard_csv_name
    scoreboard_md = run_dir / args.scoreboard_md_name
    if not args.dry_run:
        _write_csv(scoreboard_csv, scoreboard_rows, scoreboard_fields)
        _write_text(scoreboard_md, _build_scoreboard_markdown(scoreboard_rows, scoreboard_fields))

    print("")
    print("Role-mixed run complete.")
    print(f"Role manifest: {role_manifest_path}")
    if role_eval_summary is not None:
        print(f"Role eval summary: {role_eval_path}")
    print(f"Scoreboard CSV: {scoreboard_csv}")
    print(f"Scoreboard MD:  {scoreboard_md}")
    print(f"Role build failures: {len(role_manifest.get('failures') or [])}")

    if not args.dry_run and (role_manifest.get("failures") or []):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
