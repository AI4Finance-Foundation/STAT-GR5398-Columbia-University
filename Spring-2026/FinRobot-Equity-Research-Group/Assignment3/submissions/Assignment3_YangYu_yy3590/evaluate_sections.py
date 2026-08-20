#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Dict, List, Optional


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

COST_PRIORITY = {
    "gemini": 0,
    "openai": 1,
    "claude": 2,
}

SECTION_EVAL_PROMPT = """You are an independent equity-research section evaluator.

Evaluate one report section for investment-research usefulness.
Focus on section purpose fit, analytical depth, financial grounding, and actionability.
Do not judge by style alone.

Company: {ticker} ({company_name})
Section: {section}
Model tag: {model_tag}

Section text:
{section_text}

Rubric (1-5 each):
1. Section Purpose Fit
2. Analytical Depth
3. Financial Grounding
4. Actionability
5. Clarity and Structure

Return strict JSON only:
{{
  "dimensions": {{
    "purpose_fit": {{"score": 1, "justification": "..."}},
    "analytical_depth": {{"score": 1, "justification": "..."}},
    "financial_grounding": {{"score": 1, "justification": "..."}},
    "actionability": {{"score": 1, "justification": "..."}},
    "clarity_structure": {{"score": 1, "justification": "..."}}
  }},
  "overall_score": 1,
  "top_strength": "...",
  "main_weakness": "..."
}}
overall_score must be from 1 to 100.
"""


@dataclass
class ReportItem:
    ticker: str
    company_name: str
    model_provider: str
    model_name: str
    model_tag: str
    analysis_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def _extract_json(text: str) -> Dict:
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty model response.")
    try:
        return json.loads(content)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError("Could not parse JSON from model response.")
    return json.loads(match.group(0))


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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


def _load_reports(run_dir: Path) -> tuple[str, List[ReportItem]]:
    reports_index_path = run_dir / "reports_index.json"
    if not reports_index_path.exists():
        raise FileNotFoundError(f"Missing reports_index.json: {reports_index_path}")

    payload = _read_json(reports_index_path)
    run_id = payload.get("run_id") or run_dir.name
    reports = payload.get("reports") or []
    items: List[ReportItem] = []
    for row in reports:
        items.append(
            ReportItem(
                ticker=row["ticker"],
                company_name=row["company_name"],
                model_provider=row["model_provider"],
                model_name=row["model_name"],
                model_tag=row["model_tag"],
                analysis_dir=_get_analysis_dir(row, run_dir=run_dir),
            )
        )
    return run_id, items


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each report section across models and build a global section mapping."
    )
    parser.add_argument("--run-id", type=str, required=True, help="Run id under output/storage, e.g. 20260430_220342")
    parser.add_argument("--output-root", type=str, default="./output", help="Output root relative to FinRobot project.")
    parser.add_argument("--storage-subdir", type=str, default="storage", help="Storage subdir under output root.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="finrobot_equity/core/config/config.ini",
        help="Config INI used to load evaluator LLM credentials.",
    )
    parser.add_argument("--llm-provider", type=str, default="openai", help="Evaluator provider.")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="Evaluator model.")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max chars sent per section.")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="Max output tokens from evaluator.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Evaluator temperature.")
    parser.add_argument("--sections", type=str, nargs="*", default=DEFAULT_SECTIONS, help="Sections to evaluate.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap on total section evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and emit scaffolding without LLM calls.")
    parser.add_argument("--section-scores-json-name", type=str, default="section_scores.json")
    parser.add_argument("--section-scores-csv-name", type=str, default="section_scores.csv")
    parser.add_argument("--mapping-json-name", type=str, default="global_section_mapping.json")
    return parser.parse_args()


def _build_aggregates(records: List[dict]) -> dict:
    by_section_model: Dict[str, Dict[str, List[dict]]] = {}
    for row in records:
        if row.get("status") != "success":
            continue
        score = _safe_float(row.get("section_score"))
        if score is None:
            continue
        section = row["section"]
        model_tag = row["model_tag"]
        by_section_model.setdefault(section, {}).setdefault(model_tag, []).append(row)

    section_stats = {}
    global_mapping = {}

    for section, model_map in by_section_model.items():
        ranked = []
        for model_tag, rows in model_map.items():
            scores = [float(r["section_score"]) for r in rows]
            provider = rows[0]["model_provider"]
            mean_score = round(sum(scores) / len(scores), 4)
            stddev = round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0
            ranked.append(
                {
                    "section": section,
                    "model_tag": model_tag,
                    "model_provider": provider,
                    "model_name": rows[0]["model_name"],
                    "mean_score": mean_score,
                    "stddev_score": stddev,
                    "sample_count": len(scores),
                }
            )

        ranked_sorted = sorted(
            ranked,
            key=lambda x: (
                -x["mean_score"],
                x["stddev_score"],
                COST_PRIORITY.get(x["model_provider"], 99),
                x["model_tag"],
            ),
        )
        winner = ranked_sorted[0] if ranked_sorted else None
        section_stats[section] = ranked_sorted
        if winner:
            global_mapping[section] = {
                "selected_model_tag": winner["model_tag"],
                "selected_model_provider": winner["model_provider"],
                "selected_model_name": winner["model_name"],
                "mean_score": winner["mean_score"],
                "stddev_score": winner["stddev_score"],
                "sample_count": winner["sample_count"],
                "selection_rule": "highest_mean_then_lowest_stddev_then_cost_priority",
                "ranked_models": ranked_sorted,
            }

    return {
        "section_model_stats": section_stats,
        "global_section_mapping": global_mapping,
    }


def main() -> int:
    args = _parse_args()

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    run_dir = (output_root / args.storage_subdir / args.run_id).resolve()
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1

    run_id, report_items = _load_reports(run_dir)
    if not report_items:
        print("ERROR: No reports found in reports_index.json")
        return 1

    print(f"Run dir: {run_dir}")
    print(f"Run id: {run_id}")
    print(f"Reports: {len(report_items)}")
    print(f"Sections: {', '.join(args.sections)}")
    print(f"Dry-run: {args.dry_run}")

    llm_settings = None
    if not args.dry_run:
        config_path = (PROJECT_ROOT / args.config_file).resolve()
        config, config_warning = _load_config_lenient(config_path)
        if config_warning:
            print(
                "WARNING: Failed to load evaluator config cleanly; "
                "falling back to environment variables where needed. "
                f"reason={config_warning}"
            )
        llm_settings = load_llm_settings(config, provider=args.llm_provider, model=args.llm_model)
        print(f"Evaluator: {llm_settings.provider}/{llm_settings.model}")

    records: List[dict] = []
    count = 0
    for item in report_items:
        for section in args.sections:
            if args.max_items is not None and count >= args.max_items:
                break
            count += 1

            section_path = item.analysis_dir / f"{section}.txt"
            row = {
                "run_id": run_id,
                "ticker": item.ticker,
                "company_name": item.company_name,
                "section": section,
                "model_provider": item.model_provider,
                "model_name": item.model_name,
                "model_tag": item.model_tag,
                "source_path": str(section_path),
                "judge_provider": args.llm_provider if not args.dry_run else None,
                "judge_model": args.llm_model if not args.dry_run else None,
            }

            if args.dry_run:
                row.update(
                    {
                        "status": "dry_run",
                        "section_score": None,
                        "overall_score": None,
                        "top_strength": None,
                        "main_weakness": None,
                        "duration_ms": 0,
                        "error": None,
                        "llm_evaluation": None,
                    }
                )
                records.append(row)
                continue

            if not section_path.exists():
                row.update(
                    {
                        "status": "failed",
                        "section_score": None,
                        "overall_score": None,
                        "top_strength": None,
                        "main_weakness": None,
                        "duration_ms": 0,
                        "error": f"missing_file:{section_path}",
                        "llm_evaluation": None,
                    }
                )
                records.append(row)
                continue

            started = time.perf_counter()
            try:
                text = section_path.read_text(encoding="utf-8", errors="ignore")
                prompt = SECTION_EVAL_PROMPT.format(
                    ticker=item.ticker,
                    company_name=item.company_name,
                    section=section,
                    model_tag=item.model_tag,
                    section_text=_truncate(text, max_chars=args.max_chars),
                )
                llm_raw = call_llm(
                    settings=llm_settings,
                    instructions="Return strict JSON only.",
                    prompt=prompt,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                llm_json = _extract_json(llm_raw)
                overall_score = _safe_float(llm_json.get("overall_score"))
                if overall_score is None:
                    raise ValueError("overall_score missing or invalid.")
                duration_ms = int((time.perf_counter() - started) * 1000)
                row.update(
                    {
                        "status": "success",
                        "section_score": round(overall_score, 4),
                        "overall_score": round(overall_score, 4),
                        "top_strength": llm_json.get("top_strength"),
                        "main_weakness": llm_json.get("main_weakness"),
                        "duration_ms": duration_ms,
                        "error": None,
                        "llm_evaluation": llm_json,
                    }
                )
            except Exception as e:
                duration_ms = int((time.perf_counter() - started) * 1000)
                row.update(
                    {
                        "status": "failed",
                        "section_score": None,
                        "overall_score": None,
                        "top_strength": None,
                        "main_weakness": None,
                        "duration_ms": duration_ms,
                        "error": str(e),
                        "llm_evaluation": None,
                    }
                )
            records.append(row)

        if args.max_items is not None and count >= args.max_items:
            break

    aggregates = _build_aggregates(records)
    success_count = sum(1 for r in records if r["status"] == "success")
    failed_count = sum(1 for r in records if r["status"] == "failed")
    dry_count = sum(1 for r in records if r["status"] == "dry_run")

    section_scores_payload = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "judge": {
            "provider": args.llm_provider if not args.dry_run else None,
            "model": args.llm_model if not args.dry_run else None,
        },
        "sections": args.sections,
        "counts": {
            "total": len(records),
            "success": success_count,
            "failed": failed_count,
            "dry_run": dry_count,
        },
        "records": records,
        "section_model_stats": aggregates["section_model_stats"],
        "global_section_mapping": aggregates["global_section_mapping"],
    }

    section_scores_json_path = run_dir / args.section_scores_json_name
    section_scores_csv_path = run_dir / args.section_scores_csv_name
    mapping_json_path = run_dir / args.mapping_json_name

    _write_json(section_scores_json_path, section_scores_payload)
    _write_json(
        mapping_json_path,
        {
            "generated_at_utc": _utc_now_iso(),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sections": args.sections,
            "global_section_mapping": aggregates["global_section_mapping"],
        },
    )

    csv_fields = [
        "run_id",
        "ticker",
        "company_name",
        "section",
        "model_provider",
        "model_name",
        "model_tag",
        "source_path",
        "judge_provider",
        "judge_model",
        "status",
        "section_score",
        "overall_score",
        "top_strength",
        "main_weakness",
        "duration_ms",
        "error",
    ]
    csv_rows = [{k: row.get(k) for k in csv_fields} for row in records]
    _write_csv(section_scores_csv_path, csv_rows, csv_fields)

    print("")
    print("Section evaluation complete.")
    print(f"Counts: total={len(records)}, success={success_count}, failed={failed_count}, dry_run={dry_count}")
    print(f"Section scores JSON: {section_scores_json_path}")
    print(f"Section scores CSV:  {section_scores_csv_path}")
    print(f"Global mapping JSON: {mapping_json_path}")

    if failed_count > 0 and not args.dry_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
