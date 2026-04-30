#!/usr/bin/env python
# coding: utf-8
"""
Thesis Tracker — structured equity thesis snapshot (Track B enhancement).

Reads existing analysis artifacts (CSV, risks, catalyst JSON, narrative files)
and writes analysis/thesis_tracker.md. No LLM required; reproducible offline.

Usage:
    python generate_thesis_tracker.py --company-ticker NVDA --company-name "NVIDIA Corporation" \\
        --analysis-dir path/to/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any

import pandas as pd


def _read_utf8(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _thesis_summary(analysis_dir: str) -> str:
    for name in ("investment_overview.txt", "tagline.txt", "company_overview.txt"):
        p = os.path.join(analysis_dir, name)
        raw = _read_utf8(p)
        if not raw.strip():
            continue
        lines = raw.splitlines()
        body: list[str] = []
        for line in lines:
            if line.strip().startswith("#"):
                continue
            body.append(line)
        text = "\n".join(body).strip()
        if text:
            if len(text) > 1800:
                return text[:1800].rsplit(" ", 1)[0] + "..."
            return text
    return "(No investment_overview / tagline found — run Step 2 text generation first.)"


def _risk_bullets(risks_path: str) -> list[str]:
    raw = _read_utf8(risks_path)
    if not raw.strip():
        return []
    bullets: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")):
            bullets.append(s[2:].strip())
    return bullets[:12]


def _load_catalysts(path: str) -> list[dict[str, Any]]:
    raw = _read_utf8(path)
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    cats = data.get("catalysts") or []
    if not isinstance(cats, list):
        return []
    out: list[dict[str, Any]] = []
    for c in cats:
        if isinstance(c, dict) and c.get("description"):
            out.append(c)
    return out


def _parse_iso_date(s: str | None) -> tuple[str, float]:
    if not s or not isinstance(s, str):
        return ("9999-12-31", 0.0)
    s = s.strip()[:10]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return (s, dt.timestamp())
    except ValueError:
        return ("9999-12-31", 0.0)


def _impact_rank(impact: str | None) -> int:
    if not impact:
        return 1
    m = {"high": 3, "medium": 2, "low": 1}.get(str(impact).lower(), 1)
    return m


def _catalyst_rows(catalysts: list[dict[str, Any]], limit: int = 14) -> list[dict[str, Any]]:
    scored: list[tuple[tuple[float, int], dict[str, Any]]] = []
    for c in catalysts:
        d, ts = _parse_iso_date(c.get("expected_date"))
        imp = _impact_rank(c.get("impact_level"))
        scored.append(((ts, -imp), {**c, "_sort_date": d}))
    scored.sort(key=lambda x: x[0])
    rows = [x[1] for x in scored][:limit]
    return rows


def _md_escape_cell(s: str) -> str:
    s = s.replace("|", "\\|").replace("\n", " ")
    return s.strip() or "—"


def _scorecard_rows(csv_path: str) -> list[tuple[str, str, str]]:
    """Return (metric_name, latest_actual_or_na, latest_forecast_or_na) for key rows."""
    if not os.path.isfile(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if df.empty or "metrics" not in df.columns:
        return []
    want = {
        "revenue growth",
        "ebitda margin",
        "contribution margin",
        "sg&a margin",
    }
    numeric_cols = [
        c for c in df.columns if c != "metrics" and "cagr" not in str(c).lower()
    ]
    # Prefer forecast years last (2027E, 2026E, ...) then actuals
    def col_key(c: str) -> tuple[int, str]:
        ce = c.upper().replace(" ", "")
        year_match = re.search(r"(\d{4})", ce)
        y = int(year_match.group(1)) if year_match else 0
        prio = 0 if "E" in ce else 1
        return (prio, f"{y:04d}{ce}")

    numeric_cols = sorted(numeric_cols, key=col_key, reverse=True)
    if len(numeric_cols) < 2:
        tail = numeric_cols + [""]
        while len(tail) < 2:
            tail.append("")
        fc, prev = tail[0], tail[1]
    else:
        fc, prev = numeric_cols[0], numeric_cols[1]

    rows_out: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        name = str(row.get("metrics", "")).strip()
        if not name:
            continue
        if name.lower() not in want:
            continue
        v_fc = row.get(fc, "")
        v_prev = row.get(prev, "")
        rows_out.append((name, str(v_prev), str(v_fc)))
    return rows_out


def _analysis_as_of(analysis_dir: str) -> str:
    p = os.path.join(analysis_dir, "analysis_summary.json")
    raw = _read_utf8(p)
    if not raw.strip():
        return datetime.utcnow().strftime("%Y-%m-%d")
    try:
        j = json.loads(raw)
        d = j.get("analysis_date") or j.get("generated_at")
        if isinstance(d, str) and len(d) >= 10:
            return d[:10]
    except json.JSONDecodeError:
        pass
    return datetime.utcnow().strftime("%Y-%m-%d")


def build_markdown(
    ticker: str,
    company_name: str,
    analysis_dir: str,
) -> str:
    thesis = _thesis_summary(analysis_dir)
    risks = _risk_bullets(os.path.join(analysis_dir, "risks.txt"))
    cats = _load_catalysts(os.path.join(analysis_dir, "catalyst_analysis.json"))
    cat_rows = _catalyst_rows(cats)
    csv_path = os.path.join(analysis_dir, "financial_metrics_and_forecasts.csv")
    scorecard = _scorecard_rows(csv_path)
    as_of = _analysis_as_of(analysis_dir)

    lines: list[str] = [
        f"# Thesis Tracker — {company_name} ({ticker})",
        "",
        f"*As of {as_of} · Generated from FinRobot analysis artifacts (Track B: structured thesis snapshot).*",
        "",
        "## 1. Investment thesis (summary)",
        "",
        thesis,
        "",
        "## 2. Key risks (from `risks.txt`)",
        "",
    ]
    if risks:
        for b in risks:
            lines.append(f"- {b}")
    else:
        lines.append("- *(No risk bullets parsed — ensure Step 2 produced `risks.txt`.)*")
    lines.extend(["", "## 3. Financial scorecard (selected metrics)", ""])

    if scorecard:
        lines.extend(["| Metric | Prior period | Latest / forecast |", "| --- | --- | --- |"])
        for name, a, b in scorecard:
            lines.append(f"| {_md_escape_cell(name)} | {_md_escape_cell(a)} | {_md_escape_cell(b)} |")
    else:
        lines.append(
            "*Could not build scorecard — ensure `financial_metrics_and_forecasts.csv` exists with a `metrics` column.*"
        )

    lines.extend(["", "## 4. Catalyst calendar", ""])

    if cat_rows:
        lines.extend(
            [
                "| Date | Type | Impact | Sentiment | Catalyst |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for c in cat_rows:
            d = c.get("_sort_date") or "—"
            et = _md_escape_cell(str(c.get("event_type") or "—"))
            imp = _md_escape_cell(str(c.get("impact_level") or "—"))
            sent = _md_escape_cell(str(c.get("sentiment") or "—"))
            desc = _md_escape_cell(str(c.get("description") or ""))
            lines.append(f"| {d} | {et} | {imp} | {sent} | {desc} |")
    else:
        lines.append(
            "*No catalyst rows — ensure Step 1 ran with catalyst analysis and `catalyst_analysis.json` is present.*"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "*Inspired by institutional equity-research workflows (thesis definition, risk inventory, catalyst calendar, metric scorecard).*",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis_tracker.md from analysis artifacts.")
    parser.add_argument("--company-ticker", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Output path (default: <analysis-dir>/thesis_tracker.md)",
    )
    args = parser.parse_args()
    analysis_dir = os.path.abspath(args.analysis_dir)
    out = args.output.strip() or os.path.join(analysis_dir, "thesis_tracker.md")

    md = build_markdown(args.company_ticker.upper(), args.company_name, analysis_dir)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Thesis Tracker written: {out}")


if __name__ == "__main__":
    main()
