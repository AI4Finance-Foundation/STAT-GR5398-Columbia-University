#!/usr/bin/env python
# coding: utf-8
"""
Catalyst Calendar — concise upcoming-catalyst view for equity research readers.

Selects high-impact items, caps list length, and writes client-facing markdown (no
pipeline meta in the body). Workflow pattern aligns with catalyst-calendar style
scheduling + prioritization.

Reads: analysis/catalyst_analysis.json
Writes: analysis/catalyst_calendar.md (no LLM calls).

Usage:
    python generate_catalyst_calendar.py --company-ticker NVDA \\
        --company-name "NVIDIA Corporation" --analysis-dir path/to/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _read_utf8(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
    return {"high": 3, "medium": 2, "low": 1}.get(str(impact).lower(), 1)


def _norm_desc(d: str | None) -> str:
    if not d:
        return ""
    return re.sub(r"\s+", " ", d.strip())[:160]


def _gather_events(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize catalysts + top_catalysts into one list."""
    out: list[dict[str, Any]] = []

    for c in raw.get("catalysts") or []:
        if isinstance(c, dict) and (c.get("description") or c.get("catalyst")):
            out.append(dict(c))

    for tc in raw.get("top_catalysts") or []:
        if not isinstance(tc, dict):
            continue
        desc = tc.get("description") or tc.get("catalyst") or ""
        if not desc:
            continue
        merged = {
            "expected_date": tc.get("expected_date"),
            "event_type": tc.get("event_type", "event"),
            "impact_level": tc.get("impact_level") or tc.get("impact", "medium"),
            "sentiment": tc.get("sentiment", "neutral"),
            "probability": tc.get("probability"),
            "description": desc,
        }
        out.append(merged)

    # Dedupe: same date + similar description
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ev in out:
        ds, _ = _parse_iso_date(ev.get("expected_date"))
        key = (ds, _norm_desc(ev.get("description")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    scored: list[tuple[tuple[float, int], dict[str, Any]]] = []
    for ev in deduped:
        ds, ts = _parse_iso_date(ev.get("expected_date"))
        imp = _impact_rank(ev.get("impact_level"))
        scored.append(((ts, -imp), ev))
    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored]


def _pick_display_events(events: list[dict[str, Any]], max_items: int = 16) -> list[dict[str, Any]]:
    """Prefer high-impact, then earlier dates; cap count for readable calendar."""
    if len(events) <= max_items:
        return list(events)
    tmp = sorted(
        events,
        key=lambda e: (
            -_impact_rank(e.get("impact_level")),
            _parse_iso_date(e.get("expected_date"))[1],
        ),
    )
    chosen = tmp[:max_items]
    chosen.sort(key=lambda e: _parse_iso_date(e.get("expected_date"))[1])
    return chosen


def _positioning_from_sentiment(sentiment: str | None) -> str:
    s = (sentiment or "").strip().lower()
    if s == "positive":
        return "Long bias"
    if s == "negative":
        return "Defensive"
    return "Neutral"


def _month_bucket(date_str: str) -> str:
    if date_str == "9999-12-31" or not date_str:
        return "Date TBD"
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%B %Y")
    except ValueError:
        return "Date TBD"


def build_markdown(
    ticker: str,
    company_name: str,
    analysis_dir: str,
) -> str:
    path = os.path.join(analysis_dir, "catalyst_analysis.json")
    raw_text = _read_utf8(path)
    if not raw_text.strip():
        return (
            f"# Catalyst Calendar — {company_name} ({ticker})\n\n"
            "*No `catalyst_analysis.json` — run Step 1 with `--enable-catalyst-analysis`.*\n"
        )

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        return (
            f"# Catalyst Calendar — {company_name} ({ticker})\n\n"
            "*Could not parse catalyst_analysis.json.*\n"
        )

    all_events = _gather_events(raw)
    lines: list[str] = [
        f"# Catalyst Calendar — {company_name} ({ticker})",
        "",
        "**Upcoming catalysts & dates** — high-signal items for positioning and risk management.",
        "",
    ]

    if not all_events:
        lines.append("*No catalyst events found in JSON.*")
        return "\n".join(lines) + "\n"

    events = _pick_display_events(all_events, max_items=16)

    # Summary for the curated subset only (reader-facing)
    by_type: defaultdict[str, int] = defaultdict(int)
    dates_ok: list[str] = []
    for ev in events:
        et = str(ev.get("event_type") or "unknown").strip() or "unknown"
        by_type[et] += 1
        ds, ts = _parse_iso_date(ev.get("expected_date"))
        if ts < 9e11:  # not sentinel
            dates_ok.append(ds)
    date_range = ""
    if dates_ok:
        date_range = f"{min(dates_ok)} → {max(dates_ok)}"

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Catalysts covered below:** {len(events)}")
    if date_range:
        lines.append(f"- **Date span:** {date_range}")
    lines.append("- **Mix:** " + ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in sorted(by_type.items(), key=lambda x: -x[1])))
    lines.append("")

    # Timeline grouped by calendar month
    lines.append("## Calendar")
    lines.append("")

    by_month: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        ds, _ = _parse_iso_date(ev.get("expected_date"))
        bucket = _month_bucket(ds if ds != "9999-12-31" else "")
        by_month[bucket].append(ev)

    # Sort month keys chronologically where possible
    def month_sort_key(name: str) -> tuple[int, str]:
        if name == "Date TBD":
            return (1, name)
        try:
            dt = datetime.strptime(name, "%B %Y")
            return (0, dt.isoformat())
        except ValueError:
            return (2, name)

    for month in sorted(by_month.keys(), key=month_sort_key):
        lines.append(f"### {month}")
        lines.append("")
        month_ev = by_month[month]
        month_ev.sort(key=lambda e: (-_impact_rank(e.get("impact_level")), _parse_iso_date(e.get("expected_date"))[1]))
        for ev in month_ev[:6]:
            ds, _ = _parse_iso_date(ev.get("expected_date"))
            date_disp = ds if ds != "9999-12-31" else "TBD"
            et = str(ev.get("event_type") or "event").strip().replace("_", " ").title()
            imp = str(ev.get("impact_level") or ev.get("impact") or "—").title()
            desc = _norm_desc(ev.get("description"))
            if len(desc) > 120:
                desc = desc[:117].rsplit(" ", 1)[0] + "…"
            lines.append(f"- **{date_disp}** — **{et}** ({imp}) — {desc}")
        if len(month_ev) > 6:
            lines.append(f"- *…{len(month_ev) - 6} additional item(s) in this month omitted for length.*")
        lines.append("")

    # Weekly preview windows
    today = datetime.now(timezone.utc)
    this_week: list[dict[str, Any]] = []
    next_week: list[dict[str, Any]] = []
    for ev in all_events:
        ds, ts = _parse_iso_date(ev.get("expected_date"))
        if ds == "9999-12-31":
            continue
        dt = datetime.fromtimestamp(ts, timezone.utc)
        delta = (dt.date() - today.date()).days
        if 0 <= delta <= 7:
            this_week.append(ev)
        elif 8 <= delta <= 14:
            next_week.append(ev)

    lines.append("## Near-term focus")
    lines.append("")
    lines.append("### This week (0–7 days)")
    if this_week:
        this_week.sort(key=lambda e: (-_impact_rank(e.get("impact_level")), _parse_iso_date(e.get("expected_date"))[1]))
        for ev in this_week[:3]:
            ds, _ = _parse_iso_date(ev.get("expected_date"))
            et = str(ev.get("event_type") or "event").replace("_", " ").title()
            imp = str(ev.get("impact_level") or ev.get("impact") or "medium").title()
            sent = str(ev.get("sentiment") or "neutral")
            pos = _positioning_from_sentiment(sent)
            desc = _norm_desc(ev.get("description"))
            if len(desc) > 100:
                desc = desc[:97].rsplit(" ", 1)[0] + "…"
            lines.append(f"- **{ds}** — {et} ({imp}) — {desc} — *View:* {pos}.")
    else:
        lines.append("- No dated items in the next 7 days.")
    lines.append("")

    lines.append("### Next week (8–14 days)")
    if next_week:
        next_week.sort(key=lambda e: (-_impact_rank(e.get("impact_level")), _parse_iso_date(e.get("expected_date"))[1]))
        for ev in next_week[:3]:
            ds, _ = _parse_iso_date(ev.get("expected_date"))
            et = str(ev.get("event_type") or "event").replace("_", " ").title()
            imp = str(ev.get("impact_level") or ev.get("impact") or "medium").title()
            sent = str(ev.get("sentiment") or "neutral")
            pos = _positioning_from_sentiment(sent)
            desc = _norm_desc(ev.get("description"))
            if len(desc) > 100:
                desc = desc[:97].rsplit(" ", 1)[0] + "…"
            lines.append(f"- **{ds}** — {et} ({imp}) — {desc} — *View:* {pos}.")
    else:
        lines.append("- No dated items in the 8–14 day window.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate catalyst_calendar.md from catalyst_analysis.json.")
    parser.add_argument("--company-ticker", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Output path (default: <analysis-dir>/catalyst_calendar.md)",
    )
    args = parser.parse_args()
    analysis_dir = os.path.abspath(args.analysis_dir)
    out = args.output.strip() or os.path.join(analysis_dir, "catalyst_calendar.md")

    md = build_markdown(args.company_ticker.upper(), args.company_name, analysis_dir)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Catalyst Calendar written: {out}")


if __name__ == "__main__":
    main()
