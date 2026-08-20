#!/usr/bin/env python
# coding: utf-8
"""
Earnings Update (post-results digest) — Track B, workflow aligned with Anthropic
`equity-research/skills/earnings-analysis` (SKILL.md): beat/miss framing, summary tables,
forward snapshot, implications, sources. The generated `earnings_update.md` is written for
an equity-research reader (no pipeline / assignment meta in the body).

Re-implements a deterministic subset using FinRobot CSV + JSON only (stdlib csv, no pandas).

Output: analysis/earnings_update.md

Usage:
  python generate_earnings_update.py --company-ticker GOOGL --company-name "Alphabet Inc." \\
      --analysis-dir path/to/analysis

  Optional sell-side style commentary (does not replace tables; model must not invent figures):
  python generate_earnings_update.py ... --use-llm-summary [--config-file path/to/config.ini]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Optional

FMP_HOME = "https://financialmodelingprep.com"

# Soft display cap for LLM commentary in markdown: aim near this length but end on a full sentence (no "…").
LLM_COMMENTARY_DISPLAY_CHARS = 1200
# If the cap lands mid-sentence, scan at most this many extra characters to finish the sentence.
LLM_COMMENTARY_SENTENCE_OVERFLOW = 450


def _default_config_path() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.ini"))


def _load_llm_settings(config_file: str | None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (api_key, base_url, model) from config, or (None,...) if unavailable."""
    path = config_file or _default_config_path()
    if not os.path.isfile(path):
        return None, None, None
    try:
        from modules.common_utils import get_api_key, load_config

        cfg = load_config(path)
        key = get_api_key(cfg, "API_KEYS", "openai_api_key").strip()
        base = cfg.get("API_KEYS", "openai_base_url", fallback=None) or None
        mdl = cfg.get("API_KEYS", "openai_model", fallback=None) or None
        return (
            key or None,
            (base.strip() if isinstance(base, str) else base),
            (mdl.strip() if isinstance(mdl, str) else mdl),
        )
    except Exception:
        return None, None, None


def _truncate_at_sentence(out: str, max_chars: int, overflow: int) -> str:
    """Keep roughly max_chars visible length; prefer ending on a complete sentence (never add …)."""
    if len(out) <= max_chars:
        return out
    prefix = out[:max_chars]
    sentence_end = re.compile(r"[.!?](?:\s|$)")
    last_in_prefix = -1
    for m in sentence_end.finditer(prefix):
        last_in_prefix = m.end()
    # If a sentence ends reasonably inside the cap, stop there (may be well under max_chars).
    if last_in_prefix >= int(max_chars * 0.35):
        return out[:last_in_prefix].strip()
    # Cap landed mid-sentence: extend up to overflow to finish the current sentence.
    hard = min(len(out), max_chars + overflow)
    ahead = out[max_chars:hard]
    m = sentence_end.search(ahead)
    if m:
        return out[: max_chars + m.end()].strip()
    # No terminator in window: use last sentence end in whole bounded window, else soft word break.
    window = out[:hard]
    last_w = -1
    for m in sentence_end.finditer(window):
        last_w = m.end()
    if last_w > 0:
        return out[:last_w].strip()
    soft = prefix.rstrip()
    return soft.rsplit(" ", 1)[0].strip() if " " in soft else soft.strip()


def _sanitize_llm_prose(text: str, max_chars: Optional[int] = LLM_COMMENTARY_DISPLAY_CHARS) -> str:
    """Strip markdown headings / bullets; optional length cap that ends on a full sentence (no …)."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    lines_out: list[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if s.startswith(("- ", "* ", "• ")):
            s = s.lstrip("-*• ").strip()
        if s:
            lines_out.append(s)
    out = " ".join(lines_out) if lines_out else t
    out = re.sub(r"\s+", " ", out).strip()
    if max_chars is not None and len(out) > max_chars:
        return _truncate_at_sentence(out, max_chars, LLM_COMMENTARY_SENTENCE_OVERFLOW)
    return out


def _fetch_llm_analyst_commentary(
    company_name: str,
    ticker: str,
    fiscal_label: str,
    beat_miss_lines: list[str],
    api_key: str,
    base_url: Optional[str],
    model: Optional[str],
) -> str:
    """One short prose block: qualitative only; no numeric restatement."""
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ openai package not installed; skipping LLM summary.", file=sys.stderr)
        return ""

    system = (
        "You are a senior sell-side equity research analyst. Write at most TWO short paragraphs of prose "
        f"(combined hard maximum 200 words) in calm, balanced institutional English for {company_name} ({ticker}). "
        "You only know the qualitative labels vs. the firm's **prior internal model estimate row** (Beat / Miss / Inline / N/A)—treat misses as "
        "**forecast variance vs. that row**, not as proof that anyone's 'model is wrong' or broken. "
        "Tone: measured and constructive; avoid alarmist or catastrophic framing. "
        "Do NOT: criticize, question, or apologize for the house forecast process, model, or methodology; "
        "do NOT use words like deterioration, collapse, disaster, alarming, or broad-based deterioration; "
        "do NOT say the estimate framework 'reduces confidence' in the model or similar. "
        "You MAY: note mixed or directional variance vs. the prior row, mention monitoring segments or guidance as normal practice, "
        "and keep long-term thesis language neutral unless the labels alone clearly show all beats or all misses (still stay calm). "
        "Rules: "
        "(1) No dollar amounts, counts of misses as digits, EPS values, margin percentages, or other numbers—tables hold all figures. "
        "(2) No markdown headings (#) or bullet lists; prose only. "
        "(3) No fabricated guidance, dates, or filing claims. "
        "(4) If labels are mostly N/A, say a directional read was not available from the summary labels. "
        "(5) End on a complete sentence—no trailing clause left unfinished."
    )
    user = (
        f"Fiscal context: {fiscal_label}.\n\n"
        "Headline metrics vs. **prior internal model estimate row** (same fiscal year), labels only:\n"
        + "\n".join(f"- {x}" for x in beat_miss_lines)
        + "\n\nWrite concise analyst commentary: outcome vs. that prior row in professional language, without implying the forecasting approach was defective."
    )
    mdl = model or "gpt-4o-mini"
    try:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.22,
            max_tokens=700,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _sanitize_llm_prose(raw, max_chars=LLM_COMMENTARY_DISPLAY_CHARS)
    except Exception as e:
        print(f"⚠️ LLM commentary failed ({e}); continuing without AI paragraph.", file=sys.stderr)
        return ""


def _read_utf8(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_csv_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _row_by_metric(rows: list[dict[str, str]], name: str) -> Optional[dict[str, str]]:
    for r in rows:
        if str(r.get("metrics", "")).strip() == name:
            return r
    return None


def _parse_cell(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return None
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt_display(metric: str, val: Optional[float], raw: Any) -> str:
    if val is None:
        return "N/A"
    mlow = metric.lower()
    if ("margin" in mlow or "growth" in mlow) and abs(val) <= 2.0:
        return f"{val * 100:.1f}%"
    if "eps" in mlow:
        return f"{val:.2f}"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:.0f}M"
    return str(raw).strip() if raw is not None else f"{val:,.0f}"


# Beat/Miss: compare latest reported year (e.g. 2025A) to internal model column (e.g. 2025E) from the same CSV.
# A single tight relative band (e.g. 1.5%) on every line classifies most rows as Miss whenever the model
# is slightly more optimistic than reported actuals. Use materiality-style bands by metric.
_BEAT_MISS_REL_DOLLAR = 0.05  # Revenue & EBITDA: ±5% of estimate → Inline
_BEAT_MISS_REL_MARGIN = 0.04  # Margin: relative fallback
_BEAT_MISS_ABS_MARGIN = 0.012  # Margin: |actual−est| within 1.2 pp (as decimal, e.g. 0.012) → Inline
_BEAT_MISS_REL_EPS = 0.10  # EPS: avoid tiny-denominator over-sensitivity
_BEAT_MISS_ABS_EPS = 0.05  # EPS: $0.05 or smaller gap → Inline


def _beat_miss(metric: str, actual: Optional[float], est: Optional[float]) -> str:
    if actual is None or est is None or est == 0:
        return "N/A"
    m = metric.lower()
    if "margin" in m or m == "contribution margin" or m == "sg&a margin":
        # Margins in CSV are decimals; compare pp distance first, then relative.
        if abs(actual - est) <= _BEAT_MISS_ABS_MARGIN:
            return "Inline"
        rel = (actual - est) / abs(est)
        if rel > _BEAT_MISS_REL_MARGIN:
            return "Beat"
        if rel < -_BEAT_MISS_REL_MARGIN:
            return "Miss"
        return "Inline"
    if m == "eps" or m.endswith(" eps"):
        if abs(actual - est) <= _BEAT_MISS_ABS_EPS:
            return "Inline"
        rel = (actual - est) / abs(est)
        if rel > _BEAT_MISS_REL_EPS:
            return "Beat"
        if rel < -_BEAT_MISS_REL_EPS:
            return "Miss"
        return "Inline"
    # Revenue, EBITDA, etc. (dollar or large units)
    rel = (actual - est) / abs(est)
    if rel > _BEAT_MISS_REL_DOLLAR:
        return "Beat"
    if rel < -_BEAT_MISS_REL_DOLLAR:
        return "Miss"
    return "Inline"


def _next_earnings_hint(catalyst_path: str) -> str:
    raw = _read_utf8(catalyst_path)
    if not raw.strip():
        return ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    dates: list[tuple[str, str]] = []
    for c in data.get("catalysts") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("event_type", "")).lower() != "earnings":
            continue
        d = (c.get("expected_date") or "")[:10]
        desc = str(c.get("description", ""))[:80]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            dates.append((d, desc))
    if not dates:
        return ""
    dates.sort(key=lambda x: x[0])
    today = datetime.now(timezone.utc).date().isoformat()
    future = [x for x in dates if x[0] >= today]
    pick = future[0] if future else dates[-1]
    return f"**{pick[0]}** — {pick[1]}"


def build_markdown(
    ticker: str,
    company_name: str,
    analysis_dir: str,
    use_llm_summary: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    summ_path = os.path.join(analysis_dir, "analysis_summary.json")
    csv_path = os.path.join(analysis_dir, "financial_metrics_and_forecasts.csv")
    summ_raw = _read_utf8(summ_path)
    if not summ_raw.strip():
        return f"# Earnings Update — {company_name} ({ticker})\n\n*Missing analysis_summary.json.*\n"
    try:
        summary = json.loads(summ_raw)
    except json.JSONDecodeError:
        return f"# Earnings Update — {company_name} ({ticker})\n\n*Could not parse analysis_summary.json.*\n"

    if not os.path.isfile(csv_path):
        return f"# Earnings Update — {company_name} ({ticker})\n\n*Missing financial_metrics_and_forecasts.csv.*\n"

    cols, rows = _load_csv_rows(csv_path)
    if "metrics" not in cols:
        return f"# Earnings Update — {company_name} ({ticker})\n\n*CSV missing metrics column.*\n"

    latest_key = str(summary.get("latest_year") or "2025A").strip()
    m = re.match(r"^(\d{4})A$", latest_key)
    if not m:
        return f"# Earnings Update — {company_name} ({ticker})\n\n*latest_year not in YYYYA form: {latest_key}*\n"
    year = m.group(1)
    col_a = f"{year}A"
    col_e = f"{year}E"
    col_prior = f"{int(year) - 1}A"
    if col_a not in cols:
        return f"# Earnings Update — {company_name} ({ticker})\n\n*Column {col_a} not in CSV.*\n"

    analysis_date = str(summary.get("analysis_date", ""))[:19]
    as_of_disp = analysis_date[:10] if len(analysis_date) >= 10 else (analysis_date or "N/A")

    # Exclude Revenue Growth here: 2025E is a forecast assumption, not comparable to 2025A YoY % in beat/miss sense.
    metrics_watch = ["Revenue", "EBITDA", "EBITDA Margin", "EPS"]
    lines: list[str] = [
        f"# Earnings Update — {company_name} ({ticker})",
        "",
        f"**Fiscal {year} — as of {as_of_disp}**",
        "",
        f"Summary review of reported **{col_a}** results versus our **{col_e}** internal model estimate row for the same fiscal year.",
        "",
        "## 1. Executive summary",
        "",
    ]

    rows_beat: list[tuple[str, str, Optional[float], Optional[float], Any, Any]] = []
    for met in metrics_watch:
        row = _row_by_metric(rows, met)
        if row is None:
            continue
        a = _parse_cell(row.get(col_a))
        e = _parse_cell(row.get(col_e)) if col_e in cols else None
        bm = _beat_miss(met, a, e) if e is not None else "N/A"
        rows_beat.append((met, bm, a, e, row.get(col_a), row.get(col_e)))

    if not rows_beat:
        lines.append("*Key metrics for this comparison could not be assembled.*")
    else:
        beats = sum(1 for x in rows_beat if x[1] == "Beat")
        misses = sum(1 for x in rows_beat if x[1] == "Miss")
        if beats > misses:
            lines.append(
                f"On the selected operating lines, reported results **exceeded** our internal model estimates in **{beats}** area(s) "
                f"and **trailed** in **{misses}** (see table below)."
            )
        elif misses > beats:
            lines.append(
                f"On the selected operating lines, reported results **trailed** our internal model estimates in **{misses}** area(s) "
                f"and **exceeded** in **{beats}** (see table below)."
            )
        else:
            lines.append("Reported results are **mixed** relative to our internal model estimates; detail follows.")
        lines.append("")

    if use_llm_summary and api_key and rows_beat:
        fiscal_ctx = f"{year} reported ({col_a}) vs. internal model estimate row ({col_e})"
        beat_miss_lines = [f"{met}: {bm}" for met, bm, *_rest in rows_beat]
        commentary = _fetch_llm_analyst_commentary(
            company_name, ticker.upper(), fiscal_ctx, beat_miss_lines, api_key, base_url, model
        )
        if commentary:
            lines.extend(
                [
                    "### Analyst commentary",
                    "",
                    commentary,
                    "",
                ]
            )

    lines.extend(
        [
            "## 2. Key metrics — reported vs. internal model estimates",
            "",
            "| Metric | Reported (" + col_a + ") | Internal model (" + col_e + ") | vs. internal model |",
            "| --- | --- | --- | --- |",
        ]
    )
    for met, bm, a, e, raw_a, raw_e in rows_beat:
        da = _fmt_display(met, a, raw_a)
        de = _fmt_display(met, e, raw_e) if e is not None else "N/A"
        lines.append(f"| {met} | {da} | {de} | {bm} |")
    lines.append("")
    lines.append(
        f"*The **{col_e}** column reflects our internal model estimates for that fiscal year; it is not third-party sell-side consensus.*"
    )
    lines.append("")

    lines.append(f"## 3. Year-over-year change ({col_prior} → {col_a})")
    lines.append("")
    yoy_bits: list[str] = []
    r_prev = _row_by_metric(rows, "Revenue")
    r_cur = _row_by_metric(rows, "Revenue")
    if r_prev is not None and r_cur is not None and col_prior in cols:
        v0 = _parse_cell(r_prev.get(col_prior))
        v1 = _parse_cell(r_cur.get(col_a))
        if v0 and v1 and v0 > 0:
            yoy_bits.append(
                f"- **Revenue:** {_fmt_display('Revenue', v1, None)} vs prior year {_fmt_display('Revenue', v0, None)} (**{(v1 - v0) / v0 * 100:.1f}%** YoY)."
            )
    m_prev = _row_by_metric(rows, "EBITDA Margin")
    if m_prev is not None and col_prior in cols:
        p0 = _parse_cell(m_prev.get(col_prior))
        p1 = _parse_cell(m_prev.get(col_a))
        if p0 is not None and p1 is not None:
            dpt = (p1 - p0) * 100
            yoy_bits.append(f"- **EBITDA margin:** {p0 * 100:.1f}% → {p1 * 100:.1f}% (**{dpt:+.1f} pts**).")
    if not yoy_bits:
        lines.append("- *(Year-over-year bridge not available for this layout.)*")
    else:
        lines.extend(yoy_bits)
    lines.append("")

    lines.append("## 4. Forward estimates")
    lines.append("")
    fy = summary.get("forecast_years") or ["2026E", "2027E"]
    use_cols = [c for c in fy if c in cols]
    sub = ["Revenue", "EBITDA", "EPS", "EBITDA Margin"]
    hdr = "| Metric | " + " | ".join(use_cols) + " |"
    sep = "| " + " | ".join(["---"] * (1 + len(use_cols))) + " |"
    lines.append(hdr)
    lines.append(sep)
    for met in sub:
        row = _row_by_metric(rows, met)
        if row is None:
            continue
        cells = [met] + [_fmt_display(met, _parse_cell(row.get(c)), row.get(c)) for c in use_cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 5. Investment view — implications")
    lines.append("")
    rev_bm = next((x[1] for x in rows_beat if x[0] == "Revenue"), "N/A")
    eps_bm = next((x[1] for x in rows_beat if x[0] == "EPS"), "N/A")
    mar_bm = next((x[1] for x in rows_beat if x[0] == "EBITDA Margin"), "N/A")
    bullets: list[str] = []
    if rev_bm == "Beat":
        bullets.append("- **Revenue** exceeded our internal model estimate — supports a constructive read on demand and mix, subject to guidance and segment detail.")
    elif rev_bm == "Miss":
        bullets.append("- **Revenue** fell short of our internal model estimate — revisit growth drivers and segment contribution in the next update.")
    if eps_bm == "Beat":
        bullets.append("- **EPS** exceeded our internal model estimate — consider operating leverage, tax rate, and share count versus forecast.")
    elif eps_bm == "Miss":
        bullets.append("- **EPS** missed our internal model estimate — review margin bridge, below-the-line items, and forecast cadence.")
    if mar_bm == "Beat":
        bullets.append("- **Margins** ahead of our internal model estimate — positive for earnings quality; monitor sustainability vs. investment cycle.")
    elif mar_bm == "Miss":
        bullets.append("- **Margins** below our internal model estimate — focus on cost trajectory and reinvestment intensity.")
    if not bullets:
        bullets.append("- Limited variance on headline lines; maintain prior view pending updated guidance and segment disclosure.")
    lines.extend(bullets)
    lines.append("")

    hint = _next_earnings_hint(os.path.join(analysis_dir, "catalyst_analysis.json"))
    if hint:
        lines.append("## 6. Notable upcoming dates")
        lines.append("")
        lines.append(hint)
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    lines.append(
        f"- Market data and financial statement inputs: **Financial Modeling Prep** — [{FMP_HOME}]({FMP_HOME}) (accessed **{as_of_disp}**)."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate earnings_update.md (earnings-analysis skill subset).")
    parser.add_argument("--company-ticker", required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", default="", help="Default: <analysis-dir>/earnings_update.md")
    parser.add_argument(
        "--use-llm-summary",
        action="store_true",
        help="Append optional 'Analyst commentary' via LLM (Beat/Miss labels only in prompt; all figures remain from CSV).",
    )
    parser.add_argument(
        "--config-file",
        default="",
        help="Path to config.ini for openai_api_key / openai_base_url / openai_model (default: core/config/config.ini).",
    )
    args = parser.parse_args()
    analysis_dir = os.path.abspath(args.analysis_dir)
    out = args.output.strip() or os.path.join(analysis_dir, "earnings_update.md")

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    if args.use_llm_summary:
        cfg_path = args.config_file.strip() or None
        api_key, base_url, model = _load_llm_settings(cfg_path)
        if not api_key:
            print(
                "⚠️ --use-llm-summary: no openai_api_key in config (or config missing). "
                "Writing deterministic markdown without analyst commentary.",
                file=sys.stderr,
            )

    md = build_markdown(
        args.company_ticker.upper(),
        args.company_name,
        analysis_dir,
        use_llm_summary=bool(args.use_llm_summary and api_key),
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Earnings Update written: {out}")


if __name__ == "__main__":
    main()
