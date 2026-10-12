"""Claim-level grounding scorer for finrobot_equity text sections.

For one generated section it
  1. rebuilds the exact prompt the writer model received (the pipeline's own `_prepare_user_prompt`, fed with the
     tables Step 1 saved),
  2. asks an LLM judge to split the section into atomic factual claims and label each one Supported / Unsupported /
     Contradicted against that prompt, citing the evidence cells it used, and
  3. checks every number in each claim against the prompt's tables and simple values derived from them, as a
     deterministic cross-check on the judge.

Judge calls are cached in audit/data/grounding_cache.jsonl (keyed by model, judge version, evidence and section text),
so re-running costs nothing unless one of those changes.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "FinRobot" / "finrobot_equity" / "core" / "src"
OUT = ROOT / "FinRobot" / "output"
CACHE_FILE = ROOT / "audit" / "data" / "grounding_cache.jsonl"
sys.path.insert(0, str(SRC))

from modules.text_generator_agents import SYSTEM_PROMPTS, _prepare_user_prompt  # noqa: E402

SECTIONS = ["tagline", "company_overview", "investment_overview", "valuation_overview",
            "risks", "competitor_analysis", "major_takeaways", "news_summary"]
JUDGE_VERSION = "v3"

JUDGE_INSTRUCTIONS = """You audit AI-written equity research for grounding.

You receive EVIDENCE - the exact instructions and data the writer model was given, nothing else - and SECTION, the
text it wrote. Extract every factual or quantitative claim in SECTION and judge each one against EVIDENCE only.

Extracting claims
- One claim = one checkable fact. A change between two values ("rose from X in 2023 to Y in 2024") is one claim;
  anything joined to it is another. "Revenue rose from $26.9B in 2023 to $60.9B in 2024, is projected to reach $130.5B
  in 2025, a 68% CAGR" is THREE claims. Before answering, re-read every claim and split any that still joins two facts.
- Write each claim so it stands on its own, keeping the period and the wording that signals status ("projected",
  "expected", "reached", "guidance", "estimate").
- Include numbers, comparisons, trends, and evaluative statements that assert something checkable about the data
  ("margins expanded", "growth outpaces peers", "trades at a premium to peers").
- Include background statements about the company (products, history, strategy, customers, competitors, regulation,
  markets): they are claims too.
- Skip pure opinions, recommendations and advice with nothing checkable ("investors should monitor", "remains
  attractive"), and sentences that only describe the report itself.
- `quote` must be an exact substring of SECTION.

For every claim, before deciding the verdict, fill in:
- `cells`: each EVIDENCE cell the claim's numbers come from (table, row, column exactly as in EVIDENCE, value as
  shown). Leave empty when nothing in EVIDENCE bears on the claim.
- `claimed_years`: the fiscal years the claim attaches its facts to (e.g. [2023, 2025]); empty if none.
- `claimed_status`: "forecast" if the claim presents its figures as projected / expected / estimated / guided /
  forecast / "E"; "actual" if it presents them as reported / achieved / posted / past tense; "mixed" if both; "none"
  if no figures or no status.

Labels
- Supported: follows from EVIDENCE, including simple arithmetic (differences, growth rates, ratios, CAGR, "Nx") with
  sensible rounding, AND the year and status match the cells: columns ending in A are reported actuals, columns ending
  in E are the pipeline's own forecasts.
- Contradicted: EVIDENCE shows it is wrong - wrong number, wrong year, wrong direction or comparison; an A-column
  value presented as a projection/estimate/guidance or an E-column value presented as reported; a CAGR attributed to a
  period other than the one it covers; or a statement credited to a source EVIDENCE does not contain ("consensus",
  "guidance", "analysts expect") when the only forward numbers are the E columns.
- Unsupported: EVIDENCE neither establishes nor contradicts it (typically background knowledge).
- Check descriptive quantifiers against the numbers: "doubled", "nearly 6x", "each year", "consistently", "far
  outpacing", "surged", "declined". If the numbers do not show it, the claim is Contradicted even when every number it
  quotes appears in EVIDENCE.

Calibration examples (toy evidence: Revenue 2024A = 100, 2025A = 120, 2025E = 126, 2026E = 133; CAGR = 20.0%, the
2024A-2025A revenue CAGR)
- "Revenue grew 20% in 2025" -> Supported.
- "Revenue is projected to reach 120 in 2025" -> Contradicted (120 is the reported 2025A value, not a projection).
- "Revenue reached 126 in 2025" -> Contradicted (126 is a forecast).
- "Revenue is expected to grow at a 20% CAGR through 2026" -> Contradicted (the CAGR covers 2024A-2025A only).
- "Consensus expects 133 in 2026" -> Contradicted (133 is the pipeline's forecast; no consensus is in EVIDENCE).
- "Revenue is forecast to nearly double each year, reaching 133 in 2026" -> Contradicted (126 -> 133 is +6%).
- "The company was founded in 1993" -> Unsupported.

Reading EVIDENCE
- In the metrics table, the CAGR column is revenue CAGR from the first to the last A column.
- "Contribution Margin" is (revenue - cost of revenue) / revenue, i.e. gross margin; treat the two names as the same
  metric. "SG&A" excludes R&D.
- In the peer tables the year is the calendar year in which each company's fiscal year ends; a blank cell means no data.

Give a one-sentence reason for each verdict."""


class Cell(BaseModel):
    table: str
    row: str
    column: str
    value: str


class Claim(BaseModel):
    claim: str
    quote: str
    kind: Literal["quantitative", "qualitative"]
    cells: list[Cell]
    claimed_years: list[int]
    claimed_status: Literal["actual", "forecast", "mixed", "none"]
    verdict: Literal["Supported", "Unsupported", "Contradicted"]
    reason: str


class Judgment(BaseModel):
    claims: list[Claim]


# ---------------------------------------------------------------- evidence (what the writer saw)

def restore_types(df: pd.DataFrame) -> pd.DataFrame:
    """Reading the CSV back makes every column text; restore numbers, keep 'x.x%' and 'N/A' as the pipeline has them."""
    def cell(x):
        if isinstance(x, str) and (x.endswith("%") or x == "N/A"):
            return x
        try:
            return float(x)
        except (TypeError, ValueError):
            return x
    out = df.copy()
    for c in out.columns[1:]:
        out[c] = out[c].map(cell).astype(object)
    return out


def load_tables(run_dir: Path) -> dict[str, pd.DataFrame | None]:
    a = run_dir / "analysis"
    read = lambda name, **kw: pd.read_csv(a / name, **kw) if (a / name).exists() else None
    return {
        "financial_metrics": restore_types(read("financial_metrics_and_forecasts.csv")),
        "peer_ebitda": read("peer_ebitda_comparison.csv", index_col=0),
        "peer_ev_ebitda": read("peer_ev_ebitda_comparison.csv", index_col=0),
    }


def writer_prompt(run_dir: Path, ticker: str, company: str, section: str) -> tuple[str, str]:
    """(system prompt, user prompt) exactly as text_generator_agents builds them for this section."""
    data = dict(load_tables(run_dir))
    news = run_dir / "analysis" / "company_news.json"
    data["company_news"] = json.loads(news.read_text(encoding="utf-8")) if news.exists() else None
    return SYSTEM_PROMPTS[section], _prepare_user_prompt(data, section, company, ticker)


# ---------------------------------------------------------------- the texts being audited

class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "li", "br", "div", "h3", "h4", "strong"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _strip_tags(html: str) -> str:
    p = _Text()
    p.feed(html)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", "".join(p.parts))).strip()


def section_texts(run_dir: Path) -> dict[tuple[str, str], str]:
    """{(section, variant): text}. variant 'step1' is the file Step 1 wrote; 'report' is what the HTML report shows
    for the two sections Step 2 regenerates."""
    texts = {}
    for s in SECTIONS:
        f = run_dir / "analysis" / f"{s}.txt"
        if f.exists():
            texts[(s, "step1")] = f.read_text(encoding="utf-8").strip()
    report = next((run_dir / "report").glob("Professional_Equity_Report_*.html"), None)
    if report:
        html = report.read_text(encoding="utf-8")

        def between(start, end):
            i = html.index(start) + len(start)
            return _strip_tags(html[i:html.index(end, i)])
        texts[("competitor_analysis", "report")] = between('<h3 class="heading-2">Analysis</h3>',
                                                           '<h3 class="heading-2">Risk Factors</h3>')
        texts[("major_takeaways", "report")] = between('<h3 class="heading-2">Key Takeaways</h3>', "</section>")
    return texts


# ---------------------------------------------------------------- LLM judge (cached)

def _cache_key(model: str, evidence: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{JUDGE_VERSION}|{JUDGE_INSTRUCTIONS}|{evidence}|{text}".encode()).hexdigest()


_cache_lock = threading.Lock()
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    _cache[r["key"]] = r
    return _cache


def judge(client, model: str, system_prompt: str, user_prompt: str, section: str, text: str) -> tuple[list[dict], dict]:
    """Returns (claims, usage). Uses the cache when the same model, judge version, evidence and text were seen.
    Safe to call from several threads."""
    evidence = f"[writer system prompt]\n{system_prompt}\n\n[writer user prompt]\n{user_prompt}"
    key = _cache_key(model, evidence, text)
    with _cache_lock:
        cache = _load_cache()
        if key in cache:
            return cache[key]["claims"], {"cached": True}
    resp = client.chat.completions.parse(
        model=model,
        temperature=0,
        response_format=Judgment,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": f"EVIDENCE\n<<<\n{evidence}\n>>>\n\nSECTION ({section})\n<<<\n{text}\n>>>"},
        ],
    )
    claims = [c.model_dump() for c in resp.choices[0].message.parsed.claims]
    usage = {"cached": False, "input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    record = {"key": key, "model": model, "section": section, "claims": claims, "usage": usage}
    with _cache_lock:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _load_cache()[key] = record
    return claims, usage


# ---------------------------------------------------------------- deterministic number check

_MONEY = re.compile(r"\$\s?(\d[\d,]*\.?\d*)\s*(trillion|billion|million|bn|tn|[BMT])?\b", re.I)
_PCT = re.compile(r"(?<![\w.])(-?\d[\d,]*\.?\d*)\s?%")
_MULT = re.compile(r"(?<![\w.$])(\d+\.?\d*)\s?(?:x\b|times\b)", re.I)
_SCALE = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6}
_HEDGES = ("about", "approximately", "nearly", "almost", "around", "roughly", "over", "more than", "~", "close to")


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def claim_numbers(text: str) -> list[dict]:
    """Numbers in a claim with their kind and a tolerance derived from how precisely they are written."""
    out = []
    for m in _MONEY.finditer(text):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        scale = _SCALE.get(unit, 1.0)
        out.append({"kind": "money", "value": float(raw) * scale, "ulp": 10 ** -_decimals(raw) * scale, "pos": m.start()})
    for m in _PCT.finditer(text):
        raw = m.group(1).replace(",", "")
        out.append({"kind": "pct", "value": float(raw), "ulp": 10 ** -_decimals(raw), "pos": m.start()})
    for m in _MULT.finditer(text):
        raw = m.group(1)
        out.append({"kind": "mult", "value": float(raw), "ulp": 10 ** -_decimals(raw), "pos": m.start()})
    for n in out:
        before = text[max(0, n["pos"] - 18):n["pos"]].lower()
        n["hedged"] = any(h in before for h in _HEDGES)
    return out


def evidence_numbers(tables: dict[str, pd.DataFrame | None]) -> dict[str, np.ndarray]:
    """Every number in the prompt's tables, plus growth rates, CAGRs, ratios and margin differences derived from them."""
    money, pct, mult = [], [], []
    rows = []  # (row values keyed by year, is_pct_row, is_mult_row)

    fm = tables["financial_metrics"]
    if fm is not None:
        cols = [c for c in fm.columns if c[:4].isdigit()]
        for _, r in fm.iterrows():
            vals = {}
            for c in cols:
                v = r[c]
                if isinstance(v, str):
                    v = float(v.rstrip("%")) if v.endswith("%") else np.nan
                if pd.notna(v):
                    vals[int(c[:4]) + (0.5 if c.endswith("E") else 0)] = float(v)  # keep A and E columns of a year apart
            name = str(r["metrics"])
            is_pct = "Margin" in name or "Growth" in name
            is_mult = name == "PE Ratio"
            rows.append((vals, is_pct, is_mult))
            if isinstance(r.get("CAGR"), str) and r["CAGR"].endswith("%"):
                pct.append(float(r["CAGR"].rstrip("%")))
    for key, is_mult in (("peer_ebitda", False), ("peer_ev_ebitda", True)):
        df = tables.get(key)
        if df is not None:
            for tk in df.columns:
                vals = {int(y): float(v) for y, v in df[tk].items() if pd.notna(v)}
                rows.append((vals, False, is_mult))
            for y, r in df.iterrows():  # same year across companies: ratios like "10x that of AMD"
                vs = [float(v) for v in r.values if pd.notna(v)]
                for a, b in combinations(vs, 2):
                    if a and b:
                        mult += [a / b, b / a]
                        pct += [(a / b - 1) * 100, (b / a - 1) * 100]

    for vals, is_pct, is_mult in rows:
        (pct if is_pct else mult if is_mult else money).extend(vals.values())
        if not is_pct and not is_mult:  # amounts are also quoted per share or as plain numbers
            mult.extend(vals.values())
        items = sorted(vals.items())
        for (y1, v1), (y2, v2) in combinations(items, 2):
            if is_pct:
                pct.append(v2 - v1)  # margin change in pp
                continue
            if v1 and v2:
                mult += [v2 / v1, v1 / v2]
                pct += [(v2 / v1 - 1) * 100, (v1 / v2 - 1) * 100]
                n = round(y2 - y1)
                if n >= 1 and v1 > 0 and v2 > 0:
                    pct.append(((v2 / v1) ** (1 / n) - 1) * 100)
    return {"money": np.array(money), "pct": np.array(pct), "mult": np.array(mult)}


def check_numbers(claim_text: str, ev: dict[str, np.ndarray]) -> tuple[int, int, list[str]]:
    """(numbers found in evidence, numbers in claim, the ones not found)."""
    nums = claim_numbers(claim_text)
    missing = []
    for n in nums:
        pool = ev["money"] if n["kind"] == "money" else ev["pct"] if n["kind"] == "pct" else ev["mult"]
        if n["kind"] == "money" and n["value"] < 1e4:  # per-share amounts (EPS, prices) live with the plain numbers
            pool = np.concatenate([ev["mult"], ev["money"]])
        rel = 0.05 if n["hedged"] else 0.01
        tol = max(0.5 * n["ulp"], rel * abs(n["value"]))
        if not len(pool) or np.min(np.abs(pool - n["value"])) > tol:
            missing.append(f"{n['value']:g}{'%' if n['kind'] == 'pct' else 'x' if n['kind'] == 'mult' else ''}")
    return len(nums) - len(missing), len(nums), missing


# ---------------------------------------------------------------- deterministic period / status rules

_METRIC_COL = re.compile(r"^(\d{4})([AE])$")
_YEAR = re.compile(r"^(\d{4})[AE]?$")
# Wording that presents a figure as a forecast. The judge's claimed_status alone is too noisy to trigger a rule.
_FORECAST_WORDS = re.compile(r"\b(project\w*|expect(?:s|ed|ing)?|forecast\w*|estimat\w*|guid\w*|outlook|anticipat\w*|"
                             r"will|on track to)\b|\b20\d\dE\b", re.I)


def _cell_year(cell: dict) -> int | None:
    """Year of a cited cell, whichever way round the judge wrote row and column (peer tables have years as rows)."""
    for part in (cell["column"].strip(), cell["row"].strip()):
        m = _YEAR.match(part)
        if m:
            return int(m.group(1))
    return None


def _cell_status(cell: dict, peer_forecast_year: int | None) -> str | None:
    """'actual' / 'forecast' for a cited cell. Peer EV/EBITDA is all history; the peer EBITDA table's last year is
    the pipeline's projection (market_data_api.project_ebitda_for_peers)."""
    m = _METRIC_COL.match(cell["column"].strip())
    if m:
        return "actual" if m.group(2) == "A" else "forecast"
    table, year = cell["table"].lower(), _cell_year(cell)
    if "ev/ebitda" in table:
        return "actual"
    if "ebitda" in table and year:
        return "forecast" if year == peer_forecast_year else "actual"
    return None


def rule_flags(claim: dict, last_actual_year: int, peer_forecast_year: int | None = None) -> list[str]:
    """Contradictions that follow mechanically from the cells the judge cited and how the claim words them."""
    cols = [c["column"].strip() for c in claim["cells"]]
    labelled = [(f"{c['table'].strip()} / {c['row'].strip()} / {c['column'].strip()}", _cell_status(c, peer_forecast_year))
                for c in claim["cells"]]
    actuals = sorted({name for name, s in labelled if s == "actual"})
    forecasts = sorted({name for name, s in labelled if s == "forecast"})
    status = claim["claimed_status"]
    worded_as_forecast = bool(_FORECAST_WORDS.search(claim["claim"]) or _FORECAST_WORDS.search(claim["quote"]))
    flags = []
    if status in ("forecast", "mixed") and worded_as_forecast and actuals and not forecasts:
        flags.append(f"reported actuals presented as a forecast ({'; '.join(actuals)})")
    if status == "actual" and not worded_as_forecast and forecasts and not actuals:
        flags.append(f"pipeline forecasts presented as reported ({'; '.join(forecasts)})")
    if "CAGR" in cols and (status == "forecast" or any(y > last_actual_year for y in claim["claimed_years"])):
        flags.append("historical CAGR attributed to a forecast period")
    cited_years = {y for y in map(_cell_year, claim["cells"]) if y}
    if claim["claimed_years"] and cited_years and not set(claim["claimed_years"]) & cited_years:
        flags.append(f"claim says {sorted(claim['claimed_years'])}, cited cells are {sorted(cited_years)}")
    return flags


# ---------------------------------------------------------------- one run

def claim_rows_for_text(claims: list[dict], text: str, tables: dict, ids: dict) -> list[dict]:
    """Applies the deterministic rules and the number check to a judge's claims for one text."""
    ev = evidence_numbers(tables)
    last_actual = max(int(c[:4]) for c in tables["financial_metrics"].columns if _METRIC_COL.match(c) and c.endswith("A"))
    pe = tables.get("peer_ebitda")
    peer_forecast_year = int(pe.index.max()) if pe is not None and len(pe) else None
    rows = []
    for i, c in enumerate(claims, 1):
        found, total, missing = check_numbers(c["claim"], ev)
        flags = rule_flags(c, last_actual, peer_forecast_year)
        rows.append({
            **ids, "claim_id": "-".join(str(v) for v in ids.values()) + f"-{i}",
            "claim": c["claim"], "quote": c["quote"], "kind": c["kind"],
            "judge_verdict": c["verdict"], "rule_flags": "; ".join(flags),
            "verdict": "Contradicted" if flags else c["verdict"],
            "reason": c["reason"],
            "cells": "; ".join(f"{x['table']} / {x['row']} / {x['column']} = {x['value']}" for x in c["cells"]),
            "claimed_years": ",".join(map(str, c["claimed_years"])), "claimed_status": c["claimed_status"],
            "numbers_found": found, "numbers_total": total, "numbers_missing": ", ".join(missing),
            "quote_in_text": c["quote"] in text,
        })
    return rows


def score_run(client, model: str, ticker: str, company: str, run_dir: Path | None = None,
              skip: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(claims, sections) for every section text of one pipeline run."""
    run_dir = run_dir or OUT / ticker
    tables = load_tables(run_dir)
    claim_rows, section_rows = [], []
    for (section, variant), text in section_texts(run_dir).items():
        status = "scored"
        if section in skip:
            status = "skipped"
        system_prompt, user_prompt = writer_prompt(run_dir, ticker, company, section)
        claims, usage = ([], {}) if status == "skipped" else judge(client, model, system_prompt, user_prompt, section, text)
        section_rows.append({"ticker": ticker, "section": section, "variant": variant, "status": status,
                             "words": len(text.split()), "claims": len(claims), **usage, "text": text})
        claim_rows += claim_rows_for_text(claims, text, tables, {"ticker": ticker, "section": section, "variant": variant})
    return pd.DataFrame(claim_rows), pd.DataFrame(section_rows)


# ---------------------------------------------------------------- LLM-free status-error counter

_STATUS_WORDS = re.compile(r"\b(project\w*|expect(?:s|ed|ing)?|forecast\w*|estimat\w*|guid\w*|outlook|anticipat\w*|"
                           r"will (?:reach|grow|rise)|on track to)\b|\b20\d\dE\b", re.I)
_SOURCE_WORDS = re.compile(r"(?<!no )(?<!not )\b(guidance|consensus|analysts? (?:expect|estimate|forecast|project)\w*|"
                           r"street (?:estimates?|expectations?))\b", re.I)
_CAGR = re.compile(r"CAGR", re.I)
_NEGATION = re.compile(r"\b(not|no|nor|without|rather than|isn't|aren't|absent)\b", re.I)
_REPORTED_LABEL = re.compile(r"\b(?:FY)?20\d\dA\b|\breported\b", re.I)
_E_LABEL = re.compile(r"\b20\d\dE\b")
_FROM_BASE = re.compile(r"\bfrom\s+(?:about\s+|around\s+|roughly\s+|an?\s+)?$", re.I)  # "from $26.9B ... to ..."
_CLAUSE_BREAK = re.compile(r"(?<!\d)[.,](?!\d)|[.,](?=\s)|[;\n]|\s(?:but|while|whereas)\s", re.I)
_YEAR_WORD = re.compile(r"\b(?:FY)?(20\d\d)E?\b")


def status_errors(text: str, tables: dict) -> dict:
    """Counts the period/status errors found in the audit without any LLM, so it can score texts independently of
    both the judge and the verifier:
      actual_as_forecast - a $ amount or margin that matches only a reported (A) cell, with forecast wording or an
                           "E" year label within the same clause;
      cagr_period        - "CAGR" with a year after the latest reported year, or forecast wording, in the same clause;
      invented_source    - guidance / consensus / analyst expectations (the evidence contains none).
    """
    fm = tables["financial_metrics"]
    a_cols = [c for c in fm.columns if _METRIC_COL.match(c) and c.endswith("A")]
    e_cols = [c for c in fm.columns if _METRIC_COL.match(c) and c.endswith("E")]
    last_actual = max(int(c[:4]) for c in a_cols)
    a_vals, e_vals = {"money": [], "pct": []}, {"money": [], "pct": []}
    for _, r in fm.iterrows():
        for cols, store in ((a_cols, a_vals), (e_cols, e_vals)):
            for c in cols:
                v = r[c]
                if isinstance(v, str) and v.endswith("%"):
                    store["pct"].append(float(v.rstrip("%")))
                elif isinstance(v, (int, float)) and pd.notna(v) and abs(v) >= 1e8:
                    store["money"].append(float(v))

    def clause_bounds(pos: int) -> tuple[int, int]:
        """The clause around pos: bounded by . , ; but/while/whereas or a line break, but not by the point or comma
        inside a number ("$130.5", "1,234"); parentheses stay in, so "$120 billion (2026E)" is one clause."""
        left, right = 0, len(text)
        for m in _CLAUSE_BREAK.finditer(text):
            if m.start() < pos:
                left = m.end()
            else:
                right = m.start()
                break
        return left, right

    def clause(pos: int) -> str:
        left, right = clause_bounds(pos)
        return text[left:right]

    counts = {"actual_as_forecast": 0, "cagr_period": 0, "invented_source": 0}
    for n in claim_numbers(text):
        if n["kind"] not in ("money", "pct"):
            continue
        tol = max(0.5 * n["ulp"], 0.01 * abs(n["value"]))
        match = lambda vals: any(abs(v - n["value"]) <= tol for v in vals)
        before = text[max(0, n["pos"] - 30):n["pos"]]
        # "$120 billion in 2026A" is labelled correctly, unless forecast wording leads into it ("projected ... 2026A")
        explicitly_reported = _REPORTED_LABEL.search(text[n["pos"]:n["pos"] + 30]) and not _STATUS_WORDS.search(before)
        change_base = _FROM_BASE.search(before)  # the starting point of a from-to change is a reported anchor
        # The forecast wording has to lead into the figure ("projected to reach $130.5B") or an E label has to follow
        # it ("$130.5B (2025E)"); wording after it belongs to the next figure ("to 58.2% in 2024 and forecast to 62.7%").
        left, _ = clause_bounds(n["pos"])
        worded = _STATUS_WORDS.search(text[left:n["pos"]]) or _E_LABEL.search(text[n["pos"]:n["pos"] + 25])
        if (match(a_vals[n["kind"]]) and not match(e_vals[n["kind"]]) and not explicitly_reported
                and not change_base and worded):
            counts["actual_as_forecast"] += 1
    for m in _CAGR.finditer(text):
        c = clause(m.start())
        if _STATUS_WORDS.search(c) or any(int(y) > last_actual for y in _YEAR_WORD.findall(c)):
            counts["cagr_period"] += 1
    # "not company guidance or consensus" is the opposite of citing one
    counts["invented_source"] = sum(1 for m in _SOURCE_WORDS.finditer(text)
                                    if not _NEGATION.search(text[max(0, m.start() - 45):m.start()]))
    counts["total"] = sum(counts.values())
    return counts
