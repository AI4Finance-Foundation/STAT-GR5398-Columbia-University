"""Part 3 experiment: does annotating the evidence and/or adding the verifier reduce grounding errors?

2 x 2 design, same saved data for every condition (no data refetch):
  C0    original prompt (forecast columns labelled 2025E-2027E, no notes)       -> writer
  C1    annotated prompt (FY+n labels, report date, what is reported/forecast) -> writer
  C0+V  C0 text checked by the verifier (same tables as C0, plus report date and reading notes)
  C1+V  C1 text checked by the verifier against the C1 prompt
The verifier always gets the report date and notes: that is part of its design. The factor is what the writer saw.
Each text is scored against the prompt its writer saw.

Writer: exactly as text_generator_agents.generate_text_section calls it (gpt-4.1, temperature 0.7, max_tokens 1000).
Verifier: modules/claim_verifier.py from the Part 3 branch (gpt-4.1-mini).
Scoring: the Part 2.3 grounding scorer (gpt-4.1 judge + rules) against the prompt each writer saw, plus the
LLM-free status-error counter. Every call is cached in audit/data, so re-running is free.

Usage:  python audit/part3_experiment.py [--workers 8]
"""
import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WT_SRC = ROOT / "FinRobot-part3" / "finrobot_equity" / "core" / "src"
sys.path.insert(0, str(WT_SRC))  # the Part 3 branch must be imported before grounding.py adds the main checkout
from modules.text_generator_agents import SYSTEM_PROMPTS, _prepare_user_prompt  # noqa: E402
from modules.claim_verifier import VERIFIER_INSTRUCTIONS, verify_section  # noqa: E402

sys.path.insert(0, str(ROOT / "audit"))
import grounding as g  # noqa: E402
from openai import OpenAI  # noqa: E402

COMPANIES = {
    # development set: the five names the audit (and the verifier prompt) were built on
    "NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices, Inc.", "INTC": "Intel Corporation",
    "AAPL": "Apple Inc.", "GOOGL": "Alphabet Inc.",
    # held-out set: healthcare, never used while designing or tuning the changes (pilots: NVDA, AMD)
    "PFE": "Pfizer Inc.", "ABBV": "AbbVie Inc.", "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Incorporated", "MRNA": "Moderna, Inc.",
}
HELD_OUT = {"PFE", "ABBV", "JNJ", "UNH", "MRNA"}
SECTIONS = ["tagline", "company_overview", "investment_overview", "valuation_overview", "risks",
            "competitor_analysis", "major_takeaways"]
WRITER_MODEL, VERIFIER_MODEL, JUDGE_MODEL = "gpt-4.1", "gpt-4.1-mini", "gpt-4.1"
CACHE = ROOT / "audit" / "data" / "part3_cache.jsonl"
_lock = threading.Lock()
_cache: dict = {}


def _cached(kind: str, key_parts: list, fn):
    key = hashlib.sha256(json.dumps([kind] + key_parts, sort_keys=True).encode()).hexdigest()
    with _lock:
        if not _cache and CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    _cache[r["key"]] = r["value"]
        if key in _cache:
            return _cache[key]
    value = fn()
    with _lock:
        _cache[key] = value
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "kind": kind, "value": value}) + "\n")
    return value


def relabel(fm: pd.DataFrame) -> pd.DataFrame:
    """The Part 3 branch labels forecast years FY+1..FY+3 from the base year; apply that to saved tables."""
    base = max(int(c[:4]) for c in fm.columns if c.endswith("A") and c[:4].isdigit())
    return fm.rename(columns={"2025E": f"{base + 1}E", "2026E": f"{base + 2}E", "2027E": f"{base + 3}E"})


def prompts(ticker: str) -> dict:
    run = g.OUT / ticker
    t0 = g.load_tables(run)
    t1 = {**t0, "financial_metrics": relabel(t0["financial_metrics"])}
    report_date = json.loads((run / "analysis" / "analysis_summary.json").read_text())["analysis_date"][:10]
    out = {}
    for s in SECTIONS:
        out[("C0", s)] = (_prepare_user_prompt(dict(t0), s, COMPANIES[ticker], ticker, annotate=False), t0)
        # What the verifier gets when it checks a C0 text: the same tables, plus the report date and reading notes.
        # Without the date it repeats the writer's mistake (it too assumes "now" is its training cutoff).
        out[("C0n", s)] = (_prepare_user_prompt({**t0, "report_date": report_date}, s, COMPANIES[ticker], ticker,
                                                annotate=True), t0)
        out[("C1", s)] = (_prepare_user_prompt({**t1, "report_date": report_date}, s, COMPANIES[ticker], ticker,
                                               annotate=True), t1)
    return out


def write(client, section: str, prompt: str) -> str:
    def call():
        r = client.chat.completions.create(
            model=WRITER_MODEL, temperature=0.7, max_tokens=1000,
            messages=[{"role": "system", "content": SYSTEM_PROMPTS[section]}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    return _cached("writer", [WRITER_MODEL, section, prompt, 0], call)


def verify(client, section: str, prompt: str, text: str) -> dict:
    def call():
        revised, log = verify_section(text, section, prompt, api_key=None, model=VERIFIER_MODEL, client=client)
        return {"text": revised, "log": log}
    version = hashlib.sha256(VERIFIER_INSTRUCTIONS.encode()).hexdigest()[:12]
    return _cached("verifier", [VERIFIER_MODEL, version, section, prompt, text], call)


def run_unit(client, ticker: str, section: str, P: dict) -> tuple[list, list, list]:
    texts = {}
    for cond in ("C0", "C1"):
        prompt, _ = P[(cond, section)]
        texts[cond] = (write(client, section, prompt), None)
        verifier_prompt = P[("C0n" if cond == "C0" else "C1", section)][0]
        v = verify(client, section, verifier_prompt, texts[cond][0])
        texts[cond + "+V"] = (v["text"], v["log"])
    section_rows, claim_rows, text_rows = [], [], []
    for cond, (text, log) in texts.items():
        prompt, tables = P[(cond.replace("+V", ""), section)]
        claims, _ = g.judge(client, JUDGE_MODEL, SYSTEM_PROMPTS[section], prompt, section, text)
        ids = {"ticker": ticker, "section": section, "condition": cond}
        rows = g.claim_rows_for_text(claims, text, tables, ids)
        claim_rows += rows
        verdicts = pd.Series([r["verdict"] for r in rows], dtype=object)
        section_rows.append({
            **ids, "held_out": ticker in HELD_OUT, "words": len(text.split()), "claims": len(rows),
            **{v.lower(): int((verdicts == v).sum()) for v in ("Supported", "Unsupported", "Contradicted")},
            "quant_claims": sum(r["kind"] == "quantitative" for r in rows),
            "quant_contradicted": sum(r["kind"] == "quantitative" and r["verdict"] == "Contradicted" for r in rows),
            **{f"counter_{k}": v for k, v in g.status_errors(text, tables).items()},
            "verifier_corrections": len(log["corrections"]) if log else None,
            "verifier_not_applied": len(log["not_applied"]) if log else None,
            "verifier_unverified": len(log["unverified"]) if log else None,
            "verifier_error": (log or {}).get("error"),
        })
        text_rows.append({**ids, "text": text, "verifier_log": log})
    return section_rows, claim_rows, text_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    from modules.common_utils import load_config, get_api_key
    client = OpenAI(api_key=get_api_key(load_config(str(g.SRC.parent / "config" / "config.ini")),
                                        section="API_KEYS", key="openai_api_key"), max_retries=8)
    P = {t: prompts(t) for t in COMPANIES}
    units = [(t, s) for t in COMPANIES for s in SECTIONS]
    sections, claims, texts = [], [], []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_unit, client, t, s, P[t]): (t, s) for t, s in units}
        for i, f in enumerate(as_completed(futures), 1):
            s_rows, c_rows, t_rows = f.result()
            sections += s_rows
            claims += c_rows
            texts += t_rows
            if i % 10 == 0 or i == len(units):
                print(f"{i}/{len(units)} units done", flush=True)
    out = ROOT / "audit"
    pd.DataFrame(sections).sort_values(["ticker", "section", "condition"]).to_csv(out / "part3_sections.csv", index=False)
    pd.DataFrame(claims).to_csv(out / "part3_claims.csv", index=False)
    with open(out / "part3_texts.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(texts, key=lambda r: (r["ticker"], r["section"], r["condition"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved part3_sections.csv, part3_claims.csv, part3_texts.jsonl")


if __name__ == "__main__":
    main()
