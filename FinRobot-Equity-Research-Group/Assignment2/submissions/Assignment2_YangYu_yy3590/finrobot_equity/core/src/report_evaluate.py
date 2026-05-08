#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import re
from typing import Dict

from modules.common_utils import load_config
from modules.llm_gateway import call_llm, load_llm_settings


DEFAULT_SINGLE_REPORT_PROMPT = """You are an independent equity research evaluator.

Evaluate the following equity research report using the rubric below.
Do not judge writing style alone. Focus on investment-research usefulness,
financial reasoning, company-specific analysis, and factual grounding.

Company: {ticker}
Report:
{report_text}

Rubric:
1. Thesis Clarity: 1-5
2. Catalyst Quality: 1-5
3. Risk Specificity: 1-5
4. Financial Reasoning: 1-5
5. Report Structure: 1-5
6. Depth of Analysis: 1-5
7. Factual Grounding: 1-5
8. Analyst Usefulness: 1-5

For each dimension, provide:
- score
- one-sentence justification

Then provide:
- overall_score from 1-100
- top_strength
- main_weakness

Return strict JSON only, with this schema:
{{
  "dimensions": {{
    "thesis_clarity": {{"score": 1, "justification": "..."}},
    "catalyst_quality": {{"score": 1, "justification": "..."}},
    "risk_specificity": {{"score": 1, "justification": "..."}},
    "financial_reasoning": {{"score": 1, "justification": "..."}},
    "report_structure": {{"score": 1, "justification": "..."}},
    "depth_of_analysis": {{"score": 1, "justification": "..."}},
    "factual_grounding": {{"score": 1, "justification": "..."}},
    "analyst_usefulness": {{"score": 1, "justification": "..."}}
  }},
  "overall_score": 1,
  "top_strength": "...",
  "main_weakness": "..."
}}
"""


COMPARE_REPORT_PROMPT = """You are an independent equity research evaluator.

Compare the two equity research reports below and decide which one is more useful for investment decision-making.
Focus on thesis quality, financial reasoning, risk/catalyst quality, structure completeness, and factual grounding.

Company: {ticker}

Report A:
{report_a}

Report B:
{report_b}

Return strict JSON only:
{{
  "winner": "A" or "B" or "TIE",
  "score_a": 1-100,
  "score_b": 1-100,
  "winner_reason": "...",
  "a_main_weakness": "...",
  "b_main_weakness": "..."
}}
"""


def _strip_html(text: str) -> str:
    text_no_script = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text_no_style = re.sub(r"<style[\s\S]*?</style>", " ", text_no_script, flags=re.IGNORECASE)
    text_no_tags = re.sub(r"<[^>]+>", " ", text_no_style)
    return re.sub(r"\s+", " ", text_no_tags).strip()


def _load_report(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if path.lower().endswith(".html"):
        return _strip_html(raw)
    return raw


def _extract_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Could not parse JSON from model response.")
    return json.loads(match.group(0))


def evaluate_structure(report_text: str) -> Dict:
    lower = report_text.lower()

    sections = [
        ("tagline", 10),
        ("company overview", 10),
        ("investment overview", 10),
        ("valuation overview", 10),
        ("risks", 10),
        ("competitor analysis", 10),
        ("major takeaways", 10),
        ("news summary", 10),
    ]

    section_scores = {}
    total = 0
    for section_name, weight in sections:
        hit = section_name in lower
        score = weight if hit else 0
        section_scores[section_name] = score
        total += score

    word_count = len(report_text.split())
    if word_count >= 1200:
        length_score = 20
    elif word_count >= 800:
        length_score = 15
    elif word_count >= 400:
        length_score = 8
    else:
        length_score = 0
    total += length_score

    return {
        "score": total,
        "max_score": 100,
        "word_count": word_count,
        "section_scores": section_scores,
        "length_score": length_score,
        "missing_sections": [name for name, _ in sections if section_scores[name] == 0],
    }


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def evaluate_single_report(
    report_text: str,
    ticker: str,
    llm_settings,
    max_chars: int,
    evaluated_model_provider: str | None = None,
    evaluated_model_name: str | None = None,
    evaluated_model_tag: str | None = None,
) -> Dict:
    structure_eval = evaluate_structure(report_text)
    prompt = DEFAULT_SINGLE_REPORT_PROMPT.format(
        ticker=ticker,
        report_text=_truncate(report_text, max_chars=max_chars),
    )
    llm_raw = call_llm(
        settings=llm_settings,
        instructions="Return strict JSON only.",
        prompt=prompt,
        max_output_tokens=50000,
        temperature=0.1,
    )
    llm_json = _extract_json(llm_raw)
    llm_score = llm_json.get("overall_score", 0)
    combined_score = round(structure_eval["score"] * 0.3 + float(llm_score) * 0.7, 2)

    return {
        "mode": "single",
        "provider": llm_settings.provider,
        "model": llm_settings.model,
        "judge_provider": llm_settings.provider,
        "judge_model": llm_settings.model,
        "evaluated_model_provider": evaluated_model_provider,
        "evaluated_model_name": evaluated_model_name,
        "evaluated_model_tag": evaluated_model_tag,
        "ticker": ticker,
        "structure_evaluation": structure_eval,
        "llm_evaluation": llm_json,
        "combined_score": combined_score,
    }


def evaluate_report_pair(
    report_a_text: str,
    report_b_text: str,
    ticker: str,
    llm_settings,
    max_chars: int,
    evaluated_model_a_provider: str | None = None,
    evaluated_model_a_name: str | None = None,
    evaluated_model_a_tag: str | None = None,
    evaluated_model_b_provider: str | None = None,
    evaluated_model_b_name: str | None = None,
    evaluated_model_b_tag: str | None = None,
) -> Dict:
    struct_a = evaluate_structure(report_a_text)
    struct_b = evaluate_structure(report_b_text)

    prompt = COMPARE_REPORT_PROMPT.format(
        ticker=ticker,
        report_a=_truncate(report_a_text, max_chars=max_chars),
        report_b=_truncate(report_b_text, max_chars=max_chars),
    )
    llm_raw = call_llm(
        settings=llm_settings,
        instructions="Return strict JSON only.",
        prompt=prompt,
        max_output_tokens=50000,
        temperature=0.1,
    )
    llm_json = _extract_json(llm_raw)

    return {
        "mode": "compare",
        "provider": llm_settings.provider,
        "model": llm_settings.model,
        "judge_provider": llm_settings.provider,
        "judge_model": llm_settings.model,
        "evaluated_model_a_provider": evaluated_model_a_provider,
        "evaluated_model_a_name": evaluated_model_a_name,
        "evaluated_model_a_tag": evaluated_model_a_tag,
        "evaluated_model_b_provider": evaluated_model_b_provider,
        "evaluated_model_b_name": evaluated_model_b_name,
        "evaluated_model_b_tag": evaluated_model_b_tag,
        "ticker": ticker,
        "structure_a": struct_a,
        "structure_b": struct_b,
        "llm_comparison": llm_json,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or two equity research reports with structure checks + LLM scoring/comparison."
    )
    parser.add_argument("--report-a", required=True, help="Path to report A (single mode if report-b is omitted).")
    parser.add_argument("--report-b", default=None, help="Optional path to report B for comparison mode.")
    parser.add_argument("--ticker", required=True, help="Company ticker for evaluation context.")
    parser.add_argument("--config-file", default=None, help="Path to config.ini.")
    parser.add_argument("--llm-provider", default=None, help="Override provider: openai, claude, or gemini.")
    parser.add_argument("--llm-model", default=None, help="Override model name.")
    parser.add_argument("--evaluated-model-provider", default=None, help="Provider of evaluated report in single mode.")
    parser.add_argument("--evaluated-model-name", default=None, help="Model name of evaluated report in single mode.")
    parser.add_argument("--evaluated-model-tag", default=None, help="Model tag of evaluated report in single mode.")
    parser.add_argument("--evaluated-model-a-provider", default=None, help="Provider of report A in compare mode.")
    parser.add_argument("--evaluated-model-a-name", default=None, help="Model name of report A in compare mode.")
    parser.add_argument("--evaluated-model-a-tag", default=None, help="Model tag of report A in compare mode.")
    parser.add_argument("--evaluated-model-b-provider", default=None, help="Provider of report B in compare mode.")
    parser.add_argument("--evaluated-model-b-name", default=None, help="Model name of report B in compare mode.")
    parser.add_argument("--evaluated-model-b-tag", default=None, help="Model tag of report B in compare mode.")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars per report sent to LLM.")
    parser.add_argument("--output-file", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()

    report_a_path = os.path.abspath(args.report_a)
    report_b_path = os.path.abspath(args.report_b) if args.report_b else None

    report_a_text = _load_report(report_a_path)
    report_b_text = _load_report(report_b_path) if report_b_path else None

    config = load_config(args.config_file)
    llm_settings = load_llm_settings(config, provider=args.llm_provider, model=args.llm_model)

    if report_b_text:
        result = evaluate_report_pair(
            report_a_text=report_a_text,
            report_b_text=report_b_text,
            ticker=args.ticker,
            llm_settings=llm_settings,
            max_chars=args.max_chars,
            evaluated_model_a_provider=args.evaluated_model_a_provider,
            evaluated_model_a_name=args.evaluated_model_a_name,
            evaluated_model_a_tag=args.evaluated_model_a_tag,
            evaluated_model_b_provider=args.evaluated_model_b_provider,
            evaluated_model_b_name=args.evaluated_model_b_name,
            evaluated_model_b_tag=args.evaluated_model_b_tag,
        )
    else:
        result = evaluate_single_report(
            report_text=report_a_text,
            ticker=args.ticker,
            llm_settings=llm_settings,
            max_chars=args.max_chars,
            evaluated_model_provider=args.evaluated_model_provider,
            evaluated_model_name=args.evaluated_model_name,
            evaluated_model_tag=args.evaluated_model_tag,
        )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(output_json)

    if args.output_file:
        output_path = os.path.abspath(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nSaved evaluation result to: {output_path}")


if __name__ == "__main__":
    main()
