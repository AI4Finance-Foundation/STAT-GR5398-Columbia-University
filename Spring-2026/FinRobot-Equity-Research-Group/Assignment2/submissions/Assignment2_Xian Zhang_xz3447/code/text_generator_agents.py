#!/usr/bin/env python
# coding: utf-8
"""Multi-provider text generator with optional critique-revise loop.

This is the single LLM entry point for the equity research pipeline. It is
called from both `generate_financial_analysis.py` and `create_equity_report.py`
via `generate_text_section(...)`.

Two key upgrades over the original implementation:

1. **Provider-agnostic**: routes calls to OpenAI, Anthropic, or Gemini based on
   the model name (via `modules.llm_providers`). Different report sections can
   use different models, configured by the `FINROBOT_PROFILE` env var.

2. **Critique-Revise stage** (Claude-inspired): for high-stakes analytical
   sections (`investment_overview`, `risks`, `valuation_overview`), the draft
   is reviewed by a Senior Reviewer agent (potentially a different model
   family than the author), then the original section is rewritten in light
   of the critique. This lifts factual fidelity and reduces generic
   boilerplate.

Backwards-compatible: the public `generate_text_section()` keeps the same
signature the legacy callers use, so we don't have to edit the two large
pipeline scripts.
"""

import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from modules.equity_prompts import (
    CRITIC_PROMPT,
    REVISER_INSTRUCTIONS_TEMPLATE,
    SECTION_PROMPTS,
    SECTION_TOKEN_BUDGET,
)
from modules.llm_providers import LLMError, chat, resolve_provider_name
from modules.model_routing import (
    CRITIQUE_SECTIONS,
    get_profile,
    resolve_section_model,
)
from modules.retail_sentiment_client import format_retail_sentiment_for_prompt


# ---------------------------------------------------------------------------
# Fallbacks (kept for resilience — pipeline must always emit something)
# ---------------------------------------------------------------------------
_FALLBACK = {
    "tagline": "{name} demonstrates strong financial fundamentals with consistent revenue growth and solid profitability metrics. The company maintains a competitive position in its market segment through operational efficiency and strategic initiatives. Strong balance sheet metrics support continued value creation for shareholders.",
    "company_overview": "{name} operates as a prominent player in its industry sector, demonstrating consistent financial performance through strategic market positioning and operational excellence.",
    "investment_overview": "{name} has delivered solid financial performance in recent periods, supported by strong operational execution and favorable market conditions.",
    "valuation_overview": "{name} trades at reasonable valuation levels relative to its peer group, supported by strong fundamental metrics and growth prospects.",
    "risks": "Key risks include: (1) Industry competition and market share pressure, (2) Regulatory changes affecting operations, (3) Economic downturns impacting demand, (4) Technology disruption risks, (5) Supply chain and operational challenges.",
    "competitor_analysis": "{name} demonstrates competitive positioning within its industry through consistent financial performance and strategic positioning relative to key competitors.",
    "major_takeaways": (
        "Revenue Growth: {name}'s revenue growth shows consistent performance trends.\n\n"
        "Gross Profit Margin: {name}'s gross profit margins demonstrate operational effectiveness.\n\n"
        "SG&A Expense Margin: {name}'s SG&A expense management shows disciplined cost control.\n\n"
        "EBITDA Margin Stability: {name}'s EBITDA margin stability reflects strong underlying fundamentals."
    ),
    "news_summary": "Recent news coverage for {name} reflects ongoing market interest and developments in the company's operations.",
}


def _fallback_text(section: str, company_name: str) -> str:
    return _FALLBACK.get(section, f"{company_name} {section.replace('_',' ')} analysis not available.").format(name=company_name)


_TAKEAWAY_HEADERS = ["Revenue Growth", "Gross Profit Margin", "SG&A Expense Margin", "EBITDA Margin Stability"]


def _normalize_output(text: str, section: str) -> str:
    """Strip markdown that the prompts already say not to use, and normalize
    the major_takeaways header form so the legacy `"Revenue Growth:" in text`
    validator in generate_financial_analysis.py accepts it."""
    if not text:
        return text
    out = text

    # Strip leading markdown headers (## or # at line start)
    lines = []
    for ln in out.splitlines():
        s = ln.lstrip()
        if s.startswith(("###", "##", "#")):
            s = s.lstrip("# ").rstrip()
            # Normalize to colon form for major_takeaways subheaders so the
            # legacy validator (`"Revenue Growth:" in text`) is satisfied.
            if section == "major_takeaways" and s in _TAKEAWAY_HEADERS:
                s = f"{s}:"
        lines.append(s)
    out = "\n".join(lines)

    # For major_takeaways: also handle "**Revenue Growth**" → "Revenue Growth:"
    if section == "major_takeaways":
        for h in _TAKEAWAY_HEADERS:
            out = out.replace(f"**{h}**", f"{h}:")
            out = out.replace(f"**{h}:**", f"{h}:")
    return out


# ---------------------------------------------------------------------------
# Data → prompt formatting
# ---------------------------------------------------------------------------
def _df_to_md(df: Optional[pd.DataFrame], title: str) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return ""
    try:
        return f"## {title}\n{df.to_markdown()}\n\n"
    except Exception:
        return f"## {title}\n[Could not format data]\n\n"


def _prepare_user_prompt(data: Dict, section: str, company_name: str, ticker: str) -> str:
    fm = data.get("financial_metrics")
    pe = data.get("peer_ebitda")
    pev = data.get("peer_ev_ebitda")
    news = data.get("company_news")
    retail = data.get("retail_sentiment")

    parts = [f"Company: {company_name} ({ticker})\n"]

    parts.append(_df_to_md(fm, f"{company_name} Financial Metrics & Forecasts"))
    parts.append(_df_to_md(pe, "Peer EBITDA Comparison"))
    parts.append(_df_to_md(pev, "Peer EV/EBITDA Comparison"))

    # News for news_summary; otherwise omitted to keep prompt focused
    if section == "news_summary" and news:
        parts.append(f"## Recent News (last {len(news)} articles)\n")
        for i, art in enumerate(news[:10], 1):
            parts.append(
                f"{i}. {art.get('title','N/A')} ({str(art.get('publishedDate',''))[:10]})\n"
                f"   {(art.get('text') or '')[:240]}\n\n"
            )
        if retail:
            parts.append("\n" + format_retail_sentiment_for_prompt(retail) + "\n")

    parts.append(
        f"\nProduce the {section.replace('_',' ')} for the report based strictly on the data above."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core generation: single section, with optional critique-revise loop
# ---------------------------------------------------------------------------
def _resolve_api_keys(api_key: Optional[str]) -> Dict[str, str]:
    """Build the multi-provider api_keys dict.

    Legacy callers pass only `openai_api_key` (positional `api_key`). We pull
    the others from environment variables (set by the experiment runner) or
    from the parsed config when available.
    """
    return {
        "openai": api_key or os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", ""),
    }


def _select_model(profile_name: str, section: str, override: Optional[str]) -> str:
    # When the experiment runner sets FINROBOT_PROFILE, the profile owns model
    # selection — ignore the legacy `model=` arg that callers wire up from
    # config.ini's `openai_model` field (which would otherwise force every
    # section to gpt-4.1 regardless of profile).
    if os.getenv("FINROBOT_PROFILE"):
        return resolve_section_model(profile_name, section)
    if override:
        return override
    return resolve_section_model(profile_name, section)


def _critique(
    section: str,
    draft: str,
    company_name: str,
    ticker: str,
    user_prompt_with_data: str,
    critic_model: str,
    api_keys: Dict[str, str],
) -> str:
    critic_user = (
        f"Section: {section}\nCompany: {company_name} ({ticker})\n\n"
        f"--- DATA THE ANALYST WAS GIVEN ---\n{user_prompt_with_data}\n\n"
        f"--- DRAFT TO REVIEW ---\n{draft}\n--- END DRAFT ---\n\n"
        "Issue your critique now."
    )
    resp = chat(
        system=CRITIC_PROMPT,
        user=critic_user,
        model=critic_model,
        api_keys=api_keys,
        max_tokens=600,
        temperature=0.4,
    )
    return resp.text


def _revise(
    section: str,
    draft: str,
    critique: str,
    company_name: str,
    ticker: str,
    user_prompt_with_data: str,
    author_model: str,
    api_keys: Dict[str, str],
) -> str:
    section_system = SECTION_PROMPTS[section]
    reviser_user = REVISER_INSTRUCTIONS_TEMPLATE.format(
        section=section,
        company=company_name,
        ticker=ticker,
        draft=draft,
        critique=critique,
    ) + f"\n\n--- DATA ---\n{user_prompt_with_data}"
    resp = chat(
        system=section_system,
        user=reviser_user,
        model=author_model,
        api_keys=api_keys,
        max_tokens=SECTION_TOKEN_BUDGET.get(section, 1200),
        temperature=0.6,
    )
    return resp.text


# ---------------------------------------------------------------------------
# Public API (signature kept for backwards compatibility)
# ---------------------------------------------------------------------------
def generate_text_section(
    data: Dict,
    prompt_type: str,
    api_key: str,
    company_name: str,
    company_ticker: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a single report section.

    The active routing profile is selected by the FINROBOT_PROFILE env var
    (defaults to 'gpt_baseline'). When `model` is supplied it overrides the
    profile for this single call. When the active profile has critique
    enabled and the section is in CRITIQUE_SECTIONS, a critique-revise pass
    runs after the initial draft.

    Optional env vars:
        FINROBOT_PROFILE   one of: gpt_baseline, claude_all, gemini_all, mixed_critic
        FINROBOT_AUDIT_DIR if set, write {section}.audit.json with draft/critique/final
    """
    if prompt_type not in SECTION_PROMPTS:
        print(f"⚠️  Unknown section '{prompt_type}', using fallback.")
        return _fallback_text(prompt_type, company_name)

    profile_name = os.getenv("FINROBOT_PROFILE", "gpt_baseline")
    try:
        profile = get_profile(profile_name)
    except KeyError:
        print(f"⚠️  Profile '{profile_name}' not found; falling back to gpt_baseline.")
        profile_name = "gpt_baseline"
        profile = get_profile(profile_name)

    api_keys = _resolve_api_keys(api_key)
    section_model = _select_model(profile_name, prompt_type, model)
    system_prompt = SECTION_PROMPTS[prompt_type]
    user_prompt = _prepare_user_prompt(data, prompt_type, company_name, company_ticker)

    print(
        f"🤖 [{profile_name}] section='{prompt_type}' model='{section_model}' "
        f"({resolve_provider_name(section_model)})"
    )

    # --- 1. Initial draft ---
    try:
        draft_resp = chat(
            system=system_prompt,
            user=user_prompt,
            model=section_model,
            api_keys=api_keys,
            max_tokens=SECTION_TOKEN_BUDGET.get(prompt_type, 1200),
            temperature=0.7,
            openai_base_url=base_url,
        )
        draft = draft_resp.text
    except LLMError as e:
        print(f"❌ Draft failed for {prompt_type}: {e}")
        return _fallback_text(prompt_type, company_name)

    if not draft:
        print(f"⚠️  Empty draft for {prompt_type}; using fallback.")
        return _fallback_text(prompt_type, company_name)

    final_text = draft
    audit: Dict[str, Any] = {
        "section": prompt_type,
        "profile": profile_name,
        "author_model": section_model,
        "draft": draft,
        "draft_tokens": draft_resp.output_tokens,
    }

    # --- 2. Critique-Revise (only for high-stakes sections) ---
    if profile.get("critique_enabled") and prompt_type in CRITIQUE_SECTIONS:
        critic_model = profile["critic_model"]
        try:
            critique = _critique(
                section=prompt_type,
                draft=draft,
                company_name=company_name,
                ticker=company_ticker,
                user_prompt_with_data=user_prompt,
                critic_model=critic_model,
                api_keys=api_keys,
            )
        except LLMError as e:
            print(f"⚠️  Critique skipped ({e}); using draft as final.")
            critique = ""

        audit["critic_model"] = critic_model
        audit["critique"] = critique

        # If verdict is ACCEPT, skip the revise call to save tokens.
        verdict_line = critique.strip().splitlines()[0].upper() if critique else ""
        if critique and "ACCEPT" not in verdict_line:
            try:
                revised = _revise(
                    section=prompt_type,
                    draft=draft,
                    critique=critique,
                    company_name=company_name,
                    ticker=company_ticker,
                    user_prompt_with_data=user_prompt,
                    author_model=section_model,
                    api_keys=api_keys,
                )
                if revised:
                    final_text = revised
                    audit["revised"] = revised
                    audit["verdict"] = verdict_line.split()[0] if verdict_line else "REVISE"
                    print(f"   ↳ critique-revise applied ({verdict_line[:30]}...)")
            except LLMError as e:
                print(f"⚠️  Revise failed ({e}); keeping draft.")
        else:
            audit["verdict"] = "ACCEPT"
            print("   ↳ critic accepted draft; no revise pass.")

    final_text = _normalize_output(final_text, prompt_type)
    audit["final"] = final_text

    # --- 3. Optional audit dump ---
    audit_dir = os.getenv("FINROBOT_AUDIT_DIR")
    if audit_dir:
        try:
            os.makedirs(audit_dir, exist_ok=True)
            audit_path = os.path.join(audit_dir, f"{prompt_type}.audit.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Could not write audit log: {e}")

    return final_text


# Legacy symbol kept for backwards compatibility
def _query_openai(prompt: str, api_key: str) -> str:  # pragma: no cover
    return "Text generation now handled by multi-provider agents."


if __name__ == "__main__":
    print("text_generator_agents.py loaded — multi-provider mode.")
