#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
from typing import Dict, Optional, Tuple, Union

from modules.llm_gateway import LLMSettings, call_llm, normalize_provider
from modules.retail_sentiment_client import format_retail_sentiment_for_prompt


def _get_fallback_text(prompt_type: str, company_name: str) -> str:
    """Returns fallback text when agent generation fails."""
    fallbacks = {
        "tagline": f"{company_name} demonstrates strong financial fundamentals with consistent revenue growth and solid profitability metrics. The company maintains a competitive position in its market segment through operational efficiency and strategic initiatives. Strong balance sheet metrics support continued value creation for shareholders.",
        "company_overview": f"{company_name} operates as a prominent player in its industry sector, demonstrating consistent financial performance through strategic market positioning and operational excellence. The company has shown resilient growth patterns supported by strong demand dynamics and effective cost management strategies.",
        "investment_overview": f"{company_name} has delivered solid financial performance in recent periods, supported by strong operational execution and favorable market conditions. Revenue growth has been driven by robust demand and strategic initiatives, while margin improvements reflect operational efficiency gains.",
        "valuation_overview": f"{company_name} trades at reasonable valuation levels relative to its peer group, supported by strong fundamental metrics and growth prospects. The company's financial profile demonstrates consistent profitability and cash generation capabilities.",
        "risks": "Key risks include: (1) Industry competition and market share pressure, (2) Regulatory changes affecting operations, (3) Economic downturns impacting demand, (4) Technology disruption risks, (5) Supply chain and operational challenges.",
        "competitor_analysis": f"{company_name} demonstrates competitive positioning within its industry through consistent financial performance and strategic market positioning relative to key competitors in the sector.",
        "major_takeaways": f"Revenue Growth: {company_name}'s revenue growth shows consistent performance trends.\n\nGross Profit Margin: {company_name}'s gross profit margins demonstrate operational effectiveness.\n\nSG&A Expense Margin: {company_name}'s SG&A expense management shows disciplined cost control.\n\nEBITDA Margin Stability: {company_name}'s EBITDA margin stability reflects strong underlying fundamentals.",
        "news_summary": f"Recent news coverage for {company_name} reflects ongoing market interest and developments in the company's operations and strategic initiatives."
    }
    return fallbacks.get(prompt_type, f"{company_name} analysis for {prompt_type.replace('_', ' ')} section.")


# System prompts for each text section
SYSTEM_PROMPTS = {
    "tagline": "You are an equity research analyst. Create a 3-sentence professional tagline summarizing the company's financial position. Be concise and professional. Do not use markdown.",
    "company_overview": "You are a financial analyst. Write a comprehensive company overview (300-400 words) covering business model, products/services, market position, and recent performance. Use plain text, no markdown.",
    "investment_overview": "You are an investment analyst. Write an investment update (200-300 words) covering recent financial performance, growth drivers, and outlook. Use plain text, no markdown.",
    "valuation_overview": "You are a valuation analyst. Write a valuation analysis (200-300 words) covering current valuation metrics, peer comparison, and fair value assessment. Use plain text, no markdown.",
    "risks": "You are a risk analyst. List 5 key investment risks in bullet point format. Be specific and concise.",
    "competitor_analysis": "You are a competitive analyst. Write a competitor analysis (200-300 words) comparing the company to its peers. Use plain text, no markdown.",
    "major_takeaways": "You are a financial analyst. Provide 4 major takeaways covering: Revenue Growth, Gross Profit Margin, SG&A Expense Margin, and EBITDA Margin. Format each with a header (use exact header as provided) followed by 1-2 sentences.",
    "news_summary": "You are a financial news analyst. Summarize the recent news (200-300 words) highlighting key developments and their investment implications. Use plain text, no markdown."
}

# Add some limits to prevent overflow. Though these limits are higher than typical API payload as of now.
# As of now payload longer than limit should be cutoff with a line of log in console.
MAX_TABLE_ROWS = 24
MAX_NEWS_ITEMS = 10
MAX_NEWS_TEXT_CHARS = 260
MAX_PROMPT_CHARS = 14000


def _df_to_string(df: Optional[pd.DataFrame], name: str, max_rows: int = MAX_TABLE_ROWS) -> str:
    """Converts a DataFrame to a markdown string for use in a prompt with row caps."""
    if df is None or df.empty:
        return f"{name}:\n[Data not available]\n"

    try:
        df_for_prompt = df.head(max_rows)
        table_text = df_for_prompt.to_markdown()
        if len(df) > max_rows:
            table_text += f"\n[Truncated to first {max_rows} rows out of {len(df)} total rows]"
        return f"{name}:\n{table_text}\n"
    except Exception as e:
        return f"{name}:\n[Error formatting data: {e}]\n"


def _truncate_news_text(text: str, max_chars: int = MAX_NEWS_TEXT_CHARS) -> str:
    txt = (text or "").strip()
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 3].rstrip() + "..."


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if len(prompt) <= max_chars:
        return prompt

    head_chars = int(max_chars * 0.75)
    tail_chars = max_chars - head_chars - 80
    head = prompt[:head_chars]
    tail = prompt[-tail_chars:] if tail_chars > 0 else ""
    return (
        f"{head}\n\n"
        "[Prompt truncated due to size limit; middle content omitted for stability]\n\n"
        f"{tail}"
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimation fallback when provider usage is unavailable."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _build_generation_meta(
    prompt_type: str,
    source: str,
    error: Optional[str],
    generated_text: str,
    latency_ms: int,
    provider: Optional[str],
    model: Optional[str],
    tokens: Optional[Dict[str, Optional[int]]] = None,
) -> Dict:
    token_payload = tokens or {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    return {
        "section": prompt_type,
        "source": source,
        "error": error,
        "chars": len(generated_text or ""),
        "latency_ms": latency_ms,
        "tokens": token_payload,
        "provider": provider,
        "model": model,
    }


def _prepare_user_prompt(data: Dict, prompt_type: str, company_name: str, company_ticker: str) -> str:
    """Prepare user prompt with financial data and lightweight load controls."""
    financial_metrics = data.get('financial_metrics')
    peer_ebitda = data.get('peer_ebitda')
    peer_ev_ebitda = data.get('peer_ev_ebitda')
    company_news = data.get('company_news')
    retail_sentiment = data.get('retail_sentiment')

    prompt = f"Company: {company_name} ({company_ticker})\n\n"

    if financial_metrics is not None and not financial_metrics.empty:
        prompt += _df_to_string(financial_metrics, "Financial Metrics", max_rows=MAX_TABLE_ROWS)

    if peer_ebitda is not None and not peer_ebitda.empty:
        prompt += _df_to_string(peer_ebitda, "Peer EBITDA Comparison", max_rows=MAX_TABLE_ROWS)

    if peer_ev_ebitda is not None and not peer_ev_ebitda.empty:
        prompt += _df_to_string(peer_ev_ebitda, "Peer EV/EBITDA Comparison", max_rows=MAX_TABLE_ROWS)

    if prompt_type == "news_summary" and company_news:
        news_slice = company_news[:MAX_NEWS_ITEMS]
        prompt += (
            f"\n## Recent News ({MAX_NEWS_ITEMS} items, "
            f"{MAX_NEWS_TEXT_CHARS} chars each):\n"
        )
        for i, article in enumerate(news_slice, 1):
            prompt += f"{i}. {article.get('title', 'N/A')} ({article.get('publishedDate', 'N/A')[:10]})\n"
            prompt += f"   {_truncate_news_text(article.get('text', 'N/A'))}\n\n"

    if prompt_type == "news_summary" and retail_sentiment:
        prompt += "\n" + format_retail_sentiment_for_prompt(retail_sentiment) + "\n"

    prompt += f"\nPlease provide the {prompt_type.replace('_', ' ')} based on the above data."
    return _truncate_prompt(prompt)


def generate_text_section(
    data: Dict,
    prompt_type: str,
    api_key: str,
    company_name: str,
    company_ticker: str,
    base_url: str = None,
    model: str = None,
    provider: str = None,
    return_metadata: bool = False,
) -> Union[str, Tuple[str, Dict]]:
    """
    Generates a specific text section for the equity report using configurable LLM provider.

    Args:
        data: Financial data dictionary
        prompt_type: Type of text section to generate
        api_key: Provider API key
        company_name: Company name
        company_ticker: Stock ticker
        base_url: Optional API base URL
        model: Optional model name
        provider: "openai", "claude", or "gemini" (auto-inferred from model if omitted)
    """

    print(f"🤖 Generating '{prompt_type}' text section...")
    start_ts = time.perf_counter()

    model_name = model or "gpt-4o-mini"
    provider_name = normalize_provider(provider, model=model_name)

    # Validate API key
    if not api_key:
        fallback = _get_fallback_text(prompt_type, company_name)
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        meta = _build_generation_meta(
            prompt_type=prompt_type,
            source="fallback",
            error="missing_api_key",
            generated_text=fallback,
            latency_ms=latency_ms,
            provider=provider_name,
            model=model_name,
        )
        print(f"⚠️ Warning: No API key provided. Using fallback text for '{prompt_type}'.")
        return (fallback, meta) if return_metadata else fallback

    print(f"🤖 Using provider/model: {provider_name}/{model_name}")
    if base_url:
        print(f"📡 Using API base URL: {base_url}")

    # Get system prompt
    system_prompt = SYSTEM_PROMPTS.get(
        prompt_type,
        f"You are a financial analyst. Provide {prompt_type.replace('_', ' ')} analysis.")

    # Prepare user prompt with data
    user_prompt = _prepare_user_prompt(data, prompt_type, company_name, company_ticker)

    try:
        settings = LLMSettings(
            provider=provider_name,
            api_key=api_key,
            model=model_name,
            base_url=base_url,
        )
        generated_text, token_usage = call_llm(
            settings=settings,
            instructions=system_prompt,
            prompt=user_prompt,
            max_output_tokens=30000,
            temperature=0.7,
            return_meta=True,
        )

        if generated_text:
            print(f"✅ Successfully generated '{prompt_type}' ({len(generated_text)} chars)")
            source = "llm"
            error = None
            final_text = generated_text
        else:
            print(f"⚠️ Warning: Empty response for '{prompt_type}'")
            source = "fallback"
            error = "empty_response"
            final_text = _get_fallback_text(prompt_type, company_name)

        prompt_tokens = token_usage.get("prompt_tokens") if isinstance(token_usage, dict) else None
        completion_tokens = token_usage.get("completion_tokens") if isinstance(token_usage, dict) else None
        total_tokens = token_usage.get("total_tokens") if isinstance(token_usage, dict) else None

        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(system_prompt + "\n" + user_prompt)
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(final_text)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        meta = _build_generation_meta(
            prompt_type=prompt_type,
            source=source,
            error=error,
            generated_text=final_text,
            latency_ms=latency_ms,
            provider=provider_name,
            model=model_name,
            tokens={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
        return (final_text, meta) if return_metadata else final_text

    except Exception as e:
        error_msg = str(e)
        fallback = _get_fallback_text(prompt_type, company_name)
        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        print(f"❌ Error generating '{prompt_type}': {error_msg}")

        meta = _build_generation_meta(
            prompt_type=prompt_type,
            source="fallback",
            error=error_msg,
            generated_text=fallback,
            latency_ms=latency_ms,
            provider=provider_name,
            model=model_name,
            tokens={
                "prompt_tokens": _estimate_tokens(system_prompt + "\n" + user_prompt),
                "completion_tokens": _estimate_tokens(fallback),
                "total_tokens": _estimate_tokens(system_prompt + "\n" + user_prompt) + _estimate_tokens(fallback),
            },
        )
        return (fallback, meta) if return_metadata else fallback


# Backward compatibility - keep old function signature
def _query_openai(prompt: str, api_key: str) -> str:
    """Legacy function for backward compatibility."""
    return "Text generation now handled by agents."

if __name__ == '__main__':
    print("Testing agent-based text_generator...")
