#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from typing import Dict, Optional
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

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
    "investment_overview": "You are an investment analyst. Write an investment update (200-300 words) covering recent financial performance, revenue drivers, key developments from recent SEC filings, and forward outlook. Use plain text, no markdown. Ground your discussion in the provided data and do not invent facts.",
    "valuation_overview": "You are a valuation analyst. Write a valuation analysis (200-300 words) covering current valuation metrics, peer comparison, fair value context, and whether the revenue growth assumptions appear supported by identified revenue drivers. Use plain text, no markdown. Ground your discussion in the provided data and do not invent facts.",
    "risks": "You are a risk analyst. List 5 key investment risks in bullet point format. Prioritize risks supported by recent SEC filing disclosures, company news, and the company's recent financial profile. Be specific and concise.",
    "competitor_analysis": "You are a competitive analyst. Write a competitor analysis (200-300 words) comparing the company to its peers. Use plain text, no markdown.",
    "major_takeaways": "You are a financial analyst. Provide 4 major takeaways covering: Revenue Growth, Gross Profit Margin, SG&A Expense Margin, and EBITDA Margin. Format each with a header followed by 1-2 sentences. Where relevant, incorporate identified revenue drivers and key findings from recent SEC filings.",
    "news_summary": "You are a financial news analyst. Summarize the recent news (200-300 words) highlighting key developments and their investment implications. Use plain text, no markdown."
}


def _infer_provider(provider: Optional[str], model: Optional[str]) -> str:
    """Infer provider from explicit provider or model name."""
    if provider:
        return provider.lower().strip()

    model_lower = (model or "").lower()
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return "gemini"
    return "openai"


def _normalize_gemini_model_name(model: str) -> str:
    """Gemini API accepts both gemini-2.5-pro and models/gemini-2.5-pro style names."""
    if not model:
        return "gemini-2.5-pro"
    return model.replace("models/", "", 1)


def _call_openai_model(
    api_key: str,
    base_url: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call OpenAI-compatible chat completions API."""
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        print(f"📡 Using OpenAI-compatible base URL: {base_url}")

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic_model(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call Anthropic Messages API."""
    if Anthropic is None:
        raise ImportError("anthropic package is not installed. Run: pip install anthropic")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


def _call_gemini_model(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call Google Gemini generateContent API using google-genai."""
    if genai is None or genai_types is None:
        raise ImportError("google-genai package is not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    gemini_model = _normalize_gemini_model_name(model or "gemini-2.5-pro")

    response = client.models.generate_content(
        model=gemini_model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=1500,
        ),
    )

    # First try the simple helper
    if getattr(response, "text", None):
        return response.text.strip()

    # Fallback: manually extract text from candidates / parts
    text_parts = []
    candidates = getattr(response, "candidates", None) or []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None

        if not parts:
            continue

        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    extracted_text = "\n".join(text_parts).strip()

    if extracted_text:
        return extracted_text

    # Debug information if Gemini returns 200 but no text
    finish_reasons = []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            finish_reasons.append(str(finish_reason))

    print(f"⚠️ Gemini returned no text. finish_reasons={finish_reasons}")
    return ""


def _df_to_string(df: Optional[pd.DataFrame], name: str) -> str:
    """Converts a DataFrame to a markdown string for use in a prompt."""
    if df is None or df.empty:
        return f"{name}:\n[Data not available]\n"
    
    try:
        return f"{name}:\n{df.to_markdown()}\n"
    except Exception as e:
        return f"{name}:\n[Error formatting data: {e}]\n"


def _prepare_user_prompt(data: Dict, prompt_type: str, company_name: str, company_ticker: str) -> str:
    """Prepare user prompt with financial data."""
    financial_metrics = data.get('financial_metrics')
    peer_ebitda = data.get('peer_ebitda')
    peer_ev_ebitda = data.get('peer_ev_ebitda')
    company_news = data.get('company_news')
    retail_sentiment = data.get('retail_sentiment')
    sec_filing_analysis = data.get('sec_filing_analysis')
    revenue_driver_analysis = data.get('revenue_driver_analysis')
    
    prompt = f"Company: {company_name} ({company_ticker})\n\n"
    
    if financial_metrics is not None and not financial_metrics.empty:
        prompt += _df_to_string(financial_metrics, "Financial Metrics")
    
    if peer_ebitda is not None and not peer_ebitda.empty:
        prompt += _df_to_string(peer_ebitda, "Peer EBITDA Comparison")
        
    if peer_ev_ebitda is not None and not peer_ev_ebitda.empty:
        prompt += _df_to_string(peer_ev_ebitda, "Peer EV/EBITDA Comparison")

    if sec_filing_analysis and prompt_type in {"investment_overview", "risks", "major_takeaways", "valuation_overview"}:
        prompt += "\nSEC Filing Analysis:\n"
        summary = sec_filing_analysis.get("summary")
        if summary:
            prompt += f"Summary: {summary}\n"

        filings = sec_filing_analysis.get("filings_analyzed", [])
        for i, filing in enumerate(filings[:2], 1):
            prompt += f"Filing {i}: {filing.get('filing_type', 'N/A')} dated {filing.get('filing_date', 'N/A')}\n"

            key_takeaways = filing.get("key_takeaways", [])
            if key_takeaways:
                prompt += "Key Takeaways:\n"
                for takeaway in key_takeaways[:4]:
                    prompt += f"- {takeaway}\n"

            risk_signals = filing.get("risk_signals", [])
            if risk_signals:
                prompt += "Risk Signals:\n"
                for risk in risk_signals[:5]:
                    prompt += f"- {risk}\n"

            catalyst_signals = filing.get("catalyst_signals", [])
            if catalyst_signals:
                prompt += "Catalyst Signals:\n"
                for catalyst in catalyst_signals[:5]:
                    prompt += f"- {catalyst}\n"

    if revenue_driver_analysis and prompt_type in {"investment_overview", "valuation_overview", "major_takeaways"}:
        prompt += "\nRevenue Driver Analysis:\n"

        historical = revenue_driver_analysis.get("historical_revenue_analysis", {})
        historical_trend = historical.get("historical_growth_trend")
        average_growth = historical.get("average_growth")
        observations = historical.get("observations", [])

        if historical_trend:
            prompt += f"Historical Revenue Trend: {historical_trend}\n"
        if average_growth is not None:
            prompt += f"Average Historical Revenue Growth: {average_growth:.1%}\n"
        if observations:
            prompt += "Historical Observations:\n"
            for obs in observations[:4]:
                prompt += f"- {obs}\n"

        driver_summary = revenue_driver_analysis.get("driver_summary", [])
        if driver_summary:
            prompt += "Identified Revenue Drivers:\n"
            for item in driver_summary[:6]:
                prompt += f"- {item}\n"

        forecast_support = revenue_driver_analysis.get("forecast_support", {})
        if forecast_support:
            prompt += "Forecast Support:\n"
            for year, detail in forecast_support.items():
                prompt += f"- {year}: {detail}\n"

    if prompt_type == "news_summary" and company_news:
        prompt += f"\n## Recent News:\n"
        for i, article in enumerate(company_news[:10], 1):  # Limit to 10 articles
            prompt += f"{i}. {article.get('title', 'N/A')} ({article.get('publishedDate', 'N/A')[:10]})\n"
            prompt += f"   {article.get('text', 'N/A')[:200]}...\n\n"

    if prompt_type == "news_summary" and retail_sentiment:
        prompt += "\n" + format_retail_sentiment_for_prompt(retail_sentiment) + "\n"

    prompt += f"\nPlease provide the {prompt_type.replace('_', ' ')} based on the above data."
    prompt += " Use only the information provided in this prompt."
    if prompt_type in {"investment_overview", "valuation_overview", "major_takeaways", "risks"}:
        prompt += " Incorporate SEC filing analysis and revenue driver analysis when relevant."
    return prompt


def generate_text_section(
    data: Dict,
    prompt_type: str,
    api_key: str,
    company_name: str,
    company_ticker: str,
    base_url: str = None,
    model: str = None,
    provider: str = None,
) -> str:
    """
    Generates a specific text section for the equity report using OpenAI, Anthropic, or Gemini.

    Args:
        data: Financial data dictionary
        prompt_type: Type of text section to generate
        api_key: API key for the selected provider
        company_name: Company name
        company_ticker: Stock ticker
        base_url: Optional OpenAI-compatible API base URL
        model: Optional model name
        provider: Optional provider name: openai, anthropic, or gemini
    """

    print(f"🤖 Generating '{prompt_type}' text section...")

    if not api_key:
        print(f"⚠️ Warning: No API key provided. Using fallback text for '{prompt_type}'.")
        return _get_fallback_text(prompt_type, company_name)

    selected_provider = _infer_provider(provider, model)

    if selected_provider == "anthropic":
        selected_model = model or "claude-sonnet-4-6"
    elif selected_provider == "gemini":
        selected_model = model or "gemini-2.5-pro"
    else:
        selected_provider = "openai"
        selected_model = model or "gpt-4o-mini"

    print(f"🧠 Provider: {selected_provider}")
    print(f"🤖 Using model: {selected_model}")

    system_prompt = SYSTEM_PROMPTS.get(
        prompt_type,
        f"You are a financial analyst. Provide {prompt_type.replace('_', ' ')} analysis.",
    )
    user_prompt = _prepare_user_prompt(data, prompt_type, company_name, company_ticker)

    try:
        if selected_provider == "anthropic":
            generated_text = _call_anthropic_model(
                api_key=api_key,
                model=selected_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        elif selected_provider == "gemini":
            generated_text = _call_gemini_model(
                api_key=api_key,
                model=selected_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            generated_text = _call_openai_model(
                api_key=api_key,
                base_url=base_url,
                model=selected_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        if generated_text:
            print(f"✅ Successfully generated '{prompt_type}' ({len(generated_text)} chars)")
            return generated_text

        print(f"⚠️ Warning: Empty response for '{prompt_type}'")
        return _get_fallback_text(prompt_type, company_name)

    except Exception as e:
        print(f"❌ Error generating '{prompt_type}' with {selected_provider}/{selected_model}: {e}")
        return _get_fallback_text(prompt_type, company_name)

# Backward compatibility - keep old function signature
def _query_openai(prompt: str, api_key: str) -> str:
    """Legacy function for backward compatibility."""
    return "Text generation now handled by agents."

if __name__ == '__main__':
    print("Testing agent-based text_generator...")
