import json
import os
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
import anthropic
from google import genai


ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env", override=True)

OPENAI_MODEL = "gpt-4.1"
CLAUDE_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.5-pro"


def generate_with_provider(provider, system_prompt, user_input, temperature=0.2):
    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    if provider == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2500,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        return "\n".join(
            block.text for block in response.content if hasattr(block, "text")
        )

    if provider == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_prompt}\n\n{user_input}",
        )
        return response.text

    raise ValueError(f"Unknown provider: {provider}")


def build_section_prompt(text_type, company_name, company_ticker, raw_content, analysis_summary):
    system_prompt = f"""
You are an equity research analyst working inside FinRobot's equity research pipeline.

Your task is to improve the {text_type} section for an equity research report.
# 我们采用文本内容嵌入HTML,使得格式更加美观
Rules:
- Use only the provided financial data and raw content.
- Do not invent numbers.
- If data is missing, explicitly say it is not available.
- Make the section analytical, investor-oriented, and concise.
- Separate facts from interpretation.
- Improve clarity, structure, and financial usefulness.
- Do not use Markdown syntax.
- Do not use **bold**, ### headings, or bullet symbols like "-".
- Write clean HTML-ready prose.
- Use short paragraphs with <p>...</p>.
- Use <h3>...</h3> for subsection titles only when needed.
- Use <ul><li>...</li></ul> for lists if necessary.
"""

    user_input = f"""
Company: {company_name}
Ticker: {company_ticker}
Section type: {text_type}

Raw section content:
{raw_content}

Available financial analysis summary:
{analysis_summary}

Please produce an improved version of this section.
"""
    return system_prompt, user_input


def critic_prompt_for_section(text_type, company_name, company_ticker, draft, raw_content, analysis_summary):
    system_prompt = """
You are the Critic agent in a FinRobot equity research workflow.

Check the draft for:
1. unsupported claims,
2. missing-data problems,
3. invented numbers,
4. vague or promotional language,
5. weak financial reasoning,
6. formatting issues, including raw Markdown syntax and missing <strong>...</strong> emphasis on key financial figures.

Return concise revision instructions.
"""

    user_input = f"""
Company: {company_name}
Ticker: {company_ticker}
Section type: {text_type}

Raw content:
{raw_content}

Financial analysis summary:
{analysis_summary}

Draft section:
{draft}
"""
    return system_prompt, user_input


def reviser_prompt_for_section(text_type, company_name, company_ticker, draft, critique, analysis_summary):
    system_prompt = """
You are the Reviser agent in a FinRobot equity research workflow.

Revise the section using the Critic's feedback.
Keep the section grounded in the provided data.
Do not invent numbers.
If information is unavailable, disclose it clearly.

Formatting requirements:
- Return HTML-ready text, not Markdown.
- Do not use Markdown syntax such as **bold**, ### headings, or bullet lines starting with "-".
- Use <p>...</p> for paragraphs.
- Use <h3>...</h3> for subsection titles only when the section clearly benefits from a short heading.
- Use <ul><li>...</li></ul> for lists only when necessary.
- Use <strong>...</strong> to highlight important financial figures.
- Important financial figures include revenue, EBITDA, EPS, margins, growth rates, CAGR, YoY growth, valuation multiples, market cap, share price, target price, and fiscal-year estimates.
- Examples: <strong>$130.5B</strong>, <strong>68.3%</strong>, <strong>37.7x</strong>, <strong>FY2026</strong>, <strong>2027E</strong>, <strong>CAGR</strong>, <strong>YoY</strong>.
- Do not bold every number. Only bold financially meaningful numbers that support the investment argument, risk assessment, or valuation discussion.
"""

    user_input = f"""
Company: {company_name}
Ticker: {company_ticker}
Section type: {text_type}

Draft:
{draft}

Critique:
{critique}

Financial analysis summary:
{analysis_summary}

Return only the revised section text in HTML-ready format.
"""
    return system_prompt, user_input


def enhance_text_section_with_acr(
    text_type,
    company_name,
    company_ticker,
    raw_content,
    analysis_summary,
    analyst_provider="claude",
    critic_provider="openai",
    reviser_provider="claude",
):
    """
    ACR = Analyst → Critic → Reviser.
    This function can be called from FinRobot's create_equity_report.py.
    """

    analyst_system, analyst_user = build_section_prompt(
        text_type, company_name, company_ticker, raw_content, analysis_summary
    )
    draft = generate_with_provider(
        analyst_provider, analyst_system, analyst_user, temperature=0.2
    )

    critic_system, critic_user = critic_prompt_for_section(
        text_type, company_name, company_ticker, draft, raw_content, analysis_summary
    )
    critique = generate_with_provider(
        critic_provider, critic_system, critic_user, temperature=0.1
    )

    reviser_system, reviser_user = reviser_prompt_for_section(
        text_type, company_name, company_ticker, draft, critique, analysis_summary
    )
    revised = generate_with_provider(
        reviser_provider, reviser_system, reviser_user, temperature=0.1
    )

    return revised