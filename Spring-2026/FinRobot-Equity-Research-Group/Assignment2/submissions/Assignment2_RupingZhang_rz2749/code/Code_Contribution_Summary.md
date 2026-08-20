# Code Contribution Summary

This project does not replace the entire FinRobot repository. Instead, it adds a modular Assignment 2 enhancement layer on top of FinRobot's original equity research pipeline.

The main goal is to improve the reliability, groundedness, and analytical quality of FinRobot-generated equity research reports by combining frontier LLMs with a Claude-inspired Analyst-Critic-Reviser workflow.

---

## 1. Assignment 2 Text Enhancer

The main new module is the Assignment 2 text enhancer. It implements a Claude-inspired multi-agent workflow for improving FinRobot report sections.

### Main Improvement

I added an Analyst-Critic-Reviser workflow:

1. **Analyst** generates the first draft of a report section.
2. **Critic** reviews the draft for missing data, unsupported claims, invented numbers, vague language, and weak financial reasoning.
3. **Reviser** rewrites the section using the Critic's feedback and produces the final grounded version.

This design directly addresses one of the main issues I observed in FinRobot-style report generation: when financial data are missing or incomplete, LLMs may still produce confident but weakly supported analysis. The Critic and Reviser stages help reduce that risk.

---

## 2. Multi-Model Provider Support

I added a unified model-calling interface that supports multiple frontier LLM providers.

### Supported Models

The pipeline supports:

- OpenAI / GPT
- Anthropic Claude
- Google Gemini

### Main Improvement

This makes it possible to run controlled model-role experiments under the same workflow. For example:

| Experiment | Analyst | Critic | Reviser |
|---|---|---|---|
| GPT-only | GPT | GPT | GPT |
| Claude-only | Claude | Claude | Claude |
| Gemini-only | Gemini | Gemini | Gemini |
| Mixed 1 | Claude | GPT | Claude |
| Mixed 2 | Gemini | GPT | Claude |

This directly supports the Assignment 2 requirement to experiment with base models and model combinations.

---

## 3. Analyst Prompt Improvement

The Analyst prompt was designed to produce structured, investor-oriented equity research sections.

### Main Improvements

The Analyst is instructed to:

- Use only the provided financial data and raw content.
- Avoid inventing numbers.
- Clearly disclose when data are unavailable.
- Separate factual evidence from interpretation.
- Improve clarity, structure, and financial usefulness.
- Produce concise equity research writing.
- Generate HTML-ready text instead of raw Markdown.

This improves the report quality by making the first draft more grounded and more suitable for FinRobot's HTML renderer.

---

## 4. Critic Prompt Improvement

The Critic prompt was added to identify weaknesses in the Analyst's draft.

### The Critic checks for:

- Unsupported claims
- Missing-data problems
- Invented numbers
- Vague or promotional language
- Weak financial reasoning
- Formatting problems, such as raw Markdown syntax or missing emphasis on key financial figures

### Main Improvement

The Critic stage makes the workflow more reliable. Instead of allowing the first model output to become the final report, the system explicitly checks whether the analysis is supported by available data.

This is especially important for financial research, where unsupported claims or invented numbers can make a report misleading.

---

## 5. Reviser Prompt Improvement

The Reviser prompt was added to convert the Analyst draft and Critic feedback into the final report section.

### Main Improvements

The Reviser is instructed to:

- Use the Critic's feedback to improve the section.
- Keep the section grounded in the provided data.
- Avoid inventing numbers.
- Clearly disclose missing information.
- Return HTML-ready text.
- Avoid Markdown syntax such as `**bold**`, `### headings`, or raw bullet lines.
- Use `<p>...</p>` for paragraphs.
- Use `<strong>...</strong>` to highlight important financial figures.
- Highlight key financial values such as revenue, EBITDA, EPS, margins, growth rates, CAGR, YoY growth, valuation multiples, market cap, share price, target price, and fiscal-year estimates.

### Main Improvement

This improves both the analytical quality and the visual quality of the final HTML report. Instead of raw Markdown symbols appearing in the HTML page, the output is cleaner and more professional.

---

## 6. FinRobot Pipeline Integration

I modified FinRobot's report generation pipeline by adding an optional Assignment 2 enhancement flag.

### Main Improvement

When the enhancement flag is enabled, FinRobot still handles:

- FMP financial data ingestion
- Financial metrics and forecasts
- Peer comparison
- Market data fetching
- Professional HTML rendering

But the text-generation layer is enhanced by the Analyst-Critic-Reviser workflow.

This means the project is not an isolated report generator. It is integrated into FinRobot's existing equity report pipeline.

---

## 7. Optional Enhancement Flag

I added an optional flag:

```bash
--enable-assignment2-enhancement