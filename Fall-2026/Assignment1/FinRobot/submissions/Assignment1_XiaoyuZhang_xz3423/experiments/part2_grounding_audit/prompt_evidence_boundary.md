# Financial Prompt Evidence Boundary — Source Inspection

`text_generator_agents.py` defines `_prepare_user_prompt`, which adds the company name, financial metrics, peer EBITDA table and peer EV/EBITDA table when present. It optionally adds a provenance string. `company_news` and `retail_sentiment` are incorporated only for the `news_summary` section. Grounding rules explicitly distinguish Actual years (`A`) from forecasts (`E`), prevent unsupported claims and disallow inferring causality without evidence.

The code uses OpenAI Chat Completions rather than the course README’s eight-agent structured Pydantic system. Each call passes a section-specific SYSTEM_PROMPT and common financial user prompt, with temperature 0.7 and max_tokens 1000. Consequently this customized implementation should be documented as a modification of upstream, not described as the unchanged eight-agent architecture.

For the 7 non-news chapters the observed primary evidence consists of financial/peer tables plus any actual `data_provenance` text. Statements about supply chains, product strategy, market demand, regulation, competitive leadership, or reasons for numerical trends should not be marked Supported simply because plausible. Exact run-time `data_provenance` arguments should be checked before final classification.

The three news-summary texts are no-news disclosures, not evidence-backed 200-300 word news analyses. Mark them `N/A / no substantive claims` and disclose unavailable FMP news data.

The supplied numerical matching tables are an initial screening layer, not independently validated final claim classifications. Compound statements must be split to atomic claims before a per-section grounding-rate denominator is finalized.
