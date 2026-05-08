"""
agent1/prompt.py — Prompts for Agent 1.

Two prompts are used:

EXTRACTION_PROMPT
    Instructs the model to extract structured facts from the article and
    return them as a JSON object.  Guided decoding enforces the schema.
    Unchanged from previous version.

SENTIMENT_COT_PROMPT
    Instructs the DeepSeek-R1 model to write chain-of-thought reasoning
    about the article's market sentiment inside ``<think>…</think>`` tags.
    The model stops at ``</think>`` (via vLLM ``stop=["</think>"]``).

    The scoring phase in ``vllm_logits_client.get_real_choice_logits``
    then appends ``</think>\\n`` + ``SENTIMENT_DECISION_PREFIX`` and reads
    the model's genuine token-level log-probabilities for each class label
    (POSITIVE, NEGATIVE, NEUTRAL) via ``SamplingParams(prompt_logprobs=1)``.

    The model is NOT asked to output logit numbers or a JSON object.  All
    probability calculations happen in Python from the real model logprobs.

SENTIMENT_DECISION_PREFIX
    Short string appended after the CoT to create the decision context.
    The model's next-token distribution at this point is scored for each
    choice label.  Must end with a space so the choice token attaches
    cleanly (e.g. "Sentiment: POSITIVE").
"""

# ---------------------------------------------------------------------------
# Fact-extraction prompt (guided decoding — unchanged)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT: str = (
    "Extract structured facts from the article below. "
    "Use only information explicitly stated. "
    "Return only a JSON object.\n"
    "Good output example:\n"
    '{"source":"Reuters","published_at":"2024-02-25T14:30:00Z","headline":"Apple shares rise after strong iPhone demand",'
    '"companies_named":["Apple","AAPL"],"event_keywords":["iphone","demand","earnings"]}\n'
    "Now return JSON with these fields:\n"
    "- source: publisher name or empty string\n"
    "- published_at: ISO datetime if present, or empty string\n"
    "- headline: main headline; if multiple headlines are provided, write one concise combined headline\n"
    "- companies_named: list of company names or tickers\n"
    "- event_keywords: list of lowercase topic keywords"
)

# Backward-compatible alias.
SYSTEM_PROMPT: str = EXTRACTION_PROMPT

# ---------------------------------------------------------------------------
# Sentiment CoT prompt (Phase 1 — reasoning only, no answer expected)
# ---------------------------------------------------------------------------

SENTIMENT_COT_PROMPT: str = """\
You are a financial news sentiment analyst.

Reason step by step about the market sentiment conveyed by the following \
article. Write your analysis inside <think>...</think> tags. Consider:

• Which companies or sectors are mentioned and how are they affected?
• Is the news fundamentally positive, negative, or neutral for investors?
• Are there conflicting signals or ambiguity in the article?

{article_text}\
"""

# ---------------------------------------------------------------------------
# Decision prefix (Phase 2 — appended after </think> for logprob scoring)
# ---------------------------------------------------------------------------

# The model's next-token distribution at "Sentiment: " is scored for each
# of the three class labels: POSITIVE, NEGATIVE, NEUTRAL.
# Must end with a space so the choice token aligns with BPE tokenisation.
SENTIMENT_DECISION_PREFIX: str = "Sentiment: "
