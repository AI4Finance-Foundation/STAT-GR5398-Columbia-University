"""
agent2/prompt.py — Prompts for Agent 2 (trading strategy reasoner).

STRATEGY_COT_PROMPT
    Instructs the DeepSeek-R1 model to reason about the appropriate trading
    strategy given the news article and Agent 1's sentiment analysis.
    Reasoning is written inside ``<think>…</think>`` tags; vLLM stops
    generation at ``</think>``.

    After the CoT the model picks one of three lettered options (A / B / C).
    ``vllm_logits_client.get_real_choice_logits`` scores the single-character
    tokens "A", "B", "C" via ``SamplingParams(prompt_logprobs=1)``, which
    have near-equal language-model priors and therefore avoid the directional
    bias that arises when scoring the words "BUY", "HOLD", "SELL" directly.

    The model is NOT asked to output a JSON object or numeric logits.

STRATEGY_DECISION_PREFIX
    Appended after ``</think>\\n`` in Phase 2 to create the decision context.
    Scoring tokens are STRATEGY_SCORE_TOKENS = ["A", "B", "C"] which map
    positionally to STRATEGY_SET = ["BUY", "HOLD", "SELL"].

STRATEGY_SCORE_TOKENS
    The actual tokens passed to ``get_real_choice_logits`` / ``_batch``.
    Single ASCII letters are reliably single-token in all BPE vocabularies and
    have balanced priors, eliminating the HOLD-dominance artefact seen when
    scoring the words "BUY" / "HOLD" / "SELL" after "Strategy: ".
"""

# ---------------------------------------------------------------------------
# Strategy CoT prompt (Phase 1 — reasoning only, no answer expected)
# ---------------------------------------------------------------------------

STRATEGY_COT_PROMPT: str = """\
You are a quantitative trading strategist.

Based on the news article and sentiment analysis below, reason step by step \
about the appropriate one-week trading action. Write ALL of your analysis \
inside <think>...</think> tags. Consider:

• What near-term price direction does this news imply?
• Does the sentiment analysis align with the factual content?
• Is the signal strong enough to act on (A or C), or should we wait (B)?
• How confident are you given any ambiguity in the news?

At the end you will select one lettered option:
  (A) BUY  — bullish signal; expect the price to rise over the next week
  (B) HOLD — insufficient or ambiguous signal; stay flat
  (C) SELL — bearish signal; expect the price to fall over the next week

Sentiment analysis:
{sentiment_json}

News article:
{article_text}\
"""

# ---------------------------------------------------------------------------
# Decision prefix and scoring tokens (Phase 2 — prompt_logprobs scoring)
# ---------------------------------------------------------------------------

# Appended after </think>\n so the scoring context ends with an open paren.
# Single-letter tokens A / B / C are scored at this position.
STRATEGY_DECISION_PREFIX: str = "The answer is ("

# Tokens passed to get_real_choice_logits / _batch.
# Positional mapping: A → BUY (idx 0), B → HOLD (idx 1), C → SELL (idx 2).
# Must stay aligned with STRATEGY_SET in config.py.
STRATEGY_SCORE_TOKENS: list[str] = ["A", "B", "C"]

# Backward-compatible alias.
SYSTEM_PROMPT: str = STRATEGY_COT_PROMPT
