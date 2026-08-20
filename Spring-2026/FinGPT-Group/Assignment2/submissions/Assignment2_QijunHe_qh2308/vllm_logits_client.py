"""
vllm_logits_client.py — Unified vLLM wrapper for logits-based inference.

Two approaches are provided; the real-logits approach is the default.

Real logits — two-phase prompt_logprobs (DEFAULT)
--------------------------------------------------
Because this is a **local** vLLM deployment, we can read the model's true
token-level log-probabilities via ``SamplingParams(prompt_logprobs=1)``.

Phase 1 — CoT generation
    Run the chain-of-thought prompt with ``stop=["</think>"]`` so the model
    reasons freely but halts before producing an answer.

Phase 2 — Scoring at the decision point
    Append ``</think>\\n{decision_prefix}`` to the context, then construct
    one scoring prompt per choice:

        context + "POSITIVE"
        context + "NEGATIVE"
        context + "NEUTRAL"

    Send all scoring prompts in a single batched ``engine.generate()`` call
    with ``prompt_logprobs=1, max_tokens=1``.  vLLM always includes the
    logprob of the *actual* prompt token at every position, so we can
    retrieve ``log P(choice_token | context)`` regardless of where the token
    ranks in the full vocabulary.

    For multi-token choices (e.g. "POSITIVE" → ["POS", "ITIVE"]) we sum the
    per-token logprobs to obtain the full-string joint log-probability:

        log P("POSITIVE" | ctx) = log P("POS" | ctx) + log P("ITIVE" | ctx, "POS")

    These summed log-probabilities ARE the real model logits up to an additive
    constant (which is identical for all choices and therefore cancels in
    softmax).  The downstream ``softmax`` + ``CALIBRATION_T`` post-processing
    is unchanged.

Self-reported logits (legacy / fallback)
-----------------------------------------
``get_choice_logits`` / ``get_choice_logits_batch`` are kept for reference.
They instruct the LLM to output ``{"logits": [f1, f2, f3]}`` as part of its
response and parse that JSON.  These are kept but are no longer called by
the default agent pipeline.

Public API
----------
    softmax(logits, temperature)                 -> list[float]
    get_real_choice_logits(...)                  -> dict            [PRIMARY]
    get_real_choice_logits_batch(...)            -> list[dict]      [PRIMARY]
    get_choice_logits(...)                       -> dict            [legacy]
    get_choice_logits_batch(...)                 -> list[dict]      [legacy]
    extract_thinking(raw_output)                 -> str
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers (used by the legacy self-reported path)
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_LOGITS_OBJ_RE = re.compile(r'\{[^{}]*"logits"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Pure-Python post-processing (shared by both approaches)
# ---------------------------------------------------------------------------

def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """
    Numerically stable softmax with temperature scaling.

        softmax_T(x)_i = exp((x_i − max(x)) / T) / Σ_j exp((x_j − max(x)) / T)

    Works correctly whether *logits* are raw pre-softmax values or
    log-probabilities (shift-invariant property of softmax guarantees this).
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [x / temperature for x in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(x - max_val) for x in scaled]
    total = sum(exp_vals)
    return [x / total for x in exp_vals]


def extract_thinking(raw_output: str) -> str:
    """
    Extract chain-of-thought text from ``<think>…</think>`` tags.

    Falls back to everything before the first ``{"logits"`` JSON block
    (legacy path), or the full output if neither marker is found.
    """
    match = _THINK_BLOCK_RE.search(raw_output)
    if match:
        return match.group(1).strip()
    json_start = raw_output.rfind('{"logits"')
    if json_start > 0:
        return raw_output[:json_start].strip()
    return raw_output.strip()


# ---------------------------------------------------------------------------
# Real logits — two-phase prompt_logprobs approach (PRIMARY)
# ---------------------------------------------------------------------------

def _resolve_choice_token_ids(
    tokenizer: Any,
    scoring_context: str,
    choice: str,
) -> tuple[int, list[int]]:
    """
    Tokenize ``scoring_context`` and ``scoring_context + choice`` and return
    ``(context_len, choice_token_ids)`` where:

    * ``context_len``      — number of tokens in the scoring context (used as
                             the starting index into ``prompt_logprobs``).
    * ``choice_token_ids`` — token IDs of *choice* as they appear at the end
                             of the full scoring prompt.  Tokenizing the full
                             string handles BPE boundary effects correctly.

    Both tokenizations use ``add_special_tokens=False`` because the scoring
    context is already the output of ``apply_chat_template``, which embeds
    all required special tokens (BOS etc.) as literal characters in the string.
    vLLM encodes prompts the same way (no extra BOS prepended when the string
    already starts with a BOS character), so the indices align.
    """
    context_ids: list[int] = tokenizer.encode(scoring_context, add_special_tokens=False)
    full_ids: list[int] = tokenizer.encode(
        scoring_context + choice, add_special_tokens=False
    )
    choice_ids = full_ids[len(context_ids):]
    return len(context_ids), choice_ids


def _sum_choice_logprob(
    prompt_logprobs: list,
    context_len: int,
    choice_token_ids: list[int],
) -> tuple[float, bool]:
    """
    Sum ``log P(choice_token_i | context + choice_token_0..i-1)`` over all
    tokens of the choice string by reading from vLLM's ``prompt_logprobs``.

    ``prompt_logprobs[i]`` is ``Optional[Dict[int, Logprob]]``:
    * ``None``       at position 0 (first token has no prior context).
    * ``{tok_id: Logprob, …}`` at position i > 0; vLLM *always* includes the
      logprob of the actual prompt token, so the lookup never fails for tokens
      that are genuinely in the prompt.

    Returns ``(total_logprob, all_found)`` where ``all_found`` is False if any
    token was absent from the dict (indicates a vLLM version mismatch or a
    corrupted output; caller may treat this as a soft warning).
    """
    if not prompt_logprobs:
        return -100.0 * max(len(choice_token_ids), 1), False

    total = 0.0
    all_found = True

    for j, tok_id in enumerate(choice_token_ids):
        pos = context_len + j
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            total += -100.0
            all_found = False
            continue
        lp_dict = prompt_logprobs[pos]
        if tok_id in lp_dict:
            total += lp_dict[tok_id].logprob
        else:
            # Should not happen: vLLM guarantees the actual prompt token is
            # always included.  Assign a large penalty as a safe fallback.
            logger.warning(
                "Token %d not found in prompt_logprobs at position %d; "
                "assigning -100 logprob.",
                tok_id, pos,
            )
            total += -100.0
            all_found = False

    return total, all_found


def get_real_choice_logits(
    engine: Any,
    cot_prompt: str,
    choices: list[str],
    decision_prefix: str,
    tokenizer: Any,
    max_cot_tokens: int = 900,
) -> dict[str, Any]:
    """
    Extract **real** model log-probabilities for each choice using two vLLM
    calls (single-article path).

    Parameters
    ----------
    engine:
        Shared vLLM ``LLM`` instance.
    cot_prompt:
        Fully-formatted prompt (chat template already applied) that instructs
        the model to write its CoT reasoning inside ``<think>…</think>``.
    choices:
        Ordered list of class labels, e.g. ``["POSITIVE","NEGATIVE","NEUTRAL"]``.
    decision_prefix:
        Short string appended after ``</think>\\n`` that sets up the decision
        context, e.g. ``"Sentiment: "``.  The model's next-token distribution
        at this point is what we score.
    tokenizer:
        HuggingFace tokenizer for the model (same as vLLM uses internally).
        Required to resolve choice token IDs and compute ``context_len``.
    max_cot_tokens:
        Token budget for the CoT generation phase (Phase 1).

    Returns
    -------
    dict with keys:
        ``logits``        – ``list[float]``: real log P(choice_i | ctx) values,
                            aligned with *choices*.
        ``choices``       – same as input.
        ``raw_output``    – CoT text generated in Phase 1.
        ``thinking``      – extracted chain-of-thought reasoning.
        ``parse_success`` – True when all choice tokens were found in the
                            prompt_logprobs dict.

    Two vLLM calls are made:
        Phase 1: one generation call (CoT, stop at </think>).
        Phase 2: K scoring calls batched into one engine.generate() call,
                 where K = len(choices).
    """
    from vllm import SamplingParams  # type: ignore

    # ── Phase 1: CoT generation ───────────────────────────────────────────────
    cot_params = SamplingParams(
        max_tokens=max_cot_tokens,
        temperature=0.0,
        stop=["</think>"],
    )
    cot_outputs = engine.generate([cot_prompt], cot_params)
    cot_text: str = (
        cot_outputs[0].outputs[0].text if cot_outputs[0].outputs else ""
    )
    thinking = extract_thinking(cot_text + "</think>")

    # ── Phase 2: Scoring via prompt_logprobs ──────────────────────────────────
    scoring_context = cot_prompt + cot_text + "</think>\n" + decision_prefix

    scoring_prompts = [scoring_context + choice for choice in choices]
    score_params = SamplingParams(
        prompt_logprobs=1,  # vLLM always includes the actual token's logprob
        max_tokens=1,
        temperature=0.0,
    )
    score_outputs = engine.generate(scoring_prompts, score_params)

    logprobs: list[float] = []
    all_found = True
    for output, choice in zip(score_outputs, choices):
        context_len, choice_ids = _resolve_choice_token_ids(
            tokenizer, scoring_context, choice
        )
        lp, found = _sum_choice_logprob(
            output.prompt_logprobs or [], context_len, choice_ids
        )
        logprobs.append(lp)
        all_found = all_found and found

    if not all_found:
        logger.warning(
            "Some choice tokens were absent from prompt_logprobs; "
            "logprobs may be approximate."
        )

    logger.debug("Real logprobs %s for choices %s", logprobs, choices)

    return {
        "logits": logprobs,
        "choices": choices,
        "raw_output": cot_text,
        "thinking": thinking,
        "parse_success": all_found,
    }


def get_real_choice_logits_batch(
    engine: Any,
    cot_prompts: list[str],
    choices: list[str],
    decision_prefix: str,
    tokenizer: Any,
    max_cot_tokens: int = 900,
) -> list[dict[str, Any]]:
    """
    Batch version of ``get_real_choice_logits``.

    Makes exactly **two** ``engine.generate()`` calls regardless of batch size:

    Phase 1  — N CoT generations (one per article) in a single call.
    Phase 2  — N × K scoring prompts in a single call, where K = len(choices).

    Each item in the returned list has the same schema as
    ``get_real_choice_logits``.  Output logic (softmax, argmax, signal
    assembly) is unchanged and applied independently per item by the caller.
    """
    if not cot_prompts:
        return []

    from vllm import SamplingParams  # type: ignore

    n = len(cot_prompts)
    k = len(choices)

    # ── Phase 1: Batch CoT generation ────────────────────────────────────────
    cot_params = SamplingParams(
        max_tokens=max_cot_tokens,
        temperature=0.0,
        stop=["</think>"],
    )
    cot_outputs = engine.generate(cot_prompts, cot_params)
    cot_texts: list[str] = [
        o.outputs[0].text if o.outputs else "" for o in cot_outputs
    ]

    # Build scoring contexts and prompts: N × K
    scoring_contexts: list[str] = []
    scoring_prompts: list[str] = []
    for i in range(n):
        ctx = cot_prompts[i] + cot_texts[i] + "</think>\n" + decision_prefix
        scoring_contexts.append(ctx)
        for choice in choices:
            scoring_prompts.append(ctx + choice)

    # ── Phase 2: Batch scoring ────────────────────────────────────────────────
    score_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        temperature=0.0,
    )
    score_outputs = engine.generate(scoring_prompts, score_params)

    results: list[dict[str, Any]] = []
    for i in range(n):
        ctx = scoring_contexts[i]
        article_logprobs: list[float] = []
        all_found = True

        for j, choice in enumerate(choices):
            output = score_outputs[i * k + j]
            context_len, choice_ids = _resolve_choice_token_ids(
                tokenizer, ctx, choice
            )
            lp, found = _sum_choice_logprob(
                output.prompt_logprobs or [], context_len, choice_ids
            )
            article_logprobs.append(lp)
            all_found = all_found and found

        if not all_found:
            logger.warning(
                "Batch[%d]: some choice tokens absent from prompt_logprobs.", i
            )

        logger.debug(
            "Batch[%d] real logprobs %s for choices %s",
            i, article_logprobs, choices,
        )

        results.append({
            "logits": article_logprobs,
            "choices": choices,
            "raw_output": cot_texts[i],
            "thinking": extract_thinking(cot_texts[i] + "</think>"),
            "parse_success": all_found,
        })

    return results


# ---------------------------------------------------------------------------
# Legacy self-reported logits (kept for reference / ablation studies)
# ---------------------------------------------------------------------------

def _extract_logits_json(text: str) -> Optional[dict[str, Any]]:
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        try:
            candidate = json.loads(fence_match.group(1).strip())
            if "logits" in candidate:
                return candidate
        except json.JSONDecodeError:
            pass

    matches = list(_LOGITS_OBJ_RE.finditer(text))
    for match in reversed(matches):
        try:
            candidate = json.loads(match.group(0))
            if "logits" in candidate:
                return candidate
        except json.JSONDecodeError:
            continue

    idx = text.rfind('{"logits"')
    if idx >= 0:
        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def get_choice_logits(
    engine: Any,
    prompt: str,
    choices: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Legacy: single-call self-reported logits.

    Instructs the LLM to output ``{"logits": [f1, f2, …]}`` and parses it.
    Use ``get_real_choice_logits`` instead for local deployments.
    """
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    outputs = engine.generate([prompt], params)
    raw_output: str = (
        outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    )
    thinking = extract_thinking(raw_output)
    parsed = _extract_logits_json(raw_output)

    if parsed is not None and "logits" in parsed:
        logits_raw = parsed["logits"]
        if isinstance(logits_raw, list) and len(logits_raw) == len(choices):
            try:
                logits = [float(x) for x in logits_raw]
                return {
                    "logits": logits,
                    "choices": choices,
                    "raw_output": raw_output,
                    "thinking": thinking,
                    "parse_success": True,
                }
            except (TypeError, ValueError) as exc:
                logger.warning("Could not convert self-reported logits to float: %s", exc)
        else:
            logger.warning(
                "Self-reported logits length mismatch: expected %d, got %s",
                len(choices), logits_raw,
            )
    else:
        logger.warning(
            "No self-reported logits JSON found (first 300 chars): %s",
            raw_output[:300],
        )

    return {
        "logits": None,
        "choices": choices,
        "raw_output": raw_output,
        "thinking": thinking,
        "parse_success": False,
    }


def get_choice_logits_batch(
    engine: Any,
    prompts: list[str],
    choices: list[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Legacy: batch self-reported logits.  See ``get_choice_logits``.
    """
    if not prompts:
        return []

    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    outputs = engine.generate(prompts, params)

    results: list[dict[str, Any]] = []
    for i, request_output in enumerate(outputs):
        raw_output: str = (
            request_output.outputs[0].text if request_output.outputs else ""
        )
        thinking = extract_thinking(raw_output)
        parsed = _extract_logits_json(raw_output)

        if parsed is not None and "logits" in parsed:
            logits_raw = parsed["logits"]
            if isinstance(logits_raw, list) and len(logits_raw) == len(choices):
                try:
                    logits = [float(x) for x in logits_raw]
                    results.append({
                        "logits": logits,
                        "choices": choices,
                        "raw_output": raw_output,
                        "thinking": thinking,
                        "parse_success": True,
                    })
                    continue
                except (TypeError, ValueError) as exc:
                    logger.warning("Batch[%d] logits conversion failed: %s", i, exc)
            else:
                logger.warning(
                    "Batch[%d] logits length mismatch: expected %d, got %s",
                    i, len(choices), logits_raw,
                )
        else:
            logger.warning(
                "Batch[%d] no self-reported logits JSON (first 200 chars): %s",
                i, raw_output[:200],
            )

        results.append({
            "logits": None,
            "choices": choices,
            "raw_output": raw_output,
            "thinking": thinking,
            "parse_success": False,
        })

    return results


# ---------------------------------------------------------------------------
# Optional: cross-check (calibration experiments only, not used in pipeline)
# ---------------------------------------------------------------------------

def get_logits_with_vllm_logprobs(
    engine: Any,
    prompt: str,
    choices: list[str],
    max_tokens: int = 1024,
    logprobs_k: int = 50,
) -> dict[str, Any]:
    """
    Extended version that cross-checks self-reported logits against vLLM's
    top-K ``logprobs`` for generated tokens.  For calibration research only.
    """
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, top_p=1.0, logprobs=logprobs_k
    )
    outputs = engine.generate([prompt], params)
    raw_output: str = (
        outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    )
    thinking = extract_thinking(raw_output)
    parsed = _extract_logits_json(raw_output)

    self_reported: Optional[list[float]] = None
    if parsed and "logits" in parsed:
        raw = parsed["logits"]
        if isinstance(raw, list) and len(raw) == len(choices):
            try:
                self_reported = [float(x) for x in raw]
            except (TypeError, ValueError):
                pass

    token_logprobs = outputs[0].outputs[0].logprobs or []
    vllm_probs: Optional[dict[str, float]] = None
    if token_logprobs:
        choice_logprobs: dict[str, float] = {}
        for step_logprobs in token_logprobs:
            for _token_id, lp_obj in step_logprobs.items():
                decoded = getattr(lp_obj, "decoded_token", "") or ""
                for choice in choices:
                    if choice in decoded and choice not in choice_logprobs:
                        choice_logprobs[choice] = lp_obj.logprob
        if len(choice_logprobs) == len(choices):
            lp_list = [choice_logprobs[c] for c in choices]
            probs = softmax(lp_list, temperature=1.0)
            vllm_probs = dict(zip(choices, probs))

    return {
        "logits": self_reported,
        "choices": choices,
        "raw_output": raw_output,
        "thinking": thinking,
        "parse_success": self_reported is not None,
        "vllm_probs": vllm_probs,
    }
