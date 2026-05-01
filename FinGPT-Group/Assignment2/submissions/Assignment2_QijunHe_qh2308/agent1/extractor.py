"""
agent1/extractor.py — Agent 1: FinGPT fact extractor + logits-based sentiment.

Pipeline per article
--------------------
1. Fact extraction  — guided-decoding vLLM call → JSON (source, headline, …).
2. Sentiment scoring — CoT vLLM call → self-reported logits for
                       [POSITIVE, NEGATIVE, NEUTRAL].
3. Deterministic post-processing — softmax(logits / CALIBRATION_T) →
                       probabilities, confidence, label.
4. Return a NewsFingerprint that combines facts + calibrated sentiment.

The LLM never outputs a final sentiment label, probability, or confidence
value — all of those are computed in Python from the raw logits.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Any, Optional

from config import (
    CALIBRATION_T,
    FINGPT_MODEL_PATH,
    LOG_LEVEL,
    LOGITS_MAX_TOKENS,
    SENTIMENT_CLASSES,
)
from agent1.prompt import (
    EXTRACTION_PROMPT,
    SENTIMENT_COT_PROMPT,
    SENTIMENT_DECISION_PREFIX,
)
from agent1.schema import NewsFingerprint, SentimentLabel, _SENTIMENT_SCORE
from vllm_logits_client import (
    get_real_choice_logits,
    get_real_choice_logits_batch,
    softmax,
)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_DIAG_MD_DIR_ENV = "FINGPT_DIAG_MD_DIR"
_SPECIAL_TOKEN_RE = re.compile(
    r"(<\|[^>]+\|>|<｜[^｜]+｜>|<s>|</s>|\[INST\]|\[/INST\])"
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

_vllm_engine = None
_chat_tokenizer = None


# ---------------------------------------------------------------------------
# Engine management (unchanged public API)
# ---------------------------------------------------------------------------

def _ensure_chat_tokenizer() -> None:
    global _chat_tokenizer  # noqa: PLW0603
    if _chat_tokenizer is not None:
        return
    if not FINGPT_MODEL_PATH:
        return
    try:
        from transformers import AutoTokenizer  # type: ignore

        _chat_tokenizer = AutoTokenizer.from_pretrained(
            FINGPT_MODEL_PATH,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning("Failed to initialize chat tokenizer: %s", exc)


def _load_model() -> None:
    """Lazy-load one vLLM engine once."""
    global _vllm_engine, _chat_tokenizer  # noqa: PLW0603

    if _vllm_engine is not None:
        _ensure_chat_tokenizer()
        return

    if not FINGPT_MODEL_PATH:
        raise EnvironmentError(
            "FINGPT_MODEL_PATH is not set. Add it to your .env pointing to local weights."
        )

    from vllm import LLM  # type: ignore

    logger.info("Loading FinGPT model from: %s", FINGPT_MODEL_PATH)
    _ensure_chat_tokenizer()
    _vllm_engine = LLM(
        model=FINGPT_MODEL_PATH,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        enforce_eager=True,
    )
    logger.info("FinGPT vLLM engine loaded.")


def get_loaded_model_and_tokenizer():
    """Backward-compatibility shim — vLLM path returns (None, None)."""
    return None, None


def set_shared_vllm_engine(engine) -> None:
    """Inject a preloaded vLLM engine (e.g. from notebook bootstrap)."""
    global _vllm_engine  # noqa: PLW0603
    _vllm_engine = engine
    _ensure_chat_tokenizer()
    logger.info("Injected shared vLLM engine into Agent 1 extractor.")


def get_shared_vllm_engine():
    """Return Agent 1 vLLM engine if already initialized/injected."""
    return _vllm_engine


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_chat_prompt(system_prompt: str, user_content: str) -> str:
    """Format using tokenizer chat template, with a safe fallback."""
    _ensure_chat_tokenizer()
    if _chat_tokenizer is not None and hasattr(_chat_tokenizer, "apply_chat_template"):
        try:
            if system_prompt.strip():
                user_content = f"{system_prompt}\n\n{user_content}"
            messages = [{"role": "user", "content": user_content}]
            return _chat_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning("apply_chat_template failed, using fallback prompt: %s", exc)

    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_content}\n"
        "<|assistant|>\n"
    )


# ---------------------------------------------------------------------------
# Text-normalisation helpers (for fact-extraction output)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _normalize_generated_text(text: str) -> str:
    cleaned = text.replace("Ġ", " ").replace("Ċ", "\n")
    cleaned = _SPECIAL_TOKEN_RE.sub(" ", cleaned)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    marker_match = re.search(r"(?im)^###\s*facts\b", cleaned)
    if marker_match:
        cleaned = cleaned[marker_match.start():].strip()
    return cleaned


def _extract_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    match = _JSON_OBJ_RE.search(cleaned)
    return match.group(0).strip() if match else cleaned


def _extract_balanced_json_blob(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()
    return ""


def _parse_extracted_json(raw_output: str) -> dict[str, Any]:
    primary = _extract_json_blob(raw_output)
    try:
        return json.loads(primary)
    except json.JSONDecodeError:
        balanced = _extract_balanced_json_blob(raw_output)
        if balanced:
            return json.loads(balanced)
        raise


def _parse_markdown_extraction(raw_output: str) -> dict[str, Any]:
    text = _strip_code_fences(raw_output)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    result: dict[str, Any] = {
        "source": "",
        "published_at": "",
        "headline": "",
        "companies_named": [],
        "event_keywords": [],
    }

    def _add_list_items(field: str, raw: str) -> None:
        item = raw.strip().strip("-").strip()
        if not item:
            return
        item = item.replace("**", "").strip()
        if item.lower() == "none":
            return
        if "," in item:
            for part in item.split(","):
                val = part.strip()
                if val and val.lower() != "none":
                    result[field].append(val)
            return
        result[field].append(item)

    active_list: str | None = None
    for line in lines:
        striped = line.strip()
        lower = striped.lower()

        if lower.startswith("### "):
            active_list = None
            continue

        normalized = striped.replace("**", "")
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        normalized_lower = normalized.lower()

        if normalized_lower.startswith("source:"):
            result["source"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("published_at:"):
            result["published_at"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("headline:"):
            result["headline"] = normalized.split(":", 1)[1].strip()
            active_list = None
            continue
        if normalized_lower.startswith("companies_named:"):
            active_list = "companies_named"
            remainder = normalized.split(":", 1)[1].strip()
            if remainder:
                _add_list_items(active_list, remainder)
            continue
        if normalized_lower.startswith("event_keywords:"):
            active_list = "event_keywords"
            remainder = normalized.split(":", 1)[1].strip()
            if remainder:
                _add_list_items(active_list, remainder)
            continue

        if active_list is not None and (striped.startswith("-") or striped.startswith("*")):
            item = striped[1:].strip()
            _add_list_items(active_list, item)

    return result


def _parse_extraction_structured(raw_output: str) -> dict[str, Any]:
    if not raw_output.strip():
        raise ValueError("Model returned empty output after normalization")
    try:
        return _parse_extracted_json(raw_output)
    except Exception:
        parsed = _parse_markdown_extraction(raw_output)
        if parsed.get("headline") or parsed.get("companies_named"):
            return parsed
        raise


# ---------------------------------------------------------------------------
# Fact-extraction vLLM call (guided decoding, unchanged mechanism)
# ---------------------------------------------------------------------------

def _import_structured_output_params():
    try:
        from vllm.sampling_params import StructuredOutputParams  # type: ignore
        return StructuredOutputParams
    except ImportError:
        logger.warning("StructuredOutputParams not available in current vLLM version.")
    return None


def _build_guided_extraction_schema() -> dict[str, Any]:
    """Return a JSON schema for the extraction-only fields (no sentiment)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "published_at": {"type": "string"},
            "headline": {"type": "string"},
            "companies_named": {"type": "array", "items": {"type": "string"}},
            "event_keywords": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["source", "published_at", "headline", "companies_named", "event_keywords"],
    }
    return schema


def _generate_extraction_text(
    prompt: str,
    max_tokens: int,
    guided_schema: Optional[dict[str, Any]] = None,
) -> str:
    from vllm import SamplingParams  # type: ignore

    params_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if guided_schema is not None:
        StructuredOutputParams = _import_structured_output_params()
        if StructuredOutputParams is not None:
            params_kwargs["guided_decoding"] = StructuredOutputParams(json=guided_schema)
        else:
            logger.warning(
                "Structured output params unavailable; falling back to non-guided extraction."
            )
    params = SamplingParams(**params_kwargs)
    outputs = _vllm_engine.generate([prompt], params)
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


# ---------------------------------------------------------------------------
# Diagnostic / debug helpers
# ---------------------------------------------------------------------------

def _save_md_debug_output(
    article_text: str,
    raw_output: str,
    label: str,
    attempt_idx: int,
    token_budget: int,
) -> None:
    try:
        base_dir = os.getenv(_DIAG_MD_DIR_ENV, "output/diagnostics_md")
        out_dir = os.path.join(base_dir, "agent1")
        os.makedirs(out_dir, exist_ok=True)

        article_id = md5(article_text.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{timestamp}_a1_{article_id}_{label}_attempt{attempt_idx}_tok{token_budget}.md"
        )
        out_path = os.path.join(out_dir, filename)

        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(raw_output)
        logger.info("Saved Agent 1 debug output to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save Agent 1 debug output: %s", exc)


# ---------------------------------------------------------------------------
# Sentiment post-processing (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _process_sentiment_result(
    result: dict[str, Any],
    article_text: str,
    *,
    save_debug: bool = True,
) -> dict[str, Any]:
    """
    Convert a ``get_choice_logits`` / ``get_choice_logits_batch`` result dict
    into the sentiment sub-dict used when building a ``NewsFingerprint``.

    This function contains **all** output logic: softmax, argmax, confidence,
    and the uniform-NEUTRAL fallback.  It is called identically by the
    single-item path (``_score_sentiment``) and the batch path
    (``extract_fingerprint_batch``).
    """
    if save_debug:
        _save_md_debug_output(
            article_text,
            result["raw_output"],
            "sentiment",
            1,
            LOGITS_MAX_TOKENS,
        )

    if result["parse_success"] and result["logits"] is not None:
        logits: list[float] = result["logits"]
        probs = softmax(logits, temperature=CALIBRATION_T)
        best_idx = probs.index(max(probs))
        label: SentimentLabel = SENTIMENT_CLASSES[best_idx]  # type: ignore[assignment]
        confidence = probs[best_idx]
        probabilities = dict(zip(SENTIMENT_CLASSES, probs))
        logger.info(
            "Sentiment logits=%s  probs=%s  label=%s  conf=%.4f",
            logits,
            [f"{p:.4f}" for p in probs],
            label,
            confidence,
        )
    else:
        logger.warning(
            "Sentiment logits parse failed; falling back to NEUTRAL with uniform probs."
        )
        logits = [0.0, 0.0, 0.0]
        probs = [1.0 / 3] * 3
        label = "NEUTRAL"
        confidence = 1.0 / 3
        probabilities = dict(zip(SENTIMENT_CLASSES, probs))

    return {
        "sentiment_label": label,
        "sentiment_score": float(_SENTIMENT_SCORE[label]),
        "sentiment_confidence": round(confidence, 6),
        "sentiment_probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        "sentiment_logits": logits,
        "calibration_T": CALIBRATION_T,
    }


# ---------------------------------------------------------------------------
# Sentiment scoring — logits-based (single-item)
# ---------------------------------------------------------------------------

def _score_sentiment(article_text: str) -> dict[str, Any]:
    """
    Extract real model log-probabilities for each sentiment class.

    Two vLLM calls (handled inside get_real_choice_logits):
      Phase 1 — CoT generation (stop at </think>).
      Phase 2 — prompt_logprobs scoring of POSITIVE / NEGATIVE / NEUTRAL.

    Output logic is fully delegated to ``_process_sentiment_result``.
    """
    _ensure_chat_tokenizer()
    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling extract_fingerprint."
        )

    cot_user_content = SENTIMENT_COT_PROMPT.format(article_text=article_text)
    cot_prompt = _format_chat_prompt("", cot_user_content)

    result = get_real_choice_logits(
        engine=_vllm_engine,
        cot_prompt=cot_prompt,
        choices=SENTIMENT_CLASSES,
        decision_prefix=SENTIMENT_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=LOGITS_MAX_TOKENS,
    )
    return _process_sentiment_result(result, article_text)


# ---------------------------------------------------------------------------
# Fingerprint assembly (shared by single-item and batch paths)
# ---------------------------------------------------------------------------

def _assemble_fingerprint(
    extracted: dict[str, Any],
    sentiment: dict[str, Any],
    article_text: str,
) -> NewsFingerprint:
    """Build a ``NewsFingerprint`` from fact-extraction + sentiment dicts."""
    payload: dict[str, Any] = {
        "source": extracted.get("source", ""),
        "published_at": extracted.get("published_at", ""),
        "headline": extracted.get("headline", ""),
        "companies_named": extracted.get("companies_named", []),
        "event_keywords": extracted.get("event_keywords", []),
        "sentiment_label": sentiment["sentiment_label"],
        "sentiment_score": sentiment["sentiment_score"],
        "sentiment_confidence": sentiment["sentiment_confidence"],
        "sentiment_probabilities": sentiment["sentiment_probabilities"],
        "sentiment_logits": sentiment["sentiment_logits"],
        "calibration_T": sentiment["calibration_T"],
        "article_text": article_text,
    }
    return NewsFingerprint(**payload)


def _build_extraction_prompt(article_text: str) -> str:
    user_content = f"{EXTRACTION_PROMPT}\n\nArticle:\n{article_text}"
    return _format_chat_prompt("", user_content)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_fingerprint_batch(
    article_texts: list[str],
) -> list[Optional[NewsFingerprint]]:
    """
    Batch version of ``extract_fingerprint``.

    Sends all articles to vLLM in **two batched calls**:
      1. Guided fact-extraction call (all articles, single ``SamplingParams``).
      2. Sentiment CoT + logits call (all articles, single ``SamplingParams``).

    Per-item output logic (softmax, argmax, ``NewsFingerprint`` assembly) is
    identical to the single-item path — handled by ``_process_sentiment_result``
    and ``_assemble_fingerprint``.

    Returns a list of the same length as *article_texts*.  Items are ``None``
    wherever extraction or validation fails.
    """
    _load_model()
    _ensure_chat_tokenizer()

    if _chat_tokenizer is None:
        raise RuntimeError(
            "Tokenizer not loaded — cannot compute real choice logprobs. "
            "Ensure FINGPT_MODEL_PATH is set before calling extract_fingerprint_batch."
        )

    n = len(article_texts)
    if n == 0:
        return []

    from vllm import SamplingParams  # type: ignore

    # --- Step 1: Batch guided fact extraction (1 vLLM call) ---
    extraction_prompts = [_build_extraction_prompt(t) for t in article_texts]
    guided_schema = _build_guided_extraction_schema()
    ext_params_kwargs: dict[str, Any] = {"max_tokens": 2048, "temperature": 0.0}
    StructuredOutputParams = _import_structured_output_params()
    if StructuredOutputParams is not None:
        ext_params_kwargs["guided_decoding"] = StructuredOutputParams(json=guided_schema)
    else:
        logger.warning(
            "StructuredOutputParams unavailable; batch extraction runs without guided decoding."
        )
    ext_params = SamplingParams(**ext_params_kwargs)
    ext_outputs = _vllm_engine.generate(extraction_prompts, ext_params)
    raw_extractions = [
        o.outputs[0].text if o.outputs else "" for o in ext_outputs
    ]

    # --- Step 2: Batch real logprobs for sentiment (2 vLLM calls internally) ---
    # Phase 1 (CoT generation) + Phase 2 (N×K scoring prompts) are both
    # handled inside get_real_choice_logits_batch as single batched calls.
    cot_prompts = [
        _format_chat_prompt("", SENTIMENT_COT_PROMPT.format(article_text=t))
        for t in article_texts
    ]
    sentiment_results = get_real_choice_logits_batch(
        engine=_vllm_engine,
        cot_prompts=cot_prompts,
        choices=SENTIMENT_CLASSES,
        decision_prefix=SENTIMENT_DECISION_PREFIX,
        tokenizer=_chat_tokenizer,
        max_cot_tokens=LOGITS_MAX_TOKENS,
    )

    # --- Step 3: Per-item assembly (output logic unchanged) ---
    fingerprints: list[Optional[NewsFingerprint]] = []
    for i, article_text in enumerate(article_texts):
        try:
            clean_extraction = _normalize_generated_text(raw_extractions[i])
            _save_md_debug_output(article_text, clean_extraction, "extraction", 1, 2048)
            logger.info(
                "Batch extraction[%d] output (first 240 chars): %s",
                i, clean_extraction[:240],
            )
            extracted = _parse_extraction_structured(clean_extraction)

            sentiment = _process_sentiment_result(
                sentiment_results[i], article_text, save_debug=True
            )

            fingerprints.append(_assemble_fingerprint(extracted, sentiment, article_text))
        except Exception as exc:
            logger.error(
                "extract_fingerprint_batch[%d] failed: %s", i, exc, exc_info=True
            )
            fingerprints.append(None)

    return fingerprints


def extract_fingerprint(article_text: str) -> Optional[NewsFingerprint]:
    """
    Extract structured facts and logits-based sentiment from a raw news article.

    Returns None on any model, parse, or validation failure.
    """
    try:
        _load_model()

        # Fact extraction (guided decoding).
        extraction_prompt = _build_extraction_prompt(article_text)
        guided_schema = _build_guided_extraction_schema()
        token_budget = 2048
        raw_extraction = _generate_extraction_text(
            extraction_prompt,
            max_tokens=token_budget,
            guided_schema=guided_schema,
        )
        clean_extraction = _normalize_generated_text(raw_extraction)
        _save_md_debug_output(article_text, clean_extraction, "extraction", 1, token_budget)
        logger.info(
            "Agent 1 fact extraction output (max_tokens=%d): %s",
            token_budget,
            clean_extraction[:240],
        )
        extracted = _parse_extraction_structured(clean_extraction)

        # Logits-based sentiment scoring.
        sentiment = _score_sentiment(article_text)

        return _assemble_fingerprint(extracted, sentiment, article_text)

    except Exception as exc:
        logger.error("extract_fingerprint failed: %s", exc, exc_info=True)
        return None
