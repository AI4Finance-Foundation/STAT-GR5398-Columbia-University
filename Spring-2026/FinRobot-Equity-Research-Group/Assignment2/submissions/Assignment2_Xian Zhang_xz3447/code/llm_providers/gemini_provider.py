"""Google Gemini backend (google-genai SDK)."""

from google import genai
from google.genai import types

from .base import LLMError, ProviderResponse


def call_gemini(
    system: str,
    user: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> ProviderResponse:
    if not api_key:
        raise LLMError("Gemini API key is missing")

    try:
        client = genai.Client(api_key=api_key)
        # Gemini 2.5 uses internal "thinking" tokens out of the output budget.
        # - Pro requires thinking on (>=128 budget); we allow ~2k thinking and
        #   bump max_output_tokens so the caller's answer budget is preserved.
        # - Flash supports thinking_budget=0 (faster, cheaper), so disable it.
        m = model.lower()
        thinking_config = None
        effective_max_tokens = max_tokens
        if "2.5-pro" in m or "2.5-flash" in m:
            try:
                if "2.5-pro" in m:
                    thinking_config = types.ThinkingConfig(thinking_budget=2048)
                    effective_max_tokens = max_tokens + 2048
                else:  # 2.5-flash
                    thinking_config = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

        config_kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=effective_max_tokens,
        )
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        config = types.GenerateContentConfig(**config_kwargs)
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=config,
        )
    except Exception as e:
        raise LLMError(f"Gemini call failed for model={model}: {e}") from e

    text = (resp.text or "").strip() if hasattr(resp, "text") else ""
    if not text:
        try:
            parts = []
            for cand in resp.candidates or []:
                for part in (cand.content.parts or []):
                    if getattr(part, "text", None):
                        parts.append(part.text)
            text = "".join(parts).strip()
        except Exception:
            text = ""

    usage = getattr(resp, "usage_metadata", None)
    return ProviderResponse(
        text=text,
        model=model,
        provider="gemini",
        input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
        output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
    )
