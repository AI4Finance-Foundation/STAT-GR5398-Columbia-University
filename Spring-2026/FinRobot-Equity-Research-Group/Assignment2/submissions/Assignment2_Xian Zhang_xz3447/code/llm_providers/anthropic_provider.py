"""Anthropic Messages API backend."""

import anthropic

from .base import LLMError, ProviderResponse


def call_anthropic(
    system: str,
    user: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> ProviderResponse:
    if not api_key:
        raise LLMError("Anthropic API key is missing")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        raise LLMError(f"Anthropic call failed for model={model}: {e}") from e

    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    text = "".join(parts).strip()

    usage = getattr(resp, "usage", None)
    return ProviderResponse(
        text=text,
        model=model,
        provider="anthropic",
        input_tokens=getattr(usage, "input_tokens", None) if usage else None,
        output_tokens=getattr(usage, "output_tokens", None) if usage else None,
    )
