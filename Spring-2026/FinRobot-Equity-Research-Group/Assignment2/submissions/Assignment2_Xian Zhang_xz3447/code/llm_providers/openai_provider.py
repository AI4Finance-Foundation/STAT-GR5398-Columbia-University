"""OpenAI Chat Completions backend."""

from typing import Optional

from openai import OpenAI

from .base import LLMError, ProviderResponse


def call_openai(
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1500,
) -> ProviderResponse:
    if not api_key:
        raise LLMError("OpenAI API key is missing")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise LLMError(f"OpenAI call failed for model={model}: {e}") from e

    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    return ProviderResponse(
        text=text,
        model=model,
        provider="openai",
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
