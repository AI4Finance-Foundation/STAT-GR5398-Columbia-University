"""Routes a model name to the right provider."""

from typing import Dict, Optional

from .anthropic_provider import call_anthropic
from .base import LLMError, ProviderResponse
from .gemini_provider import call_gemini
from .openai_provider import call_openai


def resolve_provider_name(model: str) -> str:
    """Map model name to provider key. Used for picking the right API key."""
    name = model.lower()
    if name.startswith(("claude", "claude-")):
        return "anthropic"
    if name.startswith(("gemini", "models/gemini")):
        return "gemini"
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise LLMError(f"Cannot infer provider for model '{model}'")


def get_provider_for_model(model: str):
    """Returns the call function for the model's provider."""
    p = resolve_provider_name(model)
    return {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "gemini": call_gemini,
    }[p]


def chat(
    system: str,
    user: str,
    model: str,
    api_keys: Dict[str, str],
    temperature: float = 0.7,
    max_tokens: int = 2000,
    openai_base_url: Optional[str] = None,
) -> ProviderResponse:
    """One call site for the whole pipeline.

    api_keys keys: 'openai', 'anthropic', 'gemini'.
    """
    provider = resolve_provider_name(model)
    api_key = api_keys.get(provider) or api_keys.get(f"{provider}_api_key")

    if provider == "openai":
        return call_openai(
            system=system,
            user=user,
            model=model,
            api_key=api_key,
            base_url=openai_base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        return call_anthropic(
            system=system,
            user=user,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "gemini":
        return call_gemini(
            system=system,
            user=user,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise LLMError(f"Unhandled provider: {provider}")
