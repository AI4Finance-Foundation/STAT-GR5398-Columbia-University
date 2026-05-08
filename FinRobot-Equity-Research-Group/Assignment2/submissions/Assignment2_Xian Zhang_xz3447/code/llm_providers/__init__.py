"""Multi-provider LLM abstraction layer.

Routes chat completion calls to OpenAI, Anthropic, or Google Gemini based on
the model name. Lets the rest of the pipeline stay provider-agnostic.

Usage:
    from modules.llm_providers import chat

    text = chat(
        system="You are a financial analyst.",
        user="Summarize NVDA's Q4 earnings.",
        model="claude-opus-4-5",
        api_keys={"anthropic": "...", "openai": "...", "google": "..."},
    )
"""

from .base import LLMError, ProviderResponse
from .factory import chat, get_provider_for_model, resolve_provider_name

__all__ = [
    "chat",
    "get_provider_for_model",
    "resolve_provider_name",
    "LLMError",
    "ProviderResponse",
]
