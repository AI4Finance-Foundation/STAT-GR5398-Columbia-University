#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple, Union


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class ProviderSpec:
    default_model: str
    default_base_url: Optional[str]
    config_api_keys: Tuple[str, ...]
    config_model_keys: Tuple[str, ...]
    config_base_url_keys: Tuple[str, ...]
    env_api_keys: Tuple[str, ...]
    env_base_url_keys: Tuple[str, ...]
    model_prefixes: Tuple[str, ...]


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        default_model=DEFAULT_OPENAI_MODEL,
        default_base_url=None,
        config_api_keys=("openai_api_key", "llm_api_key"),
        config_model_keys=("openai_model", "llm_model"),
        config_base_url_keys=("openai_base_url", "llm_base_url"),
        env_api_keys=("OPENAI_API_KEY", "LLM_API_KEY"),
        env_base_url_keys=("OPENAI_BASE_URL", "LLM_BASE_URL"),
        model_prefixes=("gpt-", "o1", "o3", "o4"),
    ),
    "claude": ProviderSpec(
        default_model=DEFAULT_CLAUDE_MODEL,
        default_base_url=None,
        config_api_keys=("claude_api_key", "llm_api_key"),
        config_model_keys=("claude_model", "llm_model"),
        config_base_url_keys=("claude_base_url", "llm_base_url"),
        env_api_keys=("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "LLM_API_KEY"),
        env_base_url_keys=("ANTHROPIC_BASE_URL", "CLAUDE_BASE_URL", "LLM_BASE_URL"),
        model_prefixes=("claude-",),
    ),
    "gemini": ProviderSpec(
        default_model=DEFAULT_GEMINI_MODEL,
        default_base_url=None,
        config_api_keys=("gemini_api_key", "llm_api_key"),
        config_model_keys=("gemini_model", "llm_model"),
        config_base_url_keys=("gemini_base_url", "llm_base_url"),
        env_api_keys=("GOOGLE_API_KEY", "GEMINI_API_KEY", "LLM_API_KEY"),
        env_base_url_keys=("GEMINI_BASE_URL", "GOOGLE_BASE_URL", "LLM_BASE_URL"),
        model_prefixes=("gemini-", "models/gemini-", "gemini/"),
    ),
}
SUPPORTED_PROVIDERS = tuple(PROVIDER_SPECS.keys())


@dataclass
class LLMSettings:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str] = None


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_s = str(value).strip()
    return value_s if value_s else None


def _cfg_get(config, section: str, key: str, fallback=None):
    try:
        return _clean(config.get(section, key))
    except Exception:
        return fallback


def _first_nonempty(values) -> Optional[str]:
    for value in values:
        cleaned = _clean(value)
        if cleaned is not None:
            return cleaned
    return None


def _pick_config(config, keys: Tuple[str, ...]) -> Optional[str]:
    return _first_nonempty(_cfg_get(config, "API_KEYS", key, fallback=None) for key in keys)


def _pick_env(keys: Tuple[str, ...]) -> Optional[str]:
    return _first_nonempty(os.getenv(key) for key in keys)


def _validate_provider(provider: str) -> str:
    provider_norm = _clean(provider).lower()
    if provider_norm not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported llm provider '{provider}'. Supported providers: {SUPPORTED_PROVIDERS}"
        )
    return provider_norm


def infer_provider_from_model(model: Optional[str]) -> Optional[str]:
    model_norm = _clean(model)
    if not model_norm:
        return None

    model_lower = model_norm.lower()
    for provider, spec in PROVIDER_SPECS.items():
        if any(model_lower.startswith(prefix) for prefix in spec.model_prefixes):
            return provider

    if "claude" in model_lower:
        return "claude"
    if "gemini" in model_lower:
        return "gemini"
    if "gpt" in model_lower:
        return "openai"
    return None


def resolve_provider(config, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    if _clean(provider):
        return _validate_provider(provider)

    configured_provider = _cfg_get(config, "API_KEYS", "llm_provider", fallback=None)
    if configured_provider:
        return _validate_provider(configured_provider)

    inferred_from_model = infer_provider_from_model(model)
    if inferred_from_model:
        return inferred_from_model

    configured_model = _pick_config(config, ("llm_model", "openai_model", "claude_model", "gemini_model"))
    inferred_from_config_model = infer_provider_from_model(configured_model)
    if inferred_from_config_model:
        return inferred_from_config_model

    provider_keys = {
        name: (_pick_config(config, spec.config_api_keys) or _pick_env(spec.env_api_keys))
        for name, spec in PROVIDER_SPECS.items()
    }
    configured_with_key = [name for name, api_key in provider_keys.items() if api_key]
    if len(configured_with_key) == 1:
        return configured_with_key[0]
    return "openai"


def normalize_provider(provider: Optional[str], model: Optional[str] = None) -> str:
    if _clean(provider):
        return _validate_provider(provider)
    return infer_provider_from_model(model) or "openai"


def load_llm_settings(config, provider: Optional[str] = None, model: Optional[str] = None) -> LLMSettings:
    provider_final = resolve_provider(config, provider=provider, model=model)
    spec = PROVIDER_SPECS[provider_final]

    api_key = _pick_config(config, spec.config_api_keys) or _pick_env(spec.env_api_keys)
    base_url = _pick_config(config, spec.config_base_url_keys) or _pick_env(spec.env_base_url_keys)
    model_name = _clean(model) or _pick_config(config, spec.config_model_keys) or spec.default_model

    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider_final}'.")

    if base_url is None:
        base_url = spec.default_base_url

    return LLMSettings(provider=provider_final, api_key=api_key, model=model_name, base_url=base_url)


def _is_4o_model(model_name: Optional[str]) -> bool:
    model_lower = (_clean(model_name) or "").lower()
    return model_lower.startswith("gpt-4o")


def _extract_openai_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_blocks = getattr(response, "output", None) or []
    collected = []

    for block in output_blocks:
        contents = getattr(block, "content", None)
        if contents is None and isinstance(block, dict):
            contents = block.get("content", [])

        for content in contents or []:
            text_value = getattr(content, "text", None)
            if text_value is None and isinstance(content, dict):
                text_value = content.get("text")
            if isinstance(text_value, str) and text_value.strip():
                collected.append(text_value.strip())

    return "\n".join(collected).strip()


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _extract_openai_usage(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = _safe_int(usage.get("input_tokens", usage.get("prompt_tokens")))
            completion_tokens = _safe_int(usage.get("output_tokens", usage.get("completion_tokens")))
            total_tokens = _safe_int(usage.get("total_tokens"))
        else:
            prompt_tokens = _safe_int(getattr(usage, "input_tokens", None))
            if prompt_tokens is None:
                prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", None))

            completion_tokens = _safe_int(getattr(usage, "output_tokens", None))
            if completion_tokens is None:
                completion_tokens = _safe_int(getattr(usage, "completion_tokens", None))

            total_tokens = _safe_int(getattr(usage, "total_tokens", None))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_claude_usage(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = _safe_int(usage.get("input_tokens"))
            completion_tokens = _safe_int(usage.get("output_tokens"))
            total_tokens = _safe_int(usage.get("total_tokens"))
        else:
            prompt_tokens = _safe_int(getattr(usage, "input_tokens", None))
            completion_tokens = _safe_int(getattr(usage, "output_tokens", None))
            total_tokens = _safe_int(getattr(usage, "total_tokens", None))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_gemini_text(response) -> str:
    text = getattr(response, "text", None)
    if callable(text):
        try:
            text = text()
        except Exception:
            text = None
    if text is None and isinstance(response, dict):
        text = response.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    collected = []

    def _collect_from_parts(parts) -> None:
        for part in parts or []:
            part_text = getattr(part, "text", None)
            if part_text is None and isinstance(part, dict):
                part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                collected.append(part_text.strip())

    parts = getattr(response, "parts", None)
    if parts is None and isinstance(response, dict):
        parts = response.get("parts")
    _collect_from_parts(parts)

    if not collected:
        candidates = getattr(response, "candidates", None)
        if candidates is None and isinstance(response, dict):
            candidates = response.get("candidates")
        for candidate in candidates or []:
            content = getattr(candidate, "content", None)
            if content is None and isinstance(candidate, dict):
                content = candidate.get("content")
            candidate_parts = getattr(content, "parts", None)
            if candidate_parts is None and isinstance(content, dict):
                candidate_parts = content.get("parts")
            _collect_from_parts(candidate_parts)

    return "\n".join(collected).strip()


def _extract_gemini_usage(response) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = _safe_int(usage.get("prompt_token_count", usage.get("input_tokens")))
            completion_tokens = _safe_int(
                usage.get(
                    "candidates_token_count",
                    usage.get("response_token_count", usage.get("output_tokens")),
                )
            )
            total_tokens = _safe_int(usage.get("total_token_count", usage.get("total_tokens")))
        else:
            prompt_tokens = _safe_int(getattr(usage, "prompt_token_count", None))
            if prompt_tokens is None:
                prompt_tokens = _safe_int(getattr(usage, "input_tokens", None))

            completion_tokens = _safe_int(getattr(usage, "candidates_token_count", None))
            if completion_tokens is None:
                completion_tokens = _safe_int(getattr(usage, "response_token_count", None))
            if completion_tokens is None:
                completion_tokens = _safe_int(getattr(usage, "output_tokens", None))

            total_tokens = _safe_int(getattr(usage, "total_token_count", None))
            if total_tokens is None:
                total_tokens = _safe_int(getattr(usage, "total_tokens", None))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _call_openai(
    settings: LLMSettings,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, Optional[int]]]]:
    from openai import OpenAI

    client_kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url
    client = OpenAI(**client_kwargs)

    params = {
        "model": settings.model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }

    if _is_4o_model(settings.model):
        if temperature is not None:
            params["temperature"] = temperature
        params["text"] = {"verbosity": "medium"}
    else:
        params["reasoning"] = {"effort": "high"}
        params["text"] = {"verbosity": "low"}

    response = client.responses.create(**params)
    text = _extract_openai_text(response)
    if return_meta:
        return text, _extract_openai_usage(response)
    return text


def _call_claude(
    settings: LLMSettings,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, Optional[int]]]]:
    from anthropic import Anthropic

    client_kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        client_kwargs["base_url"] = settings.base_url
    client = Anthropic(**client_kwargs)

    params = {
        "model": settings.model,
        "system": instructions,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        params["temperature"] = temperature

    # Anthropic SDK long-request guard: use streaming by default for Claude.
    with client.messages.stream(**params) as stream:
        chunks = [text for text in stream.text_stream]
        final_message = stream.get_final_message()

    text = "".join(chunks).strip()
    if return_meta:
        return text, _extract_claude_usage(final_message)
    return text


def _call_gemini(
    settings: LLMSettings,
    instructions: str,
    prompt: str,
    max_output_tokens: int,
    temperature: Optional[float],
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, Optional[int]]]]:
    from google import genai
    from google.genai import types

    client_kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        client_kwargs["http_options"] = {"base_url": settings.base_url}
    client = genai.Client(**client_kwargs)

    config_kwargs = {
        "system_instruction": instructions,
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=settings.model,
        contents=prompt,
        config=config,
    )
    text = _extract_gemini_text(response)
    if return_meta:
        return text, _extract_gemini_usage(response)
    return text


PROVIDER_CALLERS = {
    "openai": _call_openai,
    "claude": _call_claude,
    "gemini": _call_gemini,
}


def call_llm(
    settings: LLMSettings,
    instructions: str,
    prompt: str,
    max_output_tokens: int = 1200,
    temperature: Optional[float] = 0.2,
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, Optional[int]]]]:
    provider = normalize_provider(settings.provider, settings.model)
    caller = PROVIDER_CALLERS.get(provider)
    if caller is None:
        raise ValueError(f"No caller registered for provider '{provider}'.")

    result = caller(
        settings=settings,
        instructions=instructions,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        return_meta=return_meta,
    )

    return result
