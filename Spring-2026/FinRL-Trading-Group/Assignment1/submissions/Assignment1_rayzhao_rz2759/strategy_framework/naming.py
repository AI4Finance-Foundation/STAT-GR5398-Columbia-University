from __future__ import annotations

from typing import Any


_INVALID_CONFIG_NAME_TOKENS = {"", "none", "null", "nan"}


def normalize_config_name(value: Any) -> str | None:
    """Normalize optional config name and reject common null-like tokens."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _INVALID_CONFIG_NAME_TOKENS:
        return None
    return text
