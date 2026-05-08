"""Provider-agnostic types."""

from dataclasses import dataclass
from typing import Optional


class LLMError(RuntimeError):
    """Raised when a provider call fails after the SDK reports an error."""


@dataclass
class ProviderResponse:
    text: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
