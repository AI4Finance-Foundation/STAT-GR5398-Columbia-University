"""
config.py — Global constants for the FinGPT two-agent signal pipeline.
All secrets are read from environment variables; nothing is hardcoded here.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

FINGPT_MODEL_PATH: str = os.getenv("FINGPT_MODEL_PATH", "")
FINGPT_ADAPTER_PATH: str = os.getenv("FINGPT_ADAPTER_PATH", "")
SHARE_SINGLE_LLM_BETWEEN_AGENTS: bool = _env_bool("SHARE_SINGLE_LLM_BETWEEN_AGENTS", False)

# Generation settings for FinGPT fact extraction
FINGPT_MAX_NEW_TOKENS: int = 1024
FINGPT_TEMPERATURE: float = 0.0   # deterministic extraction; no creative generation

# ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# CLAUDE_MODEL: str = "claude-sonnet-4-5"
# CLAUDE_MAX_TOKENS: int = 16_000   # must be > thinking budget
# CLAUDE_THINKING_BUDGET: int = 8_000

NEWS_PROVIDER: str = os.getenv("NEWS_PROVIDER", "alpaca").strip().lower()

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
ALPACA_NEWS_URL: str = "https://data.alpaca.markets/v1beta1/news"
ALPACA_DEFAULT_LIMIT: int = 50

FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_NEWS_URL: str = "https://finnhub.io/api/v1/company-news"
FINNHUB_TIMEOUT_SEC: float = float(os.getenv("FINNHUB_TIMEOUT_SEC", "15"))
FINNHUB_MAX_CALLS_PER_SEC: float = float(os.getenv("FINNHUB_MAX_CALLS_PER_SEC", "25"))
FINNHUB_MAX_RETRIES: int = int(os.getenv("FINNHUB_MAX_RETRIES", "3"))
FINNHUB_RETRY_BASE_DELAY_SEC: float = float(os.getenv("FINNHUB_RETRY_BASE_DELAY_SEC", "0.5"))

# Temperature for calibrating softmax probabilities computed from LLM-reported logits.
# Values > 1 soften the distribution (less confident); values < 1 sharpen it.
# Applied after softmax: probs_calibrated = softmax(logits / CALIBRATION_T).
CALIBRATION_T: float = float(os.getenv("FINGPT_CALIBRATION_T", "1.2"))

# Ordered list of sentiment classes for Agent 1.  Order is fixed and must match
# the order the prompt instructs the model to use.
SENTIMENT_CLASSES: list[str] = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

# Ordered list of trading strategies for Agent 2.  Maps index → direction:
#   BUY → long,  HOLD → neutral,  SELL → short.
STRATEGY_SET: list[str] = ["BUY", "HOLD", "SELL"]

# Max tokens for CoT + logits generation (must accommodate <think> block).
LOGITS_MAX_TOKENS: int = int(os.getenv("FINGPT_LOGITS_MAX_TOKENS", "1024"))

LOGS_DIR: str = "logs"
OUTPUT_DIR: str = "output"
LOG_LEVEL: str = "INFO"
