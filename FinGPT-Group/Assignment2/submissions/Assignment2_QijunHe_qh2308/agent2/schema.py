"""
agent2/schema.py — Pydantic schema for Agent 2 output.

TradingSignal is backward compatible with the previous version.
Optional logit/probability fields are added to support the new
logits-based confidence calculation without breaking the backtest pipeline.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):
    ticker: str
    direction: Literal["long", "short", "neutral"]
    strategy_tag: Literal["momentum", "mean_reversion", "event_driven", "macro", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    cot: str

    # -----------------------------------------------------------------------
    # Logits-derived fields (new — optional for backward compatibility)
    # -----------------------------------------------------------------------

    # Raw logits reported by the LLM for each strategy in STRATEGY_SET order.
    signal_logits: Optional[list[float]] = None

    # Calibrated probability vector, e.g. {"BUY": 0.72, "HOLD": 0.18, "SELL": 0.10}.
    signal_probabilities: Optional[dict[str, float]] = None

    # Temperature used for softmax calibration.
    calibration_T: Optional[float] = None
