from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import pandas as pd


@dataclass
class StrategySignals:
    """Container for strategy-generated signals."""

    weights_signal_df: pd.DataFrame
    regime_df: pd.DataFrame
    config_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AllocationStrategy(Protocol):
    """Strategy interface: implement signal generation only."""

    name: str

    def generate_signals(
        self,
        selected_stocks_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
        cc_ret_df: pd.DataFrame,
    ) -> StrategySignals:
        ...
