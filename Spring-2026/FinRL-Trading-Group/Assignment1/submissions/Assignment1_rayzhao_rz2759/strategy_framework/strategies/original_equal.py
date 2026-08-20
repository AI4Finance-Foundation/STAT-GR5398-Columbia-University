from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class OriginalEqualConfig:
    force_config_name: str | None = None


class OriginalEqualStrategy:
    """Original equal-weight portfolio: equal weight across selected names each trade date."""

    name = "original_equal"
    aliases = [
        "original",
        "orig_equal",
        "equal_original",
        "equal_weight_original",
    ]

    def __init__(self, config: OriginalEqualConfig | None = None) -> None:
        self.config = config or OriginalEqualConfig()

    def generate_signals(self, selected_stocks_df: pd.DataFrame, ticker_sector, cc_ret_df) -> StrategySignals:
        cfg = self.config
        selected = selected_stocks_df.copy()
        selected["trade_date"] = pd.to_datetime(selected["trade_date"]).dt.normalize()
        selected["tic"] = selected["tic"].astype(str)
        selected = selected.dropna(subset=["trade_date", "tic"])

        portfolio_dates = pd.to_datetime(sorted(selected["trade_date"].unique()))
        all_tics = sorted(selected["tic"].drop_duplicates().tolist())

        weights_dict: dict[pd.Timestamp, dict[str, float]] = {}
        regime_rows: list[dict[str, float]] = []

        prev_weights = {tic: 0.0 for tic in all_tics}

        for date in portfolio_dates:
            day_tics = (
                selected.loc[selected["trade_date"] == date, "tic"]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            current_weights = {tic: 0.0 for tic in all_tics}
            if day_tics:
                w = 1.0 / len(day_tics)
                for tic in day_tics:
                    current_weights[tic] = w

            traded = int(
                (date == portfolio_dates[0])
                or any(abs(current_weights[t] - prev_weights[t]) > 1e-12 for t in all_tics)
            )

            weights_dict[date] = current_weights
            regime_rows.append(
                {
                    "date": date,
                    "regime": 1,
                    "regime_label": "Neutral",
                    "market_vol": np.nan,
                    "market_mom": np.nan,
                    "alpha_power": np.nan,
                    "trade_executed": traded,
                }
            )

            prev_weights = current_weights

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = "original_equal"
        config_name = normalize_config_name(cfg.force_config_name) or auto_name

        return StrategySignals(
            weights_signal_df=weights_df,
            regime_df=regime_df,
            config_name=config_name,
            metadata={"strategy": self.name},
        )
