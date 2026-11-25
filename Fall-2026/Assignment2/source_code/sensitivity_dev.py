
"""Post-hoc DEV-only sensitivity analysis for Assignment 2.

Changes filter thresholds without modifying the frozen
student_valuation.py module.

Run:
    PYTHONHASHSEED=0 python sensitivity_dev.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.backtester import BacktestConfig, run_backtest
from portfolio.hooks import FilteredPredictions
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.metrics import performance
from shared.periods import resolve

from valuation.student_valuation import (
    peer_multiple_value,
    historical_ev_ebitda_value,
    dcf_value,
)

CONFIG = "selection/configs/growth.json"
OUTPUT = Path("output")

# (experiment, overvaluation cutoff, maximum method spread)
SCENARIOS = [
    ("S1_loose_bearish", -0.20, 2.0),
    ("S2_frozen_baseline", -0.30, 2.0),
    ("S3_strict_bearish", -0.40, 2.0),
    ("S4_strict_agreement", -0.30, 1.5),
    ("S5_loose_agreement", -0.30, 3.0),
]


def make_filter(bearish_cutoff, max_spread):
    """Create a threshold variant of the frozen filter."""

    def parameterized_filter(as_of, candidates, data):
        candidates = tuple(candidates)

        growth = sorted(
            (
                p for p in candidates
                if p.bucket == "growth_tech"
            ),
            key=lambda p: p.rank_score,
            reverse=True,
        )

        top_candidates = growth[:30]

        if not top_candidates:
            return candidates

        tickers = [p.symbol for p in top_candidates]

        # Same three methods as the frozen model.
        peer = peer_multiple_value(as_of, tickers, data)
        historical = historical_ev_ebitda_value(
            as_of, tickers, data
        )
        dcf = dcf_value(as_of, tickers, data)

        estimates = {}

        for frame in (peer, historical, dcf):
            for _, row in frame.iterrows():
                ticker = row["ticker"]
                fair_value = float(row["fair_value"])
                mispricing = float(row["mispricing"])

                if not (
                    np.isfinite(fair_value)
                    and fair_value > 0
                    and np.isfinite(mispricing)
                ):
                    continue

                estimates.setdefault(ticker, []).append(
                    (fair_value, mispricing)
                )

        removed = set()

        for ticker, methods in estimates.items():
            if len(methods) < 2:
                continue

            fair_values = np.asarray(
                [m[0] for m in methods], dtype=float
            )
            mispricings = np.asarray(
                [m[1] for m in methods], dtype=float
            )

            spread = fair_values.max() / fair_values.min()

            if spread > max_spread:
                continue

            bearish_votes = int(
                np.sum(mispricings < bearish_cutoff)
            )

            if bearish_votes >= 2:
                removed.add(ticker)

        # Preserve the original Prediction objects.
        return tuple(
            p for p in candidates
            if p.symbol not in removed
        )

    return parameterized_filter


def main():
    _, options, params, prices, _, universe = load(CONFIG)

    bundle = DataBundle.load(prices=prices)
    start, end = resolve("dev", None, None)

    rows = []

    saved_path = OUTPUT / "dev_B_valuation" / "equity.csv"

    for name, cutoff, max_spread in SCENARIOS:
        print(
            f"\nRunning {name}: "
            f"cutoff={cutoff}, spread={max_spread}",
            flush=True,
        )

        filter_fn = make_filter(cutoff, max_spread)

        wrapped = FilteredPredictions(
            MonthlyScores(),
            filter_fn,
            bundle,
        )

        config = BacktestConfig(
            start=start,
            end=end,
        )

        result = run_backtest(
            prices,
            wrapped,
            params,
            universe,
            config,
            global_unified=(
                options.inference_mode == "global-unified"
            ),
        )

        # Verify that the default thresholds reproduce
        # the previously frozen DEV valuation backtest.
        if name == "S2_frozen_baseline":
            saved = pd.read_csv(
                saved_path,
                index_col=0,
            )["equity"]

            saved.index = pd.to_datetime(saved.index)

            replay = result.equity.copy()
            replay.index = pd.to_datetime(replay.index)

            if not replay.index.equals(saved.index):
                raise ValueError(
                    "Baseline date index mismatch"
                )

            if not np.allclose(
                replay.to_numpy(),
                saved.to_numpy(),
                rtol=1e-8,
                atol=0.01,
            ):
                max_difference = np.max(
                    np.abs(
                        replay.to_numpy()
                        - saved.to_numpy()
                    )
                )
                raise ValueError(
                    "Frozen baseline did not reproduce. "
                    f"Maximum difference: ${max_difference:.4f}"
                )

            print("Frozen DEV baseline: EXACT MATCH")

        stats = performance(result.equity)

        traded_notional = (
            result.trades["notional"].abs().sum()
            if not result.trades.empty
            else 0.0
        )

        turnover = (
            traded_notional
            / result.equity.mean()
            / (len(result.equity) / 252)
        )

        rows.append({
            "experiment": name,
            "bearish_cutoff": cutoff,
            "max_spread": max_spread,
            "total_return_pct": stats["total_return_pct"],
            "cagr_pct": stats["cagr_pct"],
            "sharpe": stats["sharpe"],
            "max_drawdown_pct": stats["max_drawdown_pct"],
            "volatility_pct": stats["volatility_pct"],
            "turnover_per_year": round(turnover, 2),
            "avg_removed_per_week": (
                round(wrapped.removed / wrapped.calls, 2)
                if wrapped.calls else 0.0
            ),
        })

    summary = pd.DataFrame(rows)

    baseline_return = summary.loc[
        summary["experiment"] == "S2_frozen_baseline",
        "total_return_pct",
    ].iloc[0]

    summary["return_diff_vs_frozen_pp"] = (
        summary["total_return_pct"] - baseline_return
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "sensitivity_dev.csv"

    summary.to_csv(path, index=False)

    print("\n=== DEV PARAMETER SENSITIVITY ===")
    print(summary.round(3).to_string(index=False))

    print(f"\nSaved: {path}")
    print("Frozen model code was not modified.")
    print("Test period was not accessed.")


if __name__ == "__main__":
    main()