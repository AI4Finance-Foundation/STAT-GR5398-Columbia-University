
"""Replacement attribution for the current valuation model.

Replays the same A/B configurations as run_portfolio.py.
Uses weekly target holdings, not executed trades.
Does not access the test period.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.backtester import BacktestConfig, run_backtest
from portfolio.hooks import FilteredPredictions
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation.student_valuation import filter_candidates


OUTPUT = Path("output")
CONFIG = "selection/configs/growth.json"


def run_pair(period, options, params, prices, universe, bundle):
    start, end = resolve(period, None, None)
    cfg = BacktestConfig(start=start, end=end)

    selection = run_backtest(
        prices,
        MonthlyScores(),
        params,
        universe,
        cfg,
        global_unified=options.inference_mode == "global-unified",
    )

    filtered = FilteredPredictions(
        MonthlyScores(), filter_candidates, bundle
    )

    valuation = run_backtest(
        prices,
        filtered,
        params,
        universe,
        cfg,
        global_unified=options.inference_mode == "global-unified",
    )

    return selection, valuation



def verify_equity(period, name, calculated):
    path = OUTPUT / f"{period}_{name}" / "equity.csv"

    saved = pd.read_csv(path, index_col=0).iloc[:, 0]
    saved.index = pd.to_datetime(saved.index)

    actual = calculated.equity.copy()
    actual.index = pd.to_datetime(actual.index)

    if not actual.index.equals(saved.index):
        raise ValueError(f"{path}: dates do not match")

    diff = actual - saved

    max_abs_diff = float(diff.abs().max())
    max_relative_diff = float(
        (diff.abs() / saved.abs()).max() * 100
    )

    saved_return = (
        saved.iloc[-1] / saved.iloc[0] - 1
    ) * 100

    actual_return = (
        actual.iloc[-1] / actual.iloc[0] - 1
    ) * 100

    print(f"\n=== EQUITY CHECK: {period} / {name} ===")
    print(f"Maximum dollar difference: ${max_abs_diff:.4f}")
    print(f"Maximum relative difference: {max_relative_diff:.4f}%")
    print(f"Saved total return: {saved_return:.4f}%")
    print(f"Replay total return: {actual_return:.4f}%")
    print(
        f"Final NAV difference: "
        f"${float(diff.iloc[-1]):.4f}"
    )

    if np.allclose(
        actual.to_numpy(),
        saved.to_numpy(),
        rtol=1e-8,
        atol=0.01,
    ):
        print("STATUS: EXACT MATCH")
    else:
        print("STATUS: REPLAY MISMATCH — INVESTIGATE")


def get_weights(record):
    return {
        symbol: float(weight)
        for symbol, weight in record["weights"].items()
        if float(weight) > 0
    }


def analyze(period, selection, valuation, prices):
    a = {
        pd.Timestamp(x["date"]): x
        for x in selection.weekly_targets
    }

    b = {
        pd.Timestamp(x["date"]): x
        for x in valuation.weekly_targets
    }

    if set(a) != set(b):
        raise ValueError("Weekly rebalance dates do not match")

    dates = sorted(a)
    rows = []
    skipped = 0

    for i in range(len(dates) - 1):
        day = dates[i]
        next_day = dates[i + 1]

        wa = get_weights(a[day])
        wb = get_weights(b[day])

        removed = sorted(set(wa) - set(wb))
        added = sorted(set(wb) - set(wa))

        if not removed and not added:
            continue

        # Compare target substitutions only when their
        # stock counts are balanced.
        if not removed or len(removed) != len(added):
            skipped += 1
            continue

        symbols = removed + added

        start_px = prices.open.loc[day].reindex(symbols)
        end_px = prices.open.loc[next_day].reindex(symbols)

        valid = (
            np.isfinite(start_px.to_numpy()).all()
            and np.isfinite(end_px.to_numpy()).all()
            and (start_px.to_numpy() > 0).all()
            and (end_px.to_numpy() > 0).all()
        )

        if not valid:
            skipped += 1
            continue

        weekly_return = end_px / start_px - 1

        removed_return = float(
            weekly_return.loc[removed].mean()
        )
        added_return = float(
            weekly_return.loc[added].mean()
        )

        rows.append({
            "date": day.date(),
            "next_date": next_day.date(),
            "removed": ",".join(removed),
            "added": ",".join(added),
            "n_removed": len(removed),
            "n_added": len(added),
            "removed_return": removed_return,
            "added_return": added_return,
            "replacement_advantage": (
                added_return - removed_return
            ),
        })

    df = pd.DataFrame(rows)

    path = OUTPUT / f"replacement_attribution_{period}_current.csv"
    df.to_csv(path, index=False)

    print(f"\n=== {period.upper()} REPLACEMENT ATTRIBUTION ===")
    print("Comparable weeks:", len(df))
    print("Skipped weeks:", skipped)

    if df.empty:
        print("No comparable replacement weeks")
        return

    df["year"] = pd.to_datetime(df["date"]).dt.year

    annual = (
        df.groupby("year")
        .agg(
            replacement_weeks=("date", "count"),
            removed_slots=("n_removed", "sum"),
            removed_return=("removed_return", "mean"),
            replacement_return=("added_return", "mean"),
            mean_advantage=("replacement_advantage", "mean"),
            median_advantage=("replacement_advantage", "median"),
            better_fraction=(
                "replacement_advantage",
                lambda s: (s > 0).mean(),
            ),
        )
        .reset_index()
    )

    percentage_cols = [
        "removed_return",
        "replacement_return",
        "mean_advantage",
        "median_advantage",
        "better_fraction",
    ]

    annual[percentage_cols] *= 100

    print("\n=== ANNUAL SUMMARY ===")
    print(annual.round(2).to_string(index=False))

    annual_path = (
        OUTPUT / f"annual_replacement_attribution_{period}_current.csv"
    )
    annual.to_csv(annual_path, index=False)

    print(f"\nSaved: {path}")
    print(f"Saved: {annual_path}")


def main():
    _, options, params, prices, _, universe = load(CONFIG)
    bundle = DataBundle.load(prices=prices)

    for period in ["dev", "val"]:
        print(f"\nReplaying {period.upper()} A and B...")

        selection, valuation = run_pair(
            period, options, params, prices, universe, bundle
        )

        verify_equity(period, "selection", selection)
        verify_equity(period, "valuation", valuation)

        analyze(period, selection, valuation, prices)


if __name__ == "__main__":
    main()