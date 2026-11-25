"""Compare weekly-beta pairing conventions on development contexts only."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd

from selection.scores import MonthlyScores
from shared.data_view import DataBundle, DataView


OUTPUT_DIR = Path("output/model_diagnostics")
MIN_PAIRS = 52


def _weekly_counts(view: DataView, ticker: str) -> tuple[int, int, int]:
    prices = view.close([ticker, "SPY"], lookback=252 * 5)

    daily_paired = prices[[ticker, "SPY"]].dropna()
    paired_first = (
        daily_paired.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    )

    stock = pd.to_numeric(prices[ticker], errors="coerce").dropna()
    market = pd.to_numeric(prices["SPY"], errors="coerce").dropna()
    stock_returns = stock.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    market_returns = market.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    independent = pd.concat(
        [stock_returns.rename(ticker), market_returns.rename("SPY")],
        axis=1,
        join="inner",
    ).dropna()
    return int(len(paired_first)), int(len(independent)), int(stock.notna().sum())


def run() -> None:
    bundle = DataBundle.load()
    scores = MonthlyScores()
    rows = []
    for year in range(2017, 2022):
        for month in range(1, 13):
            selection_month = date(year, month, 1)
            predictions = scores.get(selection_month)
            as_of = scores.cutoff
            candidates = sorted(
                (row for row in predictions if row.bucket == "growth_tech"),
                key=lambda row: row.rank_score,
                reverse=True,
            )[:20]
            view = DataView(bundle, as_of)
            for candidate in candidates:
                paired_first, independent, daily_closes = _weekly_counts(
                    view, candidate.symbol
                )
                rows.append(
                    {
                        "selection_month": selection_month.isoformat(),
                        "as_of": pd.Timestamp(as_of).date().isoformat(),
                        "ticker": candidate.symbol,
                        "daily_paired_weekly_returns": paired_first,
                        "independent_weekly_returns": independent,
                        "paired_count_change": independent - paired_first,
                        "daily_paired_eligible": paired_first >= MIN_PAIRS,
                        "independent_eligible": independent >= MIN_PAIRS,
                        "eligibility_changed": (paired_first >= MIN_PAIRS)
                        != (independent >= MIN_PAIRS),
                        "nonmissing_daily_closes": daily_closes,
                    }
                )
    comparison = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "beta_resampling_comparison.csv", index=False)

    end_view = DataView(bundle, "2021-12-31")
    members = sorted(end_view.members())
    end_prices = end_view.close(members, lookback=252 * 5)
    zero_price = {
        ticker for ticker in members
        if pd.to_numeric(end_prices[ticker], errors="coerce").notna().sum() == 0
    }
    audit_frame = pd.read_csv("output/data_audit/beta_price_diagnostics_2021-12-31.csv")
    audited_group_a = set(audit_frame.loc[audit_frame["group"].eq("A"), "ticker"].astype(str))
    group_a_rows = []
    for ticker in sorted(zero_price | audited_group_a):
        paired_first, independent, daily_closes = _weekly_counts(end_view, ticker)
        group_a_rows.append(
            {
                "ticker": ticker,
                "audited_group_a": ticker in audited_group_a,
                "zero_price_recomputed": ticker in zero_price,
                "daily_closes": daily_closes,
                "daily_paired_weekly_returns": paired_first,
                "independent_weekly_returns": independent,
            }
        )
    pd.DataFrame(group_a_rows).to_csv(
        OUTPUT_DIR / "beta_group_a_convention_check.csv", index=False
    )

    changed = comparison["paired_count_change"].ne(0)
    crossed = comparison["eligibility_changed"]
    summary = {
        "development_only": True,
        "monthly_contexts": 60,
        "attempted_ticker_dates": int(len(comparison)),
        "ticker_dates_with_changed_pair_counts": int(changed.sum()),
        "maximum_absolute_pair_count_change": int(
            comparison["paired_count_change"].abs().max()
        ),
        "ticker_dates_crossing_52_pair_boundary": int(crossed.sum()),
        "audited_group_a_count": len(audited_group_a),
        "recomputed_zero_price_count": len(zero_price),
        "group_a_sets_identical": audited_group_a == zero_price,
        "group_a_zero_under_both_conventions": bool(
            all(
                row["daily_paired_weekly_returns"] == 0
                and row["independent_weekly_returns"] == 0
                for row in group_a_rows
            )
        ),
        "implementation_decision": (
            "keep_independent" if not crossed.any() else "use_course_daily_paired"
        ),
        "validation_or_test_accessed": False,
    }
    (OUTPUT_DIR / "beta_resampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
