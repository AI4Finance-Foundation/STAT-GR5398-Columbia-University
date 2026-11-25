"""Compare aggregation and filter absolute and quantile filters on development contexts."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selection.scores import MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from valuation.student_valuation import (
    FILTER_RULE,
    MAX_METHOD_SPREAD_FOR_FILTER,
    MIN_METHODS_FOR_FILTER,
    OVERVALUATION_THRESHOLD,
    QUANTILE_MINIMUM_POOL,
    QUANTILE_REMOVAL_FRACTION,
    filter_removals,
    valuation_details,
)


OUTPUT_DIR = Path("output/filter_checks")
RULES = ("absolute", "quantile")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    scores = MonthlyScores()
    ticker_rows, monthly_rows = [], []
    for year in range(2017, 2022):
        for month in range(1, 13):
            selection_month = date(year, month, 1)
            predictions = scores.get(selection_month)
            as_of = scores.cutoff
            candidates = tuple(sorted(
                (row for row in predictions if row.bucket == "growth_tech"),
                key=lambda row: row.rank_score,
                reverse=True,
            )[:20])
            tickers = [row.symbol for row in candidates]
            with student_call():
                details = valuation_details(as_of, tickers, DataView(bundle, as_of))
            detail_index = details.set_index("ticker") if not details.empty else pd.DataFrame()
            eligible_count = int(details["filter_eligible"].sum()) if not details.empty else 0
            for rule in RULES:
                removed = filter_removals(details, rule=rule)
                expected_quantile = (
                    int(np.floor(QUANTILE_REMOVAL_FRACTION * eligible_count))
                    if eligible_count >= QUANTILE_MINIMUM_POOL else 0
                )
                if rule == "quantile":
                    assert len(removed) == expected_quantile
                monthly_rows.append(
                    {
                        "rule": rule,
                        "selection_month": selection_month.isoformat(),
                        "selection_year": year,
                        "as_of": pd.Timestamp(as_of).date().isoformat(),
                        "attempted": len(candidates),
                        "valued": len(details),
                        "filter_eligible_pool": eligible_count,
                        "removed_count": len(removed),
                        "removed_tickers": ",".join(
                            ticker for ticker in tickers if ticker in removed
                        ),
                    }
                )
                for rank, ticker in enumerate(tickers, start=1):
                    record = {
                        "rule": rule,
                        "selection_month": selection_month.isoformat(),
                        "selection_year": year,
                        "as_of": pd.Timestamp(as_of).date().isoformat(),
                        "selection_rank": rank,
                        "ticker": ticker,
                        "valued": ticker in details["ticker"].values,
                        "removed": ticker in removed,
                    }
                    if ticker in detail_index.index:
                        row = detail_index.loc[ticker]
                        record.update(
                            {
                                "fair_value": row["fair_value"],
                                "current_price": row["current_price"],
                                "mispricing": row["mispricing"],
                                "method_count": row["method_count"],
                                "method_spread": row["method_spread"],
                                "filter_eligible": row["filter_eligible"],
                            }
                        )
                    ticker_rows.append(record)
    removals = pd.DataFrame(ticker_rows)
    monthly = pd.DataFrame(monthly_rows)
    removals.to_csv(OUTPUT_DIR / "filter_rule_removals.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "filter_rule_monthly_summary.csv", index=False)

    summary_rows = []
    for rule, rule_frame in monthly.groupby("rule", sort=True):
        for period, group in [("ALL", rule_frame), *list(rule_frame.groupby("selection_year"))]:
            summary_rows.append(
                {
                    "rule": rule,
                    "selection_year": period,
                    "months": len(group),
                    "attempted": int(group["attempted"].sum()),
                    "valued": int(group["valued"].sum()),
                    "eligible_pool_total": int(group["filter_eligible_pool"].sum()),
                    "removed": int(group["removed_count"].sum()),
                    "removed_pct_attempted": float(
                        group["removed_count"].sum() / group["attempted"].sum()
                    ),
                    "months_with_removals": int(group["removed_count"].gt(0).sum()),
                    "mean_removed_per_month": float(group["removed_count"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "filter_rule_summary.csv", index=False)
    metadata = {
        "development_only": True,
        "monthly_contexts": 60,
        "attempted_ticker_dates_per_rule": 1200,
        "default_rule": FILTER_RULE,
        "absolute_threshold": OVERVALUATION_THRESHOLD,
        "minimum_methods": MIN_METHODS_FOR_FILTER,
        "maximum_method_spread": MAX_METHOD_SPREAD_FOR_FILTER,
        "quantile_fraction": QUANTILE_REMOVAL_FRACTION,
        "quantile_minimum_pool": QUANTILE_MINIMUM_POOL,
        "validation_or_test_accessed": False,
        "portfolio_backtest_run": False,
    }
    (OUTPUT_DIR / "filter_rule_assumptions.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.loc[summary["selection_year"].astype(str).eq("ALL")].to_string(index=False))


if __name__ == "__main__":
    run()
