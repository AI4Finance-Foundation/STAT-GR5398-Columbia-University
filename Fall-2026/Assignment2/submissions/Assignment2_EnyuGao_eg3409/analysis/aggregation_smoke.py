"""Development-only smoke checks for aggregation and filter aggregation and filtering."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time

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
    filter_candidates,
    valuation_details,
)


MONTHS = (date(2017, 6, 1), date(2019, 7, 1), date(2021, 12, 1))
OUTPUT_DIR = Path("output/filter_checks")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    scores = MonthlyScores()
    rows, summaries = [], []
    final_full_context = None
    for month in MONTHS:
        predictions = scores.get(month)
        as_of = scores.cutoff
        candidates = tuple(sorted(
            (row for row in predictions if row.bucket == "growth_tech"),
            key=lambda row: row.rank_score,
            reverse=True,
        )[:20])
        tickers = [row.symbol for row in candidates]
        view = DataView(bundle, as_of)
        with student_call():
            details = valuation_details(as_of, tickers, view)
            kept = filter_candidates(as_of, candidates, view)
        kept_symbols = {row.symbol for row in kept}
        detail_by_ticker = details.set_index("ticker") if not details.empty else pd.DataFrame()
        for rank, candidate in enumerate(candidates, start=1):
            record = {
                "selection_month": month.isoformat(),
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "selection_rank": rank,
                "ticker": candidate.symbol,
                "valued": candidate.symbol in details["ticker"].values,
                "removed": candidate.symbol not in kept_symbols,
            }
            if candidate.symbol in detail_by_ticker.index:
                record.update(detail_by_ticker.loc[candidate.symbol].to_dict())
            else:
                record["filter_decision"] = "keep_unvalued"
            rows.append(record)
        method_counts = details["method_count"].value_counts() if not details.empty else pd.Series()
        summaries.append(
            {
                "selection_month": month.isoformat(),
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "attempted": len(candidates),
                "valued": len(details),
                "unvalued": len(candidates) - len(details),
                "one_method": int(method_counts.get(1, 0)),
                "two_methods": int(method_counts.get(2, 0)),
                "three_methods": int(method_counts.get(3, 0)),
                "filter_eligible": int(details["filter_eligible"].sum()) if not details.empty else 0,
                "removed_overvalued": len(candidates) - len(kept),
                "kept_for_method_disagreement_or_count": int(
                    details["filter_decision"].eq(
                        "keep_insufficient_or_disagreeing_methods"
                    ).sum()
                ) if not details.empty else 0,
            }
        )
        if month == MONTHS[-1]:
            final_full_context = (as_of, predictions, view)
    details_frame = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(summaries)
    details_frame.to_csv(OUTPUT_DIR / "aggregation_smoke.csv", index=False)
    summary_frame.to_csv(OUTPUT_DIR / "aggregation_smoke_summary.csv", index=False)
    assumptions = {
        "development_only": True,
        "aggregation": "median of available method fair values and method bands",
        "default_filter_rule": FILTER_RULE,
        "overvaluation_threshold": OVERVALUATION_THRESHOLD,
        "minimum_methods_for_filter": MIN_METHODS_FOR_FILTER,
        "maximum_method_spread_for_filter": MAX_METHOD_SPREAD_FOR_FILTER,
        "quantile_removal_fraction": QUANTILE_REMOVAL_FRACTION,
        "quantile_minimum_pool": QUANTILE_MINIMUM_POOL,
        "unvalued_candidates": "kept",
        "single_method_candidates": "valued but kept",
        "large_disagreement_candidates": "valued but kept",
        "validation_or_test_accessed": False,
        "portfolio_backtest_run": False,
    }
    (OUTPUT_DIR / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    as_of, full_candidates, full_view = final_full_context
    start = time.perf_counter()
    with student_call():
        full_kept = filter_candidates(as_of, full_candidates, full_view)
    full_benchmark = {
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "candidate_count": len(full_candidates),
        "growth_tech_count": sum(row.bucket == "growth_tech" for row in full_candidates),
        "kept_count": len(full_kept),
        "removed_count": len(full_candidates) - len(full_kept),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (OUTPUT_DIR / "full_candidate_filter_benchmark.json").write_text(
        json.dumps(full_benchmark, indent=2) + "\n"
    )
    valued = details_frame.loc[details_frame["valued"]]
    summary = {
        "development_only": True,
        "contexts": len(MONTHS),
        "attempted": int(len(details_frame)),
        "valued": int(details_frame["valued"].sum()),
        "removed": int(details_frame["removed"].sum()),
        "all_positive_prices": bool(valued["current_price"].gt(0).all()),
        "all_positive_fair_values": bool(valued["fair_value"].gt(0).all()),
        "all_ranges_ordered": bool(
            valued["low"].le(valued["fair_value"]).all()
            and valued["fair_value"].le(valued["high"]).all()
        ),
        "full_candidate_filter_benchmark": full_benchmark,
        "validation_or_test_accessed": False,
        "portfolio_backtest_run": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# aggregation and filter development-only aggregation smoke

The public valuation output is the median of available standalone fair values;
low and high are the medians of method low/high bands, not a full envelope
across method extremes. Filtering removes a
`growth_tech` candidate only when at least {MIN_METHODS_FOR_FILTER} methods are
available, method spread is at most {MAX_METHOD_SPREAD_FOR_FILTER:.1f}x, and
mispricing is strictly below {OVERVALUATION_THRESHOLD:.0%}. Unvalued,
single-method, and high-disagreement candidates are retained.

```text
{summary_frame.to_string(index=False)}
```

This is an interface and decision-rule smoke check, not valuation evaluation or
a portfolio backtest. No validation/test date is accessed.

`FILTER_RULE="absolute"` is the default. The alternative `"quantile"` rule
uses the same eligible pool, removes its bottom `floor(20%)` by mispricing, and
removes none when fewer than five names are eligible. Full 60-month comparison
outputs are in `filter_rule_removals.csv`, `filter_rule_monthly_summary.csv`,
and `filter_rule_summary.csv`. Across all 60 contexts, the absolute rule removes
187 ticker-dates and the quantile rule removes 60.
"""
    (OUTPUT_DIR / "README.md").write_text(readme)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    run()
