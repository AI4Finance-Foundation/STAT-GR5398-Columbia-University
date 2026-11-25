"""Development-only smoke checks for the three standalone valuation model models."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from selection.scores import MonthlyScores
from shared.data_view import DataBundle, DataView
from valuation.student_valuation import (
    DCF_CREDIT_SPREAD,
    DCF_EQUITY_RISK_PREMIUM,
    DCF_NWC_PCT_REVENUE,
    DCF_PROJECTION_YEARS,
    DCF_TERMINAL_GROWTH,
    EV_EBITDA_MULTIPLE_MAX,
    LEGACY_DCF_CONFIG,
    MIN_HISTORICAL_MULTIPLES,
    MIN_PEER_COUNT,
    PE_MULTIPLE_MAX,
    dcf_valuation,
    historical_ev_ebitda_valuation,
    peer_pe_valuation,
)


MONTHS = (date(2017, 6, 1), date(2019, 7, 1), date(2021, 12, 1))


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    scores = MonthlyScores()
    all_rows, coverage = [], []
    for month in MONTHS:
        predictions = scores.get(month)
        as_of = scores.cutoff
        growth = sorted(
            (row for row in predictions if row.bucket == "growth_tech"),
            key=lambda row: row.rank_score,
            reverse=True,
        )[:20]
        tickers = [row.symbol for row in growth]
        ranks = {row.symbol: i + 1 for i, row in enumerate(growth)}
        view = DataView(bundle, as_of)
        methods = {
            "dcf": dcf_valuation(as_of, tickers, view, config=LEGACY_DCF_CONFIG),
            "historical_ev_ebitda": historical_ev_ebitda_valuation(as_of, tickers, view),
            "peer_pe": peer_pe_valuation(as_of, tickers, view),
        }
        for name, frame in methods.items():
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "selection_month", month.isoformat())
            frame.insert(1, "as_of", as_of.isoformat())
            frame.insert(2, "selection_rank", frame["ticker"].map(ranks))
            all_rows.append(frame)
        coverage.append(
            {
                "selection_month": month.isoformat(),
                "as_of": as_of.isoformat(),
                "top_growth_candidates": len(tickers),
                "dcf_valid": len(methods["dcf"]),
                "historical_ev_ebitda_valid": len(methods["historical_ev_ebitda"]),
                "peer_pe_valid": len(methods["peer_pe"]),
            }
        )

    combined = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    combined.to_csv(output_dir / "model_smoke_estimates.csv", index=False)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(output_dir / "model_smoke_coverage.csv", index=False)
    assumptions = {
        "development_only": True,
        "selection_months": [month.isoformat() for month in MONTHS],
        "aggregation_or_filtering_in_this_analysis": False,
        "aggregation_implemented_after_original_model_evidence": True,
        "dcf": {
            "configuration": "legacy",
            "projection_years": DCF_PROJECTION_YEARS,
            "equity_risk_premium": DCF_EQUITY_RISK_PREMIUM,
            "terminal_growth": DCF_TERMINAL_GROWTH,
            "credit_spread": DCF_CREDIT_SPREAD,
            "nwc_pct_revenue": DCF_NWC_PCT_REVENUE,
        },
        "historical_ev_ebitda": {
            "maximum_multiple": EV_EBITDA_MULTIPLE_MAX,
            "minimum_observations": MIN_HISTORICAL_MULTIPLES,
            "window_years": 5,
        },
        "peer_pe": {"maximum_multiple": PE_MULTIPLE_MAX, "minimum_peers": MIN_PEER_COUNT},
    }
    (output_dir / "model_assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    method_summary = []
    if not combined.empty:
        combined["fair_to_price"] = combined["fair_value"] / combined["current_price"]
        for method, group in combined.groupby("method", sort=True):
            method_summary.append(
                {
                    "method": method,
                    "valid_estimates": int(len(group)),
                    "fair_to_price_min": float(group["fair_to_price"].min()),
                    "fair_to_price_median": float(group["fair_to_price"].median()),
                    "fair_to_price_max": float(group["fair_to_price"].max()),
                }
            )
    summary = {
        "development_only": True,
        "as_of_min": coverage_frame["as_of"].min(),
        "as_of_max": coverage_frame["as_of"].max(),
        "coverage": coverage,
        "method_summary": method_summary,
        "invariants": {
            "all_current_prices_positive": bool((combined["current_price"] > 0).all()),
            "all_fair_values_positive": bool((combined["fair_value"] > 0).all()),
            "all_ranges_ordered": bool(
                ((combined["low"] <= combined["fair_value"])
                 & (combined["fair_value"] <= combined["high"])).all()
            ),
        },
        "aggregation_not_exercised_by_this_analysis": True,
        "portfolio_backtest_run": False,
        "validation_or_test_accessed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    readme = f"""# valuation model development-only model smoke outputs

Generated by `python -m analysis.model_smoke` using three development-period
decision dates only ({summary['as_of_min']} through {summary['as_of_max']}). No
validation/test date or portfolio backtest is used. This script explicitly uses
`LEGACY_DCF_CONFIG`; the locked configuration is analyzed separately by the
hardening entry point.

## Files

- `model_smoke_estimates.csv`: standalone method estimates and diagnostics.
- `model_smoke_coverage.csv`: valid-method counts among each month's top 20
  `growth_tech` selection candidates.
- `model_assumptions.json`: fixed valuation model model constants.
- `summary.json`: coverage, fair-value dispersion, and invariant checks.
- `tax_fallback_dcf_sensitivity.csv`: statutory-tax versus hypothetical 0%
  DCF comparison for the 40 affected data audit inputs.
- `tax_fallback_dcf_sensitivity_summary.json`: comparison counts and direction.
- `parameter_sensitivity*.csv`: eight legacy/locked growth-fade-beta combinations.
- `erp_sensitivity*.csv`: locked DCF at 4%, 5%, and 6% ERP.
- `dcf_exclusion*.csv`: one locked status per attempted ticker and aggregates.
- `ev_ebitda_window_comparison.csv`: controlled three- versus five-year maximum lookback.
- `capex_sign_audit.csv`: source sign evidence and latest-four sequence classifications.
- `peer_pe_benchmark.csv`: component and repeated complete-call timings.
- `hardening_summary.json`: consolidated development-only hardening results.
- `beta_resampling_comparison.csv`: daily-pair-first versus independent weekly counts.
- `beta_group_a_convention_check.csv`: the 191 zero-price members under both conventions.
- `beta_resampling_summary.json`: equivalence result and implementation decision.
- `engine_rejection_subcauses.csv`: all 126 locked engine failures with raw messages.
- `engine_rejection_subcause_summary.csv`: overall and selection-year sub-cause counts.
- `original_*`: preserved evidence from before the hardening changes.

## Coverage

```text
{coverage_frame.to_string(index=False)}
```

The 2017 sample has no DCF estimate because the 52-week paired beta requirement is
not yet met and no EV/EBITDA estimate because eight point-in-time historical
multiples are not yet available. These are method-specific ramp limitations, not
performance evidence. The DCF fair-value dispersion is deliberately left untuned
for the later development/validation optimization.

This standalone-method smoke does not call the aggregation and filter aggregation or filter.
Those interfaces are now implemented and documented separately in
`aggregation and filter notes`.

Reproduce with:

```bash
python -m analysis.model_smoke
python -m analysis.tax_sensitivity
python -m analysis.model_hardening
```
"""
    (output_dir / "README.md").write_text(readme)
    print(coverage_frame.to_string(index=False))
    print(f"Wrote valuation model smoke outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/model_diagnostics"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
