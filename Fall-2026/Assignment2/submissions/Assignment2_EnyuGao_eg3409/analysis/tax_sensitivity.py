"""Development-only DCF sensitivity for the data audit tax-fallback finding."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from shared.data_view import DataBundle, DataView
from valuation.finrobot import calculate_dcf as engine_calculate_dcf
import valuation.student_valuation as student


AS_OF = "2021-12-31"
INPUT_CHANGES = Path("output/data_audit/tax_assumption_changes_2021-12-31.csv")
OUTPUT_DIR = Path("output/model_diagnostics")


def _hypothetical_zero_tax(raw_rate, _fiscal_period_end) -> float:
    """Keep a positive observation, but use 0% for a nonpositive/missing one."""
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError):
        return 0.0
    return float(min(rate, 0.35)) if np.isfinite(rate) and rate > 0 else 0.0


def run() -> None:
    affected = pd.read_csv(INPUT_CHANGES)
    tickers = affected["ticker"].astype(str).tolist()
    view = DataView(DataBundle.load(), AS_OF)

    statutory = student.dcf_valuation(
        AS_OF, tickers, view, config=student.LEGACY_DCF_CONFIG
    )
    with patch.object(student, "tax_rate_for_dcf", _hypothetical_zero_tax):
        zero_native = student.dcf_valuation(
            AS_OF, tickers, view, config=student.LEGACY_DCF_CONFIG
        )

    columns = ["ticker", "fair_value", "wacc", "tax_rate"]
    comparison = affected.merge(
        statutory[columns].rename(
            columns={
                "fair_value": "statutory_fair_value",
                "wacc": "statutory_wacc",
                "tax_rate": "statutory_model_tax_rate",
            }
        ),
        on="ticker",
        how="left",
    ).merge(
        zero_native[columns].rename(
            columns={
                "fair_value": "zero_native_fair_value",
                "wacc": "zero_native_wacc",
                "tax_rate": "zero_model_tax_rate",
            }
        ),
        on="ticker",
        how="left",
    )
    comparison["native_fair_value_change"] = (
        comparison["statutory_fair_value"] - comparison["zero_native_fair_value"]
    )
    comparison["native_fair_value_change_pct"] = (
        comparison["statutory_fair_value"] / comparison["zero_native_fair_value"] - 1
    )

    # The model-native comparison also changes the after-tax cost of debt in
    # WACC. This second comparison fixes WACC at the statutory case to isolate
    # the operating cash-flow effect of the higher tax input.
    fixed_wacc_zero = {}
    for row in statutory.itertuples(index=False):
        def fixed_wacc_calculate_dcf(
            inputs, wacc_override=None, tg_override=None, mid_year=False, *, _wacc=row.wacc
        ):
            return engine_calculate_dcf(
                inputs,
                wacc_override=float(_wacc),
                tg_override=tg_override,
                mid_year=mid_year,
            )

        with (
            patch.object(student, "tax_rate_for_dcf", _hypothetical_zero_tax),
            patch.object(student, "calculate_dcf", fixed_wacc_calculate_dcf),
        ):
            result = student.dcf_valuation(
                AS_OF, [row.ticker], view, config=student.LEGACY_DCF_CONFIG
            )
        if not result.empty:
            fixed_wacc_zero[row.ticker] = float(result.iloc[0]["fair_value"])

    comparison["zero_fixed_wacc_fair_value"] = comparison["ticker"].map(fixed_wacc_zero)
    comparison["fixed_wacc_fair_value_change"] = (
        comparison["statutory_fair_value"] - comparison["zero_fixed_wacc_fair_value"]
    )
    comparison["comparison_status"] = np.select(
        [
            comparison["statutory_fair_value"].notna()
            & comparison["zero_native_fair_value"].notna(),
            comparison["statutory_fair_value"].isna()
            & comparison["zero_native_fair_value"].notna(),
            comparison["statutory_fair_value"].notna()
            & comparison["zero_native_fair_value"].isna(),
        ],
        ["paired", "zero_only", "statutory_only"],
        default="ineligible_both",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "tax_fallback_dcf_sensitivity.csv", index=False)
    paired = comparison.loc[comparison["comparison_status"].eq("paired")]
    fixed = paired["fixed_wacc_fair_value_change"].dropna()
    summary = {
        "development_only": True,
        "configuration": "legacy",
        "as_of": AS_OF,
        "affected_ticker_inputs": int(len(comparison)),
        "comparison_status_counts": comparison["comparison_status"].value_counts().to_dict(),
        "model_native_paired": int(len(paired)),
        "model_native_lower_with_statutory_tax": int(
            paired["native_fair_value_change"].lt(0).sum()
        ),
        "model_native_higher_with_statutory_tax": int(
            paired["native_fair_value_change"].gt(0).sum()
        ),
        "fixed_wacc_paired": int(len(fixed)),
        "fixed_wacc_lower_with_statutory_tax": int(fixed.lt(0).sum()),
        "explanation": (
            "The fixed-WACC comparison isolates operating tax: every comparable fair value "
            "falls. The model-native comparison also gives debt a larger tax shield, lowering "
            "WACC; that offset reverses the net direction for two paired tickers."
        ),
        "validation_or_test_accessed": False,
    }
    (OUTPUT_DIR / "tax_fallback_dcf_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
