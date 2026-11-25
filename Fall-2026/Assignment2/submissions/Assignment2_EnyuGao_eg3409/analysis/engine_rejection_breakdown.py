"""Classify locked DCF engine rejections without discarding engine messages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT = Path("output/model_diagnostics/dcf_exclusion_reasons.csv")
OUTPUT_DIR = Path("output/model_diagnostics")
SUBCAUSES = (
    "unstable_gordon_spread",
    "nonpositive_equity_value",
    "invalid_intermediate_input",
    "other",
)


def classify_engine_rejection(message: str) -> str:
    text = str(message)
    if "Equity value is non-positive" in text:
        return "nonpositive_equity_value"
    if "Steady-state terminal FCF is non-positive" in text:
        return "invalid_intermediate_input"
    if (
        "WACC−terminal growth spread" in text
        or "Terminal growth rate" in text and "must be less than WACC" in text
    ):
        return "unstable_gordon_spread"
    return "other"


def run() -> None:
    frame = pd.read_csv(INPUT)
    rejected = frame.loc[frame["primary_status"].eq("engine_rejected")].copy()
    rejected = rejected.rename(columns={"secondary_reason": "engine_message"})
    rejected["subcause"] = rejected["engine_message"].map(classify_engine_rejection)
    if len(rejected) != 126:
        raise AssertionError(f"expected 126 locked engine rejections, found {len(rejected)}")
    if not set(rejected["subcause"]) <= set(SUBCAUSES):
        raise AssertionError("unexpected engine rejection sub-cause")
    rejected.to_csv(OUTPUT_DIR / "engine_rejection_subcauses.csv", index=False)

    rows = []
    for scope, period, group in [
        ("overall", "ALL", rejected),
        *[("selection_year", str(year), values) for year, values in rejected.groupby("year")],
    ]:
        counts = group["subcause"].value_counts()
        for subcause in SUBCAUSES:
            rows.append(
                {
                    "scope": scope,
                    "period": period,
                    "subcause": subcause,
                    "count": int(counts.get(subcause, 0)),
                    "scope_total": int(len(group)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "engine_rejection_subcause_summary.csv", index=False)
    overall = summary.loc[summary["scope"].eq("overall")]
    if int(overall["count"].sum()) != 126:
        raise AssertionError("engine rejection sub-causes do not sum to 126")
    metadata = {
        "development_only": True,
        "engine_rejections": 126,
        "subcause_counts": dict(zip(overall["subcause"], overall["count"])),
        "subcauses_sum_to_engine_rejections": int(overall["count"].sum()) == 126,
        "engine_messages_retained": True,
        "validation_or_test_accessed": False,
    }
    (OUTPUT_DIR / "engine_rejection_subcause_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
