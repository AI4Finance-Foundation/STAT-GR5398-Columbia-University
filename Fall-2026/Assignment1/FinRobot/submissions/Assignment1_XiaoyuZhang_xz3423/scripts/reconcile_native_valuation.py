from pathlib import Path
import sys
import json
import re
import pandas as pd
import numpy as np

ROOT = Path.cwd()
MODULES = ROOT / "finrobot_equity/core/src/modules"
sys.path.insert(0, str(MODULES))

from valuation_engine import ValuationEngine
from sensitivity_analyzer import SensitivityAnalyzer

TICKER = "NVDA"
CSV_PATH = ROOT / f"output/{TICKER}/analysis/financial_metrics_and_forecasts.csv"
OUT = ROOT / "experiments/part2_valuation/native_engine_reconciliation"
OUT.mkdir(parents=True, exist_ok=True)

# Illustrative assumptions, NOT independently verified market inputs.
TARGET_MULTIPLE = 12.0
SHARES = 24_500_000_000
NET_DEBT_AS_SHARE_OF_EV = 0.10

df = pd.read_csv(CSV_PATH)
if "metrics" not in df.columns:
    raise ValueError("Missing metrics column")

years = [
    str(c) for c in df.columns
    if re.fullmatch(r"20\d{2}[AE]", str(c))
]
forecast_years = sorted(y for y in years if y.endswith("E"))
if not forecast_years:
    raise ValueError("No forecast E columns found")

TARGET_YEAR = forecast_years[-1]

def metric(name, year):
    row = df[
        df["metrics"].astype(str).str.strip().str.lower()
        == name.lower()
    ]
    if row.empty:
        raise ValueError(f"Missing metric: {name}")
    value = row.iloc[0][year]
    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "")
    return float(value)

base_ebitda = metric("EBITDA", TARGET_YEAR)
base_revenue = metric("Revenue", TARGET_YEAR)
base_margin = metric("EBITDA Margin", TARGET_YEAR)

if not np.isfinite(base_ebitda) or base_ebitda <= 0:
    raise ValueError("Invalid EBITDA")

# Native valuation engine. The implementation uses the last year
# in the data table when no year is supplied.
engine = ValuationEngine({
    "analysis": df,
    "shares_outstanding": SHARES,
    "current_price": 0
})

native = engine.calculate_ev_ebitda_valuation(
    target_multiple=TARGET_MULTIPLE
)

# Completely independent manual reproduction of the engine formula.
manual_enterprise_value = base_ebitda * TARGET_MULTIPLE
manual_net_debt = manual_enterprise_value * NET_DEBT_AS_SHARE_OF_EV
manual_equity_value = manual_enterprise_value - manual_net_debt
manual_price = manual_equity_value / SHARES

difference = native.target_price - manual_price

if not np.isclose(native.target_price, manual_price, rtol=1e-10):
    raise AssertionError(
        f"Engine vs hand calculation mismatch: {difference}"
    )

print("========== NATIVE ENGINE RECONCILIATION ==========")
print("Company:", TICKER)
print("Selected forecast year:", TARGET_YEAR)
print("Forecast EBITDA (USD B):", round(base_ebitda / 1e9, 4))
print("Target multiple:", TARGET_MULTIPLE)
print("Illustrative shares (B):", SHARES / 1e9)
print("Enterprise value (USD B):", round(manual_enterprise_value / 1e9, 4))
print("Imputed net debt (USD B):", round(manual_net_debt / 1e9, 4))
print("Native engine price:", round(native.target_price, 4))
print("Manual price:", round(manual_price, 4))
print("Difference:", difference)
print("PASS: NATIVE ENGINE RECONCILED")

# Run the actual native sensitivity analyzer.
analyzer = SensitivityAnalyzer(df)
revenue_table = analyzer.analyze_revenue_sensitivity(
    growth_range=(-0.05, 0.05), steps=5
)
margin_table = analyzer.analyze_margin_sensitivity(
    margin_range=(-0.02, 0.02), steps=5
)
combined = analyzer.generate_sensitivity_table(
    revenue_range=(-0.05, 0.05),
    margin_range=(-0.02, 0.02),
    steps=5
)

revenue_table.to_csv(OUT / "native_revenue_sensitivity.csv", index=False)
margin_table.to_csv(OUT / "native_margin_sensitivity.csv", index=False)
combined.to_csv(OUT / "native_combined_ebitda_matrix.csv")

# The native analyzer outputs EBITDA, not share price.
# Convert its exact scenarios to an implied price using the
# same EV/EBITDA valuation rule as the engine.
records = []

for rev_delta in np.linspace(-0.05, 0.05, 5):
    for margin_delta in np.linspace(-0.02, 0.02, 5):
        scenario_revenue = base_revenue * (1 + rev_delta)
        scenario_margin = base_margin / 100 + margin_delta
        scenario_ebitda = scenario_revenue * scenario_margin

        scenario_ev = scenario_ebitda * TARGET_MULTIPLE
        scenario_equity = scenario_ev * (
            1 - NET_DEBT_AS_SHARE_OF_EV
        )
        scenario_price = scenario_equity / SHARES

        records.append({
            "ticker": TICKER,
            "year": TARGET_YEAR,
            "revenue_delta_pct": round(rev_delta * 100, 2),
            "margin_delta_pp": round(margin_delta * 100, 2),
            "scenario_revenue_usd_b": scenario_revenue / 1e9,
            "scenario_ebitda_usd_b": scenario_ebitda / 1e9,
            "implied_price_usd": scenario_price,
            "price_change_vs_engine_pct": (
                scenario_price / native.target_price - 1
            ) * 100,
        })

scenario_df = pd.DataFrame(records)
scenario_df.to_csv(OUT / "implied_target_price_sensitivity.csv", index=False)

base_scenario = scenario_df[
    (scenario_df["revenue_delta_pct"] == 0)
    & (scenario_df["margin_delta_pp"] == 0)
].iloc[0]

# Recomputed EBITDA may differ from the CSV EBITDA because
# the CSV EBITDA Margin could be rounded.
rounding_gap = (
    base_scenario["implied_price_usd"] - native.target_price
)

summary = {
    "ticker": TICKER,
    "forecast_year": TARGET_YEAR,
    "method": "Native ValuationEngine EV/EBITDA",
    "shares_outstanding": SHARES,
    "shares_source": "Illustrative, not verified market data",
    "target_multiple": TARGET_MULTIPLE,
    "net_debt_pct_enterprise_value": NET_DEBT_AS_SHARE_OF_EV,
    "ebitda_usd": base_ebitda,
    "revenue_usd": base_revenue,
    "ebitda_margin_pct": base_margin,
    "native_target_price": native.target_price,
    "manual_target_price": manual_price,
    "engine_manual_difference": difference,
    "sensitivity_price_min": scenario_df["implied_price_usd"].min(),
    "sensitivity_price_max": scenario_df["implied_price_usd"].max(),
    "sensitivity_baseline_rounding_gap": rounding_gap,
    "caveat": (
        "Scenario price sensitivity is derived from the native "
        "SensitivityAnalyzer EBITDA scenarios using the engine's "
        "EV/EBITDA pricing formula. These are illustrative prices."
    ),
}

(OUT / "reconciliation_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n========== SENSITIVITY ANALYSIS ==========")
print("Revenue scenarios:", len(revenue_table))
print("Margin scenarios:", len(margin_table))
print("Joint scenarios:", len(scenario_df))
print("Minimum implied price:", round(summary["sensitivity_price_min"], 4))
print("Maximum implied price:", round(summary["sensitivity_price_max"], 4))
print("Baseline price rounding gap:", round(rounding_gap, 4))
print("Outputs:", OUT)
print("PART 2B NATIVE ENGINE AUDIT COMPLETE")
