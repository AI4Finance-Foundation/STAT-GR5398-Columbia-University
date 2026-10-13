import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("output/sec_company_facts")
OUT = Path("experiments/part2_valuation")
OUT.mkdir(parents=True, exist_ok=True)

GROWTH = [0.05, 0.06, 0.04, 0.04, 0.04]
DISCOUNT_RATES = [0.09, 0.10, 0.11, 0.12, 0.13]
TERMINAL_GROWTH = [0.02, 0.03, 0.04]


def calculate_dcf(base_fcf, discount, terminal_growth):
    if discount <= terminal_growth:
        raise ValueError("Discount rate must exceed terminal growth.")

    cash_flows = []
    fcf = float(base_fcf)

    for growth in GROWTH:
        fcf *= 1 + growth
        cash_flows.append(fcf)

    pv_explicit = sum(
        value / (1 + discount) ** year
        for year, value in enumerate(cash_flows, 1)
    )

    terminal_value = (
        cash_flows[-1] * (1 + terminal_growth)
        / (discount - terminal_growth)
    )

    pv_terminal = terminal_value / (1 + discount) ** 5

    return {
        "forecast_fcf": cash_flows,
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "total": pv_explicit + pv_terminal,
    }


for ticker in ["AVGO", "QCOM"]:
    df = pd.read_csv(
        ROOT / ticker / "financials_normalized_usd.csv"
    )

    latest = df.loc[df["fiscal_year"] == 2025].iloc[0]
    fcf = float(latest["FCF Proxy"])

    assert np.isfinite(fcf) and fcf > 0

    base = calculate_dcf(fcf, 0.11, 0.03)

    forecast = pd.DataFrame({
        "year": [2026, 2027, 2028, 2029, 2030],
        "growth_assumption": GROWTH,
        "fcf_proxy_usd": base["forecast_fcf"],
    })

    forecast.to_csv(
        OUT / f"{ticker}_dcf_forecast.csv",
        index=False
    )

    sensitivity_rows = []

    for discount in DISCOUNT_RATES:
        record = {
            "discount_rate_pct": discount * 100
        }

        for growth in TERMINAL_GROWTH:
            result = calculate_dcf(
                fcf, discount, growth
            )
            record[f"terminal_growth_{growth*100:.0f}pct"] = (
                result["total"] / 1e9
            )

        sensitivity_rows.append(record)

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUT / f"{ticker}_dcf_sensitivity.csv",
        index=False
    )

    assumptions = {
        "ticker": ticker,
        "method": "Five-year illustrative FCF-proxy DCF",
        "base_fiscal_year": 2025,
        "base_fcf_proxy_usd": fcf,
        "growth_assumptions": dict(
            zip(range(2026, 2031), GROWTH)
        ),
        "discount_rate_base": 0.11,
        "terminal_growth_base": 0.03,
        "discount_rate_sensitivity": DISCOUNT_RATES,
        "terminal_growth_sensitivity": TERMINAL_GROWTH,
        "present_value_explicit_usd": base["pv_explicit"],
        "present_value_terminal_usd": base["pv_terminal"],
        "discounted_cash_flow_proxy_usd": base["total"],
        "limitations": [
            "OCF minus CapEx is an FCF proxy, not verified FCFE or FCFF.",
            "The assumed discount rate is not a calibrated WACC.",
            "This is not a stock price target or formal equity valuation.",
            "Capital expenditure classifications require 10-K review.",
            "No adjustment for net debt, excess cash, or diluted shares.",
            "Forecast growth and terminal assumptions are illustrative.",
        ],
    }

    (OUT / f"{ticker}_dcf_assumptions.json").write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8"
    )

    print(f"\n========== {ticker} DCF ==========")
    print("2025 FCF Proxy:", f"${fcf/1e9:.3f}B")
    print("Forecast:")
    print(
        forecast.assign(
            fcf_proxy_billions=
                forecast["fcf_proxy_usd"] / 1e9
        )[
            ["year", "growth_assumption", "fcf_proxy_billions"]
        ].round(3).to_string(index=False)
    )

    print(
        "\nBase discounted cash-flow proxy:",
        f"${base['total']/1e9:.3f}B"
    )

    print("\nSensitivity (USD billions):")
    print(sensitivity.round(2).to_string(index=False))

print("\nPART 2B DCF COMPLETE")
