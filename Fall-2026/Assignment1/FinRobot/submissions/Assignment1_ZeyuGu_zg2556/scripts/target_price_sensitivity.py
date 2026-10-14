"""Run native sensitivities, then propagate their shocks into EV/EBITDA value.

The valuation formula and all assumptions stay identical to the independent
reconciliation. Frozen inputs are mandatory. Native rounded-margin and exact
EBITDA/revenue results are both saved, exposing rather than hiding rounding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from independent_valuation import load_inputs, number, ev_ebitda, write_csv, write_json


def compute_grid(context: dict, revenue_deltas: list[float], margin_deltas: list[float]) -> list[dict]:
    year = context["period"]
    if not year.endswith("E"):
        raise ValueError("Native forecast sensitivity requires a forecast EBITDA period")
    revenue = number(context["metrics"]["Revenue"][year], "terminal forecast revenue")
    if revenue <= 0:
        raise ValueError("Sensitivity requires positive terminal forecast revenue")
    exact_margin = context["ebitda"] / revenue
    displayed_margin = number(context["metrics"]["EBITDA Margin"][year], "displayed EBITDA margin") / 100
    shares = context["inputs"]["shares_outstanding"]
    multiple, std = context["target_multiple"], context["historical_multiple_std"]
    base_price = ev_ebitda(context["ebitda"], multiple, std, shares)["target_price"]
    rows = []
    for rev_delta in revenue_deltas:
        for margin_delta in margin_deltas:
            scenario_revenue = revenue * (1 + rev_delta)
            scenario_margin = exact_margin + margin_delta
            scenario_ebitda = scenario_revenue * scenario_margin
            value = ev_ebitda(scenario_ebitda, multiple, std, shares)
            native_ebitda = scenario_revenue * (displayed_margin + margin_delta)
            native_value = ev_ebitda(native_ebitda, multiple, std, shares)
            rows.append({"ticker": context["inputs"]["ticker"], "run_id": context["inputs"]["run_id"],
                         "method": "EV/EBITDA", "ebitda_period": year,
                         "revenue_relative_delta": rev_delta, "revenue_relative_delta_pct": rev_delta * 100,
                         "margin_delta_fraction": margin_delta, "margin_delta_percentage_points": margin_delta * 100,
                         "scenario_revenue": scenario_revenue, "base_exact_margin": exact_margin,
                         "base_displayed_margin": displayed_margin, "scenario_exact_margin": scenario_margin,
                         "scenario_ebitda": scenario_ebitda, "target_multiple": multiple, "shares": shares,
                         "assumed_net_debt": value["assumed_net_debt"], "target_price": value["target_price"],
                         "base_target_price": base_price,
                         "target_change_pct": ((value["target_price"] / base_price - 1) * 100) if base_price else "N/A",
                         "native_rounded_margin_ebitda": native_ebitda,
                         "native_rounded_margin_target_price": native_value["target_price"],
                         "native_rounding_price_difference": native_value["target_price"] - value["target_price"]})
    return rows


def summarize(context: dict, rows: list[dict]) -> dict:
    price = rows[0]["base_target_price"]
    base_revenue = number(context["metrics"]["Revenue"][context["period"]], "base revenue")
    exact_margin = context["ebitda"] / base_revenue
    positive_interior = price > 0 and context["ebitda"] * context["target_multiple"] > 0
    return {"ticker": context["inputs"]["ticker"], "run_id": context["inputs"]["run_id"],
            "valuation_method": "EV/EBITDA", "ebitda_period": context["period"],
            "base_target_price": price,
            "minimum_target_price": min(row["target_price"] for row in rows),
            "maximum_target_price": max(row["target_price"] for row in rows),
            "maximum_native_margin_rounding_price_gap": max(abs(row["native_rounding_price_difference"]) for row in rows),
            "local_revenue_elasticity": 1.0 if positive_interior else None,
            "local_price_pct_change_per_plus_1pp_margin": (1 / exact_margin) if positive_interior else None,
            "elasticity_formula": "d log(price) / d log(revenue) = 1, and percent price change for +1pp margin = 1/base_margin_fraction, before zero clipping",
            "scenario_count": len(rows), "input_sources": context["sources"],
            "frozen_assumptions": {"target_multiple": context["target_multiple"], "shares": context["inputs"]["shares_outstanding"],
                                   "net_debt_rule": "10% of scenario enterprise value, exactly as baseline method"},
            "interpretation": [
                "Revenue shock is a relative change in forecast LEVEL, not a change in annual growth rate.",
                "Margin shock is an absolute percentage-point change; 0.02 means 2pp.",
                "The exact grid starts from EBITDA/revenue to preserve the baseline zero-shock valuation.",
                "The native rounded-margin counterpart uses the displayed one-decimal margin and is saved in each CSV row.",
                "All scenario EBITDA values are propagated without the native matrix's $0.1bn rounding.",
                "This is a deterministic scenario range, not a probability interval, investment recommendation, or calibrated fair value.",
                "The range omits uncertainty in the target multiple, shares, observed debt, forecast model structure, and timing."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--valuation-inputs", type=Path, required=True)
    parser.add_argument("--upstream-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revenue-low", type=float, default=-.05)
    parser.add_argument("--revenue-high", type=float, default=.05)
    parser.add_argument("--margin-low", type=float, default=-.02)
    parser.add_argument("--margin-high", type=float, default=.02)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 2 or args.revenue_low > args.revenue_high or args.margin_low > args.margin_high:
        parser.error("Require steps >=2 and increasing scenario ranges")
    context = load_inputs(args.analysis_dir, args.valuation_inputs)
    sys.path.insert(0, str(args.upstream_src.resolve()))
    import numpy as np
    import pandas as pd
    from modules.sensitivity_analyzer import SensitivityAnalyzer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = pd.read_csv(args.analysis_dir / "financial_metrics_and_forecasts.csv")
    native = SensitivityAnalyzer(analysis)
    revenue_range, margin_range = (args.revenue_low, args.revenue_high), (args.margin_low, args.margin_high)
    native_revenue = native.analyze_revenue_sensitivity(revenue_range, args.steps)
    native_margin = native.analyze_margin_sensitivity(margin_range, args.steps)
    native_combined = native.generate_sensitivity_table(revenue_range, margin_range, args.steps)
    native.calculate_confidence_interval("Revenue")
    native.calculate_confidence_interval("EBITDA")
    if native_revenue.empty or native_margin.empty or native_combined.empty:
        raise ValueError("Native sensitivity did not generate all required outputs")
    revenue_deltas = [float(value) for value in np.linspace(*revenue_range, args.steps)]
    margin_deltas = [float(value) for value in np.linspace(*margin_range, args.steps)]
    rows = compute_grid(context, revenue_deltas, margin_deltas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_revenue.to_csv(args.output_dir / "native_revenue_sensitivity.csv", index=False)
    native_margin.to_csv(args.output_dir / "native_margin_sensitivity.csv", index=False)
    native_combined.to_csv(args.output_dir / "native_ebitda_matrix.csv")
    (args.output_dir / "native_sensitivity_summary.md").write_text(native.generate_sensitivity_summary(), encoding="utf-8")
    write_json(args.output_dir / "native_scenario_bands.json", native.confidence_intervals)
    write_csv(args.output_dir / "target_price_sensitivity.csv", rows)
    write_json(args.output_dir / "target_price_sensitivity_summary.json", summarize(context, rows))
    values = np.array([row["target_price"] for row in rows]).reshape(args.steps, args.steps)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    picture = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(args.steps), [f"{delta * 100:+.1f} pp" for delta in margin_deltas])
    ax.set_yticks(range(args.steps), [f"{delta * 100:+.1f}%" for delta in revenue_deltas])
    ax.set_xlabel("EBITDA margin change (percentage points)")
    ax.set_ylabel("Forecast revenue level change (relative %)")
    ax.set_title(f"{context['inputs']['ticker']}: {context['period']} EV/EBITDA target-price scenarios")
    if args.steps <= 9:
        midpoint = (values.min() + values.max()) / 2
        for i in range(args.steps):
            for j in range(args.steps):
                ax.text(j, i, f"{values[i, j]:,.2f}", ha="center", va="center", color="white" if values[i, j] < midpoint else "black")
    fig.colorbar(picture, ax=ax, label="Target price (same currency per share as input financials)")
    fig.savefig(args.output_dir / "target_price_sensitivity.png", dpi=180)
    fig.savefig(args.output_dir / "target_price_sensitivity.svg")
    plt.close(fig)
    print(f"Ran native analyzer and propagated {len(rows)} scenarios into the reconciled EV/EBITDA method.")


if __name__ == "__main__":
    main()
