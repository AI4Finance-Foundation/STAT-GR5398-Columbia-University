"""Compare actual scenario artifacts with independently precomputed references."""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

D = Decimal
SCENARIOS = {"default": "default_replay", "deceleration_flat_margins": "analyst_deceleration"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--scenario-output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    inputs = {}
    checks = []

    def read(relative: str, base: Path):
        path = base / relative
        raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    def rows(relative: str, base: Path):
        return list(csv.DictReader(read(relative, base).decode("utf-8-sig").splitlines()))

    def obj(relative: str, base: Path):
        return json.loads(read(relative, base))

    def check(ticker, scenario, period, field, expected, actual, category, absolute="0", relative="0"):
        expected, actual = D(str(expected).strip("%")), D(str(actual).strip("%"))
        difference = actual - expected
        tolerance = max(D(absolute), abs(expected) * D(relative))
        checks.append({"category": category, "ticker": ticker, "scenario": scenario,
                       "period": period, "field": field, "expected": str(expected),
                       "actual": str(actual), "difference": str(difference),
                       "tolerance": str(tolerance), "passed": abs(difference) <= tolerance})

    def check_text(ticker, scenario, period, field, expected, actual, category):
        checks.append({"category": category, "ticker": ticker, "scenario": scenario,
                       "period": period, "field": field, "expected": str(expected),
                       "actual": str(actual), "difference": "N/A", "tolerance": "exact text",
                       "passed": expected == actual})

    with localcontext() as ctx:
        ctx.prec = 50
        forecast_expected = rows("precomputed_expected_forecasts.csv", args.reference_directory)
        price_expected = rows("precomputed_expected_ev_ebitda.csv", args.reference_directory)
        forecast_map = {(r["ticker"], SCENARIOS[r["scenario"]], r["fiscal_year"], r["metric"]): D(r["independent_value"]) for r in forecast_expected}
        outputs = {}
        for ticker in ("NVDA", "AMD"):
            baseline_name = f"snapshots/{ticker}/analysis/financial_metrics_and_forecasts.csv"
            baseline_bytes = read(baseline_name, args.submission_root)
            baseline_rows = list(csv.DictReader(baseline_bytes.decode("utf-8-sig").splitlines()))
            baseline = {r["metrics"]: r for r in baseline_rows}
            for scenario in SCENARIOS.values():
                raw = read(f"{ticker}/{scenario}/financial_metrics_and_forecasts.csv", args.scenario_output)
                output = {r["metrics"]: r for r in csv.DictReader(raw.decode("utf-8-sig").splitlines())}
                outputs[(ticker, scenario)] = output
                if scenario == "default_replay":
                    check_text(ticker, scenario, "entire CSV", "sha256", hashlib.sha256(baseline_bytes).hexdigest(), hashlib.sha256(raw).hexdigest(), "default raw bytes")
                for metric, row in baseline.items():
                    for period, value in row.items():
                        if period.endswith("A") or period == "CAGR":
                            check_text(ticker, scenario, period, metric, value, output[metric][period], "actuals and historical CAGR preserved")
        for row in forecast_expected:
            ticker, scenario, period, metric = row["ticker"], SCENARIOS[row["scenario"]], row["fiscal_year"], row["metric"]
            check(ticker, scenario, period, metric, row["independent_value"], outputs[(ticker, scenario)][metric][period], "independent forecast", absolute=".005" if row["unit"] in ("USD", "USD/share") else "0")
        price_map = {}
        for row in price_expected:
            ticker, scenario, period = row["ticker"], SCENARIOS[row["scenario"]], row["ebitda_fiscal_year"]
            actual = obj(f"{ticker}/{scenario}/engine_ev_ebitda.json", args.scenario_output)
            price_map[(ticker, scenario)] = row
            fields = {"terminal_ebitda": "EBITDA", "multiple": "Target Multiple", "multiple_sample_sd": "Historical Multiple Sample Std", "shares": "Shares Outstanding", "ev": "Enterprise Value", "assumed_net_debt": "Net Debt"}
            for reference_name, engine_name in fields.items():
                check(ticker, scenario, period, reference_name, row[reference_name], actual["assumptions"][engine_name], "independent valuation", absolute="1e-10", relative="1e-14")
            for reference_name, engine_name in [("target_price", "target_price"), ("low_price", "low_estimate"), ("high_price", "high_estimate")]:
                check(ticker, scenario, period, reference_name, row[reference_name], actual[engine_name], "independent valuation", absolute="1e-10", relative="1e-14")
            check_text(ticker, scenario, period, "selected EBITDA period", period, actual["assumptions"]["EBITDA Period"], "valuation scope")
        metric_slugs = {"Revenue": "revenue", "Contribution Profit": "contribution_profit", "Contribution Margin": "contribution_margin", "SG&A": "sga", "SG&A Margin": "sga_margin", "EBITDA": "ebitda", "EBITDA Margin": "ebitda_margin"}
        for row in rows("forecast_comparison.csv", args.scenario_output):
            ticker, period = row["ticker"], row["period"]
            for metric, slug in metric_slugs.items():
                old = forecast_map[(ticker, "default_replay", period, metric)]
                new = forecast_map[(ticker, "analyst_deceleration", period, metric)]
                fields = {slug + "_default": old, slug + "_adjusted": new, slug + ("_delta_pp" if metric.endswith("Margin") else "_delta_usd"): new - old}
                if not metric.endswith("Margin"):
                    fields[slug + "_delta_pct"] = (new / old - 1) * 100
                for field, expected in fields.items():
                    check(ticker, "comparison", period, field, expected, row[field], "summary forecast deltas", absolute=".00001", relative="1e-14")
        for row in rows("target_price_comparison.csv", args.scenario_output):
            ticker, period = row["ticker"], row["valuation_period"]
            old, new = price_map[(ticker, "default_replay")], price_map[(ticker, "analyst_deceleration")]
            oldp, newp = D(old["target_price"]), D(new["target_price"])
            revenue_ratio = forecast_map[(ticker, "analyst_deceleration", period, "Revenue")] / forecast_map[(ticker, "default_replay", period, "Revenue")]
            margin_ratio = forecast_map[(ticker, "analyst_deceleration", period, "EBITDA Margin")] / forecast_map[(ticker, "default_replay", period, "EBITDA Margin")]
            fields = {"default_target_usd": oldp, "adjusted_target_usd": newp,
                      "target_delta_usd": newp - oldp, "target_delta_pct": (newp / oldp - 1) * 100,
                      "fixed_target_multiple": D(old["multiple"]), "fixed_shares": D(old["shares"]),
                      "default_ebitda_usd": D(old["terminal_ebitda"]), "adjusted_ebitda_usd": D(new["terminal_ebitda"]),
                      "default_assumed_debt_usd": D(old["assumed_net_debt"]), "adjusted_assumed_debt_usd": D(new["assumed_net_debt"]),
                      "terminal_revenue_ratio": revenue_ratio, "terminal_ebitda_margin_ratio": margin_ratio,
                      "target_ratio": newp / oldp, "revenue_times_margin_ratio": revenue_ratio * margin_ratio}
            for field, expected in fields.items():
                check(ticker, "comparison", period, field, expected, row[field], "summary valuation deltas", absolute="1e-9", relative="1e-14")
        manifest = obj("manifest.json", args.scenario_output)
        for relative, expected in manifest["output_hashes"].items():
            actual = hashlib.sha256(read(relative, args.scenario_output)).hexdigest()
            check_text("all", "manifest", "N/A", relative, expected, actual, "output hash bindings")
        for relative, expected in manifest["input_hashes"].items():
            actual = hashlib.sha256(read(relative, args.submission_root)).hexdigest()
            check_text("all", "manifest", "N/A", relative, expected, actual, "input hash bindings")
    with (args.output_directory / "independent_reconciliation.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader(); writer.writerows(checks)
    failures = [row for row in checks if not row["passed"]]
    categories = {c: sum(row["category"] == c for row in checks) for c in sorted({row["category"] for row in checks})}
    result = {"passed": not failures, "checks": len(checks), "categories": categories,
              "failure_count": len(failures), "failures": failures, "inputs": inputs,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "independence": "Expected Decimal references were calculated before reading the new scenario implementation; this comparator imports neither upstream nor the scenario driver.",
              "limitations": "Arithmetic and preservation review, not a forecast-accuracy test or human financial review."}
    (args.output_directory / "review_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "inputs"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
