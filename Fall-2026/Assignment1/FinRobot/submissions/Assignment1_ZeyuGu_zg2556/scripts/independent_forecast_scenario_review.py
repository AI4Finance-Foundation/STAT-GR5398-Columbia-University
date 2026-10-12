"""Independent Decimal reference for two frozen forecast parameter scenarios.

No production forecast or valuation module is imported. Creates a fresh output
directory and refuses to overwrite it. This is arithmetic review, not a forecast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path

D = Decimal


def quantize(value: Decimal, scale: str = ".01") -> Decimal:
    return value.quantize(D(scale), rounding=ROUND_HALF_EVEN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-root", "--root", dest="root", required=True, type=Path)
    parser.add_argument("--output-directory", "--output", dest="output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read_csv(relative: str) -> list[dict[str, str]]:
        raw = (args.root / relative).read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))

    def read_json(relative: str) -> dict:
        raw = (args.root / relative).read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    rows = []
    valuations = []
    with localcontext() as ctx:
        ctx.prec = 50
        for ticker, proposed in [("NVDA", [".30", ".20", ".10"]), ("AMD", [".15", ".10", ".08"])]:
            prefix = f"snapshots/{ticker}"
            source = read_csv(f"{prefix}/analysis/financial_metrics_and_forecasts.csv")
            data = {row["metrics"]: row for row in source}
            actual = sorted(key for key in source[0] if key.endswith("A"))[-1]
            raw_multiples = read_csv(f"{prefix}/analysis/key_metrics_raw_data.csv")
            multiples = [D(row["enterpriseValueOverEBITDA"]) for row in raw_multiples if row.get("enterpriseValueOverEBITDA")]
            mean = sum(multiples) / D(len(multiples))
            sample_sd = (sum((m - mean) ** 2 for m in multiples) / D(len(multiples) - 1)).sqrt()
            shares = D(str(read_json(f"{prefix}/valuation_inputs.json")["shares_outstanding"]))
            for scenario, rates, margin_delta, sga_delta in [
                ("default", [".05", ".06", ".04"], D(".01"), D("-.005")),
                ("deceleration_flat_margins", proposed, D("0"), D("0")),
            ]:
                revenue_unrounded = D(data["Revenue"][actual])
                eps_unrounded = D(data["EPS"][actual])
                pe = D(data["PE Ratio"][actual])
                gross_base = D(data["Contribution Margin"][actual].strip("%")) / 100
                sga_base = D(data["SG&A Margin"][actual].strip("%")) / 100
                for year_number, growth_text in enumerate(rates, 1):
                    year = f"{int(actual[:-1]) + year_number}E"
                    growth = D(growth_text)
                    revenue_unrounded *= 1 + growth
                    revenue = quantize(revenue_unrounded)
                    gross_margin = quantize((gross_base + year_number * margin_delta) * 100, ".1") / 100
                    sga_margin = sga_base + year_number * sga_delta
                    sga = quantize(revenue * sga_margin)
                    gross_profit = quantize(revenue * gross_margin)
                    cost = quantize(revenue - revenue * gross_margin)
                    ebitda = quantize(revenue * gross_margin - sga)
                    eps_unrounded *= (1 + growth) * (1 + margin_delta)
                    values = {
                        "Revenue": revenue,
                        "Cost of Operations": cost,
                        "SG&A": sga,
                        "Contribution Profit": gross_profit,
                        "Contribution Margin": gross_margin * 100,
                        "EBITDA": ebitda,
                        "EBITDA Margin": quantize(ebitda / revenue * 100, ".1"),
                        "SG&A Margin": quantize(sga_margin * 100, ".1"),
                        "Revenue Growth": quantize(growth * 100, ".1"),
                        "EPS": quantize(eps_unrounded),
                        "PE Ratio": quantize(pe * D(".95") ** year_number, ".1"),
                    }
                    for metric, value in values.items():
                        unit = "percent" if metric in ["Contribution Margin", "EBITDA Margin", "SG&A Margin", "Revenue Growth"] else "USD/share" if metric == "EPS" else "x" if metric == "PE Ratio" else "USD"
                        rows.append({"ticker": ticker, "scenario": scenario, "fiscal_year": year, "metric": metric, "independent_value": str(value), "unit": unit})
                ev = ebitda * mean
                debt = ev * D(".1")
                valuations.append({
                    "ticker": ticker, "scenario": scenario, "ebitda_fiscal_year": year,
                    "terminal_ebitda": str(ebitda), "multiple": str(mean),
                    "multiple_sample_sd": str(sample_sd), "shares": str(shares),
                    "ev": str(ev), "assumed_net_debt": str(debt),
                    "target_price": str((ev - debt) / shares),
                    "low_price": str((ebitda * (mean - sample_sd) - debt) / shares),
                    "high_price": str((ebitda * (mean + sample_sd) - debt) / shares),
                    "debt_rule": "10% of modeled EV; not observed debt", "discounting": "none",
                })
    for filename, records in [("precomputed_expected_forecasts.csv", rows), ("precomputed_expected_ev_ebitda.csv", valuations)]:
        with (args.output / filename).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    result = {
        "method": "Standard-library Decimal at 50 digits; no production forecast or valuation imports",
        "inputs": inputs,
        "forecast_rows": len(rows),
        "valuation_rows": len(valuations),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.output.glob("*.csv")},
    }
    (args.output / "reference_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
