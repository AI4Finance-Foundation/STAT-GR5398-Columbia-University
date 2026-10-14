"""Independent arithmetic audit of the pinned FinRobot valuation formulas.

Computation uses csv/math/statistics, never ValuationEngine. The upstream engine
is imported only inside run_engine() to produce a separately named comparator.
Requires real frozen pipeline artifacts; no download or fabricated fallbacks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

UPSTREAM_SHA = "2717499b8e30f242640af08c4ad9afd1113c2d45"


def number(value: Any, name: str) -> float:
    try:
        result = float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit table {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(analysis_dir: Path, inputs_path: Path) -> dict:
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    for field in ("ticker", "run_id", "shares_outstanding", "current_price"):
        if field not in inputs:
            raise ValueError(f"valuation_inputs.json requires explicit {field}")
    inputs["shares_outstanding"] = number(inputs["shares_outstanding"], "shares_outstanding")
    inputs["current_price"] = number(inputs["current_price"], "current_price")
    if inputs["shares_outstanding"] <= 0:
        raise ValueError("An observed positive share count is required; one-share engine fallback is not acceptable.")
    analysis_path = analysis_dir / "financial_metrics_and_forecasts.csv"
    analysis = read_csv(analysis_path)
    if not analysis:
        raise ValueError("Empty pipeline analysis")
    columns = list(analysis[0])
    periods = [c for c in columns if len(c) == 5 and c[:4].isdigit() and c[-1] in "AE"]
    if not periods:
        raise ValueError("No annual periods in pipeline analysis")
    actuals = [int(c[:-1]) for c in periods if c.endswith("A")]
    forecasts = [int(c[:-1]) for c in periods if c.endswith("E")]
    if actuals and forecasts and min(forecasts) <= max(actuals):
        raise ValueError("Forecast labels overlap actual years; apply documented compatibility patch first.")
    metrics = {row["metrics"]: row for row in analysis}
    selected_period = periods[-1]  # exactly mirrors the documented baseline column selector
    ebitda = number(metrics["EBITDA"][selected_period], "selected EBITDA")
    ebitda_source = f"financial_metrics_and_forecasts.csv:EBITDA/{selected_period}"
    if not ebitda:
        if "ebitda" not in inputs:
            raise ValueError("Baseline treats zero EBITDA as missing; no explicit caller EBITDA fallback supplied.")
        ebitda = number(inputs["ebitda"], "fallback EBITDA")
        ebitda_source = "valuation_inputs.json:ebitda (baseline zero-value fallback)"
    key_path = analysis_dir / "key_metrics_raw_data.csv"
    key_metrics = read_csv(key_path) if key_path.exists() else []
    multiple_field_present = bool(key_metrics and "enterpriseValueOverEBITDA" in key_metrics[0])
    multiples = []
    if multiple_field_present:
        for row in key_metrics:
            value = row.get("enterpriseValueOverEBITDA")
            if value not in (None, "", "nan", "NaN"):
                multiples.append(number(value, "historical EV/EBITDA"))
        if len(multiples) < 2:
            raise ValueError("Upstream sample-standard-deviation calculation needs >=2 historical multiples; baseline would emit NaN.")
    mean_multiple = statistics.mean(multiples) if multiples else 12.0
    std_multiple = statistics.stdev(multiples) if multiples else 3.0
    target_multiple = number(inputs.get("target_multiple", mean_multiple), "target_multiple")
    sources = {"analysis": {"path": str(analysis_path), "sha256": hash_file(analysis_path)},
               "valuation_inputs": {"path": str(inputs_path), "sha256": hash_file(inputs_path)}}
    if key_path.exists():
        sources["key_metrics"] = {"path": str(key_path), "sha256": hash_file(key_path)}
    return {"inputs": inputs, "analysis": analysis, "metrics": metrics, "key_metrics": key_metrics,
            "period": selected_period, "ebitda": ebitda, "ebitda_source": ebitda_source,
            "historical_multiple_mean": mean_multiple, "historical_multiple_std": std_multiple,
            "historical_multiple_count": len(multiples), "target_multiple": target_multiple,
            "multiple_source": "observed sample mean" if multiples else "EXPLICIT BASELINE FALLBACK: mean 12, sample standard deviation 3",
            "sources": sources}


def ev_ebitda(ebitda: float, multiple: float, std: float, shares: float) -> dict:
    """Reproduce EV/EBITDA, including its problematic 10%-of-EV debt rule."""
    enterprise_value = ebitda * multiple
    assumed_debt = enterprise_value * 0.1
    equity_value = enterprise_value - assumed_debt
    return {"enterprise_value": enterprise_value, "assumed_net_debt": assumed_debt,
            "equity_value": equity_value, "target_price": max(equity_value / shares, 0),
            "low_estimate": max((ebitda * (multiple - std) - assumed_debt) / shares, 0),
            "high_estimate": max((ebitda * (multiple + std) - assumed_debt) / shares, 0)}


def dcf(base_fcf: float, shares: float, overrides: dict | None = None) -> dict:
    assumptions = {"growth_rate_1_5": .10, "growth_rate_6_10": .05, "terminal_growth": .025,
                   "wacc": .10, "projection_years": 10}
    assumptions.update(overrides or {})
    if assumptions["wacc"] <= assumptions["terminal_growth"]:
        raise ValueError("DCF WACC must exceed terminal growth")
    if not isinstance(assumptions["projection_years"], int) or assumptions["projection_years"] < 1:
        raise ValueError("projection_years must be a positive integer")
    cash_flows, running_fcf = [], base_fcf
    for horizon in range(1, assumptions["projection_years"] + 1):
        growth = assumptions["growth_rate_1_5"] if horizon <= 5 else assumptions["growth_rate_6_10"]
        running_fcf *= 1 + growth
        pv = running_fcf / ((1 + assumptions["wacc"]) ** horizon)
        cash_flows.append({"horizon": horizon, "growth": growth, "fcf": running_fcf, "present_value": pv})
    explicit_pv = sum(row["present_value"] for row in cash_flows)
    terminal_value = running_fcf * (1 + assumptions["terminal_growth"]) / (assumptions["wacc"] - assumptions["terminal_growth"])
    terminal_pv = terminal_value / ((1 + assumptions["wacc"]) ** assumptions["projection_years"])
    ev = explicit_pv + terminal_pv
    net_debt = ev * .1
    # Intentionally reproduce, rather than endorse, the baseline's partial WACC change.
    low_ev = explicit_pv + terminal_value / ((1 + assumptions["wacc"] + .01) ** assumptions["projection_years"])
    high_ev = explicit_pv + terminal_value / ((1 + assumptions["wacc"] - .01) ** assumptions["projection_years"])
    return {"assumptions": assumptions, "cash_flows": cash_flows, "base_fcf": base_fcf,
            "pv_explicit_fcf": explicit_pv, "terminal_value": terminal_value, "pv_terminal": terminal_pv,
            "enterprise_value": ev, "assumed_net_debt": net_debt, "equity_value": ev - net_debt,
            "target_price": max((ev - net_debt) / shares, 0),
            "low_estimate": max((low_ev - net_debt) / shares, 0),
            "high_estimate": max((high_ev - net_debt) / shares, 0)}


def recompute(context: dict) -> dict:
    inputs = context["inputs"]
    ev_result = ev_ebitda(context["ebitda"], context["target_multiple"],
                          context["historical_multiple_std"], inputs["shares_outstanding"])
    explicit_fcf = inputs.get("free_cash_flow", 0)
    fcf_value = number(explicit_fcf, "free_cash_flow") if explicit_fcf else context["ebitda"] * .6
    dcf_result = dcf(fcf_value, inputs["shares_outstanding"], inputs.get("dcf_assumptions"))
    dcf_result["fcf_source"] = "caller free_cash_flow" if explicit_fcf else "BASELINE FALLBACK: 60% of terminal forecast EBITDA"
    dcf_result["fcf_period"] = inputs.get("free_cash_flow_period") if explicit_fcf else context["period"]
    return {"EV/EBITDA": ev_result, "DCF": dcf_result}


def run_engine(context: dict, upstream_src: Path) -> dict:
    """Comparator only: none of its returned values feed independent arithmetic."""
    sys.path.insert(0, str(upstream_src.resolve()))
    import pandas as pd
    from modules.valuation_engine import ValuationEngine
    from dataclasses import asdict
    financial = dict(context["inputs"])
    financial["analysis"] = pd.DataFrame(context["analysis"])
    financial["key_metrics"] = pd.DataFrame(context["key_metrics"])
    if "enterpriseValueOverEBITDA" in financial["key_metrics"]:
        financial["key_metrics"]["enterpriseValueOverEBITDA"] = pd.to_numeric(
            financial["key_metrics"]["enterpriseValueOverEBITDA"], errors="coerce")
    engine = ValuationEngine(financial, financial.get("peer_data"))
    engine.calculate_ev_ebitda_valuation(financial.get("target_multiple"))
    engine.calculate_peer_comparison_valuation()
    engine.calculate_dcf_valuation(financial.get("dcf_assumptions"))
    return {"results": [asdict(result) for result in engine.valuation_results],
            "football_field": engine.generate_football_field_data(),
            "synthesis": engine.synthesize_valuation()}


def reconciliation(context: dict, independent: dict, engine: dict) -> list[dict]:
    rows = []
    mapping = {"enterprise_value": "Enterprise Value", "assumed_net_debt": "Net Debt", "equity_value": "Equity Value"}
    for result in engine["results"]:
        if result["method"] not in independent:
            continue
        ours = independent[result["method"]]
        for field in ("target_price", "low_estimate", "high_estimate", *mapping):
            expected = ours[field]
            observed = result.get(field) if field not in mapping else result["assumptions"].get(mapping[field])
            if observed is None:  # Original uninstrumented upstream does not print internal intermediates.
                continue
            observed = number(observed, field)
            delta = expected - observed
            match = math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-7)
            rows.append({"ticker": context["inputs"]["ticker"], "run_id": context["inputs"]["run_id"],
                         "method": result["method"], "field": field, "engine_value": observed,
                         "independent_value": expected, "delta": delta,
                         "delta_pct": (delta / abs(observed) * 100) if observed else "N/A",
                         "verdict": "MATCH" if match else "MISMATCH",
                         "explanation": "Independent Python standard-library arithmetic; identical explicit and hidden baseline assumptions. Numerical agreement does not validate assumptions."})
    return rows


def assumptions_rows(context: dict, results: dict) -> list[dict]:
    inputs = context["inputs"]
    pairs = [
        ("EBITDA", context["ebitda"], context["ebitda_source"], "terminal forecast, not latest historical EBITDA"),
        ("EBITDA period", context["period"], "last A/E column in analysis CSV", "preserves baseline selector"),
        ("Shares", inputs["shares_outstanding"], inputs.get("source_metadata", {}).get("shares_outstanding", "caller must document share source"), "actual share count must match valuation units"),
        ("Historical mean EV/EBITDA", context["historical_multiple_mean"], context["multiple_source"], "unweighted sample mean; fiscal/calendar labels require source audit"),
        ("Historical sample standard deviation", context["historical_multiple_std"], "sample standard deviation, denominator n-1", f"observations={context['historical_multiple_count']}"),
        ("Target multiple", context["target_multiple"], "caller override" if "target_multiple" in inputs else context["multiple_source"], "same multiple across sensitivity cases"),
        ("Net debt rule", .1, "hard-coded baseline", "10% of each modeled enterprise value; NOT balance-sheet debt; low/high multiples hold central debt fixed"),
        ("Price floor", 0, "hard-coded baseline", "negative equity valuations clipped to zero"),
        ("EV/EBITDA discount rate", "N/A", "no discounting in baseline multiple method", "terminal forecast is not discounted back to observation date"),
        ("EV/EBITDA terminal growth", "N/A", "multiple method", "not a DCF assumption"),
        ("Base FCF", results["DCF"]["base_fcf"], results["DCF"]["fcf_source"], f"period={results['DCF']['fcf_period']}; caller FCF definition must be disclosed"),
        ("DCF low/high range", "+/-1pp terminal discount denominator only", "hard-coded baseline", "not a full WACC sensitivity: explicit PV, terminal-value denominator and net debt remain fixed"),
        ("Confidence weights", "EV/EBITDA 0.7; peers 0.6; DCF 0.5", "hard-coded baseline", "heuristic weights, not statistical confidence"),
    ]
    pairs.extend((name, value, "caller/default DCF assumption", "ten-year horizon growth defaults 10% then 5%; terminal default 2.5%; WACC default 10%")
                 for name, value in results["DCF"]["assumptions"].items())
    return [{"ticker": inputs["ticker"], "run_id": inputs["run_id"], "assumption": name,
             "value": value, "source": json.dumps(source) if isinstance(source, dict) else source,
             "interpretation": note} for name, value, source, note in pairs]


def write_notebook(path: Path, context: dict) -> None:
    """Emit a small, rerunnable notebook beside the actual experiment outputs."""
    submission = Path(__file__).resolve().parents[1]
    inputs = Path(context["sources"]["valuation_inputs"]["path"]).resolve()
    analysis = Path(context["sources"]["analysis"]["path"]).resolve().parent
    try:
        inputs_path = inputs.relative_to(submission).as_posix()
        analysis_dir = analysis.relative_to(submission).as_posix()
    except ValueError:
        inputs_path, analysis_dir = str(inputs), str(analysis)
    code = ("from pathlib import Path\nimport sys\n"
            "# Start Jupyter in this submission or in this notebook's folder.\n"
            "submission = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts' / 'independent_valuation.py').exists())\n"
            "sys.path.insert(0, str(submission / 'scripts'))\n"
            "from independent_valuation import load_inputs, recompute\n"
            f"context = load_inputs(submission / {analysis_dir!r}, submission / {inputs_path!r})\n"
            "results = recompute(context)\nresults\n")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [
        "# Independent valuation reproduction\n",
        "This notebook reruns the independent standard-library equations in the submitted script. No ValuationEngine is imported by the calculation.\n",
        "EV = terminal forecast EBITDA × target multiple; assumed net debt = 10% × EV; price = max((EV − assumed net debt)/shares, 0).\n",
        "DCF sums explicit discounted FCF plus discounted perpetuity. The hidden baseline assumptions are reproduced for reconciliation, not endorsed.\n",
        "Input file hashes and full line-by-line reconciliation are saved alongside this notebook. Snapshot paths are relative to the submission and remain portable when the complete submission is moved.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}]
    write_json(path, {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--valuation-inputs", type=Path, required=True)
    parser.add_argument("--upstream-src", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    context = load_inputs(args.analysis_dir, args.valuation_inputs)
    results = recompute(context)
    engine = run_engine(context, args.upstream_src)
    rows = reconciliation(context, results, engine)
    if not rows:
        raise ValueError("No engine outputs available for reconciliation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "independent_valuation.json", {"upstream_sha": UPSTREAM_SHA,
        "ticker": context["inputs"]["ticker"], "run_id": context["inputs"]["run_id"],
        "input_sources": context["sources"], "independent_results": results})
    write_json(args.output_dir / "engine_valuation.json", engine)
    write_csv(args.output_dir / "valuation_reconciliation.csv", rows)
    write_csv(args.output_dir / "valuation_assumptions.csv", assumptions_rows(context, results))
    write_csv(args.output_dir / "dcf_cash_flows.csv", results["DCF"]["cash_flows"])
    write_notebook(args.output_dir / "independent_valuation.ipynb", context)
    failures = sum(row["verdict"] != "MATCH" for row in rows)
    print(f"Independent reconciliation: {len(rows)} values; {failures} mismatches.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
