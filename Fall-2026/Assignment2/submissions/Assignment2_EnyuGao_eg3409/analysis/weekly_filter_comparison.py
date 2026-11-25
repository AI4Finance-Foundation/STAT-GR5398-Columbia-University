"""Reconcile development-only absolute and quantile portfolio evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("output")
CORE_METRICS = (
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "max_drawdown_pct",
)
DISPLAY_METRICS = (
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "max_drawdown_pct",
    "turnover_per_year",
    "avg_removed_per_week",
    "avg_weekly_removed_fraction",
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing required development output: {path}")
    return json.loads(path.read_text())


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _validate_audit(rule: str, audit: pd.DataFrame) -> dict:
    required = {
        "filter_call", "filter_rule", "growth_tech_candidates",
        "eligible_candidates", "expected_removed_count", "removed_count",
        "removed_fraction", "eligible_removed_fraction",
        "all_removed_eligible", "all_removed_below_minus_20",
        "rule_count_check",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise AssertionError(f"{rule} weekly audit is missing columns: {missing}")
    if audit.empty:
        raise AssertionError(f"{rule} weekly audit is empty")
    if not audit["filter_rule"].eq(rule).all():
        raise AssertionError(f"{rule} weekly audit contains another filter rule")
    if not audit["filter_call"].equals(pd.Series(range(1, len(audit) + 1))):
        raise AssertionError(f"{rule} weekly audit calls are not consecutive")
    if not _truthy(audit["all_removed_eligible"]).all():
        raise AssertionError(f"{rule} removed at least one ineligible ticker")
    if not _truthy(audit["rule_count_check"]).all():
        raise AssertionError(f"{rule} removed count does not match its definition")
    if not audit["removed_count"].eq(audit["expected_removed_count"]).all():
        raise AssertionError(f"{rule} actual and expected removal counts differ")

    growth = audit["growth_tech_candidates"].to_numpy(dtype=float)
    removed = audit["removed_count"].to_numpy(dtype=float)
    expected_fraction = np.divide(
        removed,
        growth,
        out=np.zeros_like(removed),
        where=growth != 0,
    )
    if not np.allclose(
        audit["removed_fraction"].to_numpy(dtype=float),
        expected_fraction,
        atol=1e-12,
    ):
        raise AssertionError(f"{rule} removed fractions use the wrong denominator")

    eligible = audit["eligible_candidates"].to_numpy(dtype=int)
    expected_eligible_fraction = np.divide(
        removed,
        eligible,
        out=np.zeros_like(removed),
        where=eligible != 0,
    )
    if not np.allclose(
        audit["eligible_removed_fraction"].to_numpy(dtype=float),
        expected_eligible_fraction,
        atol=1e-12,
    ):
        raise AssertionError(
            f"{rule} eligible removed fractions use the wrong denominator"
        )

    if rule == "absolute":
        if not _truthy(audit["all_removed_below_minus_20"]).all():
            raise AssertionError("absolute rule has a removal at or above -20%")
    else:
        expected_quantile_count = np.where(
            eligible < 5,
            0,
            np.floor(0.20 * eligible).astype(int),
        )
        if not np.array_equal(removed.astype(int), expected_quantile_count):
            raise AssertionError("quantile weekly counts do not equal floor(20% of eligible)")

    removed_weeks = audit.loc[audit["removed_count"].gt(0)]
    return {
        "weeks": int(len(audit)),
        "weeks_with_removals": int(len(removed_weeks)),
        "growth_tech_candidates": int(audit["growth_tech_candidates"].sum()),
        "eligible_candidates": int(audit["eligible_candidates"].sum()),
        "removed": int(audit["removed_count"].sum()),
        "all_removed_eligible": True,
        "all_removed_below_minus_20": bool(
            removed_weeks.empty
            or _truthy(removed_weeks["all_removed_below_minus_20"]).all()
        ),
        "rule_definition_check": True,
        "fraction_denominator_check": True,
    }


def run(period: str = "dev") -> None:
    if period != "dev":
        raise ValueError("earlier development filter comparison is development-only")

    run_dirs = {
        "baseline_A": OUTPUT_DIR / f"{period}_selection",
        "absolute": OUTPUT_DIR / f"{period}_valuation_absolute",
        "quantile": OUTPUT_DIR / f"{period}_valuation_quantile",
    }
    stats = {
        name: _load_json(path / "stats.json")
        for name, path in run_dirs.items()
    }
    audits = {}
    audit_summaries = {}
    for rule in ("absolute", "quantile"):
        audit_path = run_dirs[rule] / "weekly_filter_audit.csv"
        if not audit_path.exists():
            raise FileNotFoundError(f"missing required development output: {audit_path}")
        audits[rule] = pd.read_csv(audit_path)
        audit_summaries[rule] = _validate_audit(rule, audits[rule])

    baseline = stats["baseline_A"]
    rows = []
    for label in ("baseline_A", "absolute", "quantile"):
        source = stats[label]
        row = {"run": label, "filter_rule": source.get("filter_rule", "none")}
        for metric in DISPLAY_METRICS:
            default = 0.0 if metric.startswith("avg_") and label == "baseline_A" else np.nan
            row[metric] = source.get(metric, default)
        for metric in CORE_METRICS:
            row[f"{metric}_gap_vs_A"] = float(source[metric] - baseline[metric])
            row[f"{metric}_absolute_gap_vs_A"] = abs(
                float(source[metric] - baseline[metric])
            )
        row["turnover_per_year_gap_vs_A"] = float(
            source["turnover_per_year"] - baseline["turnover_per_year"]
        )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = OUTPUT_DIR / f"filter_rule_comparison_{period}.csv"
    comparison.to_csv(comparison_path, index=False)

    by_run = comparison.set_index("run")
    gap_checks = {
        metric: bool(
            by_run.at["quantile", f"{metric}_absolute_gap_vs_A"]
            < by_run.at["absolute", f"{metric}_absolute_gap_vs_A"]
        )
        for metric in CORE_METRICS
    }
    summary = {
        "period": period,
        "development_only": True,
        "audit": audit_summaries,
        "quantile_has_smaller_removal_set": (
            audit_summaries["quantile"]["removed"]
            < audit_summaries["absolute"]["removed"]
        ),
        "quantile_smaller_absolute_gap_vs_A": gap_checks,
        "quantile_smaller_performance_gap_vs_A_all_core_metrics": all(
            gap_checks.values()
        ),
        "validation_or_test_accessed": False,
    }
    summary_path = OUTPUT_DIR / f"filter_rule_comparison_{period}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=["dev"], default="dev")
    args = parser.parse_args(argv)
    run(args.period)


if __name__ == "__main__":
    main()
