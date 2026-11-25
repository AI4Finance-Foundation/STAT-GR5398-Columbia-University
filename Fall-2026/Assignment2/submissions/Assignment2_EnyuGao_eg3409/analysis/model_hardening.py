"""Reproducible development-only hardening analyses for valuation model valuation."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from selection.scores import MonthlyScores
from shared.data_view import DataBundle, DataView
from valuation.finrobot import calculate_dcf as engine_calculate_dcf
import valuation.student_valuation as student


OUTPUT_DIR = Path("output/model_diagnostics")
END_DEV = "2021-12-31"
TOP_N = 20
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _months() -> list[date]:
    return [date(year, month, 1) for year in range(2017, 2022) for month in range(1, 13)]


def _contexts(bundle: DataBundle) -> list[dict]:
    scores = MonthlyScores()
    contexts = []
    for month in _months():
        predictions = scores.get(month)
        as_of = scores.cutoff
        growth = sorted(
            (row for row in predictions if row.bucket == "growth_tech"),
            key=lambda row: row.rank_score,
            reverse=True,
        )[:TOP_N]
        contexts.append(
            {
                "selection_month": month.isoformat(),
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                # Attribute the December-2016 cutoff to its January-2017
                # selection month when reporting development-year coverage.
                "year": month.year,
                "tickers": [row.symbol for row in growth],
                "view": DataView(bundle, as_of),
            }
        )
    return contexts


def _parameter_configs() -> list[student.DCFConfig]:
    configs = []
    for clip in ("legacy", "locked"):
        for fade in ("linear", "exponential"):
            for beta in ("legacy", "locked"):
                if (clip, fade, beta) == ("legacy", "linear", "legacy"):
                    configs.append(student.LEGACY_DCF_CONFIG)
                    continue
                if (clip, fade, beta) == ("locked", "exponential", "locked"):
                    configs.append(student.LOCKED_DCF_CONFIG)
                    continue
                configs.append(
                    replace(
                        student.LEGACY_DCF_CONFIG,
                        name=f"clip_{clip}__fade_{fade}__beta_{beta}",
                        initial_growth_min=(-0.05 if clip == "locked" else -0.10),
                        initial_growth_max=(0.35 if clip == "locked" else 0.30),
                        growth_fade=fade,
                        beta_min=(0.5 if beta == "locked" else 0.0),
                        beta_max=(2.5 if beta == "locked" else 5.0),
                        clip_raw_beta_before_blume=(beta == "legacy"),
                    )
                )
    return configs


def _run_dcf_panel(contexts: list[dict], configs: list[student.DCFConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        for context in contexts:
            _, diagnostics = student.dcf_valuation_diagnostics(
                context["as_of"],
                context["tickers"],
                context["view"],
                config=config,
            )
            diagnostics.insert(0, "selection_month", context["selection_month"])
            diagnostics.insert(1, "year", context["year"])
            diagnostics["fair_to_price"] = (
                diagnostics["fair_value"] / diagnostics["current_price"]
            )
            diagnostics["mispricing_sign"] = np.sign(diagnostics["fair_to_price"] - 1)
            rows.append(diagnostics)
    return pd.concat(rows, ignore_index=True, sort=False)


def _distribution(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    result = {"min": np.nan, "p05": np.nan, "p25": np.nan, "median": np.nan,
              "p75": np.nan, "p95": np.nan, "max": np.nan}
    if values.empty:
        return result
    q = values.quantile(QUANTILES)
    return {
        "min": float(values.min()),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "max": float(values.max()),
    }


def _dcf_summary(panel: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    key = ["selection_month", "as_of", "ticker"]
    baseline = panel.loc[panel["config"].eq(baseline_name)].copy()
    baseline = baseline[key + [
        "primary_status", "fair_value", "fair_to_price", "mispricing_sign",
        "wacc", "adjusted_beta",
    ]].rename(columns={
        "primary_status": "baseline_status",
        "fair_value": "baseline_fair_value",
        "fair_to_price": "baseline_fair_to_price",
        "mispricing_sign": "baseline_mispricing_sign",
        "wacc": "baseline_wacc",
        "adjusted_beta": "baseline_adjusted_beta",
    })
    summary = []
    for config_name, config_group in panel.groupby("config", sort=True):
        merged = config_group.merge(baseline, on=key, how="left")
        for year_label, group in [("ALL", merged), *list(merged.groupby("year"))]:
            valued = group["primary_status"].eq("valued")
            base_valued = group["baseline_status"].eq("valued")
            paired = group.loc[valued & base_valued].copy()
            ratio_dist = _distribution(group.loc[valued, "fair_to_price"])
            summary.append(
                {
                    "config": config_name,
                    "year": year_label,
                    "attempted": int(len(group)),
                    "valid": int(valued.sum()),
                    "excluded": int((~valued).sum()),
                    "coverage_pct": float(valued.mean() * 100),
                    "newly_valid_vs_legacy": int((valued & ~base_valued).sum()),
                    "lost_valid_vs_legacy": int((~valued & base_valued).sum()),
                    "paired_vs_legacy": int(len(paired)),
                    "paired_fair_value_change_median": float(
                        (paired["fair_value"] - paired["baseline_fair_value"]).median()
                    ) if not paired.empty else np.nan,
                    "paired_fair_value_change_pct_median": float(
                        (paired["fair_value"] / paired["baseline_fair_value"] - 1).median()
                    ) if not paired.empty else np.nan,
                    "mispricing_sign_changes_vs_legacy": int(
                        paired["mispricing_sign"].ne(paired["baseline_mispricing_sign"]).sum()
                    ),
                    "wacc_change_median": float(
                        (paired["wacc"] - paired["baseline_wacc"]).median()
                    ) if not paired.empty else np.nan,
                    "adjusted_beta_change_median": float(
                        (paired["adjusted_beta"] - paired["baseline_adjusted_beta"]).median()
                    ) if not paired.empty else np.nan,
                    **{f"fair_to_price_{name}": value for name, value in ratio_dist.items()},
                }
            )
    return pd.DataFrame(summary)


def _exclusion_summary(locked_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "ALL", locked_panel)]
    scopes.extend(("year", str(year), group) for year, group in locked_panel.groupby("year"))
    scopes.extend(("date", as_of, group) for as_of, group in locked_panel.groupby("as_of"))
    for scope, period, group in scopes:
        attempted = len(group)
        valued = int(group["primary_status"].eq("valued").sum())
        excluded = attempted - valued
        for status, count in group["primary_status"].value_counts().sort_index().items():
            rows.append(
                {
                    "scope": scope,
                    "period": period,
                    "primary_status": status,
                    "count": int(count),
                    "attempted": int(attempted),
                    "valued": valued,
                    "excluded": excluded,
                    "identity_verified": attempted == valued + excluded,
                }
            )
    return pd.DataFrame(rows)


def _ev_window_analysis(contexts: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for years in (3, 5):
        for context in contexts:
            _, diagnostics = student.historical_ev_ebitda_diagnostics(
                context["as_of"], context["tickers"], context["view"],
                lookback_years=years,
            )
            diagnostics.insert(0, "selection_month", context["selection_month"])
            diagnostics.insert(1, "as_of", context["as_of"])
            diagnostics.insert(2, "year", context["year"])
            for column in ("current_price", "fair_value", "low", "high"):
                if column not in diagnostics:
                    diagnostics[column] = np.nan
            diagnostics["fair_to_price"] = (
                diagnostics["fair_value"] / diagnostics["current_price"]
            )
            rows.append(diagnostics)
    panel = pd.concat(rows, ignore_index=True, sort=False)
    key = ["selection_month", "as_of", "ticker"]
    three = panel.loc[panel["lookback_years"].eq(3), key + ["status"]].rename(
        columns={"status": "status_3y"}
    )
    five = panel.loc[panel["lookback_years"].eq(5), key + ["status"]].rename(
        columns={"status": "status_5y"}
    )
    compare = three.merge(five, on=key)
    summary = {
        "attempted_per_window": int(len(three)),
        "eligible_3y": int(three["status_3y"].eq("eligible").sum()),
        "eligible_5y": int(five["status_5y"].eq("eligible").sum()),
        "eligible_only_3y": int(
            (compare["status_3y"].eq("eligible") & ~compare["status_5y"].eq("eligible")).sum()
        ),
        "eligible_only_5y": int(
            (~compare["status_3y"].eq("eligible") & compare["status_5y"].eq("eligible")).sum()
        ),
    }
    return panel, summary


def _capex_audit(bundle: DataBundle) -> tuple[pd.DataFrame, dict]:
    view = DataView(bundle, END_DEV)
    fundamentals = view.fundamentals(view.members()).sort_values(["ticker", "datadate"])
    values = pd.to_numeric(fundamentals["capex_per_share"], errors="coerce")
    overall = {
        "record_type": "overall",
        "ticker": "ALL",
        "quarters": int(len(values)),
        "positive_quarters": int(values.gt(0).sum()),
        "negative_quarters": int(values.lt(0).sum()),
        "zero_quarters": int(values.eq(0).sum()),
        "missing_quarters": int(values.isna().sum()),
    }
    rows = [overall]
    for ticker, group in fundamentals.groupby("ticker", sort=True):
        quarter = pd.to_numeric(group.tail(4)["capex_per_share"], errors="coerce")
        magnitude, convention = student.capex_spending_magnitude(quarter)
        rows.append(
            {
                "record_type": "ticker_latest_four",
                "ticker": ticker,
                "quarters": int(len(quarter)),
                "positive_quarters": int(quarter.gt(0).sum()),
                "negative_quarters": int(quarter.lt(0).sum()),
                "zero_quarters": int(quarter.eq(0).sum()),
                "missing_quarters": int(quarter.isna().sum()),
                "sequence_class": convention,
                "sum_abs": float(quarter.abs().sum()) if quarter.notna().all() else np.nan,
                "abs_sum": float(abs(quarter.sum())) if quarter.notna().all() else np.nan,
                "converted_spending_magnitude": magnitude,
            }
        )
    audit = pd.DataFrame(rows)
    ticker_rows = audit.loc[audit["record_type"].eq("ticker_latest_four")]
    summary = {
        "quarter_counts": {
            "positive": overall["positive_quarters"],
            "negative": overall["negative_quarters"],
            "zero": overall["zero_quarters"],
            "missing": overall["missing_quarters"],
        },
        "latest_four_sequence_counts": ticker_rows["sequence_class"].value_counts().to_dict(),
        "sum_abs_differs_from_abs_sum": int(
            (~np.isclose(ticker_rows["sum_abs"], ticker_rows["abs_sum"], equal_nan=True)).sum()
        ),
        "finrobot_expected_convention": "nonnegative capex_pct_revenue subtracted from FCF",
    }
    return audit, summary


def _peer_component_timing(view: DataView, tickers: list[str]) -> dict:
    start = time.perf_counter()
    members = set(view.members())
    universe = sorted(members | set(tickers))
    membership_s = time.perf_counter() - start

    start = time.perf_counter()
    fundamentals = view.fundamentals(universe)
    fundamentals_s = time.perf_counter() - start

    start = time.perf_counter()
    ttm = student.build_ttm_fundamentals(fundamentals).set_index("ticker")
    ttm_s = time.perf_counter() - start

    start = time.perf_counter()
    prices = pd.Series(view.last_close(ttm.index), dtype=float).reindex(ttm.index)
    prices_s = time.perf_counter() - start

    start = time.perf_counter()
    eps = pd.to_numeric(ttm["eps_ttm"], errors="coerce")
    pe = prices / eps
    valid_peer = (
        eps.gt(0)
        & prices.gt(0)
        & pe.gt(student.PE_MULTIPLE_MIN)
        & pe.le(student.PE_MULTIPLE_MAX)
        & ttm.index.to_series().isin(members).values
        & ttm["sector"].notna()
    )
    calculated = 0
    for ticker in tickers:
        if ticker not in ttm.index or pd.isna(ttm.at[ticker, "sector"]):
            continue
        mask = valid_peer & ttm["sector"].eq(ttm.at[ticker, "sector"])
        mask &= ttm.index.to_series().ne(ticker).values
        peer_multiples = pe.loc[mask]
        if len(peer_multiples) >= student.MIN_PEER_COUNT:
            peer_multiples.quantile([0.25, 0.5, 0.75])
            calculated += 1
    peer_filter_s = time.perf_counter() - start
    return {
        "universe_size": len(universe),
        "membership_ms": membership_s * 1000,
        "fundamentals_ms": fundamentals_s * 1000,
        "ttm_ms": ttm_s * 1000,
        "price_ms": prices_s * 1000,
        "peer_filter_ms": peer_filter_s * 1000,
        "peer_groups_calculated": calculated,
    }


def _peer_benchmark(contexts: list[dict]) -> tuple[pd.DataFrame, dict]:
    chosen = [contexts[5], contexts[29], contexts[59]]
    rows = []
    for context in chosen:
        component = _peer_component_timing(context["view"], context["tickers"])
        timings, outputs = [], []
        for _ in range(3):
            start = time.perf_counter()
            result = student.peer_pe_valuation(
                context["as_of"], context["tickers"], context["view"]
            )
            timings.append((time.perf_counter() - start) * 1000)
            outputs.append(result)
        pd.testing.assert_frame_equal(outputs[0], outputs[1])
        pd.testing.assert_frame_equal(outputs[1], outputs[2])
        rows.append(
            {
                "selection_month": context["selection_month"],
                "as_of": context["as_of"],
                "candidate_count": len(context["tickers"]),
                "valid_estimates": len(outputs[0]),
                **component,
                "unmeasured_total_remainder_ms": max(
                    0.0,
                    float(np.median(timings))
                    - component["membership_ms"]
                    - component["fundamentals_ms"]
                    - component["ttm_ms"]
                    - component["price_ms"]
                    - component["peer_filter_ms"],
                ),
                "total_ms_run_1": timings[0],
                "total_ms_run_2": timings[1],
                "total_ms_run_3": timings[2],
                "total_ms_median": float(np.median(timings)),
                "deterministic_outputs": True,
                "optimization_applied": False,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "median_complete_call_ms": float(frame["total_ms_median"].median()),
        "maximum_complete_call_ms": float(frame["total_ms_median"].max()),
        "optimization_applied": False,
        "reason": "Representative complete calls were sub-second and no redundant cross-date cache was justified.",
    }
    return frame, summary


def _hypothetical_zero_tax(raw_rate, _fiscal_period_end) -> float:
    rate = student._finite_float(raw_rate)
    return float(min(rate, 0.35)) if rate is not None and rate > 0 else 0.0


def _locked_tax_sensitivity(bundle: DataBundle) -> dict:
    affected = pd.read_csv("output/data_audit/tax_assumption_changes_2021-12-31.csv")
    tickers = affected["ticker"].astype(str).tolist()
    view = DataView(bundle, END_DEV)
    statutory = student.dcf_valuation(
        END_DEV, tickers, view, config=student.LOCKED_DCF_CONFIG
    )
    with patch.object(student, "tax_rate_for_dcf", _hypothetical_zero_tax):
        zero = student.dcf_valuation(
            END_DEV, tickers, view, config=student.LOCKED_DCF_CONFIG
        )
    keep = ["ticker", "fair_value", "wacc", "tax_rate"]
    result = affected.merge(
        statutory[keep].add_prefix("statutory_").rename(columns={"statutory_ticker": "ticker"}),
        on="ticker", how="left",
    ).merge(
        zero[keep].add_prefix("zero_").rename(columns={"zero_ticker": "ticker"}),
        on="ticker", how="left",
    )
    result["fair_value_change"] = result["statutory_fair_value"] - result["zero_fair_value"]
    result["status"] = np.select(
        [
            result["statutory_fair_value"].notna() & result["zero_fair_value"].notna(),
            result["statutory_fair_value"].isna() & result["zero_fair_value"].notna(),
            result["statutory_fair_value"].notna() & result["zero_fair_value"].isna(),
        ],
        ["paired", "zero_only", "statutory_only"],
        default="ineligible_both",
    )
    result.to_csv(OUTPUT_DIR / "tax_fallback_dcf_sensitivity_locked.csv", index=False)
    paired = result.loc[result["status"].eq("paired")]
    summary = {
        "configuration": "locked",
        "affected": len(result),
        "status_counts": result["status"].value_counts().to_dict(),
        "paired_lower_statutory": int(paired["fair_value_change"].lt(0).sum()),
        "paired_higher_statutory": int(paired["fair_value_change"].gt(0).sum()),
    }
    (OUTPUT_DIR / "tax_fallback_dcf_sensitivity_locked_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _assert_consistency(*frames: pd.DataFrame) -> None:
    for frame in frames:
        eligible = frame.loc[
            frame.get("primary_status", frame.get("status", pd.Series(index=frame.index))).isin(
                ["valued", "eligible"]
            )
        ]
        if eligible.empty:
            continue
        assert eligible["current_price"].gt(0).all()
        assert np.isfinite(eligible["fair_value"]).all()
        assert eligible["fair_value"].gt(0).all()
        assert eligible["low"].le(eligible["fair_value"]).all()
        assert eligible["fair_value"].le(eligible["high"]).all()


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    contexts = _contexts(bundle)
    assert len(contexts) == 60
    assert max(context["as_of"] for context in contexts) <= END_DEV

    print("Running eight DCF parameter configurations over 60 development months...", flush=True)
    parameter_panel = _run_dcf_panel(contexts, _parameter_configs())
    parameter_summary = _dcf_summary(parameter_panel, "legacy")
    parameter_panel.to_csv(OUTPUT_DIR / "parameter_sensitivity.csv", index=False)
    parameter_summary.to_csv(OUTPUT_DIR / "parameter_sensitivity_summary.csv", index=False)

    print("Running locked-configuration ERP sensitivity...", flush=True)
    erp_configs = [
        replace(student.LOCKED_DCF_CONFIG, name=f"locked_erp_{erp:.2f}", equity_risk_premium=erp)
        for erp in (0.04, 0.05, 0.06)
    ]
    erp_panel = _run_dcf_panel(contexts, erp_configs)
    erp_summary = _dcf_summary(erp_panel, "locked_erp_0.05")
    erp_summary = erp_summary.rename(
        columns={
            column: column.replace("vs_legacy", "vs_erp_5pct")
            for column in erp_summary.columns
            if "vs_legacy" in column
        }
    )
    erp_panel.to_csv(OUTPUT_DIR / "erp_sensitivity.csv", index=False)
    erp_summary.to_csv(OUTPUT_DIR / "erp_sensitivity_summary.csv", index=False)

    locked = parameter_panel.loc[parameter_panel["config"].eq("locked")].copy()
    locked.to_csv(OUTPUT_DIR / "dcf_exclusion_reasons.csv", index=False)
    exclusion_summary = _exclusion_summary(locked)
    exclusion_summary.to_csv(OUTPUT_DIR / "dcf_exclusion_summary.csv", index=False)

    print("Comparing three- and five-year EV/EBITDA maximum lookbacks...", flush=True)
    ev_panel, ev_summary = _ev_window_analysis(contexts)
    ev_panel.to_csv(OUTPUT_DIR / "ev_ebitda_window_comparison.csv", index=False)

    capex_audit, capex_summary = _capex_audit(bundle)
    capex_audit.to_csv(OUTPUT_DIR / "capex_sign_audit.csv", index=False)

    print("Benchmarking complete peer-P/E calls...", flush=True)
    peer_benchmark, peer_summary = _peer_benchmark(contexts)
    peer_benchmark.to_csv(OUTPUT_DIR / "peer_pe_benchmark.csv", index=False)

    locked_tax_summary = _locked_tax_sensitivity(bundle)
    _assert_consistency(parameter_panel, erp_panel, ev_panel)

    all_parameter = parameter_summary.loc[parameter_summary["year"].astype(str).eq("ALL")]
    all_erp = erp_summary.loc[erp_summary["year"].astype(str).eq("ALL")]
    overall_exclusions = exclusion_summary.loc[exclusion_summary["scope"].eq("overall")]
    hardening_summary = {
        "development_only": True,
        "as_of_min": min(context["as_of"] for context in contexts),
        "as_of_max": max(context["as_of"] for context in contexts),
        "monthly_cutoffs": len(contexts),
        "attempted_ticker_dates": len(contexts) * TOP_N,
        "locked_parameters_are_course_prescribed": False,
        "locked_parameters_source": "valuation model hardening project specification",
        "parameter_sensitivity_all_years": all_parameter.to_dict(orient="records"),
        "erp_sensitivity_all_years": all_erp.to_dict(orient="records"),
        "locked_exclusion_counts": dict(
            zip(overall_exclusions["primary_status"], overall_exclusions["count"])
        ),
        "attempted_equals_valued_plus_excluded": bool(
            overall_exclusions["identity_verified"].all()
        ),
        "ev_ebitda_window": ev_summary,
        "capex": capex_summary,
        "peer_pe_benchmark": peer_summary,
        "locked_tax_sensitivity": locked_tax_summary,
        "consistency_checks_passed": True,
        "aggregation_not_part_of_hardening_analysis": True,
        "aggregation_implemented_after_hardening": True,
        "validation_or_test_accessed": False,
        "portfolio_backtest_or_ablation_run": False,
        "selection_scores_regenerated": False,
    }
    (OUTPUT_DIR / "hardening_summary.json").write_text(
        json.dumps(_json_safe(hardening_summary), indent=2) + "\n"
    )
    print(json.dumps({
        "parameter_rows": len(parameter_panel),
        "erp_rows": len(erp_panel),
        "locked_status": hardening_summary["locked_exclusion_counts"],
        "ev_window": ev_summary,
        "peer_benchmark": peer_summary,
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
