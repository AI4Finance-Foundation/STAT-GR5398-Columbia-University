"""Build the development-only report-evidence package.

This script deliberately accepts only development-period work.  It has three
subcommands so the expensive growth-clip portfolio runs can be reproduced
independently (and, if desired, in parallel):

    python -m analysis.dev_report_evidence core
    python -m analysis.dev_report_evidence growth-clip --upper-clip 0.25
    python -m analysis.dev_report_evidence growth-clip --upper-clip 0.35
    python -m analysis.dev_report_evidence growth-clip --upper-clip 0.45
    python -m analysis.dev_report_evidence finalize

All generated files live below output/dev_analysis/report_evidence/.  No
default valuation constant is mutated outside the process-local patch used by
the explicitly labelled growth-clip ablation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.model_hardening import _contexts, _run_dcf_panel
from run_portfolio import CONFIG, run_one
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle, DataView
from shared.metrics import performance
from shared.periods import PERIODS
from valuation.finrobot import calculate_dcf as engine_calculate_dcf
import valuation.student_valuation as student


OUTPUT_DIR = Path("output/dev_analysis/report_evidence")
END_DEV = pd.Timestamp("2021-12-31")
HORIZONS = {"3m": 63, "6m": 126, "12m": 252}
RULES = ("absolute", "quantile")
RUN_DIRS = {
    "baseline_A": Path("output/dev_selection"),
    "absolute": Path("output/dev_valuation_absolute"),
    "quantile": Path("output/dev_valuation_quantile"),
}
PRE_ENGINE_STATUSES = (
    "risk_free_missing",
    "nonpositive_price",
    "insufficient_quarters",
    "growth_history_missing",
    "beta_missing",
    "beta_insufficient_pairs",
    "invalid_market_variance",
    "missing_component",
    "nonpositive_revenue",
    "invalid_margin",
    "invalid_capex",
    "invalid_balance_sheet",
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _development_context():
    start, end = PERIODS["dev"]
    _, options, params, prices, _, universe = load(CONFIG)
    if pd.Timestamp(end) != END_DEV:
        raise AssertionError("development end date changed; review before running")
    bundle = DataBundle.load(prices=prices)
    return start, end, options, params, prices, universe, bundle


def _weekly_decision_dates(prices, start, end) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    sessions = [day for day in prices.sessions if start <= day.date() <= end]
    rows = []
    prior_week = None
    for day in sessions:
        week = day.isocalendar()[:2]
        if week == prior_week:
            continue
        prior_week = week
        as_of = pd.Timestamp(
            prices.last_session_on_or_before(day.date() - timedelta(days=1))
        )
        rows.append((pd.Timestamp(day), as_of))
    return rows


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def _forward_return(
    close: pd.DataFrame,
    session_positions: dict[pd.Timestamp, int],
    as_of: pd.Timestamp,
    ticker: str,
    horizon_sessions: int,
) -> tuple[str | None, float | None]:
    position = session_positions.get(pd.Timestamp(as_of))
    if position is None:
        return None, None
    target_position = position + horizon_sessions
    if target_position >= len(close.index):
        return None, None
    target = close.index[target_position]
    if target > END_DEV or ticker not in close.columns:
        return None, None
    start_price = pd.to_numeric(pd.Series([close.at[as_of, ticker]]), errors="coerce").iloc[0]
    end_price = pd.to_numeric(pd.Series([close.at[target, ticker]]), errors="coerce").iloc[0]
    if not (np.isfinite(start_price) and np.isfinite(end_price) and start_price > 0):
        return target.date().isoformat(), None
    return target.date().isoformat(), float(end_price / start_price - 1.0)


def _filter_context_and_cohorts(bundle, prices, start, end) -> dict:
    scores = MonthlyScores()
    close = prices.close.loc[:END_DEV].copy()
    session_positions = {pd.Timestamp(day): i for i, day in enumerate(close.index)}
    context_rows: list[dict] = []
    cohort_rows: list[dict] = []

    for call, (rebalance_day, as_of) in enumerate(
        _weekly_decision_dates(prices, start, end), start=1
    ):
        predictions = scores.get(rebalance_day.date())
        growth = [row for row in predictions if row.bucket == "growth_tech"]
        growth_symbols = list(dict.fromkeys(row.symbol for row in growth))
        top10 = {
            row.symbol
            for row in sorted(growth, key=lambda item: item.rank_score, reverse=True)[:10]
        }
        details = student.valuation_details(
            as_of,
            growth_symbols,
            DataView(bundle, as_of),
        )
        eligible = details.loc[details["filter_eligible"].fillna(False)].copy()

        for rule in RULES:
            removed = student.filter_removals(details, rule=rule)
            removed_rows = eligible.loc[eligible["ticker"].astype(str).isin(removed)]
            if rule == "absolute" and not (
                removed_rows.empty
                or removed_rows["mispricing"].lt(student.OVERVALUATION_THRESHOLD).all()
            ):
                raise AssertionError("absolute rule removed a ticker at or above -20%")
            expected = (
                len(
                    eligible.loc[
                        eligible["mispricing"].lt(student.OVERVALUATION_THRESHOLD)
                    ]
                )
                if rule == "absolute"
                else (
                    int(np.floor(student.QUANTILE_REMOVAL_FRACTION * len(eligible)))
                    if len(eligible) >= student.QUANTILE_MINIMUM_POOL
                    else 0
                )
            )
            if len(removed) != expected:
                raise AssertionError(f"{rule} removal count does not match its definition")

            removed_top10 = sum(ticker in top10 for ticker in removed)
            context_rows.append(
                {
                    "filter_call": call,
                    "rebalance_date": rebalance_day.date().isoformat(),
                    "decision_as_of": as_of.date().isoformat(),
                    "selection_year": rebalance_day.year,
                    "selection_month": rebalance_day.strftime("%Y-%m"),
                    "filter_rule": rule,
                    "total_candidates": len(predictions),
                    "growth_tech_candidates": len(growth_symbols),
                    "valued_candidates": len(details),
                    "eligible_candidates": len(eligible),
                    "removed_count": len(removed),
                    "removed_fraction_of_growth": (
                        len(removed) / len(growth_symbols) if growth_symbols else 0.0
                    ),
                    "removed_fraction_of_eligible": (
                        len(removed) / len(eligible) if len(eligible) else 0.0
                    ),
                    "removed_in_month_top10": removed_top10,
                    "removed_top10_fraction": (
                        removed_top10 / len(removed) if removed else 0.0
                    ),
                    "removed_tickers": ",".join(sorted(removed)),
                    "eligible_tickers": ",".join(
                        eligible["ticker"].astype(str).tolist()
                    ),
                }
            )

            for row in eligible.itertuples(index=False):
                cohort = "removed" if row.ticker in removed else "kept_eligible"
                for horizon, sessions in HORIZONS.items():
                    target, result = _forward_return(
                        close,
                        session_positions,
                        as_of,
                        str(row.ticker),
                        sessions,
                    )
                    cohort_rows.append(
                        {
                            "filter_call": call,
                            "rebalance_date": rebalance_day.date().isoformat(),
                            "decision_as_of": as_of.date().isoformat(),
                            "selection_year": rebalance_day.year,
                            "filter_rule": rule,
                            "ticker": str(row.ticker),
                            "cohort": cohort,
                            "horizon": horizon,
                            "horizon_sessions": sessions,
                            "target_date": target,
                            "forward_return": result,
                            "mispricing": float(row.mispricing),
                            "method_count": int(row.method_count),
                            "method_spread": float(row.method_spread),
                            "in_month_top10": str(row.ticker) in top10,
                        }
                    )

    contexts = pd.DataFrame(context_rows)
    cohorts = pd.DataFrame(cohort_rows)
    contexts.to_csv(OUTPUT_DIR / "filter_context_audit_by_call.csv", index=False)
    cohorts.to_csv(OUTPUT_DIR / "filter_cohort_ticker_returns.csv", index=False)

    distribution_metrics = (
        "total_candidates",
        "growth_tech_candidates",
        "eligible_candidates",
        "removed_count",
        "removed_fraction_of_growth",
        "removed_fraction_of_eligible",
    )
    distribution_rows = []
    for rule, group in contexts.groupby("filter_rule", sort=True):
        for metric in distribution_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            distribution_rows.append(
                {
                    "filter_rule": rule,
                    "metric": metric,
                    "calls": len(values),
                    "p25": float(values.quantile(0.25)),
                    "p50": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "mean": float(values.mean()),
                    "sum": float(values.sum()),
                }
            )
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(OUTPUT_DIR / "filter_context_distribution.csv", index=False)

    call_level = (
        cohorts.dropna(subset=["forward_return"])
        .groupby(
            [
                "filter_call",
                "rebalance_date",
                "decision_as_of",
                "selection_year",
                "filter_rule",
                "horizon",
                "cohort",
            ],
            as_index=False,
        )
        .agg(
            ticker_count=("ticker", "nunique"),
            equal_weight_forward_return=("forward_return", "mean"),
        )
    )
    call_level.to_csv(OUTPUT_DIR / "filter_cohort_returns_by_call.csv", index=False)

    cohort_summary = (
        call_level.groupby(
            ["filter_rule", "selection_year", "horizon", "cohort"],
            as_index=False,
        )
        .agg(
            cohorts=("filter_call", "nunique"),
            ticker_dates=("ticker_count", "sum"),
            mean_equal_weight_forward_return=("equal_weight_forward_return", "mean"),
            median_equal_weight_forward_return=("equal_weight_forward_return", "median"),
        )
    )
    overall = (
        call_level.groupby(["filter_rule", "horizon", "cohort"], as_index=False)
        .agg(
            cohorts=("filter_call", "nunique"),
            ticker_dates=("ticker_count", "sum"),
            mean_equal_weight_forward_return=("equal_weight_forward_return", "mean"),
            median_equal_weight_forward_return=("equal_weight_forward_return", "median"),
        )
    )
    overall.insert(1, "selection_year", "ALL")
    cohort_summary["selection_year"] = cohort_summary["selection_year"].astype(str)
    cohort_summary = pd.concat([cohort_summary, overall], ignore_index=True)
    paired = cohort_summary.pivot(
        index=["filter_rule", "selection_year", "horizon"],
        columns="cohort",
        values="mean_equal_weight_forward_return",
    ).reset_index()
    paired["removed_minus_kept_return"] = (
        paired.get("removed") - paired.get("kept_eligible")
    )
    cohort_summary = cohort_summary.merge(
        paired[
            ["filter_rule", "selection_year", "horizon", "removed_minus_kept_return"]
        ],
        on=["filter_rule", "selection_year", "horizon"],
        how="left",
    )
    cohort_summary.to_csv(OUTPUT_DIR / "filter_cohort_return_summary.csv", index=False)

    # Reconcile the freshly computed calls against the previously saved weekly
    # portfolio audit, rather than silently treating a second calculation as
    # independent evidence.
    reconciliations = {}
    for rule in RULES:
        existing = pd.read_csv(RUN_DIRS[rule] / "weekly_filter_audit.csv")
        fresh = contexts.loc[contexts["filter_rule"].eq(rule)].reset_index(drop=True)
        if len(existing) != len(fresh):
            raise AssertionError(f"{rule} call count changed")
        column_pairs = {
            "growth_tech_candidates": "growth_tech_candidates",
            "eligible_candidates": "eligible_candidates",
            "removed_count": "removed_count",
        }
        for old, new in column_pairs.items():
            if not existing[old].astype(int).equals(fresh[new].astype(int)):
                raise AssertionError(f"{rule} {old} does not match saved weekly audit")
        removed_total = int(fresh["removed_count"].sum())
        top10_total = int(fresh["removed_in_month_top10"].sum())
        reconciliations[rule] = {
            "calls": len(fresh),
            "total_candidates": int(fresh["total_candidates"].sum()),
            "growth_observations": int(fresh["growth_tech_candidates"].sum()),
            "eligible_observations": int(fresh["eligible_candidates"].sum()),
            "removals": removed_total,
            "removed_in_month_top10": top10_total,
            "fraction_removed_names_in_month_top10": (
                top10_total / removed_total if removed_total else 0.0
            ),
            "matches_existing_weekly_audit": True,
        }
    expected_absolute = {
        "growth_observations": 24829,
        "eligible_observations": 8584,
        "removals": 2563,
    }
    for key, expected in expected_absolute.items():
        if reconciliations["absolute"][key] != expected:
            raise AssertionError(
                f"absolute {key}={reconciliations['absolute'][key]}, expected {expected}"
            )
    return {
        "reconciliation": reconciliations,
        "distribution_rows": len(distributions),
        "cohort_ticker_rows": len(cohorts),
        "cohort_summary_rows": len(cohort_summary),
    }


def _annual_returns_and_concentration(prices, start, end) -> dict:
    yearly_rows = []
    weekly_concentration_rows = []
    concentration_summary_rows = []
    close_ff = prices.close.loc[:END_DEV].ffill()
    sessions = [day for day in prices.sessions if start <= day.date() <= end]

    for run, directory in RUN_DIRS.items():
        equity_frame = pd.read_csv(directory / "equity.csv", index_col=0, parse_dates=True)
        equity = equity_frame.iloc[:, 0].astype(float).sort_index()
        daily_returns = equity.pct_change()
        stats = _load_json(directory / "stats.json")

        previous_value = 100_000.0
        for year, group in equity.groupby(equity.index.year):
            year_return = float(group.iloc[-1] / previous_value - 1.0)
            volatility = float(
                daily_returns.loc[daily_returns.index.year == year].dropna().std()
                * np.sqrt(252)
            )
            yearly_rows.append(
                {
                    "run": run,
                    "year": int(year),
                    "calendar_return": year_return,
                    "volatility_annualized": volatility,
                    "year_end_equity": float(group.iloc[-1]),
                }
            )
            previous_value = float(group.iloc[-1])

        trades = pd.read_csv(directory / "trades.csv")
        if not trades.empty:
            trades["date"] = pd.to_datetime(trades["date"])
        trade_groups = {
            pd.Timestamp(day): group for day, group in trades.groupby("date")
        } if not trades.empty else {}
        positions: dict[str, int] = {}
        prior_week = None
        for day in sessions:
            group = trade_groups.get(pd.Timestamp(day))
            if group is not None:
                for trade in group.itertuples(index=False):
                    symbol = str(trade.symbol)
                    positions[symbol] = positions.get(symbol, 0) + int(trade.qty)
                    if positions[symbol] == 0:
                        positions.pop(symbol)
            week = day.isocalendar()[:2]
            if week == prior_week:
                continue
            prior_week = week
            values = []
            for symbol, quantity in positions.items():
                if symbol not in close_ff.columns:
                    continue
                price = close_ff.at[day, symbol]
                if np.isfinite(price) and quantity > 0:
                    values.append(float(quantity * price))
            equity_value = float(equity.loc[day])
            values.sort(reverse=True)
            weekly_concentration_rows.append(
                {
                    "run": run,
                    "date": day.date().isoformat(),
                    "year": day.year,
                    "holding_count": len(values),
                    "top10_market_value": sum(values[:10]),
                    "top10_weight_concentration": (
                        sum(values[:10]) / equity_value if equity_value else np.nan
                    ),
                    "all_positions_weight": (
                        sum(values) / equity_value if equity_value else np.nan
                    ),
                }
            )

        full_vol = float(equity.pct_change().dropna().std() * np.sqrt(252))
        concentration_summary_rows.append(
            {
                "run": run,
                "year": "ALL",
                "mean_top10_weight_concentration": np.nan,
                "p50_top10_weight_concentration": np.nan,
                "p75_top10_weight_concentration": np.nan,
                "mean_holding_count": np.nan,
                "volatility_annualized": full_vol,
                "reported_volatility_pct": stats.get("volatility_pct"),
            }
        )

    yearly = pd.DataFrame(yearly_rows)
    baseline = yearly.loc[yearly["run"].eq("baseline_A"), ["year", "calendar_return"]].rename(
        columns={"calendar_return": "baseline_A_return"}
    )
    yearly = yearly.merge(baseline, on="year", how="left")
    yearly["B_minus_A_return_gap"] = yearly["calendar_return"] - yearly["baseline_A_return"]
    yearly.to_csv(OUTPUT_DIR / "yearly_portfolio_attribution.csv", index=False)

    weekly = pd.DataFrame(weekly_concentration_rows)
    weekly.to_csv(OUTPUT_DIR / "weekly_portfolio_concentration.csv", index=False)
    for (run, year), group in weekly.groupby(["run", "year"], sort=True):
        volatility = float(
            yearly.loc[
                yearly["run"].eq(run) & yearly["year"].eq(year),
                "volatility_annualized",
            ].iloc[0]
        )
        concentration_summary_rows.append(
            {
                "run": run,
                "year": str(year),
                "mean_top10_weight_concentration": float(
                    group["top10_weight_concentration"].mean()
                ),
                "p50_top10_weight_concentration": float(
                    group["top10_weight_concentration"].median()
                ),
                "p75_top10_weight_concentration": float(
                    group["top10_weight_concentration"].quantile(0.75)
                ),
                "mean_holding_count": float(group["holding_count"].mean()),
                "volatility_annualized": volatility,
                "reported_volatility_pct": np.nan,
            }
        )
    concentration_summary = pd.DataFrame(concentration_summary_rows)
    for run in RUN_DIRS:
        mask = concentration_summary["run"].eq(run) & concentration_summary["year"].eq("ALL")
        group = weekly.loc[weekly["run"].eq(run)]
        concentration_summary.loc[mask, "mean_top10_weight_concentration"] = float(
            group["top10_weight_concentration"].mean()
        )
        concentration_summary.loc[mask, "p50_top10_weight_concentration"] = float(
            group["top10_weight_concentration"].median()
        )
        concentration_summary.loc[mask, "p75_top10_weight_concentration"] = float(
            group["top10_weight_concentration"].quantile(0.75)
        )
        concentration_summary.loc[mask, "mean_holding_count"] = float(
            group["holding_count"].mean()
        )
    concentration_summary.to_csv(
        OUTPUT_DIR / "portfolio_concentration_volatility.csv", index=False
    )
    return {
        "year_rows": len(yearly),
        "weekly_concentration_rows": len(weekly),
        "all_period": concentration_summary.loc[
            concentration_summary["year"].eq("ALL")
        ].to_dict("records"),
    }


def _dcf_funnel() -> dict:
    panel = pd.read_csv("output/model_diagnostics/dcf_exclusion_reasons.csv")
    if len(panel) != 1200 or not panel["config"].eq("locked").all():
        raise AssertionError("locked 1,200-ticker-date DCF panel is unavailable")
    counts = panel["primary_status"].value_counts().to_dict()
    rows = []
    for status in PRE_ENGINE_STATUSES:
        count = int(counts.get(status, 0))
        rows.append(
            {
                "stage": "pre_engine_exclusion",
                "status": status,
                "count": count,
                "pct_of_attempts": count / len(panel),
            }
        )
    for status in ("valued", "engine_rejected", "invalid_sensitivity_grid"):
        count = int(counts.get(status, 0))
        rows.append(
            {
                "stage": "engine_call_outcome",
                "status": status,
                "count": count,
                "pct_of_attempts": count / len(panel),
            }
        )
    funnel = pd.DataFrame(rows)
    pre_engine = int(
        funnel.loc[funnel["stage"].eq("pre_engine_exclusion"), "count"].sum()
    )
    engine_calls = int(
        funnel.loc[funnel["stage"].eq("engine_call_outcome"), "count"].sum()
    )
    valued = int(counts.get("valued", 0))
    rejected = int(counts.get("engine_rejected", 0))
    invalid_grid = int(counts.get("invalid_sensitivity_grid", 0))
    if not (
        pre_engine + engine_calls == 1200
        and engine_calls == valued + rejected + invalid_grid
        and engine_calls == 752
        and valued == 626
        and rejected == 126
        and invalid_grid == 0
    ):
        raise AssertionError("locked DCF funnel does not match actual panel")
    funnel.to_csv(OUTPUT_DIR / "dcf_funnel.csv", index=False)
    summary = {
        "attempted_ticker_dates": 1200,
        "pre_engine_exclusions": pre_engine,
        "pre_engine_status_codes_reported": len(PRE_ENGINE_STATUSES),
        "engine_calls": engine_calls,
        "valued": valued,
        "engine_rejected": rejected,
        "invalid_sensitivity_grid": invalid_grid,
        "engine_rejection_rate": rejected / engine_calls,
        "identity_verified": True,
    }
    _write_json(OUTPUT_DIR / "dcf_funnel_summary.json", summary)
    return summary


def _assumption_direction_evidence(bundle) -> dict:
    affected = pd.read_csv("output/data_audit/da_affected_tickers.csv")
    affected = affected.loc[
        affected["as_of"].eq("2021-12-31")
        & affected["paired_median_difference"].lt(0)
    ].copy()
    if len(affected) != 15:
        raise AssertionError("expected 15 end-development D&A-floor inputs")
    tickers = affected["ticker"].astype(str).tolist()
    view = DataView(bundle, END_DEV)
    # Capture the exact locked DCFInputs that reach FinRobot under the
    # production floor.  A negative D&A value cannot be passed through the
    # engine as an economic comparator: besides violating DCFInputs' bounds,
    # FinRobot's terminal normalization would interpret it as negative
    # maintenance capex.  Instead isolate the explicit-period D&A tax shield,
    # holding WACC and terminal economics fixed at the valid floored case.
    comparison_rows = []
    raw_by_ticker = affected.set_index("ticker")["paired_median_difference"]
    for ticker in tickers:
        captured = []

        def capture_calculate_dcf(inputs, wacc_override=None, tg_override=None, mid_year=False):
            if wacc_override is None and tg_override is None:
                captured.append(inputs)
            return engine_calculate_dcf(
                inputs,
                wacc_override=wacc_override,
                tg_override=tg_override,
                mid_year=mid_year,
            )

        with patch.object(student, "calculate_dcf", capture_calculate_dcf):
            _, diagnostic = student.dcf_valuation_diagnostics(
                END_DEV,
                [ticker],
                view,
                config=student.LOCKED_DCF_CONFIG,
            )
        row = diagnostic.iloc[0]
        result = {
            "ticker": ticker,
            "paired_median_difference": float(raw_by_ticker.loc[ticker]),
            "floored_policy_value": 0.0,
            "floored_status": str(row["primary_status"]),
            "floored_fair_value": row["fair_value"],
            "floored_da_pct_revenue": row["da_pct_revenue"],
            "unfloored_status": "did_not_reach_engine",
            "unfloored_fair_value": np.nan,
            "unfloored_da_pct_revenue": float(raw_by_ticker.loc[ticker]),
            "explicit_tax_shield_fair_value_change": np.nan,
            "comparator_definition": (
                "negative D&A proxy in explicit forecast only; WACC and terminal "
                "economics fixed at the floored locked case"
            ),
        }
        if captured:
            floored_inputs = captured[0]
            try:
                floored_result = engine_calculate_dcf(floored_inputs)
            except (ValueError, ArithmeticError) as exc:
                result["unfloored_status"] = f"engine_rejected:{exc}"
            else:
                raw_da = float(raw_by_ticker.loc[ticker])
                tax = float(floored_inputs.tax_rate)
                wacc = float(floored_result.wacc)
                shield_change = sum(
                    (-raw_da * tax * float(revenue)) / (1.0 + wacc) ** year
                    for year, revenue in enumerate(
                        floored_result.projected_revenue, start=1
                    )
                ) / float(floored_inputs.shares_outstanding)
                result["unfloored_status"] = "valued_fixed_terminal_comparator"
                result["floored_fair_value"] = float(floored_result.implied_price)
                result["unfloored_fair_value"] = float(
                    floored_result.implied_price - shield_change
                )
                result["explicit_tax_shield_fair_value_change"] = float(shield_change)
        comparison_rows.append(result)
    comparison = pd.DataFrame(comparison_rows)
    comparison["fair_value_change_from_floor"] = (
        comparison["floored_fair_value"] - comparison["unfloored_fair_value"]
    )
    comparison["fair_value_change_pct_from_floor"] = (
        comparison["floored_fair_value"] / comparison["unfloored_fair_value"] - 1
    )
    comparison.to_csv(OUTPUT_DIR / "da_floor_direction_sensitivity.csv", index=False)
    paired = comparison.dropna(
        subset=["floored_fair_value", "unfloored_fair_value"]
    )

    tax = pd.read_csv("output/model_diagnostics/tax_fallback_dcf_sensitivity_locked.csv")
    if len(tax) != 40 or not tax["input_change"].eq(0.21).all():
        raise AssertionError("expected 40 aggregated statutory-tax fallback inputs")
    tax.to_csv(OUTPUT_DIR / "tax_fallback_direction_sensitivity.csv", index=False)
    tax_paired = tax.loc[tax["status"].eq("paired")]
    notes = {
        "da_floor": {
            "affected_aggregated_ticker_inputs": len(comparison),
            "input_direction": "negative paired D&A/revenue median raised to 0%",
            "mechanism": (
                "The non-negative floor raises the modeled D&A input relative to a "
                "negative proxy, lowers taxable EBIT, raises the D&A tax shield, and "
                "therefore raises fair value when the DCF is otherwise comparable."
            ),
            "paired_dcf_comparisons": len(paired),
            "paired_higher_fair_value_with_floor": int(
                paired["fair_value_change_from_floor"].gt(0).sum()
            ),
            "paired_median_fair_value_change": (
                float(paired["fair_value_change_from_floor"].median())
                if len(paired)
                else None
            ),
            "paired_median_fair_value_change_pct": (
                float(paired["fair_value_change_pct_from_floor"].median())
                if len(paired)
                else None
            ),
        },
        "tax_fallback": {
            "affected_aggregated_ticker_inputs": len(tax),
            "input_direction": "nonpositive raw rate replaced by 21% statutory rate",
            "expected_operating_direction": "higher tax and lower fair value",
            "paired_dcf_comparisons": len(tax_paired),
            "paired_lower_fair_value_model_native": int(
                tax_paired["fair_value_change"].lt(0).sum()
            ),
            "paired_higher_fair_value_model_native_due_to_wacc_offset": int(
                tax_paired["fair_value_change"].gt(0).sum()
            ),
            "caveat": (
                "All 40 inputs move toward higher operating tax. Only 14 are DCF-eligible "
                "under both comparators; model-native WACC financing effects reverse two "
                "of those, while the fixed-WACC operating comparison is lower for all 14."
            ),
        },
    }
    _write_json(OUTPUT_DIR / "assumption_direction_notes.json", notes)
    return notes


def run_core() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start, end, _, _, prices, _, bundle = _development_context()
    filter_summary = _filter_context_and_cohorts(bundle, prices, start, end)
    portfolio_summary = _annual_returns_and_concentration(prices, start, end)
    funnel_summary = _dcf_funnel()
    assumption_notes = _assumption_direction_evidence(bundle)
    summary = {
        "period": "dev",
        "start": str(start),
        "end": str(end),
        "filter": filter_summary,
        "portfolio": portfolio_summary,
        "dcf_funnel": funnel_summary,
        "assumption_directions": assumption_notes,
        "validation_or_test_accessed": False,
    }
    _write_json(OUTPUT_DIR / "core_summary.json", summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def run_growth_clip(upper_clip: float) -> None:
    if upper_clip not in {0.25, 0.35, 0.45}:
        raise ValueError("upper clip must be one of 0.25, 0.35, 0.45")
    start, end, options, params, prices, universe, bundle = _development_context()
    label = f"{int(round(upper_clip * 100)):02d}"
    out = OUTPUT_DIR / f"growth_clip_{label}"
    out.mkdir(parents=True, exist_ok=True)
    config = replace(
        student.LOCKED_DCF_CONFIG,
        name=f"locked_growth_clip_{label}",
        initial_growth_max=upper_clip,
    )

    if upper_clip == 0.35:
        panel = pd.read_csv("output/model_diagnostics/dcf_exclusion_reasons.csv").copy()
        panel["config"] = config.name
        stats = _load_json(Path("output/dev_valuation_absolute/stats.json"))
        portfolio_source = "output/dev_valuation_absolute (existing exact default run)"
    else:
        panel = _run_dcf_panel(_contexts(bundle), [config])
        ctx = {
            "options": options,
            "params": params,
            "prices": prices,
            "universe": universe,
            "scores": MonthlyScores(),
            "bundle": bundle,
        }
        relative_name = f"dev_analysis/report_evidence/growth_clip_{label}/portfolio_absolute"
        with patch.object(student, "DEFAULT_DCF_CONFIG", config):
            stats = run_one(
                relative_name,
                start,
                end,
                valuation=True,
                filter_rule="absolute",
                timing="none",
                stops=False,
                ctx=ctx,
            )
        portfolio_source = f"output/{relative_name}"
    panel.to_csv(out / "dcf_panel.csv", index=False)
    _write_json(out / "portfolio_stats.json", stats)
    _write_json(
        out / "run_manifest.json",
        {
            "period": "dev",
            "start": str(start),
            "end": str(end),
            "upper_growth_clip": upper_clip,
            "lower_growth_clip": config.initial_growth_min,
            "default_configuration_changed": False,
            "portfolio_filter_rule": "absolute",
            "portfolio_source": portfolio_source,
            "dcf_ticker_dates": len(panel),
            "validation_or_test_accessed": False,
        },
    )
    print(json.dumps({"clip": upper_clip, "stats": stats}, indent=2))


def _format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def finalize() -> None:
    core = _load_json(OUTPUT_DIR / "core_summary.json")
    panels = {}
    stats = {}
    for label in ("25", "35", "45"):
        panel_path = OUTPUT_DIR / f"growth_clip_{label}/dcf_panel.csv"
        stats_path = OUTPUT_DIR / f"growth_clip_{label}/portfolio_stats.json"
        if not panel_path.exists() or not stats_path.exists():
            raise FileNotFoundError(
                f"run growth-clip --upper-clip 0.{label} before finalize"
            )
        panels[label] = pd.read_csv(panel_path)
        stats[label] = _load_json(stats_path)

    baseline_stats = _load_json(RUN_DIRS["baseline_A"] / "stats.json")
    filter_stats = {
        rule: _load_json(RUN_DIRS[rule] / "stats.json") for rule in RULES
    }
    key = ["selection_month", "as_of", "ticker"]
    base = panels["35"][
        key + ["primary_status", "fair_value", "initial_growth"]
    ].rename(
        columns={
            "primary_status": "status_35",
            "fair_value": "fair_value_35",
            "initial_growth": "initial_growth_35",
        }
    )
    hit_35 = int(base["initial_growth_35"].eq(0.35).sum())
    rows = []
    for label in ("25", "35", "45"):
        clip = int(label) / 100
        panel = panels[label]
        merged = panel.merge(base, on=key, how="left")
        paired = merged.loc[
            merged["primary_status"].eq("valued")
            & merged["status_35"].eq("valued")
        ].copy()
        paired["fair_value_change_vs_35"] = (
            paired["fair_value"] - paired["fair_value_35"]
        )
        paired["fair_value_change_pct_vs_35"] = (
            paired["fair_value"] / paired["fair_value_35"] - 1
        )
        changed_growth = paired.loc[
            ~np.isclose(
                pd.to_numeric(paired["initial_growth"], errors="coerce"),
                pd.to_numeric(paired["initial_growth_35"], errors="coerce"),
                equal_nan=True,
            )
        ].copy()
        row_stats = stats[label]
        rows.append(
            {
                "upper_growth_clip": clip,
                "attempted_ticker_dates": len(panel),
                "valued_ticker_dates": int(panel["primary_status"].eq("valued").sum()),
                "ticker_dates_hitting_default_35pct_clip": hit_35,
                "paired_valued_vs_35": len(paired),
                "paired_fair_value_change_median": float(
                    paired["fair_value_change_vs_35"].median()
                ),
                "paired_fair_value_change_pct_median": float(
                    paired["fair_value_change_pct_vs_35"].median()
                ),
                "paired_fair_value_increases": int(
                    changed_growth["fair_value_change_vs_35"].gt(1e-8).sum()
                ),
                "paired_fair_value_decreases": int(
                    changed_growth["fair_value_change_vs_35"].lt(-1e-8).sum()
                ),
                "paired_growth_inputs_changed_vs_35": len(changed_growth),
                "changed_input_fair_value_change_median": (
                    float(changed_growth["fair_value_change_vs_35"].median())
                    if len(changed_growth)
                    else 0.0
                ),
                "changed_input_fair_value_change_pct_median": (
                    float(changed_growth["fair_value_change_pct_vs_35"].median())
                    if len(changed_growth)
                    else 0.0
                ),
                "status_changes_vs_35": int(
                    merged["primary_status"].ne(merged["status_35"]).sum()
                ),
                "portfolio_total_return_pct": row_stats["total_return_pct"],
                "portfolio_cagr_pct": row_stats["cagr_pct"],
                "portfolio_sharpe": row_stats["sharpe"],
                "portfolio_max_drawdown_pct": row_stats["max_drawdown_pct"],
                "portfolio_turnover_per_year": row_stats["turnover_per_year"],
                "B_minus_A_total_return_gap_pct_points": (
                    row_stats["total_return_pct"] - baseline_stats["total_return_pct"]
                ),
                "B_minus_A_cagr_gap_pct_points": (
                    row_stats["cagr_pct"] - baseline_stats["cagr_pct"]
                ),
            }
        )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(OUTPUT_DIR / "growth_clip_ablation.csv", index=False)

    contexts = pd.read_csv(OUTPUT_DIR / "filter_context_audit_by_call.csv")
    distributions = pd.read_csv(OUTPUT_DIR / "filter_context_distribution.csv")
    yearly = pd.read_csv(OUTPUT_DIR / "yearly_portfolio_attribution.csv")
    cohorts = pd.read_csv(OUTPUT_DIR / "filter_cohort_return_summary.csv")
    concentration = pd.read_csv(OUTPUT_DIR / "portfolio_concentration_volatility.csv")

    abs_context = contexts.loc[contexts["filter_rule"].eq("absolute")]
    quant_context = contexts.loc[contexts["filter_rule"].eq("quantile")]
    cohort_all = cohorts.loc[cohorts["selection_year"].astype(str).eq("ALL")]
    concentration_all = concentration.loc[concentration["year"].astype(str).eq("ALL")]
    yearly_filters = yearly.loc[yearly["run"].isin(["absolute", "quantile"])]

    dist_report = distributions[["filter_rule", "metric", "p25", "p50", "p75"]].copy()
    dist_report["metric"] = dist_report["metric"].replace(
        {
            "total_candidates": "total candidates",
            "growth_tech_candidates": "growth subset",
            "eligible_candidates": "eligible",
            "removed_count": "removals",
            "removed_fraction_of_growth": "removed / growth",
            "removed_fraction_of_eligible": "removed / eligible",
        }
    )
    fraction_mask = dist_report["metric"].str.startswith("removed /")
    for column in ("p25", "p50", "p75"):
        dist_report[column] = np.where(
            fraction_mask,
            dist_report[column].map(lambda value: f"{value * 100:.2f}%"),
            dist_report[column].map(lambda value: f"{value:.1f}"),
        )
    dist_table = dist_report.to_markdown(index=False)

    yearly_report = yearly_filters[
        ["run", "year", "calendar_return", "B_minus_A_return_gap"]
    ].copy()
    yearly_report["calendar_return"] = yearly_report["calendar_return"].map(
        lambda value: f"{value * 100:.2f}%"
    )
    yearly_report["B_minus_A_return_gap"] = yearly_report[
        "B_minus_A_return_gap"
    ].map(lambda value: f"{value * 100:+.2f} pp")
    yearly_table = yearly_report.to_markdown(index=False)

    cohort_pivot = cohort_all.pivot(
        index=["filter_rule", "horizon"],
        columns="cohort",
        values="mean_equal_weight_forward_return",
    ).reset_index()
    cohort_gap = cohort_all.drop_duplicates(["filter_rule", "horizon"])[
        ["filter_rule", "horizon", "removed_minus_kept_return"]
    ]
    cohort_report = cohort_pivot.merge(
        cohort_gap, on=["filter_rule", "horizon"], how="left"
    ).sort_values(["filter_rule", "horizon"])
    for column in ("removed", "kept_eligible"):
        cohort_report[column] = cohort_report[column].map(
            lambda value: f"{value * 100:.2f}%"
        )
    cohort_report["removed_minus_kept_return"] = cohort_report[
        "removed_minus_kept_return"
    ].map(lambda value: f"{value * 100:+.2f} pp")
    cohort_table = cohort_report.to_markdown(index=False)

    concentration_report = concentration_all[
        [
            "run",
            "mean_top10_weight_concentration",
            "p50_top10_weight_concentration",
            "volatility_annualized",
        ]
    ].copy()
    for column in concentration_report.columns[1:]:
        concentration_report[column] = concentration_report[column].map(
            lambda value: f"{value * 100:.2f}%"
        )
    concentration_table = concentration_report.to_markdown(index=False)

    ablation_report = ablation[
        [
            "upper_growth_clip",
            "paired_growth_inputs_changed_vs_35",
            "changed_input_fair_value_change_median",
            "changed_input_fair_value_change_pct_median",
            "portfolio_total_return_pct",
            "B_minus_A_total_return_gap_pct_points",
        ]
    ].copy()
    ablation_report["upper_growth_clip"] = ablation_report["upper_growth_clip"].map(
        lambda value: f"{value * 100:.0f}%"
    )
    ablation_report["changed_input_fair_value_change_median"] = ablation_report[
        "changed_input_fair_value_change_median"
    ].map(lambda value: f"${value:,.2f}")
    ablation_report["changed_input_fair_value_change_pct_median"] = ablation_report[
        "changed_input_fair_value_change_pct_median"
    ].map(lambda value: f"{value * 100:+.2f}%")
    ablation_report["portfolio_total_return_pct"] = ablation_report[
        "portfolio_total_return_pct"
    ].map(lambda value: f"{value:.2f}%")
    ablation_report["B_minus_A_total_return_gap_pct_points"] = ablation_report[
        "B_minus_A_total_return_gap_pct_points"
    ].map(lambda value: f"{value:+.2f} pp")
    ablation_table = ablation_report.to_markdown(index=False)

    funnel_report = pd.read_csv(OUTPUT_DIR / "dcf_funnel.csv")[
        ["stage", "status", "count"]
    ]
    funnel_table = funnel_report.to_markdown(index=False)

    report = f"""# Development report evidence

## Scope and safeguards

This package uses only the fixed development window, {core['start']} through
{core['end']}. Forward cohort endpoints are required to fall on or before
2021-12-31. No validation or test rows were accessed, no portfolio backtest was
run outside development, and the default 35% growth cap remains unchanged.

## Filter context and removal reconciliation

The per-call audit contains {len(abs_context)} weekly calls per rule. The
absolute run reconciles exactly to 24,829 growth observations, 8,584 eligible
observations, and 2,563 removals. The quantile run has
{int(quant_context['removed_count'].sum()):,} removals from the same contexts.
For absolute filtering, every removed name satisfies the sign convention
`mispricing < -0.20`, where negative means overvalued.

Across absolute removals,
{int(abs_context['removed_in_month_top10'].sum()):,} of
{int(abs_context['removed_count'].sum()):,} removed ticker-dates
({_format_pct(abs_context['removed_in_month_top10'].sum() / abs_context['removed_count'].sum())})
were members of that month's top-10 growth ranking. For quantile filtering the
corresponding fraction is
{_format_pct(quant_context['removed_in_month_top10'].sum() / quant_context['removed_count'].sum())}.
P25/P50/P75 values for total candidates, growth candidates, eligible names,
removals, and both removal-fraction denominators are:

{dist_table}

## Year and cohort attribution

`yearly_portfolio_attribution.csv` reports each calendar-year B-A return gap
for both rules. `filter_cohort_return_summary.csv` compares same-as-of removed
and kept-eligible equal-weight returns at 3, 6, and 12 months, both by selection
year and overall. A cohort is included only when its horizon ends inside
development, so periods are never mixed. These are descriptive conditional
cohorts, not causal estimates: removal status is determined by valuation and
the two groups need not have matched risk or sector composition.

Calendar-year attribution:

{yearly_table}

Overall point-in-time cohort comparison:

{cohort_table}

Removed names do not show a stable forward-return disadvantage: signs vary by
rule and horizon. That weakens a causal interpretation of the filter and is
consistent with both filtered portfolios lagging baseline A in development.
The absolute rule's total return is {filter_stats['absolute']['total_return_pct']:.2f}%
({filter_stats['absolute']['total_return_pct'] - baseline_stats['total_return_pct']:+.2f}
percentage points versus A); quantile returns
{filter_stats['quantile']['total_return_pct']:.2f}%
({filter_stats['quantile']['total_return_pct'] - baseline_stats['total_return_pct']:+.2f}
points). Thus the quantile rule's smaller removal set produces the smaller
development performance gap, but it still does not beat baseline A.

Actual post-trade weekly holdings produce top-10 weight concentration in
`weekly_portfolio_concentration.csv`; all-period and annual summaries, together
with realized annualized portfolio volatility, are in
`portfolio_concentration_volatility.csv`.

{concentration_table}

The filters barely change concentration or volatility; their main development
effect is which names occupy the ten-stock portfolio.

## Growth-cap ablation

Exactly {hit_35} of the 1,200 locked monthly ticker-dates hit the default 35%
upper growth clip. `growth_clip_ablation.csv` compares locked DCF outputs under
25%, 35%, and 45% caps, including paired fair-value changes and the absolute
filter portfolio's B-A return gap. This is an ablation only; the default remains
35%.

{ablation_table}

Lowering the cap to 25% changes 103 paired growth inputs and reduces their
median fair value by 26.39%. Raising it to 45% changes the 24 default-capped
inputs and raises their median fair value by 37.24%. Neither alternative
improves the absolute-filter portfolio's development B-A gap relative to the
35% default.

## DCF funnel

The locked funnel is verified directly from the 1,200-row diagnostic panel:
448 pre-engine exclusions and 752 calls reaching `calculate_dcf`. Engine calls
split into 626 valued and 126 rejected, an engine rejection rate of
{_format_pct(core['dcf_funnel']['engine_rejection_rate'], 1)}. All 12 possible
pre-engine status codes are shown in `dcf_funnel.csv`, including six zero-count
codes; this is why the table has more status rows than the observed panel.

{funnel_table}

## Assumption directions

The D&A non-negative floor affects 15 end-development aggregated ticker inputs.
It raises the modeled D&A proxy from a negative value to zero, increases the
D&A tax shield relative to that hypothetical negative comparator, and modestly
raises fair value whenever both DCFs are valid. Detailed paired results and
ineligible cases are retained in `da_floor_direction_sensitivity.csv`. Seven
of the 15 inputs are otherwise DCF-comparable; all seven increase, with a
median $0.21 per-share change (0.21%) in the fixed-terminal comparator.

The statutory tax fallback affects 40 aggregated ticker inputs, replacing a
non-positive raw rate with 21%. Its operating direction is higher tax and lower
fair value. Only 14 inputs are DCF-valid under both assumptions; at fixed WACC
all 14 fair values fall. In the model-native comparison, the tax effect on debt
cost lowers WACC enough to reverse the net direction for two tickers. Thus the
40 count describes changed inputs, not 40 observable paired fair-value changes.

## Reproducibility

Run from `source_code/`:

```bash
python -m analysis.dev_report_evidence core
python -m analysis.dev_report_evidence growth-clip --upper-clip 0.25
python -m analysis.dev_report_evidence growth-clip --upper-clip 0.35
python -m analysis.dev_report_evidence growth-clip --upper-clip 0.45
python -m analysis.dev_report_evidence finalize
```

The 25% and 45% growth-clip commands write separate directories and may be run
in parallel. The 35% command snapshots the existing exact-default development
run and locked 1,200-row panel. Every output is under this directory.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report)
    _write_json(
        OUTPUT_DIR / "reproducibility.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "period": "dev",
            "start": core["start"],
            "end": core["end"],
            "commands": [
                "python -m analysis.dev_report_evidence core",
                "python -m analysis.dev_report_evidence growth-clip --upper-clip 0.25",
                "python -m analysis.dev_report_evidence growth-clip --upper-clip 0.35",
                "python -m analysis.dev_report_evidence growth-clip --upper-clip 0.45",
                "python -m analysis.dev_report_evidence finalize",
            ],
            "default_configuration_changed": False,
            "validation_or_test_accessed": False,
        },
    )
    print(ablation.to_string(index=False))
    print(f"wrote {OUTPUT_DIR / 'REPORT.md'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("core")
    clip = subparsers.add_parser("growth-clip")
    clip.add_argument("--upper-clip", type=float, required=True, choices=(0.25, 0.35, 0.45))
    subparsers.add_parser("finalize")
    args = parser.parse_args(argv)
    if args.command == "core":
        run_core()
    elif args.command == "growth-clip":
        run_growth_clip(args.upper_clip)
    else:
        finalize()


if __name__ == "__main__":
    main()
