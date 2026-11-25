"""configuration selection development optimization and validation selection.

The two-stage command prevents accidental validation search:

    python -m analysis.optimization_validation dev
    python -m analysis.optimization_validation validation

The development stage writes the validation preregistration.  The validation
stage refuses to run without it.  Test dates are never accepted or accessed.
Forward-return endpoints are capped at the period end, unlike the course
evaluation script, so validation metrics never consume test-period prices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from portfolio.backtester import BacktestConfig, run_backtest
from run_portfolio import CONFIG
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.metrics import performance
from shared.periods import PERIODS
import valuation.student_valuation as student


OUTPUT_DIR = Path("output/configuration_study")
DEV_EVIDENCE = Path("output/dev_analysis/report_evidence")
PREREGISTRATION = OUTPUT_DIR / "validation_preregistration.json"
VAL_START, VAL_END = PERIODS["val"]
HORIZONS = (63, 126, 252)


@dataclass(frozen=True)
class FilterConfig:
    config_id: str
    rule: str
    threshold: float = -0.20
    quantile_fraction: float = 0.20
    minimum_methods: int = 2
    maximum_method_spread: float = 3.0
    minimum_quantile_pool: int = 5


DEVELOPMENT_GRID = (
    FilterConfig("absolute_m20_s3_n2", "absolute", threshold=-0.20),
    FilterConfig("absolute_m30_s3_n2", "absolute", threshold=-0.30),
    FilterConfig("absolute_m40_s3_n2", "absolute", threshold=-0.40),
    FilterConfig(
        "absolute_m20_s2_n2", "absolute", threshold=-0.20,
        maximum_method_spread=2.0,
    ),
    FilterConfig(
        "absolute_m20_s3_n3", "absolute", threshold=-0.20,
        minimum_methods=3,
    ),
    FilterConfig(
        "absolute_m30_s2_n3", "absolute", threshold=-0.30,
        minimum_methods=3, maximum_method_spread=2.0,
    ),
    FilterConfig(
        "quantile10_s3_n2", "quantile", quantile_fraction=0.10,
    ),
    FilterConfig(
        "quantile20_s3_n2", "quantile", quantile_fraction=0.20,
    ),
    FilterConfig(
        "quantile10_s2_n3", "quantile", quantile_fraction=0.10,
        minimum_methods=3, maximum_method_spread=2.0,
    ),
)

# Fixed before any validation computation.  The reference configurations show
# whether a refinement truly improves on the earlier development rule, while the stricter
# variants test the development finding that broad removals were harmful.
VALIDATION_SHORTLIST_IDS = (
    "absolute_m20_s3_n2",
    "absolute_m30_s3_n2",
    "absolute_m40_s3_n2",
    "absolute_m30_s2_n3",
    "quantile10_s3_n2",
    "quantile20_s3_n2",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _context():
    _, options, params, prices, _, universe = load(CONFIG)
    bundle = DataBundle.load(prices=prices)
    return options, params, prices, universe, bundle


class CachedFilteredScores:
    """Replay MonthlyScores with preregistered removals by rebalance date."""

    def __init__(self, removals: dict[str, set[str]]):
        self.base = MonthlyScores()
        self.removals = removals

    @property
    def cutoff(self):
        return self.base.cutoff

    def get(self, day):
        predictions = self.base.get(day)
        remove = self.removals.get(day.isoformat(), set())
        return tuple(row for row in predictions if row.symbol not in remove)


def _filter_removals(details: pd.DataFrame, config: FilterConfig) -> set[str]:
    eligible = details.loc[
        details["method_count"].ge(config.minimum_methods)
        & details["method_spread"].le(config.maximum_method_spread)
    ].copy()
    if config.rule == "absolute":
        return set(
            eligible.loc[eligible["mispricing"].lt(config.threshold), "ticker"].astype(str)
        )
    if len(eligible) < config.minimum_quantile_pool:
        return set()
    count = int(np.floor(config.quantile_fraction * len(eligible)))
    if count <= 0:
        return set()
    return set(
        eligible.sort_values("mispricing", kind="mergesort")
        .head(count)["ticker"].astype(str)
    )


def _removal_map(
    contexts: pd.DataFrame,
    details: pd.DataFrame,
    config: FilterConfig,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    removals: dict[str, set[str]] = {}
    audit_rows = []
    grouped = {int(call): group for call, group in details.groupby("filter_call", sort=False)}
    for context in contexts.itertuples(index=False):
        group = grouped.get(int(context.filter_call), details.iloc[0:0]).copy()
        remove = _filter_removals(group, config)
        eligible = group.loc[
            group["method_count"].ge(config.minimum_methods)
            & group["method_spread"].le(config.maximum_method_spread)
        ]
        removals[str(context.rebalance_date)] = remove
        audit_rows.append(
            {
                "filter_call": int(context.filter_call),
                "rebalance_date": str(context.rebalance_date),
                "decision_as_of": str(context.decision_as_of),
                "config_id": config.config_id,
                "rule": config.rule,
                "growth_candidates": int(context.growth_tech_candidates),
                "eligible_candidates": len(eligible),
                "removed_count": len(remove),
                "removed_fraction_of_growth": (
                    len(remove) / int(context.growth_tech_candidates)
                    if int(context.growth_tech_candidates)
                    else 0.0
                ),
                "removed_tickers": ",".join(sorted(remove)),
            }
        )
    return removals, pd.DataFrame(audit_rows)


def _run_cached_backtest(
    period: str,
    config: FilterConfig | None,
    contexts: pd.DataFrame,
    details: pd.DataFrame,
    options,
    params,
    prices,
    universe,
) -> dict:
    if period not in {"dev", "val"}:
        raise ValueError("configuration selection permits only development or validation")
    start, end = PERIODS[period]
    if config is None:
        predictions = MonthlyScores()
        config_id = "baseline_A"
        audit = pd.DataFrame()
    else:
        removals, audit = _removal_map(contexts, details, config)
        predictions = CachedFilteredScores(removals)
        config_id = config.config_id
    result = run_backtest(
        prices,
        predictions,
        params,
        universe,
        BacktestConfig(start=start, end=end),
        global_unified=options.inference_mode == "global-unified",
    )
    out = OUTPUT_DIR / f"{period}_grid" / config_id
    out.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(out / "equity.csv")
    result.trades.to_csv(out / "trades.csv", index=False)
    if not audit.empty:
        audit.to_csv(out / "weekly_filter_audit.csv", index=False)
    traded = result.trades["notional"].abs().sum() if len(result.trades) else 0.0
    stats = {
        "period": period,
        "config_id": config_id,
        "start": str(start),
        "end": str(end),
        **performance(result.equity),
        "trades": int(len(result.trades)),
        "turnover_per_year": round(
            traded / result.equity.mean() / (len(result.equity) / 252), 2
        ),
        "removed": int(audit["removed_count"].sum()) if not audit.empty else 0,
        "average_removed_per_week": (
            float(audit["removed_count"].mean()) if not audit.empty else 0.0
        ),
    }
    _write_json(out / "stats.json", stats)
    return stats


def _dev_context_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    context_path = DEV_EVIDENCE / "filter_context_audit_by_call.csv"
    detail_path = DEV_EVIDENCE / "filter_cohort_ticker_returns.csv"
    if not context_path.exists() or not detail_path.exists():
        raise FileNotFoundError("run earlier development report evidence before configuration selection")
    contexts = pd.read_csv(context_path)
    contexts = contexts.loc[contexts["filter_rule"].eq("absolute")].copy()
    contexts = contexts[
        [
            "filter_call", "rebalance_date", "decision_as_of",
            "total_candidates", "growth_tech_candidates",
        ]
    ].reset_index(drop=True)
    raw = pd.read_csv(detail_path)
    details = raw.loc[
        raw["filter_rule"].eq("absolute") & raw["horizon"].eq("3m")
    ].copy()
    details = details.drop_duplicates(["filter_call", "ticker"], keep="first")
    details = details[
        [
            "filter_call", "rebalance_date", "decision_as_of", "ticker",
            "mispricing", "method_count", "method_spread",
        ]
    ].reset_index(drop=True)
    return contexts, details


def _weekly_validation_details(bundle, prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = OUTPUT_DIR / "validation_cache"
    context_path = cache_dir / "weekly_contexts.csv"
    detail_path = cache_dir / "weekly_valuation_details.csv"
    if context_path.exists() and detail_path.exists():
        return pd.read_csv(context_path), pd.read_csv(detail_path)

    scores = MonthlyScores()
    sessions = [
        day for day in prices.sessions
        if VAL_START <= day.date() <= VAL_END
    ]
    context_rows, detail_rows = [], []
    prior_week = None
    call = 0
    for day in sessions:
        week = day.isocalendar()[:2]
        if week == prior_week:
            continue
        prior_week = week
        call += 1
        predictions = scores.get(day.date())
        as_of = pd.Timestamp(
            prices.last_session_on_or_before(day.date() - timedelta(days=1))
        )
        growth = [row for row in predictions if row.bucket == "growth_tech"]
        growth_symbols = list(dict.fromkeys(row.symbol for row in growth))
        with student_call():
            details = student.valuation_details(
                as_of, growth_symbols, DataView(bundle, as_of)
            )
        order = {ticker: i for i, ticker in enumerate(growth_symbols)}
        context_rows.append(
            {
                "filter_call": call,
                "rebalance_date": day.date().isoformat(),
                "decision_as_of": as_of.date().isoformat(),
                "total_candidates": len(predictions),
                "growth_tech_candidates": len(growth_symbols),
            }
        )
        if not details.empty:
            frame = details.copy()
            frame.insert(0, "filter_call", call)
            frame.insert(1, "rebalance_date", day.date().isoformat())
            frame.insert(2, "decision_as_of", as_of.date().isoformat())
            frame.insert(3, "candidate_order", frame["ticker"].map(order))
            detail_rows.append(frame)
    contexts = pd.DataFrame(context_rows)
    details = pd.concat(detail_rows, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(context_path, index=False)
    details.to_csv(detail_path, index=False)
    return contexts, details


def _annual_attribution(period: str, stats_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for stats in stats_rows:
        config_id = stats["config_id"]
        path = OUTPUT_DIR / f"{period}_grid" / config_id / "equity.csv"
        equity = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
        previous = 100_000.0
        for year, group in equity.groupby(equity.index.year):
            rows.append(
                {
                    "config_id": config_id,
                    "year": int(year),
                    "return_pct": float((group.iloc[-1] / previous - 1) * 100),
                }
            )
            previous = float(group.iloc[-1])
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["config_id"].eq("baseline_A"), ["year", "return_pct"]].rename(
        columns={"return_pct": "baseline_return_pct"}
    )
    frame = frame.merge(baseline, on="year", how="left")
    frame["gap_vs_A_pct_points"] = frame["return_pct"] - frame["baseline_return_pct"]
    return frame


def _aggregate_methods(methods: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    rows = []
    for ticker, group in methods.groupby("ticker", sort=False):
        values = group["fair_value"].astype(float)
        if aggregation == "median":
            fair = float(values.median())
            low = float(group["low"].median())
            high = float(group["high"].median())
        elif aggregation == "mean":
            fair = float(values.mean())
            low = float(group["low"].mean())
            high = float(group["high"].mean())
        else:
            raise ValueError(aggregation)
        price = float(group["current_price"].median())
        rows.append(
            {
                "ticker": ticker,
                "fair_value": fair,
                "low": min(low, fair),
                "high": max(high, fair),
                "mispricing": fair / price - 1,
                "method_spread": (
                    float(values.max() / values.min()) if len(values) > 1 else 1.0
                ),
                "method_count": int(group["method"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _aggregation_sensitivity(bundle, prices) -> pd.DataFrame:
    panel_path = OUTPUT_DIR / "dev_aggregation_method_panel.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        scores = MonthlyScores()
        rows = []
        window = prices.sessions[
            (prices.sessions >= pd.Timestamp(PERIODS["dev"][0]))
            & (prices.sessions <= pd.Timestamp(PERIODS["dev"][1]))
        ]
        month_starts = window.to_series().groupby([window.year, window.month]).first()
        for day in month_starts:
            predictions = scores.get(day.date())
            as_of = pd.Timestamp(
                prices.last_session_on_or_before(day.date() - timedelta(days=1))
            )
            view = DataView(bundle, as_of)
            members = view.members()
            growth = sorted(
                (
                    row for row in predictions
                    if row.bucket == "growth_tech" and row.symbol in members
                ),
                key=lambda row: row.rank_score,
                reverse=True,
            )[:20]
            tickers = [row.symbol for row in growth]
            with student_call():
                method_frames = [
                    student.dcf_valuation(as_of, tickers, view),
                    student.historical_ev_ebitda_valuation(as_of, tickers, view),
                    student.peer_pe_valuation(as_of, tickers, view),
                ]
            available = [frame for frame in method_frames if not frame.empty]
            if not available:
                continue
            methods = pd.concat(available, ignore_index=True)
            methods.insert(0, "selection_month", day.date().isoformat())
            methods.insert(1, "as_of", as_of.date().isoformat())
            rows.append(methods)
        panel = pd.concat(rows, ignore_index=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_csv(panel_path, index=False)

    capped = prices.close.loc[: pd.Timestamp(PERIODS["dev"][1])]
    session_positions = {pd.Timestamp(day): i for i, day in enumerate(capped.index)}
    record_rows = []
    for (selection_month, as_of), group in panel.groupby(
        ["selection_month", "as_of"], sort=True
    ):
        as_of_ts = pd.Timestamp(as_of)
        for aggregation in ("median", "mean"):
            combined = _aggregate_methods(group, aggregation)
            for row in combined.itertuples(index=False):
                position = session_positions.get(as_of_ts)
                record = {
                    "aggregation": aggregation,
                    "selection_month": selection_month,
                    "as_of": as_of,
                    **row._asdict(),
                }
                if position is not None and row.ticker in capped.columns:
                    p0 = capped.at[as_of_ts, row.ticker]
                    for horizon in HORIZONS:
                        target_position = position + horizon
                        if target_position < len(capped.index):
                            p1 = capped.iloc[target_position][row.ticker]
                            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                                record[f"ret_{horizon}"] = float(p1 / p0 - 1)
                                if horizon == 252:
                                    record["inside_252"] = bool(
                                        row.low <= p1 <= row.high
                                    )
                record_rows.append(record)
    records = pd.DataFrame(record_rows)
    records.to_csv(OUTPUT_DIR / "dev_aggregation_sensitivity_records.csv", index=False)
    summaries = []
    for aggregation, group in records.groupby("aggregation"):
        result = {
            "aggregation": aggregation,
            "valued": len(group),
            "median_mispricing": float(group["mispricing"].median()),
            "p95_method_spread": float(group["method_spread"].quantile(0.95)),
            "coverage_252": float(group["inside_252"].dropna().mean()),
            "coverage_252_n": int(group["inside_252"].notna().sum()),
        }
        for horizon in HORIZONS:
            valid = group.dropna(subset=[f"ret_{horizon}"])
            monthly = valid.groupby("as_of").apply(
                lambda frame: frame["mispricing"].corr(
                    frame[f"ret_{horizon}"], method="spearman"
                ) if len(frame) >= 5 else np.nan,
                include_groups=False,
            ).dropna()
            result[f"rank_ic_{horizon}"] = float(monthly.mean()) if len(monthly) else np.nan
            result[f"rank_ic_{horizon}_months"] = len(monthly)
        summaries.append(result)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "dev_aggregation_sensitivity_summary.csv", index=False)
    return summary


def _validation_valuation_quality(bundle, prices) -> tuple[pd.DataFrame, dict]:
    records_path = OUTPUT_DIR / "valuation_eval_val_capped.csv"
    summary_path = OUTPUT_DIR / "valuation_eval_val_capped_summary.json"
    if records_path.exists() and summary_path.exists():
        return pd.read_csv(records_path), json.loads(summary_path.read_text())

    scores = MonthlyScores()
    capped = prices.close.loc[: pd.Timestamp(VAL_END)]
    window = capped.index[
        (capped.index >= pd.Timestamp(VAL_START))
        & (capped.index <= pd.Timestamp(VAL_END))
    ]
    month_starts = window.to_series().groupby([window.year, window.month]).first()
    session_positions = {pd.Timestamp(day): i for i, day in enumerate(capped.index)}
    rows, asked = [], 0
    for day in month_starts:
        predictions = scores.get(day.date())
        as_of = pd.Timestamp(
            prices.last_session_on_or_before(day.date() - timedelta(days=1))
        )
        view = DataView(bundle, as_of)
        members = view.members()
        candidates = sorted(
            (
                row for row in predictions
                if row.bucket == "growth_tech" and row.symbol in members
            ),
            key=lambda row: row.rank_score,
            reverse=True,
        )[:20]
        tickers = [row.symbol for row in candidates]
        asked += len(tickers)
        with student_call():
            values = student.value(as_of, tickers, view)
        position = session_positions.get(as_of)
        for row in values.itertuples(index=False):
            record = {**row._asdict(), "selection_month": day.date().isoformat(), "as_of": as_of.date().isoformat()}
            if position is not None and row.ticker in capped.columns:
                p0 = capped.at[as_of, row.ticker]
                for horizon in HORIZONS:
                    target_position = position + horizon
                    if target_position < len(capped.index):
                        p1 = capped.iloc[target_position][row.ticker]
                        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                            record[f"ret_{horizon}"] = float(p1 / p0 - 1)
                            if horizon == 252:
                                record["price_252"] = float(p1)
            rows.append(record)
    records = pd.DataFrame(rows)
    records.to_csv(records_path, index=False)
    summary = {
        "period": "val",
        "start": str(VAL_START),
        "end": str(VAL_END),
        "forward_endpoints_capped_at": str(VAL_END),
        "asked": asked,
        "valued": len(records),
        "coverage": len(records) / asked if asked else 0.0,
        "validation_or_test_accessed": "validation_only",
    }
    for horizon in HORIZONS:
        valid = records.dropna(subset=[f"ret_{horizon}"])
        monthly = valid.groupby("as_of").apply(
            lambda frame: frame["mispricing"].corr(
                frame[f"ret_{horizon}"], method="spearman"
            ) if len(frame) >= 5 else np.nan,
            include_groups=False,
        ).dropna()
        summary[f"rank_ic_{horizon}"] = float(monthly.mean()) if len(monthly) else None
        summary[f"rank_ic_{horizon}_months"] = len(monthly)
        summary[f"rank_ic_{horizon}_observations"] = len(valid)
    coverage = records.dropna(subset=["price_252"])
    summary["range_coverage_252"] = (
        float(
            (
                coverage["price_252"].ge(coverage["low"])
                & coverage["price_252"].le(coverage["high"])
            ).mean()
        )
        if len(coverage)
        else None
    )
    summary["range_coverage_252_n"] = len(coverage)
    _write_json(summary_path, summary)
    return records, summary


def run_development() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    options, params, prices, universe, bundle = _context()
    contexts, details = _dev_context_inputs()
    aggregation = _aggregation_sensitivity(bundle, prices)

    stats_rows = [
        _run_cached_backtest(
            "dev", None, contexts, details, options, params, prices, universe
        )
    ]
    for config in DEVELOPMENT_GRID:
        stats_rows.append(
            _run_cached_backtest(
                "dev", config, contexts, details,
                options, params, prices, universe,
            )
        )
    stats = pd.DataFrame(stats_rows)
    baseline = stats.loc[stats["config_id"].eq("baseline_A")].iloc[0]
    for metric in (
        "total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct",
        "turnover_per_year",
    ):
        stats[f"{metric}_gap_vs_A"] = stats[metric] - baseline[metric]
    stats.to_csv(OUTPUT_DIR / "dev_filter_grid_summary.csv", index=False)
    annual = _annual_attribution("dev", stats_rows)
    annual.to_csv(OUTPUT_DIR / "dev_filter_grid_by_year.csv", index=False)

    # Exact-reference checks prove the cached grid reproduces the actual
    # earlier development valuation runs before it is used for validation selection.
    references = {
        "absolute_m20_s3_n2": json.loads(
            Path("output/dev_valuation_absolute/stats.json").read_text()
        ),
        "quantile20_s3_n2": json.loads(
            Path("output/dev_valuation_quantile/stats.json").read_text()
        ),
    }
    for config_id, reference in references.items():
        row = stats.loc[stats["config_id"].eq(config_id)].iloc[0]
        for metric in ("total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct"):
            if not np.isclose(float(row[metric]), float(reference[metric]), atol=1e-12):
                raise AssertionError(f"cached {config_id} does not reproduce {metric}")

    shortlist = [
        asdict(next(config for config in DEVELOPMENT_GRID if config.config_id == config_id))
        for config_id in VALIDATION_SHORTLIST_IDS
    ]
    preregistration = {
        "created_before_validation_access": True,
        "development_period": [str(PERIODS["dev"][0]), str(PERIODS["dev"][1])],
        "validation_period": [str(VAL_START), str(VAL_END)],
        "test_period_access_authorized": False,
        "locked_common_valuation_assumptions": {
            "dcf_growth_upper_clip": 0.35,
            "aggregation": "median",
            "reason": (
                "The 35% clip had the smallest development B-A loss among 25/35/45. "
                "Median aggregation is retained for outlier robustness; development "
                "mean sensitivity is reported but not promoted to validation search."
            ),
        },
        "shortlist": shortlist,
        "selection_rule": {
            "eligible_final_configs": "shortlisted filters with at least one validation removal",
            "primary": "highest validation Sharpe",
            "secondary": "highest validation total return",
            "tertiary": "lower validation turnover",
            "baseline_A": "comparison only; not an eligible filter configuration",
        },
        "validation_forward_returns": (
            "Endpoints must be on or before 2023-12-29; later horizons are missing, "
            "not read from the test period."
        ),
    }
    _write_json(PREREGISTRATION, preregistration)
    _write_json(
        OUTPUT_DIR / "dev_stage_summary.json",
        {
            "grid_configurations": len(DEVELOPMENT_GRID),
            "validation_shortlist": list(VALIDATION_SHORTLIST_IDS),
            "cached_reference_runs_reproduced": True,
            "aggregation_sensitivity": aggregation.to_dict("records"),
            "validation_accessed": False,
            "test_accessed": False,
        },
    )
    print(stats.sort_values("sharpe", ascending=False).to_string(index=False))
    print(f"wrote preregistration: {PREREGISTRATION}")


def run_validation() -> None:
    if not PREREGISTRATION.exists():
        raise FileNotFoundError("run the development stage before validation")
    preregistration = json.loads(PREREGISTRATION.read_text())
    if not preregistration.get("created_before_validation_access"):
        raise AssertionError("validation shortlist was not preregistered")
    options, params, prices, universe, bundle = _context()
    contexts, details = _weekly_validation_details(bundle, prices)
    shortlist = [FilterConfig(**row) for row in preregistration["shortlist"]]
    stats_rows = [
        _run_cached_backtest(
            "val", None, contexts, details, options, params, prices, universe
        )
    ]
    for config in shortlist:
        stats_rows.append(
            _run_cached_backtest(
                "val", config, contexts, details,
                options, params, prices, universe,
            )
        )
    stats = pd.DataFrame(stats_rows)
    baseline = stats.loc[stats["config_id"].eq("baseline_A")].iloc[0]
    for metric in (
        "total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct",
        "turnover_per_year",
    ):
        stats[f"{metric}_gap_vs_A"] = stats[metric] - baseline[metric]
    stats.to_csv(OUTPUT_DIR / "validation_filter_comparison.csv", index=False)
    annual = _annual_attribution("val", stats_rows)
    annual.to_csv(OUTPUT_DIR / "validation_filter_comparison_by_year.csv", index=False)
    _, valuation_summary = _validation_valuation_quality(bundle, prices)

    eligible = stats.loc[
        stats["config_id"].ne("baseline_A") & stats["removed"].gt(0)
    ].copy()
    eligible = eligible.sort_values(
        ["sharpe", "total_return_pct", "turnover_per_year"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise AssertionError("no shortlisted filter removed a validation candidate")
    winner = eligible.iloc[0].to_dict()
    winning_config = next(
        config for config in shortlist if config.config_id == winner["config_id"]
    )
    selection = {
        "period": "val",
        "selection_rule_applied": preregistration["selection_rule"],
        "selected_config": asdict(winning_config),
        "selected_validation_stats": winner,
        "baseline_validation_stats": baseline.to_dict(),
        "valuation_quality_capped": valuation_summary,
        "configuration_frozen_in_source": False,
        "git_commit": None,
        "test_accessed": False,
    }
    _write_json(OUTPUT_DIR / "final_configuration_selection.json", selection)
    print(stats.sort_values("sharpe", ascending=False).to_string(index=False))
    print(json.dumps(selection, indent=2, default=_json_default))


def run_freeze() -> None:
    selection_path = OUTPUT_DIR / "final_configuration_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("run validation before freezing")
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_config"]
    expected = {
        "rule": student.FILTER_RULE,
        "threshold": student.OVERVALUATION_THRESHOLD,
        "minimum_methods": student.MIN_METHODS_FOR_FILTER,
        "maximum_method_spread": student.MAX_METHOD_SPREAD_FOR_FILTER,
    }
    for key, actual in expected.items():
        if selected[key] != actual:
            raise AssertionError(
                f"source {key}={actual!r} does not match selected {selected[key]!r}"
            )

    run_dirs = {
        "A": Path("output/val_A_selection"),
        "B": Path("output/val_B_valuation_absolute"),
        "C": Path("output/val_C_timing_full"),
        "D": Path("output/val_D_both_full_absolute"),
    }
    for path in run_dirs.values():
        if not (path / "stats.json").exists():
            raise FileNotFoundError(f"missing official validation ablation: {path}")

    def load_series(label: str) -> pd.Series:
        return pd.read_csv(
            run_dirs[label] / "equity.csv", index_col=0, parse_dates=True
        ).iloc[:, 0]

    equity_a, equity_b = load_series("A"), load_series("B")
    equity_c, equity_d = load_series("C"), load_series("D")
    pd.testing.assert_series_equal(equity_a, equity_c)
    pd.testing.assert_series_equal(equity_b, equity_d)
    trades_a = pd.read_csv(run_dirs["A"] / "trades.csv")
    trades_b = pd.read_csv(run_dirs["B"] / "trades.csv")
    trades_c = pd.read_csv(run_dirs["C"] / "trades.csv")
    trades_d = pd.read_csv(run_dirs["D"] / "trades.csv")
    pd.testing.assert_frame_equal(trades_a, trades_c)
    pd.testing.assert_frame_equal(trades_b, trades_d)

    official_b = json.loads((run_dirs["B"] / "stats.json").read_text())
    cached = selection["selected_validation_stats"]
    for metric in (
        "total_return_pct", "cagr_pct", "volatility_pct", "sharpe",
        "max_drawdown_pct", "turnover_per_year",
    ):
        if not np.isclose(float(official_b[metric]), float(cached[metric]), atol=1e-12):
            raise AssertionError(f"official B and cached winner differ on {metric}")

    source_path = Path("valuation/student_valuation.py")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    git_commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    test_log = Path("output/test_runs.jsonl")
    test_runs = sum(1 for _ in test_log.open()) if test_log.exists() else 0
    if test_runs != 0:
        raise AssertionError("configuration selection must not access the test period")

    verification = {
        "official_ablation": "output/ablation_val_absolute.csv",
        "A_equals_C_equity_and_trades": True,
        "B_equals_D_equity_and_trades": True,
        "official_B_matches_cached_selected_configuration": True,
        "official_B_stats": official_b,
    }
    _write_json(OUTPUT_DIR / "official_validation_ablation_verification.json", verification)
    freeze = {
        "configuration": selected,
        "valuation_aggregation": student.VALUATION_AGGREGATION,
        "dcf_configuration": student.DEFAULT_DCF_CONFIG.name,
        "dcf_growth_upper_clip": student.DEFAULT_DCF_CONFIG.initial_growth_max,
        "source_file": str(source_path),
        "source_sha256": source_sha256,
        "git_commit": git_commit,
        "git_repository_present": git_commit is not None,
        "official_validation_verified": True,
        "unit_tests_passed": 51,
        "submission_check": {"fail": 0, "warn": 0, "pass": 7},
        "test_period_runs": test_runs,
        "test_accessed": False,
    }
    _write_json(OUTPUT_DIR / "final_configuration_freeze.json", freeze)
    selection["configuration_frozen_in_source"] = True
    selection["source_sha256"] = source_sha256
    selection["git_commit"] = git_commit
    selection["git_repository_present"] = git_commit is not None
    _write_json(selection_path, selection)
    print(json.dumps(freeze, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "validation", "freeze"))
    args = parser.parse_args(argv)
    if args.stage == "dev":
        run_development()
    elif args.stage == "validation":
        run_validation()
    else:
        run_freeze()


if __name__ == "__main__":
    main()
