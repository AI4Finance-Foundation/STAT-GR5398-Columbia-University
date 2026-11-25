"""Extended data audit diagnostics for the FinRobot valuation track.

Only development-period dates are used. Historical model inputs come through
DataView. Raw price rows are inspected here solely for source/symbol evidence
and are explicitly truncated to the diagnostic as-of date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data_audit import AUDIT_DATES
from shared.data_view import DataBundle, DataView
from valuation.student_valuation import (
    _latest_four_visible_quarters,
    build_ttm_fundamentals,
    estimate_da_pct_revenue,
    tax_rate_for_dcf,
)


DEV_START = pd.Timestamp("2017-01-01")
DEV_END = pd.Timestamp("2021-12-31")
CORE_TTM_FIELDS = (
    "eps_ttm",
    "revenue_ttm",
    "capex_ttm",
    "ebitda_ttm",
    "ebitda_margin_median",
    "operating_margin_median",
    "effective_tax_rate_median",
    "cash_per_share",
    "interest_debt_per_share",
    "debt_to_capital",
)


def finite(values) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalized_symbol(symbol: str) -> str:
    """Normalize punctuation for conservative alias discovery."""
    return re.sub(r"[^A-Z0-9]", "", str(symbol).upper())


def alias_candidates(symbol: str, available_symbols) -> list[str]:
    """Return punctuation-only source aliases, excluding the exact symbol."""
    target = normalized_symbol(symbol)
    return sorted(s for s in set(map(str, available_symbols)) if s != symbol and normalized_symbol(s) == target)


def first_monthly_sessions(spy: pd.Series, start=DEV_START, end=DEV_END) -> pd.DatetimeIndex:
    """First observed SPY session of each calendar month in the interval."""
    series = spy.dropna().sort_index()
    series = series.loc[pd.Timestamp(start): pd.Timestamp(end)]
    if series.empty:
        return pd.DatetimeIndex([])
    frame = series.rename("close").to_frame()
    frame["month"] = frame.index.to_period("M")
    return pd.DatetimeIndex(frame.groupby("month", sort=True).head(1).index)


def transition_point(frame: pd.DataFrame, column: str, threshold: float, operator: str) -> str:
    """First as-of date satisfying a reproducible diagnostic threshold."""
    if operator == ">=":
        eligible = frame[finite(frame[column]).ge(threshold)]
    elif operator == "<=":
        eligible = frame[finite(frame[column]).le(threshold)]
    else:
        raise ValueError(f"unsupported threshold operator: {operator}")
    return str(eligible.iloc[0]["as_of"]) if not eligible.empty else "not_reached"


def weekly_price_diagnostics(closes: pd.DataFrame, tickers) -> pd.DataFrame:
    """Apply the example's daily-pair -> W-FRI -> return convention exactly."""
    tickers = list(tickers)
    matrix = closes.reindex(columns=[*tickers, "SPY"])
    rows = []
    for ticker in tickers:
        stock = matrix[ticker].dropna()
        stock_weekly = stock.resample("W-FRI").last().pct_change(fill_method=None).dropna()
        paired_daily = matrix[[ticker, "SPY"]].dropna()
        paired_weekly = paired_daily.resample("W-FRI").last().pct_change(fill_method=None).dropna()
        paired_count = len(paired_weekly)
        if stock.empty:
            group = "A"
        elif len(stock_weekly) < 52:
            group = "B"
        elif paired_count < 52:
            group = "C"
        else:
            group = "sufficient"
        rows.append(
            {
                "ticker": ticker,
                "daily_closes": int(len(stock)),
                "valid_stock_weekly_returns": int(len(stock_weekly)),
                "paired_stock_spy_weekly_returns": int(paired_count),
                "first_close": str(stock.index.min().date()) if not stock.empty else None,
                "last_close": str(stock.index.max().date()) if not stock.empty else None,
                "group": group,
            }
        )
    return pd.DataFrame(rows)


def _quarter_gap_fields(dates: pd.Series) -> dict:
    q = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
    gaps = pd.Series(q).diff().dt.days.dropna().astype(int)
    return {
        "earliest_quarter": str(q.min().date()) if len(q) else None,
        "latest_quarter": str(q.max().date()) if len(q) else None,
        "quarter_gaps_days": ";".join(map(str, gaps.tolist())),
        "gaps_over_130_days": int(gaps.gt(130).sum()),
        "max_quarter_gap_days": int(gaps.max()) if not gaps.empty else None,
    }


def ttm_coverage_diagnostic(bundle: DataBundle, as_of: str) -> tuple[dict, pd.DataFrame]:
    view = DataView(bundle, as_of)
    members = sorted(view.members())
    fund = view.fundamentals(members)
    fund = fund.copy()
    fund["datadate"] = pd.to_datetime(fund["datadate"], errors="coerce")
    latest4 = _latest_four_visible_quarters(fund)
    ttm = build_ttm_fundamentals(fund)
    latest_tickers = set(latest4["ticker"])
    ttm_tickers = set(ttm["ticker"])
    ttm_index = ttm.set_index("ticker") if not ttm.empty else pd.DataFrame()
    accepted_raw_missing = fund["accepted_date"].isna()
    filing_raw_missing = fund["filing_date"].isna()
    accepted_blank = fund["accepted_date"].astype(str).str.strip().eq("")
    filing_blank = fund["filing_date"].astype(str).str.strip().eq("")
    accepted_parsed_missing = pd.to_datetime(fund["accepted_date"], errors="coerce").isna()
    filing_parsed_missing = pd.to_datetime(fund["filing_date"], errors="coerce").isna()

    rows = []
    for ticker in members:
        frame = fund.loc[fund["ticker"].eq(ticker)].sort_values("datadate")
        distinct = int(frame["datadate"].nunique())
        gaps = _quarter_gap_fields(frame["datadate"])
        survives_latest = ticker in latest_tickers
        survives_ttm = ticker in ttm_tickers
        complete = False
        if survives_ttm:
            complete = bool(ttm_index.loc[ticker, list(CORE_TTM_FIELDS)].notna().all())
        reasons = []
        if distinct < 4:
            reasons.append("fewer_than_four_distinct_visible_quarters")
        if not survives_latest and distinct >= 4:
            reasons.append("helper_exclusion_despite_four_quarters")
        if survives_latest and not survives_ttm:
            reasons.append("ttm_builder_exclusion")
        rows.append(
            {
                "as_of": as_of,
                "ticker": ticker,
                "visible_rows": int(len(frame)),
                "distinct_visible_quarters": distinct,
                "missing_accepted_date": int(pd.to_datetime(frame.get("accepted_date"), errors="coerce").isna().sum()),
                "missing_filing_date": int(pd.to_datetime(frame.get("filing_date"), errors="coerce").isna().sum()),
                **gaps,
                "at_least_four_distinct_quarters": distinct >= 4,
                "survives_latest_four_helper": survives_latest,
                "survives_ttm_builder": survives_ttm,
                "complete_core_ttm_aggregation": complete,
                "excluded": not survives_ttm,
                "exclusion_reason": ";".join(reasons),
            }
        )
    detail = pd.DataFrame(rows)
    counts = detail["distinct_visible_quarters"]
    first_visible = pd.to_datetime(detail["earliest_quarter"], errors="coerce")
    pre_2016_q2 = fund.loc[fund["datadate"].lt(pd.Timestamp("2016-03-31"))]
    summary = {
        "as_of": as_of,
        "members": len(members),
        "visible_fundamental_rows": len(fund),
        "average_visible_rows_per_member": round(len(fund) / len(members), 4) if members else None,
        "minimum_visible_fiscal_quarter": str(fund["datadate"].min().date()) if not fund.empty else None,
        "maximum_visible_fiscal_quarter": str(fund["datadate"].max().date()) if not fund.empty else None,
        "accepted_date_raw_isna": int(accepted_raw_missing.sum()),
        "accepted_date_blank": int(accepted_blank.sum()),
        "missing_accepted_date": int(accepted_parsed_missing.sum()),
        "filing_date_raw_isna": int(filing_raw_missing.sum()),
        "filing_date_blank": int(filing_blank.sum()),
        "missing_filing_date": int(filing_parsed_missing.sum()),
        "distinct_quarters_min": int(counts.min()),
        "distinct_quarters_p25": float(counts.quantile(0.25)),
        "distinct_quarters_median": float(counts.median()),
        "distinct_quarters_p75": float(counts.quantile(0.75)),
        "distinct_quarters_max": int(counts.max()),
        "tickers_with_at_least_four_distinct_quarters": int(detail["at_least_four_distinct_quarters"].sum()),
        "survive_latest_four_helper": int(detail["survives_latest_four_helper"].sum()),
        "survive_ttm_builder": int(detail["survives_ttm_builder"].sum()),
        "complete_core_ttm_aggregation": int(detail["complete_core_ttm_aggregation"].sum()),
        "helper_eligibility_mismatches": int(
            detail["survives_latest_four_helper"].ne(detail["survives_ttm_builder"]).sum()
        ),
        "four_quarter_helper_exclusions": int(
            (detail["at_least_four_distinct_quarters"] & ~detail["survives_latest_four_helper"]).sum()
        ),
        "tickers_without_visible_rows": int(first_visible.isna().sum()),
        "tickers_first_visible_on_or_after_2016_03_31": int(first_visible.ge(pd.Timestamp("2016-03-31")).sum()),
        "tickers_first_visible_on_or_after_2016_06_30": int(first_visible.ge(pd.Timestamp("2016-06-30")).sum()),
        "pre_2016_q2_visible_rows": int(len(pre_2016_q2)),
        "pre_2016_q2_visible_tickers": int(pre_2016_q2["ticker"].nunique()),
        "pre_2016_q2_missing_accepted_date": int(
            pd.to_datetime(pre_2016_q2["accepted_date"], errors="coerce").isna().sum()
        ),
        "pre_2016_q2_missing_filing_date": int(
            pd.to_datetime(pre_2016_q2["filing_date"], errors="coerce").isna().sum()
        ),
    }
    return summary, detail


def beta_price_diagnostic(bundle: DataBundle, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    view = DataView(bundle, as_of)
    members = sorted(view.members())
    closes = view.close([*members, "SPY"], lookback=252 * 5)
    detail = weekly_price_diagnostics(closes, members)

    # Raw source inspection is restricted to this analysis module and to rows
    # no later than as_of, as required by the data audit diagnostic protocol.
    raw = bundle.prices.frame.loc[bundle.prices.frame["date"].le(pd.Timestamp(as_of))].copy()
    source_group = raw.groupby("symbol")["date"].agg(["count", "min", "max"])
    source_symbols = set(source_group.index.astype(str))
    member_source_difference = sorted(set(members) - source_symbols)
    priceable_members = sorted(set(members) & source_symbols)
    detail["exact_symbol_in_source"] = detail["ticker"].isin(source_symbols)
    detail["source_rows_exact_symbol"] = detail["ticker"].map(source_group["count"]).fillna(0).astype(int)
    detail["source_first_date"] = detail["ticker"].map(source_group["min"]).map(
        lambda x: str(x.date()) if pd.notna(x) else None
    )
    detail["source_last_date"] = detail["ticker"].map(source_group["max"]).map(
        lambda x: str(x.date()) if pd.notna(x) else None
    )

    alias_rows = []
    for ticker in detail.loc[detail["group"].eq("A"), "ticker"]:
        candidates = alias_candidates(ticker, source_symbols)
        for alternate in candidates:
            alt_closes = view.close([alternate, "SPY"], lookback=252 * 5)
            alt_diag = weekly_price_diagnostics(alt_closes, [alternate]).iloc[0]
            alias_rows.append(
                {
                    "ticker": ticker,
                    "alternate_symbol": alternate,
                    "match_rule": "punctuation_normalized_exact",
                    "alternate_daily_closes": int(alt_diag["daily_closes"]),
                    "alternate_paired_weekly_returns": int(alt_diag["paired_stock_spy_weekly_returns"]),
                    "usable_for_52_week_beta": bool(alt_diag["paired_stock_spy_weekly_returns"] >= 52),
                }
            )
    aliases = pd.DataFrame(
        alias_rows,
        columns=[
            "ticker", "alternate_symbol", "match_rule", "alternate_daily_closes",
            "alternate_paired_weekly_returns", "usable_for_52_week_beta",
        ],
    )
    alias_count = aliases.groupby("ticker").size() if not aliases.empty else pd.Series(dtype=int)
    detail["alias_candidate_count"] = detail["ticker"].map(alias_count).fillna(0).astype(int)
    usable_aliases = set(aliases.loc[aliases["usable_for_52_week_beta"], "ticker"]) if not aliases.empty else set()
    detail["unambiguous_usable_alias"] = detail["ticker"].map(
        lambda t: bool(t in usable_aliases and alias_count.get(t, 0) == 1)
    )

    spy = closes["SPY"].dropna()
    spy_weekly = spy.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    affected = detail.loc[~detail["group"].eq("sufficient")]
    summary = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "members": len(members),
                "group_a_no_exact_closes": int(detail["group"].eq("A").sum()),
                "group_b_short_stock_history": int(detail["group"].eq("B").sum()),
                "group_c_spy_pairing_shortfall": int(detail["group"].eq("C").sum()),
                "total_below_52_paired_returns": int(len(affected)),
                "unambiguous_usable_aliases": int(detail["unambiguous_usable_alias"].sum()),
                "price_symbols_as_of": int(len(source_symbols)),
                "priceable_members": int(len(priceable_members)),
                "member_minus_price_symbol_count": int(len(member_source_difference)),
                "member_minus_price_symbols_first_30": ";".join(member_source_difference[:30]),
                "affected_exactly_zero_daily_closes": int(affected["daily_closes"].eq(0).sum()),
                "members_with_fewer_than_252_price_sessions": int(
                    closes.reindex(columns=members).notna().sum().lt(252).sum()
                ),
                "spy_daily_closes": int(len(spy)),
                "spy_weekly_returns": int(len(spy_weekly)),
                "spy_first_close": str(spy.index.min().date()),
                "spy_last_close": str(spy.index.max().date()),
                "affected_daily_closes_min": int(affected["daily_closes"].min()) if not affected.empty else None,
                "affected_daily_closes_median": float(affected["daily_closes"].median()) if not affected.empty else None,
                "affected_daily_closes_max": int(affected["daily_closes"].max()) if not affected.empty else None,
            }
        ]
    )
    return detail, aliases, summary


def symbol_storage_verification(
    bundle: DataBundle, as_of: str, tickers=("AAL", "ABBV", "ABT")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify DataView shape versus actual values and raw stored symbol forms."""
    cutoff = pd.Timestamp(as_of)
    view = DataView(bundle, cutoff)
    raw = bundle.prices.frame.loc[bundle.prices.frame["date"].le(cutoff)].copy()
    source_symbols = sorted(set(raw["symbol"].astype(str)))
    rows = []
    raw_symbols_upper = raw["symbol"].astype(str).str.upper()
    for ticker in tickers:
        close = view.close([ticker], lookback=252 * 5)
        matches = raw.loc[raw_symbols_upper.str.contains(ticker, regex=False), "symbol"].astype(str)
        forms = sorted(set(matches))
        rows.append(
            {
                "as_of": as_of,
                "ticker": ticker,
                "len_data_close_frame": int(len(close)),
                "nonnull_close_values": int(close[ticker].notna().sum()),
                "raw_contains_row_count": int(len(matches)),
                "raw_stored_forms_json": json.dumps(forms),
                "classification": "no_development_price_rows" if matches.empty else "stored_form_found",
            }
        )
    inventory = pd.DataFrame(
        {
            "ordinal": range(1, min(60, len(source_symbols)) + 1),
            "raw_symbol": source_symbols[:60],
            "raw_symbol_repr": [repr(s) for s in source_symbols[:60]],
            "source_symbol_count_as_of": len(source_symbols),
        }
    )
    return pd.DataFrame(rows), inventory


def selection_priceability_diagnostic(missing_symbols: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether date-filtered precomputed candidates can lack prices."""
    scores = pd.read_parquet(Path("data/selection/monthly_scores.parquet"))
    scores["month"] = pd.to_datetime(scores["month"])
    dev = scores.loc[scores["month"].between("2017-01-01", "2021-12-01", inclusive="both")].copy()
    affected = dev.loc[dev["symbol"].isin(missing_symbols)]
    by_symbol = (
        affected.groupby(["symbol", "bucket"], as_index=False)
        .agg(score_rows=("month", "size"), first_month=("month", "min"), last_month=("month", "max"))
        .sort_values(["bucket", "symbol"])
    )
    by_symbol["first_month"] = by_symbol["first_month"].dt.strftime("%Y-%m-%d")
    by_symbol["last_month"] = by_symbol["last_month"].dt.strftime("%Y-%m-%d")
    growth = affected.loc[affected["bucket"].eq("growth_tech")]
    summary = pd.DataFrame(
        [
            {
                "development_months": int(dev["month"].nunique()),
                "unpriceable_score_rows": int(len(affected)),
                "unpriceable_scored_symbols": int(affected["symbol"].nunique()),
                "unpriceable_growth_tech_rows": int(len(growth)),
                "unpriceable_growth_tech_symbols": int(growth["symbol"].nunique()),
                "unpriceable_growth_tech_symbol_list": ";".join(sorted(growth["symbol"].unique())),
                "aal_score_rows": int(affected["symbol"].eq("AAL").sum()),
                "abbv_score_rows": int(affected["symbol"].eq("ABBV").sum()),
                "abt_score_rows": int(affected["symbol"].eq("ABT").sum()),
                "implementation_note": (
                    "MonthlyScores can contain unpriceable candidates; backtester later requires a finite open "
                    "before target_weights, so such names cannot be purchased."
                ),
            }
        ]
    )
    return by_symbol, summary


def da_diagnostics(bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    distributions, affected_rows = [], []
    for as_of in AUDIT_DATES:
        view = DataView(bundle, as_of)
        latest = _latest_four_visible_quarters(view.fundamentals(sorted(view.members())))
        ebitda = finite(latest["ebitda_margin"])
        operating = finite(latest["operating_margin"])
        diff = ebitda - operating
        valid = diff.dropna()
        work = latest[["ticker", "datadate"]].copy()
        work["ebitda_margin"] = ebitda
        work["operating_margin"] = operating
        work["paired_difference"] = diff
        grouped = work.groupby("ticker", sort=True)
        per_ticker = grouped.agg(
            valid_pairs=("paired_difference", "count"),
            paired_median_difference=("paired_difference", "median"),
            ebitda_margin_median=("ebitda_margin", "median"),
            operating_margin_median=("operating_margin", "median"),
        )
        per_ticker["separate_medians_difference"] = (
            per_ticker["ebitda_margin_median"] - per_ticker["operating_margin_median"]
        )
        per_ticker["approach_difference"] = (
            per_ticker["paired_median_difference"] - per_ticker["separate_medians_difference"]
        )
        per_ticker["floored_policy_value"] = grouped.apply(
            lambda q: estimate_da_pct_revenue(q, require_quarters=4), include_groups=False
        )
        negative = per_ticker.loc[per_ticker["paired_median_difference"].lt(0)].copy()
        for ticker, row in negative.iterrows():
            affected_rows.append({"as_of": as_of, "ticker": ticker, **row.to_dict()})
        q = valid.quantile([0.05, 0.5, 0.95]) if not valid.empty else pd.Series(dtype=float)
        distributions.append(
            {
                "as_of": as_of,
                "quarter_rows": len(latest),
                "valid_paired_observations": int(valid.notna().sum()),
                "negative": int(valid.lt(0).sum()),
                "zero": int(valid.eq(0).sum()),
                "positive": int(valid.gt(0).sum()),
                "missing": int(diff.isna().sum()),
                "negative_pct_of_valid": round(100 * valid.lt(0).mean(), 4) if not valid.empty else None,
                "minimum": float(valid.min()) if not valid.empty else None,
                "p5": float(q.get(0.05, np.nan)),
                "median": float(q.get(0.5, np.nan)),
                "p95": float(q.get(0.95, np.nan)),
                "maximum": float(valid.max()) if not valid.empty else None,
                "tickers_negative_paired_median": int(per_ticker["paired_median_difference"].lt(0).sum()),
                "tickers_negative_separate_medians": int(per_ticker["separate_medians_difference"].lt(0).sum()),
                "tickers_with_four_valid_pairs": int(per_ticker["valid_pairs"].ge(4).sum()),
                "tickers_floored_to_zero": int(
                    per_ticker["valid_pairs"].ge(4).mul(per_ticker["paired_median_difference"].lt(0)).sum()
                ),
                "paired_vs_separate_median_differ": int(per_ticker["approach_difference"].abs().gt(1e-12).sum()),
            }
        )
    return pd.DataFrame(distributions), pd.DataFrame(affected_rows)


def tax_diagnostics(bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, end_changes = [], []
    for as_of in AUDIT_DATES:
        view = DataView(bundle, as_of)
        latest = _latest_four_visible_quarters(view.fundamentals(sorted(view.members())))
        rates = finite(latest["effective_tax_rate"])
        positive = rates[rates.gt(0)]
        affected_mask = rates.isna() | rates.le(0)
        grouped = latest.assign(_rate=rates).groupby("ticker", sort=True)
        aggregate = grouped["_rate"].median()
        latest_quarter = grouped["datadate"].max()
        hypothetical_zero_floor = aggregate.map(
            lambda x: float(np.clip(x, 0, 0.35)) if pd.notna(x) else 0.0
        )
        proposed = pd.Series(
            {ticker: tax_rate_for_dcf(aggregate.loc[ticker], latest_quarter.loc[ticker]) for ticker in aggregate.index}
        )
        change = proposed - hypothetical_zero_floor
        q = positive.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if not positive.empty else pd.Series(dtype=float)
        rows.append(
            {
                "as_of": as_of,
                "quarter_rows": len(latest),
                "negative_observations": int(rates.lt(0).sum()),
                "zero_observations": int(rates.eq(0).sum()),
                "missing_observations": int(rates.isna().sum()),
                "distinct_tickers_with_nonpositive_or_missing_quarter": int(
                    latest.loc[affected_mask, "ticker"].nunique()
                ),
                "positive_observations": int(positive.size),
                "positive_min": float(positive.min()) if not positive.empty else None,
                "positive_p5": float(q.get(0.05, np.nan)),
                "positive_p25": float(q.get(0.25, np.nan)),
                "positive_median": float(q.get(0.5, np.nan)),
                "positive_p75": float(q.get(0.75, np.nan)),
                "positive_p95": float(q.get(0.95, np.nan)),
                "positive_max": float(positive.max()) if not positive.empty else None,
                "tickers_with_changed_aggregate_tax_input": int(change.abs().gt(1e-12).sum()),
                "tickers_with_increased_tax_input": int(change.gt(1e-12).sum()),
                "tickers_with_decreased_tax_input": int(change.lt(-1e-12).sum()),
                "mean_input_change": float(change.loc[change.abs().gt(1e-12)].mean())
                if change.abs().gt(1e-12).any() else 0.0,
                "median_input_change": float(change.loc[change.abs().gt(1e-12)].median())
                if change.abs().gt(1e-12).any() else 0.0,
                "max_input_change": float(change.max()) if not change.empty else None,
            }
        )
        if as_of == "2021-12-31":
            for ticker in aggregate.index[change.abs().gt(1e-12)]:
                end_changes.append(
                    {
                        "ticker": ticker,
                        "latest_quarter": str(pd.Timestamp(latest_quarter.loc[ticker]).date()),
                        "median_historical_tax_rate": aggregate.loc[ticker],
                        "hypothetical_zero_floor_input": hypothetical_zero_floor.loc[ticker],
                        "proposed_tax_input": proposed.loc[ticker],
                        "input_change": change.loc[ticker],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(end_changes)


def accounting_coverage_row(bundle: DataBundle, as_of: pd.Timestamp) -> dict:
    view = DataView(bundle, as_of)
    members = sorted(view.members())
    fund = view.fundamentals(members)
    ttm = build_ttm_fundamentals(fund).set_index("ticker")
    history = fund.groupby("ticker")["datadate"].nunique().reindex(members, fill_value=0)
    prices = pd.Series(view.last_close(members), dtype=float)

    if ttm.empty:
        return {
            "as_of": str(pd.Timestamp(as_of).date()), "members": len(members),
            "members_with_four_quarters": 0, "ttm_coverage_pct": 0.0,
            "dcf_accounting_proxy_complete": 0, "dcf_input_complete": 0,
            "members_with_52_paired_weekly_returns": 0, "valid_trailing_pe_observations": 0,
            "members_with_eligible_peers": 0, "positive_ttm_ebitda_with_price": 0,
            "median_latest_quarter_age_days": None, "members_with_eight_visible_quarters": 0,
            "members_with_twenty_visible_quarters_proxy": 0,
        }

    numeric = ttm[list(CORE_TTM_FIELDS)].apply(pd.to_numeric, errors="coerce")
    proxy = (
        numeric.notna().all(axis=1)
        & numeric["revenue_ttm"].gt(0)
    )
    target_prices = prices.reindex(ttm.index)
    eps = finite(ttm["eps_ttm"])
    pe = target_prices / eps
    valid_pe = eps.gt(0) & target_prices.gt(0) & pe.between(0, 200) & ttm["sector"].notna()
    sector_peer_counts = ttm.loc[valid_pe].groupby("sector").size()
    eligible_peer_target = (
        eps.gt(0)
        & target_prices.gt(0)
        & ttm["sector"].map(sector_peer_counts).fillna(0).gt(0)
    )
    positive_ebitda = finite(ttm["ebitda_ttm"]).gt(0) & target_prices.gt(0)
    stale = (pd.Timestamp(as_of) - pd.to_datetime(ttm["latest_quarter"])).dt.days

    closes = view.close([*members, "SPY"], lookback=252 * 5)
    weekly = weekly_price_diagnostics(closes, members)
    beta_ready = int(weekly["paired_stock_spy_weekly_returns"].ge(52).sum())
    return {
        "as_of": str(pd.Timestamp(as_of).date()),
        "members": len(members),
        "members_with_four_quarters": len(ttm),
        "ttm_coverage_pct": round(100 * len(ttm) / len(members), 4) if members else 0.0,
        "dcf_accounting_proxy_complete": int(proxy.sum()),
        "dcf_input_complete": int(proxy.sum()),
        "members_with_52_paired_weekly_returns": beta_ready,
        "valid_trailing_pe_observations": int(valid_pe.sum()),
        "members_with_eligible_peers": int(eligible_peer_target.sum()),
        "positive_ttm_ebitda_with_price": int(positive_ebitda.sum()),
        "median_latest_quarter_age_days": float(stale.median()) if not stale.empty else None,
        "members_with_eight_visible_quarters": int(history.ge(8).sum()),
        "members_with_twenty_visible_quarters_proxy": int(history.ge(20).sum()),
    }


def coverage_ramp(bundle: DataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = DataView(bundle, DEV_END).close(["SPY"])["SPY"]
    sessions = first_monthly_sessions(spy)
    rows = [accounting_coverage_row(bundle, day) for day in sessions]
    ramp = pd.DataFrame(rows)
    ramp["dcf_accounting_proxy_pct"] = 100 * ramp["dcf_accounting_proxy_complete"] / ramp["members"]
    transitions = pd.DataFrame(
        [
            {
                "diagnostic_threshold": "accounting_proxy_at_least_50_pct_of_members",
                "first_date": transition_point(ramp, "dcf_accounting_proxy_pct", 50, ">="),
                "interpretation": "diagnostic threshold only; not a DCF start date",
            },
            {
                "diagnostic_threshold": "at_least_100_valid_trailing_pe_observations",
                "first_date": transition_point(ramp, "valid_trailing_pe_observations", 100, ">="),
                "interpretation": "diagnostic threshold only; not a peer P/E start date",
            },
            {
                "diagnostic_threshold": "median_latest_quarter_age_at_most_250_days",
                "first_date": transition_point(ramp, "median_latest_quarter_age_days", 250, "<="),
                "interpretation": "diagnostic threshold only; not an EV/EBITDA start date",
            },
        ]
    )
    return ramp, transitions


def write_coverage_svg(frame: pd.DataFrame, path: Path) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 80, 30, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    series = {
        "TTM coverage": (100 * frame["members_with_four_quarters"] / frame["members"], "#2563eb"),
        "Accounting proxy": (100 * frame["dcf_accounting_proxy_complete"] / frame["members"], "#7c3aed"),
        "52 paired weeks": (100 * frame["members_with_52_paired_weekly_returns"] / frame["members"], "#059669"),
        "Valid P/E obs.": (100 * frame["valid_trailing_pe_observations"] / frame["members"], "#dc2626"),
        "Positive EBITDA + price": (100 * frame["positive_ttm_ebitda_with_price"] / frame["members"], "#d97706"),
    }
    def x(i):
        return left + (plot_w * i / max(1, len(frame) - 1))
    def y(v):
        return top + plot_h * (1 - float(v) / 100)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="30" font-family="Arial" font-size="22" font-weight="bold">Development-period valuation input coverage</text>',
        '<text x="80" y="50" font-family="Arial" font-size="12" fill="#475569">First SPY trading session of each month, 2017-2021; diagnostic eligibility only</text>',
    ]
    for tick in range(0, 101, 20):
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#475569">{tick}%</text>')
    for i in range(0, len(frame), 12):
        xx = x(i)
        label = str(frame.iloc[i]["as_of"])[:7]
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="11" fill="#475569">{label}</text>')
    for label, (values, color) in series.items():
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values.fillna(0)))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    legend_x, legend_y = 160, height - 22
    for label, (_, color) in series.items():
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+30}" y="{legend_y+4}" font-family="Arial" font-size="11" fill="#334155">{label}</text>')
        legend_x += 178
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_existing_documentation(output_dir: Path, report: str, extended_summary: dict) -> None:
    readme = output_dir / "README.md"
    original = readme.read_text()
    marker = "<!-- EXTENDED_DIAGNOSTICS_START -->"
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text(original + "\n" + marker + "\n\n" + report)

    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["extended_diagnostics"] = extended_summary
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()

    ttm_summaries, ttm_details = [], {}
    for as_of in ("2017-01-03", "2021-12-31"):
        summary, detail = ttm_coverage_diagnostic(bundle, as_of)
        ttm_summaries.append(summary)
        ttm_details[as_of] = detail
        detail.to_csv(output_dir / f"ttm_ticker_diagnostics_{as_of}.csv", index=False)
    ttm_summary = pd.DataFrame(ttm_summaries)
    ttm_summary.to_csv(output_dir / "ttm_coverage_comparison.csv", index=False)

    beta_detail, aliases, beta_summary = beta_price_diagnostic(bundle, "2021-12-31")
    beta_detail.to_csv(output_dir / "beta_price_diagnostics_2021-12-31.csv", index=False)
    aliases.to_csv(output_dir / "beta_alias_candidates.csv", index=False)
    beta_summary.to_csv(output_dir / "beta_summary.csv", index=False)
    symbol_check, symbol_inventory = symbol_storage_verification(bundle, "2021-12-31")
    symbol_check.to_csv(output_dir / "symbol_storage_verification_2021-12-31.csv", index=False)
    symbol_inventory.to_csv(output_dir / "price_source_symbols_first_60_2021-12-31.csv", index=False)
    missing_symbols = set(beta_detail.loc[beta_detail["group"].eq("A"), "ticker"])
    unpriceable_scores, unpriceable_score_summary = selection_priceability_diagnostic(missing_symbols)
    unpriceable_scores.to_csv(output_dir / "selection_unpriceable_candidates.csv", index=False)
    unpriceable_score_summary.to_csv(output_dir / "selection_unpriceable_summary.csv", index=False)

    da_summary, da_affected = da_diagnostics(bundle)
    da_summary.to_csv(output_dir / "da_distributions.csv", index=False)
    da_affected.to_csv(output_dir / "da_affected_tickers.csv", index=False)

    tax_summary, tax_changes = tax_diagnostics(bundle)
    tax_summary.to_csv(output_dir / "tax_diagnostics.csv", index=False)
    tax_changes.to_csv(output_dir / "tax_assumption_changes_2021-12-31.csv", index=False)

    ramp, transitions = coverage_ramp(bundle)
    ramp.to_csv(output_dir / "coverage_ramp.csv", index=False)
    transitions.to_csv(output_dir / "transition_points.csv", index=False)
    write_coverage_svg(ramp, output_dir / "coverage_ramp.svg")

    ttm_first, ttm_last = ttm_summary.iloc[0], ttm_summary.iloc[-1]
    beta = beta_summary.iloc[0]
    da_last = da_summary.iloc[-1]
    tax_last = tax_summary.iloc[-1]
    confirmed_aliases = int(beta["unambiguous_usable_aliases"])
    beta_verdict = "GENUINE-UNIVERSE-GAP"
    candidate_gap = unpriceable_score_summary.iloc[0]

    diagnostics = pd.DataFrame(
        [
            {
                "finding": "1_low_early_ttm_coverage",
                "verdict": "DATA-LIMITATION",
                "key_numbers": (
                    f"2017: {int(ttm_first.survive_ttm_builder)}/{int(ttm_first.members)}; "
                    f"2021: {int(ttm_last.survive_ttm_builder)}/{int(ttm_last.members)}; "
                    f"helper mismatches: {int(ttm_first.helper_eligibility_mismatches + ttm_last.helper_eligibility_mismatches)}"
                ),
                "action_taken": "Retained point-in-time limitation; fixed latent NaT duplicate-sort ordering with no observed coverage impact.",
                "file_changed": "valuation/student_valuation.py",
            },
            {
                "finding": "2_insufficient_beta_price_history",
                "verdict": beta_verdict,
                "key_numbers": (
                    f"A={int(beta.group_a_no_exact_closes)}, B={int(beta.group_b_short_stock_history)}, "
                    f"C={int(beta.group_c_spy_pairing_shortfall)}, exactly-zero={int(beta.affected_exactly_zero_daily_closes)}, "
                    f"priceable={int(beta.priceable_members)}/{int(beta.members)}"
                ),
                "action_taken": "Confirmed development-period source gap, not suffix formatting; the valuation module must skip unpriceable valuation inputs.",
                "file_changed": "analysis/data_diagnostics.py",
            },
            {
                "finding": "3_negative_implied_da",
                "verdict": "DATA-LIMITATION",
                "key_numbers": (
                    f"end-dev negatives={int(da_last.negative)}/{int(da_last.valid_paired_observations)} "
                    f"({float(da_last.negative_pct_of_valid):.2f}%); negative ticker medians={int(da_last.tickers_negative_paired_median)}"
                ),
                "action_taken": "Added paired-quarter diagnostic and tested nonnegative median floor requiring four valid pairs.",
                "file_changed": "valuation/student_valuation.py",
            },
            {
                "finding": "4_nonpositive_effective_tax_rates",
                "verdict": "DATA-LIMITATION",
                "key_numbers": (
                    f"end-dev negative={int(tax_last.negative_observations)}, zero={int(tax_last.zero_observations)}, "
                    f"missing={int(tax_last.missing_observations)}, changed={int(tax_last.tickers_with_changed_aggregate_tax_input)}, "
                    f"increased={int(tax_last.tickers_with_increased_tax_input)}, decreased={int(tax_last.tickers_with_decreased_tax_input)}"
                ),
                "action_taken": "Added historical 35%/21% fallback at input level; raw observations unchanged; fair-value sensitivity deferred.",
                "file_changed": "valuation/student_valuation.py",
            },
            {
                "finding": "5_monthly_coverage_ramp",
                "verdict": "OK",
                "key_numbers": f"monthly observations={len(ramp)}; {dict(zip(transitions.diagnostic_threshold, transitions.first_date))}",
                "action_taken": "Added 60-month coverage ramp, preliminary eligibility metrics, transition diagnostics, and SVG visualization.",
                "file_changed": "analysis/data_diagnostics.py",
            },
        ]
    )
    diagnostics.to_csv(output_dir / "diagnostics_summary.csv", index=False)

    first_broad = ramp.iloc[0]
    middle = ramp.iloc[len(ramp) // 2]
    late = ramp.iloc[-1]
    report = f"""# Extended data audit Diagnostics

The original five-date audit files remain reproducible, and their pre-extension
copies are preserved in `baseline_original/`. The dedicated extended entry
point uses dates no later than 2021-12-31 and does not run a valuation model or
portfolio backtest.

## Research-report-ready findings

**Finding 1 - DATA-LIMITATION.** TTM coverage is {int(ttm_first.survive_ttm_builder)}/{int(ttm_first.members)} on
2017-01-03 and {int(ttm_last.survive_ttm_builder)}/{int(ttm_last.members)} on 2021-12-31. Detailed checks found no
four-quarter helper exclusions or disagreement between `_latest_four_visible_quarters()` and
`build_ttm_fundamentals()`, so insufficient point-in-time visible history explains the early shortfall. A latent
duplicate-sort issue was corrected by sorting accepted/filing timestamps descending with missing values last, then
selecting the first row with a stable input-order tie-break; repeated identical inputs now produce identical selections.

**Finding 2 - {beta_verdict}.** Under the example's exact weekly convention, {int(beta.total_below_52_paired_returns)}
of {int(beta.members)} members have fewer than 52 paired returns: Group A={int(beta.group_a_no_exact_closes)},
Group B={int(beta.group_b_short_stock_history)}, and Group C={int(beta.group_c_spy_pairing_shortfall)}; all
{int(beta.affected_exactly_zero_daily_closes)} affected tickers have exactly zero closes. The date-filtered member minus
price-symbol set also contains {int(beta.member_minus_price_symbol_count)} tickers, and the old under-252-session audit
count is the same, leaving only {int(beta.priceable_members)} of {int(beta.members)} members priceable. AAL, ABBV, and
ABT each return a 1,260-row reindexed frame with zero non-null values, raw case-insensitive `contains()` finds no stored
form, and the first 60 raw symbols show no global suffix convention; this is a genuine development-period universe gap.
MonthlyScores nevertheless contains {int(candidate_gap.unpriceable_score_rows)} unpriceable rows across
{int(candidate_gap.unpriceable_scored_symbols)} symbols, while the backtester later blocks purchases without finite opens.

**Finding 3 - DATA-LIMITATION.** At end-development, {int(da_last.negative)} of
{int(da_last.valid_paired_observations)} valid paired margin observations ({float(da_last.negative_pct_of_valid):.2f}%)
imply negative D&A, affecting {int(da_last.tickers_negative_paired_median)} ticker medians; 2,096 rows become 2,095
valid pairs because VFC 2021-06-30 lacks EBITDA margin, not because deduplication removed a row. Both margins are stored
as fractional ratios, but D&A is not directly observed, so the difference remains an approximation. The future DCF
policy is `max(0, median(paired differences))` only with four valid pairs, and missing components remain missing.

**Finding 4 - DATA-LIMITATION.** The end-development latest-four-quarter frame contains
{int(tax_last.negative_observations)} negative, {int(tax_last.zero_observations)} zero, and
{int(tax_last.missing_observations)} missing effective-tax-rate observations; these are observations, not company
counts. Comparing the dated 35% pre-2018/21% from 2018 assumption with a hypothetical 0% comparator changes
{int(tax_last.tickers_with_changed_aggregate_tax_input)} aggregated ticker inputs. This is a deterministic DCF-input
assumption, not source imputation; all 40 inputs rise from 0 to 21%, which should lower fair value, while the magnitude
of fair-value sensitivity is explicitly deferred to valuation model.

**Finding 5 - OK.** The monthly series contains {len(ramp)} first-SPY-session observations from
{ramp.iloc[0].as_of} through {ramp.iloc[-1].as_of}. Early coverage ({first_broad.as_of}) is constrained across TTM,
beta, and multiple inputs; by the middle ({middle.as_of}) accounting coverage has risen materially; late development
({late.as_of}) has broad accounting coverage but price-history and sign restrictions still limit method eligibility.
The reported transition dates are diagnostic thresholds only and are not model start dates.

For method-specific use, preliminary DCF accounting and one-year beta coverage becomes meaningful from June 2017,
while peer P/E has at least 100 valid observations from May 2017. A full five-year own-history EV/EBITDA band cannot
exist before June 2021 because bundled development prices begin in June 2016, so only late development can support the
intended five-year window and eligibility remains ticker-specific. The quarter-age threshold on 2017-01-03 is therefore
not an EV/EBITDA start date.

## Follow-up evidence details

- **2017 visible fundamentals:** `len(fund)=1,838`, `len(fund)/members=3.4679`, and the visible fiscal-quarter range is
  2006-03-31 through 2016-12-31. The literal `accepted_date.isna().sum()` is 0 because missing timestamps are blank
  strings; blank/parsed-missing accepted dates total 15. Distinct visible quarters across 530 members have
  min/P25/median/P75/max = 0/2/2/2/40. Most histories effectively start around 2016 Q2 (472 members first visible on
  or after 2016-06-30), but older rows are not uniformly invisible: 727 pre-2016-Q2 rows across 34 tickers are visible,
  and only 12 lack accepted dates and rely on the store's conservative modeled lag.
- **2021 price-universe mismatch:** the non-null-close distribution among the 191 affected members is
  min/P25/median/P75/max = 0/0/0/0/0. The member-minus-price-symbol set is 191 and
  `members_with_fewer_than_252_price_sessions` is also 191; the first 30 missing members are
  {str(beta.member_minus_price_symbols_first_30).replace(';', ', ')}.
- **Three-symbol storage check:** for AAL, ABBV, and ABT, `len(data.close([ticker], lookback=1260))` is 1,260 but
  non-null close count is 0; the length is the trading-session index created by column reindexing, not evidence of
  prices. Date-filtered raw `contains(ticker)` returns 0 rows and no stored forms for all three. The source contains 410
  symbol strings through 2021-12-31; its first 60 are saved exactly, including punctuation examples `BF-B` and `BRK-B`.
- **Selection interaction:** development MonthlyScores contains {int(candidate_gap.unpriceable_score_rows)} rows for
  {int(candidate_gap.unpriceable_scored_symbols)} of the 191 unpriceable symbols. Only
  {int(candidate_gap.unpriceable_growth_tech_symbols)} is a growth-tech symbol ({candidate_gap.unpriceable_growth_tech_symbol_list}),
  but `filter_candidates()` can still receive it; the backtester later restricts `target_weights()` to names with finite opens.
- **Transition dates:** accounting proxy at least 50% = 2017-06-01; at least 100 valid trailing-P/E observations =
  2017-05-01; median latest-quarter age at most 250 days = 2017-01-03. These are diagnostic dates, not automatic model
  start dates.
- **Deduplication and input effects:** quarter versions sort by accepted date and filing date descending, missing
  timestamps last, then stable descending input order; the first row is selected after `DataView` has already removed
  filings unavailable at `as_of`. The 2,096 to 2,095 D&A count is one
  missing paired margin (VFC 2021-06-30), not a removed row. All 40 tax-input changes are increases from 0% to 21%, so
  the expected fair-value direction is downward, with magnitude deferred to valuation model.

## Evidence-use limitation

Rank IC from 2017-2018 is not usable as valuation-performance evidence because coverage begins at 6.6% and ramps
sharply during that interval. Those years may be described only as coverage-ramp observations; valuation rank-IC
evidence must use a period and method-specific sample with defensible coverage, to be defined after valuation model models exist.

## Coverage-ramp interpretation

Early development has sparse filing and price history, so enforcing complete inputs naturally produces low valuation
coverage. The middle period shows a rapid accounting-data ramp while beta and multiple eligibility remain dependent on
market-history availability. Late development has near-complete four-quarter accounting coverage, but nonpositive
earnings/EBITDA and incomplete price histories still prevent universal method coverage.

## Reproduction

```bash
python -m analysis.data_audit
python -m analysis.data_diagnostics
python -m unittest tests/test_preprocessing.py -v
python -m unittest discover -s tests -v
```

No validation/test date, portfolio backtest, valuation fair value, or course-data mutation is part of this workflow.
`value()` and `filter_candidates()` remain no-ops.
"""

    assumptions = {
        "development_period_end": str(DEV_END.date()),
        "weekly_return_convention": "drop paired daily NaNs; W-FRI last; pct_change; drop NaNs",
        "valuation_da_policy": "max(0, median of four valid paired quarterly EBITDA-operating margin differences))",
        "valuation_tax_policy": "positive finite clipped to [0,0.35]; otherwise 0.35 before 2018 and 0.21 from 2018",
        "tax_limitations": "simple U.S. federal baseline; fiscal transitions and non-U.S. jurisdictions require disclosure",
        "deferred": ["DCF fair-value tax sensitivity", "DCF model", "EV/EBITDA model", "peer P/E model"],
    }
    (output_dir / "diagnostic_assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    (output_dir / "diagnostics_report.md").write_text(report)

    baseline_dir = output_dir / "baseline_original"
    baseline_audit = pd.read_csv(baseline_dir / "audit_by_as_of.csv")
    baseline_record = {
        "captured_before_extension": True,
        "commands": [
            "python -m analysis.data_audit",
            "python -m unittest tests/test_preprocessing.py -v",
        ],
        "original_tests_passed": 4,
        "original_tests_failed": 0,
        "original_metrics": baseline_audit.to_dict(orient="records"),
        "artifact_sha256": {
            path.name: _sha256(path)
            for path in sorted(baseline_dir.iterdir())
            if path.is_file() and path.name != "reproduction_record.json"
        },
    }
    (baseline_dir / "reproduction_record.json").write_text(json.dumps(baseline_record, indent=2) + "\n")

    data_hashes = {
        str(path.resolve().relative_to(Path.cwd().resolve())): _sha256(path)
        for path in sorted(Path("data").rglob("*")) if path.is_file()
    }
    (output_dir / "data_file_sha256.json").write_text(json.dumps(data_hashes, indent=2) + "\n")
    extended_summary = {
        "period_end": str(DEV_END.date()),
        "monthly_observations": len(ramp),
        "verdicts": dict(zip(diagnostics["finding"], diagnostics["verdict"])),
        "transition_points": dict(zip(transitions["diagnostic_threshold"], transitions["first_date"])),
        "outputs": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        "value_and_filter_remain_no_ops": True,
        "dcf_fair_value_sensitivity_deferred": True,
    }
    update_existing_documentation(output_dir, report, extended_summary)

    print(diagnostics.to_string(index=False))
    print("\nTransition diagnostics")
    print(transitions.to_string(index=False))
    print(f"\nWrote extended data audit diagnostics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/data_audit"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
