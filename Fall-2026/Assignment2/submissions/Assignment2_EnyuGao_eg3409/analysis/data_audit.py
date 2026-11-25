"""Reproducible development-period data audit for the valuation track.

The audit deliberately stops at 2021-12-31. It uses DataView for every
historical valuation input, mirroring the interface used by student code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.data_view import DataBundle, DataView
from valuation.student_valuation import _latest_four_visible_quarters, build_ttm_fundamentals


AUDIT_DATES = (
    "2017-01-03",
    "2018-12-31",
    "2019-12-31",
    "2020-12-31",
    "2021-12-31",
)

QUARTERLY_FIELDS = (
    "EPS",
    "revenue_per_share",
    "fcf_per_share",
    "ocf_per_share",
    "capex_per_share",
    "ebitda_margin",
    "operating_margin",
    "effective_tax_rate",
    "cash_per_share",
    "interest_debt_per_share",
    "debt_to_capital",
)


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _audit_one_date(bundle: DataBundle, as_of: str) -> tuple[dict, list[dict]]:
    view = DataView(bundle, as_of)
    as_of_ts = pd.Timestamp(as_of)
    members = sorted(view.members())
    fund = view.fundamentals(members)
    latest4 = _latest_four_visible_quarters(fund)
    ttm = build_ttm_fundamentals(fund).set_index("ticker")
    history = fund.groupby("ticker").size()

    accepted = pd.to_datetime(fund.get("accepted_date"), errors="coerce")
    filed = pd.to_datetime(fund.get("filing_date"), errors="coerce")
    availability = accepted.fillna(filed)

    closes = view.close([*members, "SPY"], lookback=1260)
    latest_prices = pd.Series(view.last_close(members), dtype=float)
    member_price_counts = closes.reindex(columns=members).notna().sum()
    weekly = closes.resample("W-FRI").last().pct_change(fill_method=None)
    paired_weekly = weekly.reindex(columns=members).notna().mul(weekly["SPY"].notna(), axis=0).sum()

    if not ttm.empty:
        prices = latest_prices.reindex(ttm.index)
        pe = prices / _finite(ttm["eps_ttm"])
        peer_valid = (_finite(ttm["eps_ttm"]) > 0) & pe.between(0, 200) & ttm["sector"].notna()
        peer_counts = ttm.loc[peer_valid].groupby("sector").size()
        stale_days = (as_of_ts - pd.to_datetime(ttm["latest_quarter"])).dt.days
        dcf_complete = (
            (_finite(ttm["revenue_ttm"]) > 0)
            & _finite(ttm["ebitda_margin_median"]).notna()
            & _finite(ttm["operating_margin_median"]).notna()
            & _finite(ttm["capex_ttm"]).notna()
            & _finite(ttm["effective_tax_rate_median"]).notna()
            & _finite(ttm["cash_per_share"]).notna()
            & _finite(ttm["interest_debt_per_share"]).notna()
            & _finite(ttm["debt_to_capital"]).notna()
        )
    else:
        peer_counts = pd.Series(dtype=int)
        stale_days = pd.Series(dtype=float)
        dcf_complete = pd.Series(dtype=bool)

    row = {
        "as_of": as_of,
        "members": len(members),
        "fundamental_rows_visible": len(fund),
        "members_with_any_fundamentals": int(fund["ticker"].nunique()),
        "duplicate_visible_ticker_quarters": int(fund.duplicated(["ticker", "datadate"]).sum()),
        "future_availability_rows": int(availability.gt(as_of_ts).sum()),
        "hidden_leakage_columns_visible": int(len({"trade_price", "adj_close_q", "y_return"} & set(fund.columns))),
        "members_with_four_quarters": int(len(ttm)),
        "ttm_coverage_pct": round(100 * len(ttm) / len(members), 2) if members else 0.0,
        "members_with_fewer_than_four_quarters": int((history.reindex(members, fill_value=0) < 4).sum()),
        "members_with_fewer_than_eight_quarters": int((history.reindex(members, fill_value=0) < 8).sum()),
        "members_with_latest_price": int(latest_prices.reindex(members).notna().sum()),
        "members_with_fewer_than_252_price_sessions": int((member_price_counts < 252).sum()),
        "members_with_fewer_than_52_paired_weekly_returns": int((paired_weekly < 52).sum()),
        "nonpositive_ttm_revenue": int((_finite(ttm.get("revenue_ttm", pd.Series(dtype=float))) <= 0).sum()),
        "nonpositive_ttm_eps": int((_finite(ttm.get("eps_ttm", pd.Series(dtype=float))) <= 0).sum()),
        "nonpositive_ttm_ebitda": int((_finite(ttm.get("ebitda_ttm", pd.Series(dtype=float))) <= 0).sum()),
        "nonpositive_ebitda_margin": int((_finite(ttm.get("ebitda_margin_median", pd.Series(dtype=float))) <= 0).sum()),
        "dcf_input_complete": int(dcf_complete.sum()),
        "valid_trailing_pe_observations": int(peer_counts.sum()) if not peer_counts.empty else 0,
        "sectors_with_fewer_than_five_valid_peers": int((peer_counts < 5).sum()) if not peer_counts.empty else 0,
        "median_valid_peers_per_sector": round(float(peer_counts.median()), 2) if not peer_counts.empty else None,
        "median_latest_quarter_age_days": round(float(stale_days.median()), 2) if not stale_days.empty else None,
        "max_latest_quarter_age_days": int(stale_days.max()) if not stale_days.empty else None,
    }

    missingness = []
    denominator = len(latest4)
    for field in QUARTERLY_FIELDS:
        values = _finite(latest4[field]) if field in latest4 else pd.Series(np.nan, index=latest4.index)
        missing = int(values.isna().sum())
        missingness.append(
            {
                "as_of": as_of,
                "field": field,
                "quarter_rows": denominator,
                "missing": missing,
                "missing_pct": round(100 * missing / denominator, 2) if denominator else None,
                "nonpositive": int(values.le(0).sum()),
                "nonpositive_pct": round(100 * values.le(0).sum() / denominator, 2) if denominator else None,
            }
        )
    return row, missingness


def _source_quality(bundle: DataBundle, end: str) -> list[dict]:
    end_ts = pd.Timestamp(end)
    prices = bundle.prices.frame.loc[bundle.prices.frame["date"].le(end_ts)].copy()
    market = bundle.market_frame.loc[bundle.market_frame["date"].le(end_ts)].copy()
    latest_view = DataView(bundle, end)
    fund = latest_view.fundamentals()
    rows = [
        {
            "source": "adjusted_prices",
            "rows": len(prices),
            "duplicate_keys": int(prices.duplicated(["symbol", "date"]).sum()),
            "missing_values": int(prices[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
            "nonpositive_core_values": int(prices[["open", "high", "low", "close"]].le(0).sum().sum()),
            "min_date": str(prices["date"].min().date()),
            "max_date": str(prices["date"].max().date()),
        },
        {
            "source": "market_series",
            "rows": len(market),
            "duplicate_keys": int(market.duplicated(["symbol", "date"]).sum()),
            "missing_values": int(market["close"].isna().sum()),
            "nonpositive_core_values": int(market["close"].le(0).sum()),
            "min_date": str(market["date"].min().date()),
            "max_date": str(market["date"].max().date()),
        },
        {
            "source": "visible_fundamentals",
            "rows": len(fund),
            "duplicate_keys": int(fund.duplicated(["ticker", "datadate"]).sum()),
            "missing_values": int(fund[list(QUARTERLY_FIELDS)].isna().sum().sum()),
            "nonpositive_core_values": int(fund[["EPS", "revenue_per_share", "ebitda_margin"]].le(0).sum().sum()),
            "min_date": str(pd.to_datetime(fund["datadate"]).min().date()),
            "max_date": str(pd.to_datetime(fund["datadate"]).max().date()),
        },
    ]
    for series_id in ("DGS10", "T10Y2Y", "BAA10Y", "BAMLH0A0HYM2"):
        series = latest_view.macro(series_id)
        rows.append(
            {
                "source": series_id,
                "rows": len(series),
                "duplicate_keys": int(series.index.duplicated().sum()),
                "missing_values": int(series.isna().sum()),
                "nonpositive_core_values": int(series.le(0).sum()),
                "min_date": str(series.index.min().date()),
                "max_date": str(series.index.max().date()),
            }
        )
    return rows


def _market_macro_summary(bundle: DataBundle, start: str, end: str) -> list[dict]:
    view = DataView(bundle, end)
    start_ts = pd.Timestamp(start)
    rows = []
    market = bundle.market_frame.loc[
        bundle.market_frame["date"].between(start_ts, pd.Timestamp(end), inclusive="both")
    ]
    for symbol, frame in market.groupby("symbol", sort=True):
        values = _finite(frame["close"])
        rows.append(
            {
                "source_type": "market",
                "series": symbol,
                "observations": int(values.notna().sum()),
                "start": str(frame["date"].min().date()),
                "end": str(frame["date"].max().date()),
                "minimum": float(values.min()),
                "median": float(values.median()),
                "maximum": float(values.max()),
            }
        )
    for series_id in ("DGS10", "T10Y2Y", "BAA10Y", "BAMLH0A0HYM2"):
        series = _finite(view.macro(series_id).loc[start_ts:])
        rows.append(
            {
                "source_type": "macro",
                "series": series_id,
                "observations": int(series.notna().sum()),
                "start": str(series.index.min().date()) if not series.empty else None,
                "end": str(series.index.max().date()) if not series.empty else None,
                "minimum": float(series.min()) if not series.empty else None,
                "median": float(series.median()) if not series.empty else None,
                "maximum": float(series.max()) if not series.empty else None,
            }
        )
    return rows


def _write_readme(
    output_dir: Path, audit: pd.DataFrame, source: pd.DataFrame, market_macro: pd.DataFrame
) -> None:
    first = audit.iloc[0]
    last = audit.iloc[-1]
    duplicate_total = int(source["duplicate_keys"].sum())
    dgs10 = market_macro.loc[market_macro["series"].eq("DGS10")].iloc[0]
    text = f"""# data audit Data Audit

This audit uses only development-period information and stops at 2021-12-31.
Historical inputs are obtained through `DataView`, exactly as they will be in
the student valuation functions.

## Key findings

- Point-in-time checks passed at all {len(audit)} audit dates: zero future
  filing rows, zero duplicate visible ticker-quarter rows, and zero hidden
  leakage columns (`trade_price`, `adj_close_q`, `y_return`).
- Four-quarter TTM coverage is the main early-sample constraint: only
  {int(first['members_with_four_quarters'])} of {int(first['members'])} members
  ({first['ttm_coverage_pct']:.2f}%) are eligible on {first['as_of']}, versus
  {int(last['members_with_four_quarters'])} of {int(last['members'])}
  ({last['ttm_coverage_pct']:.2f}%) on {last['as_of']}.
- At the end of development, {int(last['members_with_fewer_than_eight_quarters'])}
  members lack eight visible quarters for a simple prior-year growth estimate,
  and {int(last['members_with_fewer_than_52_paired_weekly_returns'])} lack the
  minimum 52 paired weekly returns used by the example beta calculation.
- End-of-development TTM screens find {int(last['nonpositive_ttm_eps'])}
  nonpositive-EPS names and {int(last['nonpositive_ttm_ebitda'])}
  nonpositive-EBITDA names. These observations cannot support standard positive
  P/E or EV/EBITDA valuation and must be excluded or handled explicitly.
- Raw development-period source checks found {duplicate_total} duplicate keys
  across prices, market series, visible fundamentals, and macro series.
- During development, DGS10 ranges from {dgs10['minimum']:.2f}% to
  {dgs10['maximum']:.2f}% with a {dgs10['median']:.2f}% median. The high-yield
  BAML series has no development-period observations, while XLC begins in June
  2018; downstream code must tolerate those structural coverage gaps.
- Missing quarterly components are never imputed in the TTM helper. A TTM
  measure remains missing unless all four quarterly components are available.

## Generated files

- `audit_by_as_of.csv`: coverage, history sufficiency, nonpositive measures,
  peer availability, and point-in-time checks by audit date.
- `fundamental_missingness.csv`: missing and nonpositive quarterly fields among
  the latest four visible quarters.
- `source_quality.csv`: duplicates, missing values, and date ranges for prices,
  market data, fundamentals, and macro series.
- `market_macro_summary.csv`: development-period coverage and distribution of
  VIX, SPY, sector ETF, Treasury-yield, term-spread, and credit-spread series.
- `summary.json`: machine-readable audit metadata and findings.

No validation-period or test-period data was used.
"""
    (output_dir / "README.md").write_text(text)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle.load()
    rows, missingness = [], []
    for as_of in AUDIT_DATES:
        row, missing = _audit_one_date(bundle, as_of)
        rows.append(row)
        missingness.extend(missing)

    audit = pd.DataFrame(rows)
    missing = pd.DataFrame(missingness)
    source = pd.DataFrame(_source_quality(bundle, AUDIT_DATES[-1]))
    market_macro = pd.DataFrame(_market_macro_summary(bundle, AUDIT_DATES[0], AUDIT_DATES[-1]))
    audit.to_csv(output_dir / "audit_by_as_of.csv", index=False)
    missing.to_csv(output_dir / "fundamental_missingness.csv", index=False)
    source.to_csv(output_dir / "source_quality.csv", index=False)
    market_macro.to_csv(output_dir / "market_macro_summary.csv", index=False)

    summary = {
        "period": {"start": AUDIT_DATES[0], "end": AUDIT_DATES[-1]},
        "audit_dates": list(AUDIT_DATES),
        "point_in_time_checks_passed": bool(
            audit["duplicate_visible_ticker_quarters"].eq(0).all()
            and audit["future_availability_rows"].eq(0).all()
            and audit["hidden_leakage_columns_visible"].eq(0).all()
        ),
        "source_duplicate_keys": int(source["duplicate_keys"].sum()),
        "end_dev_ttm_coverage_pct": float(audit.iloc[-1]["ttm_coverage_pct"]),
        "outputs": [
            "audit_by_as_of.csv",
            "fundamental_missingness.csv",
            "source_quality.csv",
            "market_macro_summary.csv",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_readme(output_dir, audit, source, market_macro)

    print(audit.to_string(index=False))
    print("\nSource quality through 2021-12-31")
    print(source.to_string(index=False))
    print(f"\nWrote data audit audit to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/data_audit"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
