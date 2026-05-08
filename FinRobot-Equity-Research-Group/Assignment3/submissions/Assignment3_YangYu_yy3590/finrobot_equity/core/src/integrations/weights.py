"""Portfolio weight utilities for FinRL + FinRobot integration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .flag_rules import score_flags


@dataclass(frozen=True)
class BaseWeight:
    ticker: str
    weight: float
    asset_type: str = "stock"


def _infer_asset_type(ticker: str) -> str:
    upper = str(ticker).upper()
    if upper in {"CASH", "USD"}:
        return "cash"
    etf_tickers = {
        "DIA",
        "GLD",
        "IAU",
        "IEF",
        "IWM",
        "QQQ",
        "SHY",
        "SLV",
        "SPY",
        "TLT",
        "UUP",
        "XLE",
        "XLF",
        "XLK",
        "XLU",
        "XLV",
    }
    if upper.endswith(("ETF",)) or upper in etf_tickers:
        return "etf"
    return "stock"


def _normalize_long_weights(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame | None:
    ticker_col = columns.get("ticker") or columns.get("symbol") or columns.get("gvkey")
    weight_col = columns.get("weight")
    if not ticker_col or not weight_col:
        return None

    date_col = columns.get("date") or columns.get("as_of_date")
    keep_cols = [ticker_col, weight_col]
    if date_col:
        keep_cols.insert(0, date_col)
    out = df[keep_cols].copy()
    rename_map = {ticker_col: "ticker", weight_col: "weight"}
    if date_col:
        rename_map[date_col] = "date"
    return out.rename(columns=rename_map)


def _normalize_wide_finrl_weights(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame | None:
    date_col = columns.get("date") or columns.get("as_of_date")
    if not date_col:
        return None

    metadata_cols = {date_col, columns.get("regime"), columns.get("decision_type")}
    metadata_cols = {c for c in metadata_cols if c}
    value_cols: list[str] = []
    work = df.copy()
    for col in df.columns:
        if col in metadata_cols:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            work[col] = numeric.fillna(0.0)
            value_cols.append(col)

    if not value_cols:
        return None

    out = work[[date_col] + value_cols].melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="ticker",
        value_name="weight",
    )
    out = out.rename(columns={date_col: "date"})
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out.loc[out["ticker"] == "CASH", "ticker"] = "CASH"
    return out


def _finalize_weights(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["asset_type"] = out["ticker"].map(_infer_asset_type)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return out


def load_base_weights(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    columns = {c.lower(): c for c in df.columns}
    out = _normalize_long_weights(df, columns)
    if out is None:
        out = _normalize_wide_finrl_weights(df, columns)
    if out is None:
        raise ValueError(
            "Base weights file must be either long format with ticker/symbol/gvkey and weight columns "
            "or FinRL wide format with date plus asset weight columns"
        )
    return _finalize_weights(out)


def _is_high_conviction(flags: dict[str, Any] | None) -> bool:
    if not flags:
        return False
    risk = str(flags.get("risk_flag", "")).lower()
    valuation = str(flags.get("valuation_flag", "")).lower()
    catalyst = str(flags.get("catalyst_flag", "")).lower()
    sentiment = str(flags.get("sentiment_flag", "")).lower()
    thesis = str(flags.get("thesis_strength", "")).lower()

    positive = thesis == "strong" and risk != "high" and (catalyst == "positive" or sentiment == "positive")
    negative = thesis == "weak" and (
        risk == "high" or valuation == "extreme" or catalyst == "negative" or sentiment == "negative"
    )
    return positive or negative


def _effective_multiplier(
    *,
    raw_multiplier: float,
    flags: dict[str, Any] | None,
    defensive_base: bool,
    no_op_band: tuple[float, float],
    max_relative_adjustment: float,
    defensive_max_relative_adjustment: float,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    effective = float(raw_multiplier)

    if no_op_band[0] <= effective <= no_op_band[1]:
        effective = 1.0
        reasons.append("no_op_band")

    high_conviction = _is_high_conviction(flags)
    if defensive_base and flags and not high_conviction:
        effective = 1.0
        reasons.append("defensive_guard_noop")

    cap = defensive_max_relative_adjustment if defensive_base else max_relative_adjustment
    lower = 1.0 - cap
    upper = 1.0 + cap
    capped = max(lower, min(upper, effective))
    if capped != effective:
        reasons.append("relative_adjustment_cap")
        effective = capped

    if defensive_base and high_conviction:
        reasons.append("defensive_guard_high_conviction")

    return effective, reasons


def _decision_from_effective_multiplier(multiplier: float) -> str:
    if abs(multiplier - 1.0) <= 1e-9:
        return "keep"
    if multiplier < 1.0:
        return "reduce"
    return "boost"


def _final_multiplier(base_weight: float, adjusted_weight: float) -> float:
    if abs(base_weight) <= 1e-12:
        return 1.0
    return adjusted_weight / base_weight


def _append_guard(audit_item: dict[str, Any], reason: str) -> None:
    reasons = audit_item.setdefault("guard_reasons", [])
    if reason not in reasons:
        reasons.append(reason)


def _assign_cash_weight(result: pd.DataFrame, audit: dict[str, dict], cash_total: float) -> pd.DataFrame:
    cash_total = max(0.0, float(cash_total))
    cash_mask = result["asset_type"] == "cash"
    if cash_mask.any():
        base_cash_weights = result.loc[cash_mask, "base_weight"].astype(float)
        base_cash_sum = float(base_cash_weights.sum())
        if base_cash_sum > 0:
            result.loc[cash_mask, "adjusted_weight"] = cash_total * base_cash_weights / base_cash_sum
        else:
            result.loc[cash_mask, "adjusted_weight"] = cash_total / int(cash_mask.sum())
        return result

    if cash_total <= 1e-12:
        return result

    cash_row = pd.DataFrame(
        [{"ticker": "CASH", "base_weight": 0.0, "adjusted_weight": cash_total, "asset_type": "cash"}]
    )
    audit["CASH"] = {
        "asset_type": "cash",
        "base_weight": 0.0,
        "adjusted_weight": cash_total,
        "raw_multiplier": 1.0,
        "pre_budget_multiplier": 1.0,
        "multiplier": 1.0,
        "raw_decision": "keep",
        "decision": "keep",
        "guard_reasons": ["cash_residual_created"],
        "flags": None,
    }
    return pd.concat([result, cash_row], ignore_index=True)


def _finalize_audit_from_result(result: pd.DataFrame, audit: dict[str, dict]) -> None:
    for _, row in result.iterrows():
        ticker = str(row["ticker"]).upper()
        item = audit.get(ticker)
        if item is None:
            continue
        base_weight = float(row["base_weight"])
        adjusted_weight = float(row["adjusted_weight"])
        final_multiplier = _final_multiplier(base_weight, adjusted_weight)
        item["adjusted_weight"] = adjusted_weight
        item["multiplier"] = final_multiplier
        item["decision"] = _decision_from_effective_multiplier(final_multiplier)


def apply_research_adjustment(
    base_weights: pd.DataFrame,
    flags_by_ticker: dict[str, dict],
    *,
    no_op_band: tuple[float, float] = (0.90, 1.10),
    max_relative_adjustment: float = 0.20,
    defensive_stock_threshold: float = 0.50,
    defensive_cash_threshold: float = 0.20,
    defensive_max_relative_adjustment: float = 0.10,
    stock_boost_cash_funding_limit: float = 0.0,
    stock_boost_etf_funding_limit: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    audit = {}
    cash_release = 0.0
    base_stock_total = float(base_weights.loc[base_weights["asset_type"] == "stock", "weight"].sum())
    base_cash_total = float(base_weights.loc[base_weights["asset_type"] == "cash", "weight"].sum())
    defensive_base = base_stock_total <= defensive_stock_threshold or base_cash_total >= defensive_cash_threshold

    for _, row in base_weights.iterrows():
        ticker = str(row["ticker"]).upper()
        base_weight = float(row["weight"])
        asset_type = str(row.get("asset_type", _infer_asset_type(ticker)))

        if asset_type in {"cash", "etf"}:
            rows.append({"ticker": ticker, "base_weight": base_weight, "adjusted_weight": base_weight, "asset_type": asset_type})
            audit[ticker] = {
                "asset_type": asset_type,
                "base_weight": base_weight,
                "adjusted_weight": base_weight,
                "raw_multiplier": 1.0,
                "pre_budget_multiplier": 1.0,
                "multiplier": 1.0,
                "raw_decision": "keep",
                "decision": "keep",
                "guard_reasons": [],
                "flags": None,
            }
            continue

        flags = flags_by_ticker.get(ticker)
        if not flags:
            raw_multiplier = 1.0
            effective_multiplier = 1.0
            detail = {"decision": "keep", "multiplier": 1.0}
            guard_reasons: list[str] = []
        else:
            raw_multiplier, detail = score_flags(flags)
            effective_multiplier, guard_reasons = _effective_multiplier(
                raw_multiplier=raw_multiplier,
                flags=flags,
                defensive_base=defensive_base,
                no_op_band=no_op_band,
                max_relative_adjustment=max_relative_adjustment,
                defensive_max_relative_adjustment=defensive_max_relative_adjustment,
            )

        adjusted = base_weight * effective_multiplier
        cash_release += max(0.0, base_weight - adjusted)
        rows.append(
            {
                "ticker": ticker,
                "base_weight": base_weight,
                "adjusted_weight": adjusted,
                "asset_type": asset_type,
            }
        )
        audit[ticker] = {
            "asset_type": asset_type,
            "base_weight": base_weight,
            "adjusted_weight": adjusted,
            "raw_multiplier": raw_multiplier,
            "pre_budget_multiplier": effective_multiplier,
            "multiplier": effective_multiplier,
            "raw_decision": detail.get("decision", "keep"),
            "decision": _decision_from_effective_multiplier(effective_multiplier),
            "guard_reasons": guard_reasons,
            "flags": flags,
        }

    result = pd.DataFrame(rows)
    stock_mask = result["asset_type"] == "stock"
    etf_mask = result["asset_type"] == "etf"
    etf_total = float(result.loc[etf_mask, "adjusted_weight"].sum())

    funding_enabled = stock_boost_cash_funding_limit > 0 or stock_boost_etf_funding_limit > 0
    cash_funding = 0.0
    etf_funding = 0.0
    stock_budget_scale = 1.0
    stock_extra_scale = 1.0
    stock_extra_requested = 0.0

    if funding_enabled and stock_mask.any():
        stock_base = result.loc[stock_mask, "base_weight"].astype(float)
        stock_adjusted = result.loc[stock_mask, "adjusted_weight"].astype(float)
        stock_extra = (stock_adjusted - stock_base).clip(lower=0.0)
        stock_reduction = (stock_base - stock_adjusted).clip(lower=0.0)
        stock_extra_requested = float(stock_extra.sum())
        cash_release = float(stock_reduction.sum())

        cash_limit = max(0.0, float(stock_boost_cash_funding_limit))
        etf_limit = max(0.0, float(stock_boost_etf_funding_limit))
        cash_funding = min(base_cash_total, cash_limit, stock_extra_requested)
        remaining_extra = max(0.0, stock_extra_requested - cash_funding)
        etf_funding = min(etf_total, etf_limit, remaining_extra)
        funded_extra = cash_funding + etf_funding

        if stock_extra_requested > 0:
            stock_extra_scale = min(1.0, funded_extra / stock_extra_requested)
            boost_indices = stock_extra[stock_extra > 0].index
            result.loc[boost_indices, "adjusted_weight"] = (
                result.loc[boost_indices, "base_weight"].astype(float)
                + stock_extra.loc[boost_indices] * stock_extra_scale
            )
            if stock_extra_scale < 1.0:
                for ticker in result.loc[boost_indices, "ticker"].astype(str).str.upper():
                    _append_guard(audit[ticker], "stock_boost_funding_limit")

        if etf_funding > 0 and etf_total > 0:
            etf_scale = max(0.0, (etf_total - etf_funding) / etf_total)
            result.loc[etf_mask, "adjusted_weight"] = result.loc[etf_mask, "adjusted_weight"] * etf_scale
            for ticker in result.loc[etf_mask, "ticker"].astype(str).str.upper():
                _append_guard(audit[ticker], "etf_funding_source")

        cash_total = base_cash_total - cash_funding + cash_release
    else:
        cash_total = float(result.loc[result["asset_type"] == "cash", "adjusted_weight"].sum())
        stock_total = float(result.loc[stock_mask, "adjusted_weight"].sum())
        max_stock_budget = max(0.0, 1.0 - etf_total - cash_total)
        if stock_total > max_stock_budget and stock_total > 0:
            stock_budget_scale = max_stock_budget / stock_total
            result.loc[stock_mask, "adjusted_weight"] = result.loc[stock_mask, "adjusted_weight"] * stock_budget_scale
            for ticker in result.loc[stock_mask, "ticker"].astype(str).str.upper():
                _append_guard(audit[ticker], "portfolio_stock_budget_scale")
            stock_total = float(result.loc[stock_mask, "adjusted_weight"].sum())
        cash_total = max(0.0, 1.0 - stock_total - etf_total)

    result = _assign_cash_weight(result, audit, cash_total)
    _finalize_audit_from_result(result, audit)
    audit["__portfolio__"] = {
        "cash_release": cash_release,
        "total_adjusted_weight": float(result["adjusted_weight"].sum()),
        "cash_weight": float(result.loc[result["asset_type"] == "cash", "adjusted_weight"].sum()),
        "base_stock_weight": base_stock_total,
        "base_cash_weight": base_cash_total,
        "defensive_base": defensive_base,
        "no_op_band": list(no_op_band),
        "max_relative_adjustment": max_relative_adjustment,
        "defensive_max_relative_adjustment": defensive_max_relative_adjustment,
        "stock_boost_cash_funding_limit": stock_boost_cash_funding_limit,
        "stock_boost_etf_funding_limit": stock_boost_etf_funding_limit,
        "stock_extra_requested": stock_extra_requested,
        "cash_funding": cash_funding,
        "etf_funding": etf_funding,
        "stock_extra_scale": stock_extra_scale,
        "stock_budget_scale": stock_budget_scale,
    }
    return result.sort_values(["asset_type", "ticker"]).reset_index(drop=True), audit
