"""Valuation module: is a stock overpriced or underpriced?

The valuation group edits only this file. Implement two functions.

value(as_of, tickers, data) -> DataFrame
    One row per ticker you can value, with columns
        ticker, fair_value, low, high, mispricing, method_spread
    fair_value     your per-share estimate
    low, high      a range you believe contains the value (low <= fair_value <= high)
    mispricing     fair_value / price - 1, where price = data.last_close([ticker])
                   positive = undervalued, negative = overvalued
    method_spread  highest / lowest fair value across your methods (1.0 for one method)
    Leave out tickers you cannot value (negative earnings, missing data).

filter_candidates(as_of, candidates, data) -> candidates
    Called once a week with every stock the selection model scored that month
    (a tuple of selection.Prediction(symbol, bucket, rank_score, weight_score)).
    Return the ones you keep. The portfolio then takes the 10 best-ranked
    growth_tech names among them, so a removed name is replaced by the next one.
    You may only remove rows. Only growth_tech names can enter the portfolio,
    so valuing other buckets is wasted time.

Rules
  * Read data only through ``data`` (shared.data_view.DataView). It returns
    nothing dated after as_of. Reading files or the network inside these
    functions raises an error.
  * Do not use an LLM to produce numbers for past dates: it has seen the future.
  * Do not use the database columns pe and ev_multiple; their basis is not
    verified. Compute multiples from EPS, revenue_per_share and prices.

Useful calls
    data.fundamentals(tickers)    filings known by as_of; per-share values are single-quarter
    data.last_close(tickers)      latest close on or before as_of
    data.close(tickers, 1260)     about 5 years of daily closes (beta, history bands)
    data.members()                index members on as_of, for peer groups
    data.macro("DGS10")           10-year Treasury yield, percent

Building blocks
    valuation/finrobot/           FinRobot's DCF engine: calculate_dcf, calculate_sensitivity, adjust_beta_blume
    valuation/examples.py         a worked trailing-P/E method and a per-share DCF

Run
    python evaluate_valuation.py --period dev               rank IC and range coverage
    python run_portfolio.py --period dev --valuation        effect on the portfolio
    python check_submission.py --valuation
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from valuation.finrobot import DCFInputs, adjust_beta_blume, calculate_dcf, calculate_sensitivity

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]


@dataclass(frozen=True)
class DCFConfig:
    """Immutable, explicit assumptions for the standalone DCF method."""

    name: str
    projection_years: int = 10
    initial_growth_min: float = -0.10
    initial_growth_max: float = 0.30
    growth_fade: str = "linear"
    growth_half_life_years: float = 5.0
    beta_min: float = 0.0
    beta_max: float = 5.0
    clip_raw_beta_before_blume: bool = True
    equity_risk_premium: float = 0.05
    terminal_growth: float = 0.025
    credit_spread: float = 0.02
    nwc_pct_revenue: float = 0.0
    wacc_sensitivity: float = 0.01
    terminal_growth_sensitivity: float = 0.005

    def __post_init__(self) -> None:
        if self.projection_years <= 0:
            raise ValueError("projection_years must be positive")
        if self.initial_growth_min > self.initial_growth_max:
            raise ValueError("initial growth bounds are reversed")
        if self.growth_fade not in {"linear", "exponential"}:
            raise ValueError("growth_fade must be 'linear' or 'exponential'")
        if self.growth_half_life_years <= 0:
            raise ValueError("growth_half_life_years must be positive")
        if self.beta_min > self.beta_max:
            raise ValueError("beta bounds are reversed")


LEGACY_DCF_CONFIG = DCFConfig(name="legacy")
LOCKED_DCF_CONFIG = DCFConfig(
    name="locked",
    initial_growth_min=-0.05,
    initial_growth_max=0.35,
    growth_fade="exponential",
    growth_half_life_years=5.0,
    beta_min=0.5,
    beta_max=2.5,
    clip_raw_beta_before_blume=False,
)
DEFAULT_DCF_CONFIG = LOCKED_DCF_CONFIG

# Backward-compatible names used by the original Phase 3 evidence scripts.
DCF_PROJECTION_YEARS = LEGACY_DCF_CONFIG.projection_years
DCF_EQUITY_RISK_PREMIUM = LEGACY_DCF_CONFIG.equity_risk_premium
DCF_TERMINAL_GROWTH = LEGACY_DCF_CONFIG.terminal_growth
DCF_CREDIT_SPREAD = LEGACY_DCF_CONFIG.credit_spread
DCF_NWC_PCT_REVENUE = LEGACY_DCF_CONFIG.nwc_pct_revenue
DCF_WACC_SENSITIVITY = LEGACY_DCF_CONFIG.wacc_sensitivity
DCF_TG_SENSITIVITY = LEGACY_DCF_CONFIG.terminal_growth_sensitivity
MIN_BETA_WEEKLY_RETURNS = 52
PE_MULTIPLE_MIN = 0.0
PE_MULTIPLE_MAX = 200.0
MIN_PEER_COUNT = 5
EV_EBITDA_MULTIPLE_MIN = 0.0
EV_EBITDA_MULTIPLE_MAX = 100.0
MIN_HISTORICAL_MULTIPLES = 8

# Phase 6 frozen aggregation/filter defaults.  The development grid and the
# preregistered 2022-2023 validation comparison selected the stricter
# three-method consensus rule; see output/phase6/final_configuration_selection.json.
VALUATION_AGGREGATION = "median"
FILTER_RULE = "absolute"
OVERVALUATION_THRESHOLD = -0.30
MIN_METHODS_FOR_FILTER = 3
MAX_METHOD_SPREAD_FOR_FILTER = 2.0
QUANTILE_REMOVAL_FRACTION = 0.20
QUANTILE_MINIMUM_POOL = 5

# Single-quarter values that become trailing-twelve-month values by summing
# the latest four filings visible through DataView at the decision date.
_TTM_SUM_COLUMNS = (
    "EPS",
    "revenue_per_share",
    "fcf_per_share",
    "ocf_per_share",
    "capex_per_share",
)


def _deduplicate_visible_quarters(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic, point-in-time-visible row per ticker-quarter."""
    required = {"ticker", "datadate"}
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"fundamentals is missing required columns: {sorted(missing)}")
    if fundamentals.empty:
        return fundamentals.copy()

    frame = fundamentals.copy()
    frame["datadate"] = pd.to_datetime(frame["datadate"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "datadate"])
    frame["ticker"] = frame["ticker"].astype(str)

    # When duplicate versions are supplied, mirror FundamentalStore's ranking:
    # known timestamps descending, missing timestamps last, and a stable input
    # order tie-break. DataView has already excluded filings unavailable at
    # as_of, so this ranks only visible versions and normally is a no-op.
    version_columns = [c for c in ("accepted_date", "filing_date") if c in frame]
    for column in version_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["_input_order"] = range(len(frame))
    sort_columns = ["ticker", "datadate", *version_columns, "_input_order"]
    ascending = [True, True, *([False] * len(version_columns)), False]
    frame = frame.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )
    frame = frame.drop_duplicates(["ticker", "datadate"], keep="first").drop(columns="_input_order")

    return frame.sort_values(["ticker", "datadate"]).reset_index(drop=True)


def _latest_four_visible_quarters(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return the latest four distinct visible quarters for eligible tickers.

    ``DataView.fundamentals`` already applies the point-in-time filing filter
    and selects the latest filing version visible at ``as_of``. The defensive
    de-duplication makes the TTM calculation deterministic for synthetic tests
    and future course-data refreshes. Tickers with fewer than four visible
    quarters are excluded rather than imputed; quarter consecutiveness is not
    required.
    """
    frame = _deduplicate_visible_quarters(fundamentals)
    if frame.empty:
        return frame
    latest = frame.groupby("ticker", group_keys=False).tail(4)
    counts = latest.groupby("ticker")["datadate"].transform("size")
    return latest.loc[counts.eq(4)].sort_values(["ticker", "datadate"]).reset_index(drop=True)


def build_ttm_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Construct filing-aware TTM measures from visible quarterly records.

    The caller must pass the result of ``data.fundamentals(...)`` for the
    decision date. Additive per-share measures are summed. EBITDA and
    operating income per share are reconstructed quarter by quarter as
    revenue per share times the corresponding margin, then summed. Balance
    sheet measures use the latest visible quarter.

    Missing components remain missing (``min_count=4``) and no future or
    cross-sectional values are used for imputation.
    """
    latest = _latest_four_visible_quarters(fundamentals)
    output_columns = [
        "ticker", "quarter_count", "first_quarter", "latest_quarter", "quarter_span_days",
        "sector", "eps_ttm", "revenue_ttm", "fcf_ttm", "ocf_ttm", "capex_ttm",
        "ebitda_ttm", "operating_income_ttm", "ebitda_margin_median",
        "operating_margin_median", "effective_tax_rate_median", "cash_per_share",
        "interest_debt_per_share", "net_debt_per_share", "debt_to_capital",
        "latest_filing_date", "latest_accepted_date",
    ]
    if latest.empty:
        return pd.DataFrame(columns=output_columns)

    numeric_inputs = set(_TTM_SUM_COLUMNS) | {
        "ebitda_margin", "operating_margin", "effective_tax_rate", "cash_per_share",
        "interest_debt_per_share", "debt_to_capital",
    }
    for column in numeric_inputs & set(latest.columns):
        latest[column] = pd.to_numeric(latest[column], errors="coerce")

    def product(left: str, right: str) -> pd.Series:
        if left not in latest or right not in latest:
            return pd.Series(float("nan"), index=latest.index)
        return latest[left] * latest[right]

    latest["_ebitda_per_share"] = product("revenue_per_share", "ebitda_margin")
    latest["_operating_income_per_share"] = product("revenue_per_share", "operating_margin")
    grouped = latest.groupby("ticker", sort=True)

    result = pd.DataFrame(index=grouped.size().index)
    result["quarter_count"] = grouped.size()
    result["first_quarter"] = grouped["datadate"].min()
    result["latest_quarter"] = grouped["datadate"].max()
    result["quarter_span_days"] = (result["latest_quarter"] - result["first_quarter"]).dt.days

    def sum_four(source: str) -> pd.Series:
        if source not in latest:
            return pd.Series(float("nan"), index=result.index)
        return grouped[source].sum(min_count=4)

    result["eps_ttm"] = sum_four("EPS")
    result["revenue_ttm"] = sum_four("revenue_per_share")
    result["fcf_ttm"] = sum_four("fcf_per_share")
    result["ocf_ttm"] = sum_four("ocf_per_share")
    result["capex_ttm"] = sum_four("capex_per_share")
    result["ebitda_ttm"] = sum_four("_ebitda_per_share")
    result["operating_income_ttm"] = sum_four("_operating_income_per_share")

    for source, target in (
        ("ebitda_margin", "ebitda_margin_median"),
        ("operating_margin", "operating_margin_median"),
        ("effective_tax_rate", "effective_tax_rate_median"),
    ):
        result[target] = grouped[source].median() if source in latest else float("nan")

    latest_rows = latest.sort_values(["ticker", "datadate"]).groupby("ticker").tail(1).set_index("ticker")
    result["sector"] = latest_rows["gsector"] if "gsector" in latest_rows else pd.NA
    for source in ("cash_per_share", "interest_debt_per_share", "debt_to_capital"):
        result[source] = latest_rows[source] if source in latest_rows else float("nan")
    result["net_debt_per_share"] = result["interest_debt_per_share"] - result["cash_per_share"]
    result["latest_filing_date"] = latest_rows["filing_date"] if "filing_date" in latest_rows else pd.NaT
    result["latest_accepted_date"] = latest_rows["accepted_date"] if "accepted_date" in latest_rows else pd.NaT

    return result.reset_index().reindex(columns=output_columns)


def estimate_da_pct_revenue(quarters: pd.DataFrame, *, require_quarters: int = 4) -> float | None:
    """Estimate D&A/revenue from paired quarterly margins, floored at zero.

    This is a deterministic modeling assumption for a future DCF input, not an
    imputation of either historical margin. A result is returned only when the
    requested number of quarters has both margins available.
    """
    required = {"ebitda_margin", "operating_margin"}
    if not required <= set(quarters.columns):
        return None
    ebitda = pd.to_numeric(quarters["ebitda_margin"], errors="coerce")
    operating = pd.to_numeric(quarters["operating_margin"], errors="coerce")
    paired = (ebitda - operating).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(paired) < require_quarters:
        return None
    return max(0.0, float(paired.median()))


def statutory_federal_tax_rate(fiscal_period_end) -> float:
    """Simple point-in-time U.S. federal statutory-rate assumption.

    The Tax Cuts and Jobs Act reduced the top federal corporate rate from 35%
    to 21% for tax years beginning after 2017. Fiscal-year transitions and
    non-U.S. jurisdictions require disclosure and remain a Phase 3 limitation.
    """
    period = pd.Timestamp(fiscal_period_end)
    if pd.isna(period):
        raise ValueError("fiscal_period_end is required for the statutory-rate assumption")
    return 0.35 if period.year < 2018 else 0.21


def tax_rate_for_dcf(raw_rate, fiscal_period_end) -> float:
    """Prepare a future DCF tax input without changing the historical source.

    Positive finite observations are clipped to [0, 35%]. Zero, negative, or
    missing observations use the date-dependent statutory-rate assumption.
    """
    rate = pd.to_numeric(pd.Series([raw_rate]), errors="coerce").iloc[0]
    if pd.notna(rate) and float(rate) > 0:
        return float(min(float(rate), 0.35))
    return statutory_federal_tax_rate(fiscal_period_end)


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def capex_spending_magnitude(quarterly_capex) -> tuple[float | None, str]:
    """Convert a consistent source sign convention to positive cash spending.

    FinRobot subtracts ``capex_pct_revenue`` from projected cash flow, so its
    input must be a nonnegative spending magnitude. Four all-negative source
    values are outflows and are negated; four all-positive values are already
    magnitudes. A mixed-sign or incomplete sequence is ambiguous and rejected.
    """
    values = pd.to_numeric(pd.Series(quarterly_capex), errors="coerce")
    if len(values) != 4 or values.notna().sum() != 4:
        return None, "missing"
    if (values < 0).any() and (values > 0).any():
        return None, "mixed_sign"
    if values.eq(0).all():
        return 0.0, "zero"
    if values.le(0).all():
        return float(-values.sum()), "negative_outflow"
    if values.ge(0).all():
        return float(values.sum()), "positive_magnitude"
    return None, "invalid"


def _weekly_beta_diagnostic(
    ticker: str, data, config: DCFConfig
) -> tuple[dict | None, str | None, str | None]:
    """Pair independently constructed contemporaneous Friday returns."""
    prices = data.close([ticker, "SPY"], lookback=252 * 5)
    stock = pd.to_numeric(prices.get(ticker), errors="coerce").dropna()
    market = pd.to_numeric(prices.get("SPY"), errors="coerce").dropna()
    if stock.empty or market.empty:
        missing = ticker if stock.empty else "SPY"
        return None, "beta_missing", f"no_daily_prices:{missing}"

    stock_weekly = stock.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    market_weekly = market.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    weekly = pd.concat(
        [stock_weekly.rename(ticker), market_weekly.rename("SPY")], axis=1, join="inner"
    ).dropna()
    if len(weekly) < MIN_BETA_WEEKLY_RETURNS:
        return None, "beta_insufficient_pairs", f"paired_weekly_returns:{len(weekly)}"
    market_variance = float(weekly["SPY"].var())
    if not np.isfinite(market_variance) or market_variance <= 0:
        return None, "invalid_market_variance", f"market_variance:{market_variance}"
    raw = float(weekly[ticker].cov(weekly["SPY"]) / market_variance)
    if not np.isfinite(raw):
        return None, "invalid_market_variance", "nonfinite_covariance_beta"
    beta_for_blume = float(np.clip(raw, 0.0, 5.0)) if config.clip_raw_beta_before_blume else raw
    adjusted_unclipped = float(adjust_beta_blume(beta_for_blume))
    adjusted = float(np.clip(adjusted_unclipped, config.beta_min, config.beta_max))
    return (
        {
            "raw_beta": raw,
            "adjusted_beta_unclipped": adjusted_unclipped,
            "adjusted_beta": adjusted,
            "beta_weeks": int(len(weekly)),
        },
        None,
        None,
    )


def _weekly_beta(
    ticker: str, data, config: DCFConfig = DEFAULT_DCF_CONFIG
) -> tuple[float, float, int] | None:
    """Backward-compatible compact beta result for callers and tests."""
    result, _, _ = _weekly_beta_diagnostic(ticker, data, config)
    if result is None:
        return None
    return result["raw_beta"], result["adjusted_beta"], result["beta_weeks"]


def _historical_growth_path(
    quarters: pd.DataFrame,
    *,
    config: DCFConfig = DEFAULT_DCF_CONFIG,
) -> tuple[list[float], float, int] | None:
    """TTM YoY growth using only complete eight-consecutive-quarter windows."""
    frame = _deduplicate_visible_quarters(quarters).sort_values("datadate")
    if len(frame) < 8 or "revenue_per_share" not in frame:
        return None
    revenue = pd.to_numeric(frame["revenue_per_share"], errors="coerce")
    ttm = revenue.rolling(4, min_periods=4).sum()
    yoy_all = (ttm / ttm.shift(4) - 1).replace([np.inf, -np.inf], np.nan)
    quarter_ordinals = frame["datadate"].dt.to_period("Q").astype("int64").to_numpy()
    consecutive = pd.Series(False, index=frame.index)
    for position in range(7, len(frame)):
        if np.all(np.diff(quarter_ordinals[position - 7: position + 1]) == 1):
            consecutive.iloc[position] = True
    yoy = yoy_all.loc[consecutive].dropna()
    if yoy.empty:
        return None
    recent = yoy.tail(4)
    initial = float(
        np.clip(recent.median(), config.initial_growth_min, config.initial_growth_max)
    )
    if config.growth_fade == "linear":
        path = np.linspace(
            initial, config.terminal_growth, config.projection_years
        ).astype(float).tolist()
    else:
        path = [
            float(
                config.terminal_growth
                + (initial - config.terminal_growth)
                * 2 ** (-(year - 1) / config.growth_half_life_years)
            )
            for year in range(1, config.projection_years + 1)
        ]
    return path, initial, int(len(yoy))


DCF_VALUE_COLUMNS = [
    "ticker", "method", "fair_value", "low", "high", "current_price",
    "wacc", "risk_free_rate", "raw_beta", "adjusted_beta", "beta_weeks",
    "initial_growth", "growth_observations", "terminal_growth", "tax_rate",
    "da_pct_revenue", "capex_pct_revenue",
]

DCF_DIAGNOSTIC_COLUMNS = [
    "ticker", "as_of", "config", "primary_status", "secondary_reason",
    "current_price", "fair_value", "low", "high", "wacc", "risk_free_rate",
    "raw_beta", "adjusted_beta_unclipped", "adjusted_beta", "beta_weeks",
    "initial_growth", "growth_observations", "terminal_growth", "tax_rate",
    "da_pct_revenue", "capex_pct_revenue", "capex_sign_convention",
]


def _dcf_valuation_with_diagnostics(
    as_of, tickers, data, config: DCFConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(dict.fromkeys(map(str, tickers)))
    if not tickers:
        return (
            pd.DataFrame(columns=DCF_VALUE_COLUMNS),
            pd.DataFrame(columns=DCF_DIAGNOSTIC_COLUMNS),
        )
    fundamentals = _deduplicate_visible_quarters(data.fundamentals(tickers))
    latest4 = _latest_four_visible_quarters(fundamentals)
    prices = pd.Series(data.last_close(tickers), dtype=float)
    treasury = data.macro("DGS10").dropna()
    risk_free_raw = _finite_float(treasury.iloc[-1]) if not treasury.empty else None
    risk_free = (
        float(np.clip(risk_free_raw / 100.0, 0.0, 0.15))
        if risk_free_raw is not None
        else None
    )

    rows, diagnostics = [], []

    def exclude(ticker: str, status: str, reason: str | None = None, **details) -> None:
        diagnostics.append(
            {
                "ticker": ticker,
                "as_of": pd.Timestamp(as_of).date().isoformat(),
                "config": config.name,
                "primary_status": status,
                "secondary_reason": reason,
                **details,
            }
        )

    for ticker in tickers:
        if risk_free is None:
            exclude(ticker, "risk_free_missing", "DGS10 has no finite observation")
            continue
        current_price = _finite_float(prices.get(ticker))
        if current_price is None or current_price <= 0:
            exclude(
                ticker,
                "nonpositive_price",
                "missing_current_price" if current_price is None else "price_not_positive",
                current_price=current_price,
                risk_free_rate=risk_free,
            )
            continue
        history = fundamentals.loc[fundamentals["ticker"].eq(ticker)].sort_values("datadate")
        quarters = latest4.loc[latest4["ticker"].eq(ticker)].sort_values("datadate")
        if len(history) < 8 or len(quarters) != 4:
            exclude(
                ticker,
                "insufficient_quarters",
                f"visible:{len(history)};latest_four:{len(quarters)}",
                current_price=current_price,
                risk_free_rate=risk_free,
            )
            continue
        growth = _historical_growth_path(history, config=config)
        if growth is None:
            exclude(
                ticker,
                "growth_history_missing",
                "no_valid_consecutive_eight_quarter_yoy_window",
                current_price=current_price,
                risk_free_rate=risk_free,
            )
            continue
        beta, beta_status, beta_reason = _weekly_beta_diagnostic(ticker, data, config)
        if beta is None:
            exclude(
                ticker,
                beta_status or "beta_missing",
                beta_reason,
                current_price=current_price,
                risk_free_rate=risk_free,
            )
            continue

        required = {
            "revenue_per_share", "capex_per_share", "ebitda_margin",
            "operating_margin", "effective_tax_rate",
        }
        if not required <= set(quarters.columns):
            exclude(
                ticker, "missing_component", "missing_required_columns",
                current_price=current_price, risk_free_rate=risk_free, **beta,
            )
            continue
        revenue_q = pd.to_numeric(quarters["revenue_per_share"], errors="coerce")
        ebitda_margin_q = pd.to_numeric(quarters["ebitda_margin"], errors="coerce")
        tax_q = pd.to_numeric(quarters["effective_tax_rate"], errors="coerce")
        if revenue_q.notna().sum() != 4 or ebitda_margin_q.notna().sum() != 4:
            exclude(
                ticker, "missing_component", "incomplete_revenue_or_ebitda_margin",
                current_price=current_price, risk_free_rate=risk_free, **beta,
            )
            continue
        revenue_ttm = _finite_float(revenue_q.sum())
        if revenue_ttm is None or revenue_ttm <= 0:
            exclude(
                ticker, "nonpositive_revenue", f"revenue_ttm:{revenue_ttm}",
                current_price=current_price, risk_free_rate=risk_free, **beta,
            )
            continue
        ebitda_margin = _finite_float(ebitda_margin_q.median())
        da_pct = estimate_da_pct_revenue(quarters, require_quarters=4)
        if ebitda_margin is None or da_pct is None:
            exclude(
                ticker, "missing_component", "incomplete_paired_margin_history",
                current_price=current_price, risk_free_rate=risk_free, **beta,
            )
            continue
        if ebitda_margin <= 0:
            exclude(
                ticker, "invalid_margin", f"ebitda_margin:{ebitda_margin}",
                current_price=current_price, risk_free_rate=risk_free, **beta,
            )
            continue
        capex_magnitude, capex_convention = capex_spending_magnitude(
            quarters["capex_per_share"]
        )
        if capex_magnitude is None:
            exclude(
                ticker, "invalid_capex", capex_convention,
                current_price=current_price, risk_free_rate=risk_free, **beta,
                capex_sign_convention=capex_convention,
            )
            continue
        capex_pct = _finite_float(capex_magnitude / revenue_ttm)
        if capex_pct is None or capex_pct < 0:
            exclude(
                ticker, "invalid_capex", f"capex_pct:{capex_pct}",
                current_price=current_price, risk_free_rate=risk_free, **beta,
                capex_sign_convention=capex_convention,
            )
            continue

        latest = quarters.iloc[-1]
        debt_ratio = _finite_float(latest.get("debt_to_capital"))
        debt = _finite_float(latest.get("interest_debt_per_share"))
        cash = _finite_float(latest.get("cash_per_share"))
        if None in (debt_ratio, debt, cash):
            exclude(
                ticker, "invalid_balance_sheet", "missing_debt_cash_or_debt_ratio",
                current_price=current_price, risk_free_rate=risk_free, **beta,
                capex_sign_convention=capex_convention,
            )
            continue

        ebitda_margin = float(np.clip(ebitda_margin, 0.0, 1.0))
        da_pct = float(np.clip(da_pct, 0.0, 0.5))
        capex_pct = float(np.clip(capex_pct, 0.0, 1.0))
        debt_ratio = float(np.clip(debt_ratio, 0.0, 0.95))
        raw_tax = _finite_float(tax_q.median())
        tax_rate = tax_rate_for_dcf(raw_tax, latest["datadate"])
        growth_path, initial_growth, growth_observations = growth

        try:
            inputs = DCFInputs(
                revenue_base=revenue_ttm,
                revenue_growth_rates=growth_path,
                ebitda_margin=ebitda_margin,
                da_pct_revenue=da_pct,
                capex_pct_revenue=capex_pct,
                nwc_pct_revenue=config.nwc_pct_revenue,
                tax_rate=tax_rate,
                risk_free_rate=risk_free,
                beta=beta["adjusted_beta"],
                equity_risk_premium=config.equity_risk_premium,
                cost_of_debt=float(np.clip(risk_free + config.credit_spread, 0.0, 0.20)),
                debt_ratio=debt_ratio,
                terminal_growth_rate=config.terminal_growth,
                shares_outstanding=1.0,
                net_debt=debt - cash,
            )
            result = calculate_dcf(inputs)
        except (ValueError, ArithmeticError) as exc:
            exclude(
                ticker, "engine_rejected", str(exc), current_price=current_price,
                risk_free_rate=risk_free, initial_growth=initial_growth,
                growth_observations=growth_observations,
                terminal_growth=config.terminal_growth, tax_rate=tax_rate,
                da_pct_revenue=da_pct, capex_pct_revenue=capex_pct,
                capex_sign_convention=capex_convention, **beta,
            )
            continue
        wacc_values = [
            max(0.001, result.wacc - config.wacc_sensitivity),
            result.wacc,
            result.wacc + config.wacc_sensitivity,
        ]
        tg_values = [
            max(0.0, config.terminal_growth - config.terminal_growth_sensitivity),
            config.terminal_growth,
            min(0.05, config.terminal_growth + config.terminal_growth_sensitivity),
        ]
        try:
            grid = calculate_sensitivity(inputs, wacc_values, tg_values)
        except (ValueError, ArithmeticError) as exc:
            exclude(
                ticker, "invalid_sensitivity_grid", str(exc), current_price=current_price,
                fair_value=float(result.implied_price), wacc=float(result.wacc),
                risk_free_rate=risk_free, **beta,
            )
            continue
        cells = [
            float(value)
            for row in grid["implied_prices"]
            for value in row
            if value is not None and np.isfinite(value) and value > 0
        ]
        if (
            not cells
            or not np.isfinite(result.implied_price)
            or result.implied_price <= 0
        ):
            exclude(
                ticker, "invalid_sensitivity_grid", "no_positive_finite_grid_cells",
                current_price=current_price, fair_value=_finite_float(result.implied_price),
                wacc=float(result.wacc), risk_free_rate=risk_free, **beta,
            )
            continue
        cells.append(float(result.implied_price))
        value_row = {
            "ticker": ticker,
            "method": "dcf",
            "fair_value": float(result.implied_price),
            "low": min(cells),
            "high": max(cells),
            "current_price": current_price,
            "wacc": float(result.wacc),
            "risk_free_rate": risk_free,
            "raw_beta": beta["raw_beta"],
            "adjusted_beta": beta["adjusted_beta"],
            "beta_weeks": beta["beta_weeks"],
            "initial_growth": initial_growth,
            "growth_observations": growth_observations,
            "terminal_growth": config.terminal_growth,
            "tax_rate": tax_rate,
            "da_pct_revenue": da_pct,
            "capex_pct_revenue": capex_pct,
        }
        rows.append(value_row)
        exclude(
            ticker, "valued", None, current_price=current_price,
            fair_value=value_row["fair_value"], low=value_row["low"], high=value_row["high"],
            wacc=value_row["wacc"], risk_free_rate=risk_free,
            initial_growth=initial_growth, growth_observations=growth_observations,
            terminal_growth=config.terminal_growth, tax_rate=tax_rate,
            da_pct_revenue=da_pct, capex_pct_revenue=capex_pct,
            capex_sign_convention=capex_convention, **beta,
        )
    return (
        pd.DataFrame(rows, columns=DCF_VALUE_COLUMNS),
        pd.DataFrame(diagnostics).reindex(columns=DCF_DIAGNOSTIC_COLUMNS),
    )


def dcf_valuation(
    as_of, tickers, data, *, config: DCFConfig = DEFAULT_DCF_CONFIG
) -> pd.DataFrame:
    """Standalone point-in-time per-share DCF estimates using FinRobot."""
    values, _ = _dcf_valuation_with_diagnostics(as_of, tickers, data, config)
    return values


def dcf_valuation_diagnostics(
    as_of, tickers, data, *, config: DCFConfig = DEFAULT_DCF_CONFIG
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normal DCF estimates plus one primary status per attempted ticker."""
    return _dcf_valuation_with_diagnostics(as_of, tickers, data, config)


def _modeled_availability_date(datadate) -> pd.Timestamp | pd.NaT:
    quarter = pd.Timestamp(datadate)
    month_day = quarter.strftime("%m-%d")
    if month_day == "03-31":
        return pd.Timestamp(quarter.year, 6, 1)
    if month_day == "06-30":
        return pd.Timestamp(quarter.year, 9, 1)
    if month_day == "09-30":
        return pd.Timestamp(quarter.year, 12, 1)
    if month_day == "12-31":
        return pd.Timestamp(quarter.year + 1, 3, 1)
    return pd.NaT


def _availability_dates(frame: pd.DataFrame) -> pd.Series:
    accepted = pd.to_datetime(frame.get("accepted_date"), errors="coerce")
    filed = pd.to_datetime(frame.get("filing_date"), errors="coerce")
    modeled = frame["datadate"].map(_modeled_availability_date)
    return accepted.fillna(filed).fillna(modeled)


def _price_on_or_before(series: pd.Series, day, *, max_days: int = 14) -> float | None:
    timestamp = pd.Timestamp(day)
    window = series.loc[timestamp - pd.Timedelta(days=max_days): timestamp].dropna()
    return _finite_float(window.iloc[-1]) if not window.empty else None


def _historical_ev_ebitda_multiples(
    ticker: str,
    fundamentals: pd.DataFrame,
    prices: pd.Series,
    as_of,
    *,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Own-history point-in-time observations within a maximum lookback."""
    frame = _deduplicate_visible_quarters(fundamentals)
    if frame.empty:
        return pd.DataFrame(columns=["date", "multiple"])
    frame = frame.copy()
    frame["availability_date"] = _availability_dates(frame)
    as_of_ts = pd.Timestamp(as_of)
    frame = frame.loc[frame["availability_date"].notna() & frame["availability_date"].le(as_of_ts)]
    start = as_of_ts - pd.DateOffset(years=lookback_years)
    rows = []
    for event_date in sorted(frame["availability_date"].unique()):
        event_date = pd.Timestamp(event_date)
        if event_date < start:
            continue
        visible = frame.loc[frame["availability_date"].le(event_date)].sort_values("datadate").tail(4)
        if len(visible) != 4:
            continue
        revenue = pd.to_numeric(visible["revenue_per_share"], errors="coerce")
        margin = pd.to_numeric(visible["ebitda_margin"], errors="coerce")
        if revenue.notna().sum() != 4 or margin.notna().sum() != 4:
            continue
        ebitda_ttm = _finite_float((revenue * margin).sum())
        latest = visible.iloc[-1]
        debt = _finite_float(latest.get("interest_debt_per_share"))
        cash = _finite_float(latest.get("cash_per_share"))
        price = _price_on_or_before(prices, event_date)
        if None in (ebitda_ttm, debt, cash, price) or ebitda_ttm <= 0 or price <= 0:
            continue
        multiple = (price + debt - cash) / ebitda_ttm
        if np.isfinite(multiple) and EV_EBITDA_MULTIPLE_MIN < multiple <= EV_EBITDA_MULTIPLE_MAX:
            rows.append({"ticker": ticker, "date": event_date, "multiple": float(multiple)})
    return pd.DataFrame(rows, columns=["ticker", "date", "multiple"])


EV_EBITDA_VALUE_COLUMNS = [
    "ticker", "method", "fair_value", "low", "high", "current_price",
    "multiple_p25", "multiple_median", "multiple_p75", "history_count",
    "history_span_days",
]


def _historical_ev_ebitda_with_diagnostics(
    as_of,
    tickers,
    data,
    *,
    lookback_years: int,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(dict.fromkeys(map(str, tickers)))
    if not tickers:
        return pd.DataFrame(columns=EV_EBITDA_VALUE_COLUMNS), pd.DataFrame()
    fundamentals = _deduplicate_visible_quarters(data.fundamentals(tickers))
    ttm = build_ttm_fundamentals(fundamentals).set_index("ticker")
    current_prices = pd.Series(data.last_close(tickers), dtype=float)
    rows, diagnostics = [], []
    for ticker in tickers:
        if ticker not in ttm.index:
            diagnostics.append(
                {
                    "ticker": ticker, "status": "insufficient_current_ttm",
                    "lookback_years": lookback_years, "history_count": 0,
                }
            )
            continue
        current_price = _finite_float(current_prices.get(ticker))
        ebitda_ttm = _finite_float(ttm.at[ticker, "ebitda_ttm"])
        net_debt = _finite_float(ttm.at[ticker, "net_debt_per_share"])
        if current_price is None or current_price <= 0:
            diagnostics.append(
                {
                    "ticker": ticker, "status": "nonpositive_price",
                    "lookback_years": lookback_years, "history_count": 0,
                    "current_price": current_price,
                }
            )
            continue
        if ebitda_ttm is None or ebitda_ttm <= 0:
            diagnostics.append(
                {
                    "ticker": ticker, "status": "nonpositive_ebitda",
                    "lookback_years": lookback_years, "history_count": 0,
                    "current_price": current_price,
                }
            )
            continue
        if net_debt is None:
            diagnostics.append(
                {
                    "ticker": ticker, "status": "invalid_balance_sheet",
                    "lookback_years": lookback_years, "history_count": 0,
                    "current_price": current_price,
                }
            )
            continue
        price_history = data.close([ticker], lookback=252 * (lookback_years + 1))[ticker]
        history = _historical_ev_ebitda_multiples(
            ticker,
            fundamentals.loc[fundamentals["ticker"].eq(ticker)],
            price_history,
            as_of,
            lookback_years=lookback_years,
        )
        diagnostic = {
            "ticker": ticker,
            "lookback_years": lookback_years,
            "current_price": current_price,
            "history_count": int(len(history)),
            "history_span_days": (
                int((history["date"].max() - history["date"].min()).days)
                if not history.empty else 0
            ),
        }
        if not history.empty:
            p25, median, p75 = history["multiple"].quantile([0.25, 0.5, 0.75])
            diagnostic.update(
                {
                    "multiple_p25": float(p25),
                    "multiple_median": float(median),
                    "multiple_p75": float(p75),
                }
            )
        if len(history) < minimum_observations:
            diagnostic["status"] = "insufficient_history"
            diagnostics.append(diagnostic)
            continue
        low = max(0.0, float(p25 * ebitda_ttm - net_debt))
        fair = float(median * ebitda_ttm - net_debt)
        high = float(p75 * ebitda_ttm - net_debt)
        if fair <= 0 or high <= 0:
            diagnostic.update(
                {"status": "nonpositive_fair_value", "fair_value": fair, "low": low, "high": high}
            )
            diagnostics.append(diagnostic)
            continue
        value_row = {
            "ticker": ticker,
            "method": "historical_ev_ebitda",
            "fair_value": fair,
            "low": low,
            "high": high,
            "current_price": current_price,
            "multiple_p25": float(p25),
            "multiple_median": float(median),
            "multiple_p75": float(p75),
            "history_count": int(len(history)),
            "history_span_days": diagnostic["history_span_days"],
        }
        rows.append(value_row)
        diagnostic.update(
            {"status": "eligible", "fair_value": fair, "low": low, "high": high}
        )
        diagnostics.append(diagnostic)
    return pd.DataFrame(rows, columns=EV_EBITDA_VALUE_COLUMNS), pd.DataFrame(diagnostics)


def historical_ev_ebitda_valuation(
    as_of,
    tickers,
    data,
    *,
    lookback_years: int = 5,
    minimum_observations: int = MIN_HISTORICAL_MULTIPLES,
) -> pd.DataFrame:
    """Fair value from an own-history band within a maximum lookback."""
    values, _ = _historical_ev_ebitda_with_diagnostics(
        as_of,
        tickers,
        data,
        lookback_years=lookback_years,
        minimum_observations=minimum_observations,
    )
    return values


def historical_ev_ebitda_diagnostics(
    as_of,
    tickers,
    data,
    *,
    lookback_years: int = 5,
    minimum_observations: int = MIN_HISTORICAL_MULTIPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return EV/EBITDA estimates plus one diagnostic row per attempted target."""
    return _historical_ev_ebitda_with_diagnostics(
        as_of,
        tickers,
        data,
        lookback_years=lookback_years,
        minimum_observations=minimum_observations,
    )


def peer_pe_valuation(as_of, tickers, data) -> pd.DataFrame:
    """Trailing P/E valuation from same-sector point-in-time index peers."""
    tickers = list(dict.fromkeys(map(str, tickers)))
    columns = [
        "ticker", "method", "fair_value", "low", "high", "current_price",
        "eps_ttm", "sector", "peer_count", "pe_p25", "pe_median", "pe_p75",
    ]
    if not tickers:
        return pd.DataFrame(columns=columns)
    members = set(data.members())
    universe = sorted(members | set(tickers))
    ttm = build_ttm_fundamentals(data.fundamentals(universe)).set_index("ticker")
    if ttm.empty:
        return pd.DataFrame(columns=columns)
    prices = pd.Series(data.last_close(ttm.index), dtype=float).reindex(ttm.index)
    eps = pd.to_numeric(ttm["eps_ttm"], errors="coerce")
    pe = prices / eps
    in_members = pd.Series(ttm.index.isin(members), index=ttm.index)
    valid_peer = (
        eps.gt(0)
        & prices.gt(0)
        & pe.gt(PE_MULTIPLE_MIN)
        & pe.le(PE_MULTIPLE_MAX)
        #& ttm.index.to_series().isin(members).values
        #& ttm["sector"].notna()
        & ttm["sector"].notna()
        & in_members
    )
    rows = []
    for ticker in tickers:
        if ticker not in ttm.index:
            continue
        target_eps = _finite_float(eps.get(ticker))
        current_price = _finite_float(prices.get(ticker))
        sector = ttm.at[ticker, "sector"]
        if target_eps is None or target_eps <= 0 or current_price is None or current_price <= 0 or pd.isna(sector):
            continue
        #mask = valid_peer & ttm["sector"].eq(sector) & ttm.index.to_series().ne(ticker).values
        mask = valid_peer & ttm["sector"].eq(sector) & (ttm.index != ticker)
        peer_multiples = pe.loc[mask]
        if len(peer_multiples) < MIN_PEER_COUNT:
            continue
        p25, median, p75 = peer_multiples.quantile([0.25, 0.5, 0.75])
        # modification
        if not (np.isfinite(median) and median > 0):
            continue
        if not (np.isfinite(p25) and p25 > 0):
            p25 = median
        if not (np.isfinite(p75) and p75 > 0):
            p75 = median
        
        rows.append(
            {
                "ticker": ticker,
                "method": "peer_pe",
                "fair_value": float(median * target_eps),
                "low": float(p25 * target_eps),
                "high": float(p75 * target_eps),
                "current_price": current_price,
                "eps_ttm": target_eps,
                "sector": sector,
                "peer_count": int(len(peer_multiples)),
                "pe_p25": float(p25),
                "pe_median": float(median),
                "pe_p75": float(p75),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def valuation_details(as_of, tickers, data) -> pd.DataFrame:
    """Combine standalone estimates and retain transparent decision metadata.

    The public ``value`` function projects this frame to the six assignment
    columns. The filter uses the metadata to avoid acting on a single method or
    on methods whose fair values disagree by more than the configured spread.
    """
    tickers = list(dict.fromkeys(map(str, tickers)))
    columns = [
        *COLUMNS,
        "current_price", "method_count", "methods", "filter_eligible",
        "overvaluation_flag", "filter_decision",
    ]
    if not tickers:
        return pd.DataFrame(columns=columns)
    method_frames = [
        dcf_valuation(as_of, tickers, data, config=DEFAULT_DCF_CONFIG),
        historical_ev_ebitda_valuation(as_of, tickers, data),
        peer_pe_valuation(as_of, tickers, data),
    ]
    available = []
    for frame in method_frames:
        if frame.empty:
            continue
        required = {"ticker", "method", "fair_value", "low", "high", "current_price"}
        if not required <= set(frame.columns):
            raise ValueError(f"standalone valuation output is missing {sorted(required - set(frame.columns))}")
        clean = frame[["ticker", "method", "fair_value", "low", "high", "current_price"]].copy()
        for column in ("fair_value", "low", "high", "current_price"):
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.loc[
            clean["fair_value"].gt(0)
            & clean["current_price"].gt(0)
            & clean["low"].le(clean["fair_value"])
            & clean["fair_value"].le(clean["high"])
            & np.isfinite(clean[["fair_value", "low", "high", "current_price"]]).all(axis=1)
        ]
        available.append(clean)
    if not available:
        return pd.DataFrame(columns=columns)
    estimates = pd.concat(available, ignore_index=True)
    estimates = estimates.drop_duplicates(["ticker", "method"], keep="first")

    order = {ticker: position for position, ticker in enumerate(tickers)}
    rows = []
    for ticker, group in estimates.groupby("ticker", sort=False):
        fair_values = group["fair_value"]
        current_price = float(group["current_price"].median())
        if current_price <= 0 or not np.isfinite(current_price):
            continue
        fair_value = float(fair_values.median())
        low = float(group["low"].median())
        high = float(group["high"].median())
        low = min(low, fair_value)
        high = max(high, fair_value)
        method_count = int(group["method"].nunique())
        method_spread = (
            float(fair_values.max() / fair_values.min()) if method_count > 1 else 1.0
        )
        mispricing = float(fair_value / current_price - 1)
        filter_eligible = (
            method_count >= MIN_METHODS_FOR_FILTER
            and method_spread <= MAX_METHOD_SPREAD_FOR_FILTER
        )
        overvaluation_flag = mispricing < OVERVALUATION_THRESHOLD
        if not filter_eligible:
            decision = "keep_insufficient_or_disagreeing_methods"
        elif overvaluation_flag:
            decision = "remove_overvalued"
        else:
            decision = "keep"
        rows.append(
            {
                "ticker": ticker,
                "fair_value": fair_value,
                "low": low,
                "high": high,
                "mispricing": mispricing,
                "method_spread": method_spread,
                "current_price": current_price,
                "method_count": method_count,
                "methods": ",".join(sorted(group["method"].astype(str))),
                "filter_eligible": filter_eligible,
                "overvaluation_flag": overvaluation_flag,
                "filter_decision": decision,
                "_order": order.get(ticker, len(order)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values("_order", kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
        .reindex(columns=columns)
    )


def value(as_of, tickers, data) -> pd.DataFrame:
    return valuation_details(as_of, tickers, data).reindex(columns=COLUMNS)


def filter_removals(details: pd.DataFrame, *, rule: str = FILTER_RULE) -> set[str]:
    """Return tickers removed by a transparent absolute or quantile rule."""
    if rule not in {"absolute", "quantile"}:
        raise ValueError("FILTER_RULE must be 'absolute' or 'quantile'")
    if details.empty:
        return set()
    eligible = details.loc[details["filter_eligible"]].copy()
    if eligible.empty:
        return set()
    if rule == "absolute":
        return set(
            eligible.loc[
                eligible["mispricing"].lt(OVERVALUATION_THRESHOLD), "ticker"
            ].astype(str)
        )
    if len(eligible) < QUANTILE_MINIMUM_POOL:
        return set()
    remove_count = int(np.floor(QUANTILE_REMOVAL_FRACTION * len(eligible)))
    if remove_count <= 0:
        return set()
    ranked = eligible.sort_values("mispricing", ascending=True, kind="mergesort")
    return set(ranked.head(remove_count)["ticker"].astype(str))


def _filter_audit_row(as_of, candidates, growth_tickers, details, remove, rule):
    """Validate one filter decision and return its weekly audit record."""
    if details.empty:
        eligible = details
        removed_rows = details
    else:
        eligible = details.loc[details["filter_eligible"].fillna(False)].copy()
        removed_rows = details.loc[
            details["ticker"].astype(str).isin(remove)
        ].copy()

    all_removed_eligible = bool(
        removed_rows.empty or removed_rows["filter_eligible"].fillna(False).all()
    )
    if not all_removed_eligible:
        raise AssertionError("filter removed a ticker outside the eligible pool")

    threshold_eligible = eligible.loc[
        eligible["mispricing"].lt(OVERVALUATION_THRESHOLD)
    ] if not eligible.empty else eligible
    all_removed_below_threshold = bool(
        removed_rows.empty
        or removed_rows["mispricing"].lt(OVERVALUATION_THRESHOLD).all()
    )

    if rule == "absolute":
        expected = set(threshold_eligible["ticker"].astype(str))
        if remove != expected:
            raise AssertionError(
                "absolute removals do not match the configured mispricing threshold"
            )
        if not all_removed_below_threshold:
            raise AssertionError(
                "absolute rule removed a ticker at or above the configured threshold"
            )
    else:
        expected_count = (
            int(np.floor(QUANTILE_REMOVAL_FRACTION * len(eligible)))
            if len(eligible) >= QUANTILE_MINIMUM_POOL
            else 0
        )
        expected = set(
            eligible.sort_values(
                "mispricing", ascending=True, kind="mergesort"
            ).head(expected_count)["ticker"].astype(str)
        )
        if remove != expected:
            raise AssertionError(
                "quantile removals do not match the lowest eligible mispricing ranks"
            )

    growth_count = len(growth_tickers)
    eligible_count = len(eligible)
    removed_count = len(remove)
    removed_mispricing = removed_rows["mispricing"] if not removed_rows.empty else None
    timestamp = pd.Timestamp(as_of)
    return {
        "decision_as_of": timestamp.date().isoformat(),
        "decision_iso_year": int(timestamp.isocalendar().year),
        "decision_iso_week": int(timestamp.isocalendar().week),
        "filter_rule": rule,
        "input_candidates": len(candidates),
        "growth_tech_candidates": growth_count,
        "valued_candidates": len(details),
        "eligible_candidates": eligible_count,
        "threshold_eligible_candidates": len(threshold_eligible),
        "expected_removed_count": len(expected),
        "removed_count": removed_count,
        "removed_fraction": (
            removed_count / growth_count if growth_count else 0.0
        ),
        "eligible_removed_fraction": (
            removed_count / eligible_count if eligible_count else 0.0
        ),
        "removed_tickers": ",".join(
            ticker for ticker in growth_tickers if ticker in remove
        ),
        "removed_mispricing_min": (
            float(removed_mispricing.min()) if removed_mispricing is not None else np.nan
        ),
        "removed_mispricing_max": (
            float(removed_mispricing.max()) if removed_mispricing is not None else np.nan
        ),
        "all_removed_eligible": all_removed_eligible,
        "absolute_threshold": OVERVALUATION_THRESHOLD,
        "all_removed_below_threshold": all_removed_below_threshold,
        "all_removed_below_minus_20": all_removed_below_threshold,
        "rule_count_check": removed_count == len(expected),
    }


def filter_candidates_by_rule(
    as_of, candidates, data, *, rule: str, audit_rows: list[dict] | None = None
):
    """Filter candidates under an explicit rule and optionally log its audit."""
    if rule not in {"absolute", "quantile"}:
        raise ValueError("FILTER_RULE must be 'absolute' or 'quantile'")
    growth_tickers = list(
        dict.fromkeys(
            candidate.symbol
            for candidate in candidates
            if candidate.bucket == "growth_tech"
        )
    )
    details = (
        valuation_details(as_of, growth_tickers, data)
        if growth_tickers
        else pd.DataFrame(columns=["ticker", "mispricing", "filter_eligible"])
    )
    remove = filter_removals(details, rule=rule)
    audit_row = _filter_audit_row(
        as_of, candidates, growth_tickers, details, remove, rule
    )
    if audit_rows is not None:
        audit_rows.append(audit_row)
    kept = [
        candidate
        for candidate in candidates
        if candidate.bucket != "growth_tech" or candidate.symbol not in remove
    ]
    if isinstance(candidates, tuple):
        return tuple(kept)
    if isinstance(candidates, list):
        return kept
    return tuple(kept)


def filter_candidates(as_of, candidates, data):
    return filter_candidates_by_rule(as_of, candidates, data, rule=FILTER_RULE)
