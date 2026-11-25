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

"""Assignment 2: FinRobot Valuation Module.

Point-in-time multi-method fundamental valuation.

Methods:
- Sector peer trailing P/E
- Historical company EV/EBITDA
- FinRobot discounted cash flow (DCF)

The module combines available fair-value estimates and applies
a conservative, agreement-based candidate filter.
"""

import numpy as np
import pandas as pd

from valuation.finrobot import (
    DCFInputs,
    adjust_beta_blume,
    calculate_dcf,
    calculate_sensitivity,
)
from valuation.examples import weekly_beta

COLUMNS = [
    "ticker",
    "fair_value",
    "low",
    "high",
    "mispricing",
    "method_spread",
]


def peer_multiple_value(as_of, tickers, data):
    """Estimate fair value using sector trailing P/E multiples."""

    members = data.members()
    universe = sorted(members | set(tickers))

    fundamentals = data.fundamentals(universe)

    if fundamentals.empty:
        return pd.DataFrame(columns=COLUMNS)

    fundamentals = fundamentals.sort_values(
        ["ticker", "datadate"]
    )

    # Obtain the most recent four reported quarters.
    last4 = fundamentals.groupby("ticker").tail(4)

    # Keep only stocks with four quarterly observations.
    counts = last4.groupby("ticker")["ticker"].transform("size")
    last4 = last4[counts == 4]


    # Require four consecutive quarters for trailing EPS.
    last4 = last4.copy()
    last4["quarter_number"] = (
        pd.to_datetime(last4["datadate"]).dt.year * 4
        + pd.to_datetime(last4["datadate"]).dt.quarter
    )

    consecutive = last4.groupby("ticker")[
        "quarter_number"
    ].transform(
        lambda x: (x.diff().dropna() == 1).all()
    )

    last4 = last4[consecutive].drop(
        columns=["quarter_number"]
    )

    # Reject missing quarterly EPS values.
    valid_eps = last4.groupby("ticker")["EPS"].transform(
        lambda x: x.notna().all()
    )

    last4 = last4[valid_eps]

    if last4.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Compute trailing twelve-month EPS.
    financials = last4.groupby("ticker").agg(
        eps_ttm=("EPS", "sum"),
        sector=("gsector", "last"),
    )

    # Retrieve prices known at the valuation date.
    prices = data.last_close(financials.index.tolist())

    financials["price"] = pd.Series(prices)

    # Calculate trailing P/E from verified inputs.
    financials["trailing_pe"] = (
        financials["price"] / financials["eps_ttm"]
    )

    # Select valid index constituents as peers.
    peers = financials[
        (financials.index.isin(members))
        & (financials["eps_ttm"] > 0)
        & (financials["trailing_pe"] > 0)
        & (financials["trailing_pe"] < 200)
    ]

    if peers.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Sector-level valuation multiples.
    sector_bands = (
        peers.groupby("sector")["trailing_pe"]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
    )

    results = []

    for ticker in tickers:
        if ticker not in financials.index:
            continue

        eps = financials.at[ticker, "eps_ttm"]
        sector = financials.at[ticker, "sector"]
        price = financials.at[ticker, "price"]

        if (
            not np.isfinite(eps)
            or eps <= 0
            or not np.isfinite(price)
            or price <= 0
            or sector not in sector_bands.index
        ):
            continue

        multiples = sector_bands.loc[sector]

        low = float(eps * multiples[0.25])
        fair_value = float(eps * multiples[0.50])
        high = float(eps * multiples[0.75])

        mispricing = fair_value / price - 1.0

        results.append({
            "ticker": ticker,
            "fair_value": fair_value,
            "low": low,
            "high": high,
            "mispricing": mispricing,
            "method_spread": 1.0,
        })

    return pd.DataFrame(results, columns=COLUMNS)


def historical_ev_ebitda_value(as_of, tickers, data):
    """Historical own-company EV/EBITDA valuation.

    All historical multiples are constructed using financial
    statements and prices available at each historical date.
    """

    rows = []

    for ticker in tickers:
        f = data.fundamentals([ticker])

        if f.empty or len(f) < 8:
            continue

        f = f.sort_values("datadate").copy()
        f["datadate"] = pd.to_datetime(f["datadate"])

        # Conservative historical availability date.
        available = f["datadate"].copy()

        for col in ("filing_date", "accepted_date"):
            dates = pd.to_datetime(f[col], errors="coerce")
            available = available.where(
                available >= dates, dates
            )

        f["available_date"] = available

        prices = data.close([ticker])[ticker].dropna()

        if prices.empty:
            continue

        historical_multiples = []

        # Construct historical trailing-four-quarter EBITDA.
        for end_idx in range(3, len(f)):
            q = f.iloc[end_idx - 3:end_idx + 1]

            observation_date = q["available_date"].max()

            if pd.isna(observation_date):
                continue

            # Do not use a future observation.
            if observation_date > pd.Timestamp(as_of):
                continue

            # Require four consecutive quarterly observations.
            quarter_dates = q["datadate"].dt.to_period("Q")
            quarter_numbers = (
                quarter_dates.dt.year * 4
                + quarter_dates.dt.quarter
            )

            if not (quarter_numbers.diff().dropna() == 1).all():
                continue

            revenues = q["revenue_per_share"]
            margins = q["ebitda_margin"]

            ebitda_ttm = float((revenues * margins).sum())

            if (
                not np.isfinite(ebitda_ttm)
                or ebitda_ttm <= 0
                or revenues.isna().any()
                or margins.isna().any()
            ):
                continue


            # Data-quality guard: unusually large EBITDA margins.
            # Such observations may reflect non-recurring items
            # and can distort rolling TTM valuation multiples.
            if (
                (q["ebitda_margin"] > 1.0).any()
                or (q["ebitda_margin"] < -1.0).any()
            ):
                continue

            # The latest quarter's balance-sheet values.
            latest = q.iloc[-1]

            net_debt = (
                latest["interest_debt_per_share"]
                - latest["cash_per_share"]
            )

            if not np.isfinite(net_debt):
                continue

            # Use the last price available at the observation date.
            past_prices = prices.loc[:observation_date]

            if past_prices.empty:
                continue

            price = float(past_prices.iloc[-1])

            if not np.isfinite(price) or price <= 0:
                continue

            enterprise_value = price + net_debt

            if enterprise_value <= 0:
                continue

            multiple = enterprise_value / ebitda_ttm

            # Preliminary plausibility limits.
            if 1.0 <= multiple <= 150.0:
                historical_multiples.append(
                    (observation_date, float(multiple))
                )

        if len(historical_multiples) < 4:
            continue

        # Compute current TTM EBITDA.
        current_q = f.tail(4)


        # Do not value the current company using an
        # anomalous or incomplete EBITDA-margin window.
        if (
            current_q["ebitda_margin"].isna().any()
            or (current_q["ebitda_margin"] > 1.0).any()
            or (current_q["ebitda_margin"] < -1.0).any()
        ):
            continue


        # Require consecutive quarters for the current TTM.
        current_quarters = current_q["datadate"].dt.to_period("Q")

        current_qnums = (
            current_quarters.dt.year * 4
            + current_quarters.dt.quarter
        )

        if not (current_qnums.diff().dropna() == 1).all():
            continue

        # Confirm all current financial data were available.
        if (
            current_q["available_date"].isna().any()
            or current_q["available_date"].max()
            > pd.Timestamp(as_of)
        ):
            continue


        current_ebitda = float(
            (
                current_q["revenue_per_share"]
                * current_q["ebitda_margin"]
            ).sum()
        )

        latest = current_q.iloc[-1]

        current_net_debt = (
            latest["interest_debt_per_share"]
            - latest["cash_per_share"]
        )

        current_price = data.last_close([ticker]).get(ticker)

        if (
            not np.isfinite(current_ebitda)
            or current_ebitda <= 0
            or not np.isfinite(current_net_debt)
            or current_price is None
            or not np.isfinite(current_price)
            or current_price <= 0
        ):
            continue

        # Use only historical observations before the current
        # latest financial snapshot to avoid self-anchoring.
        current_available = current_q["available_date"].max()

        multiples = np.array([
            m for d, m in historical_multiples
            if d < current_available
        ])

        if len(multiples) < 4:
            continue

        low_multiple, median_multiple, high_multiple = (
            np.quantile(multiples, [0.25, 0.50, 0.75])
        )

        low = low_multiple * current_ebitda - current_net_debt
        fair_value = (
            median_multiple * current_ebitda - current_net_debt
        )
        high = high_multiple * current_ebitda - current_net_debt

        if (
            not np.isfinite(low)
            or not np.isfinite(fair_value)
            or not np.isfinite(high)
            or low <= 0
            or fair_value <= 0
            or high <= 0
            or not (low <= fair_value <= high)
        ):
            continue

        rows.append({
            "ticker": ticker,
            "fair_value": float(fair_value),
            "low": float(low),
            "high": float(high),
            "mispricing": float(
                fair_value / current_price - 1
            ),
            "method_spread": 1.0,
        })

    return pd.DataFrame(rows, columns=COLUMNS)



def dcf_value(as_of, tickers, data):
    """Point-in-time per-share DCF valuation."""

    results = []
    erp = 0.05
    terminal_growth = 0.025
    credit_spread = 0.02
    years = 10

    def bounded(x, lo, hi):
        if not np.isfinite(x):
            return lo
        return float(np.clip(x, lo, hi))

    rf_series = data.macro("DGS10")

    if rf_series.empty:
        return pd.DataFrame(columns=COLUMNS)

    rf = float(rf_series.iloc[-1]) / 100.0

    for ticker in tickers:
        f = data.fundamentals([ticker])

        if len(f) < 8:
            continue

        f = f.sort_values("datadate")
        current = f.tail(4)
        previous = f.iloc[-8:-4]


        # Require eight consecutive quarters for YoY growth.
        last8 = f.tail(8).copy()

        quarter_dates = pd.to_datetime(
            last8["datadate"]
        ).dt.to_period("Q")

        quarter_numbers = (
            quarter_dates.dt.year * 4
            + quarter_dates.dt.quarter
        )

        if not (
            quarter_numbers.diff().dropna() == 1
        ).all():
            continue

        # Reject missing or invalid revenue observations.
        if (
            current["revenue_per_share"].isna().any()
            or previous["revenue_per_share"].isna().any()
        ):
            continue

        revenue = current["revenue_per_share"].sum()
        previous_revenue = previous["revenue_per_share"].sum()

        if not (
            np.isfinite(revenue)
            and np.isfinite(previous_revenue)
            and revenue > 0
            and previous_revenue > 0
        ):
            continue

        # Historical TTM revenue growth, with conservative bounds.
        historical_growth = revenue / previous_revenue - 1
        initial_growth = bounded(historical_growth, -0.10, 0.30)

        # Growth gradually converges toward terminal growth.
        growth_path = np.linspace(
            initial_growth, terminal_growth, years
        ).tolist()

        beta = weekly_beta(ticker, data)

        if beta is None or not np.isfinite(beta):
            continue

        beta = bounded(adjust_beta_blume(beta), 0, 5)

        margin = bounded(
            current["ebitda_margin"].median(), 0, 1
        )

        da = bounded(
            (
                current["ebitda_margin"]
                - current["operating_margin"]
            ).median(),
            0, 0.5
        )

        capex = bounded(
            current["capex_per_share"].abs().sum() / revenue,
            0, 1
        )

        tax = bounded(
            current["effective_tax_rate"].median(),
            0, 0.35
        )

        debt_ratio = bounded(
            current["debt_to_capital"].iloc[-1],
            0, 0.95
        )

        net_debt = (
            current["interest_debt_per_share"].iloc[-1]
            - current["cash_per_share"].iloc[-1]
        )

        if not np.isfinite(net_debt):
            continue

        inputs = DCFInputs(
            revenue_base=float(revenue),
            revenue_growth_rates=growth_path,
            ebitda_margin=margin,
            da_pct_revenue=da,
            capex_pct_revenue=capex,
            nwc_pct_revenue=0.0,
            tax_rate=tax,
            risk_free_rate=bounded(rf, 0, 0.15),
            beta=beta,
            equity_risk_premium=erp,
            cost_of_debt=bounded(rf + credit_spread, 0, 0.20),
            debt_ratio=debt_ratio,
            terminal_growth_rate=terminal_growth,
            shares_outstanding=1.0,
            net_debt=float(net_debt),
        )

        try:
            result = calculate_dcf(inputs)
            wacc = result.wacc

            grid = calculate_sensitivity(
                inputs,
                wacc_range=[wacc - 0.01, wacc, wacc + 0.01],
                tg_range=[
                    terminal_growth - 0.005,
                    terminal_growth,
                    terminal_growth + 0.005,
                ],
            )
        except (ValueError, ArithmeticError):
            continue

        values = [
            float(x)
            for row in grid["implied_prices"]
            for x in row
            if x is not None and np.isfinite(x) and x > 0
        ]

        fair_value = float(result.implied_price)
        price = data.last_close([ticker]).get(ticker)

        if (
            not values
            or not np.isfinite(fair_value)
            or fair_value <= 0
            or price is None
            or not np.isfinite(price)
            or price <= 0
        ):
            continue

        results.append({
            "ticker": ticker,
            "fair_value": fair_value,
            "low": min(min(values), fair_value),
            "high": max(max(values), fair_value),
            "mispricing": fair_value / price - 1.0,
            "method_spread": 1.0,
        })

    return pd.DataFrame(results, columns=COLUMNS)



def value(as_of, tickers, data) -> pd.DataFrame:
    """Combine three point-in-time valuation methods."""

    peer = peer_multiple_value(as_of, tickers, data)
    historical = historical_ev_ebitda_value(as_of, tickers, data)
    dcf = dcf_value(as_of, tickers, data)

    method_frames = {
        "peer": peer,
        "historical": historical,
        "dcf": dcf,
    }

    estimates = {}

    for method, frame in method_frames.items():
        for _, row in frame.iterrows():
            ticker = row["ticker"]
            fv = float(row["fair_value"])

            if not np.isfinite(fv) or fv <= 0:
                continue

            estimates.setdefault(ticker, []).append({
                "method": method,
                "fair_value": fv,
                "low": float(row["low"]),
                "high": float(row["high"]),
            })

    prices = data.last_close(tickers)
    results = []

    for ticker in tickers:
        methods = estimates.get(ticker, [])

        # Require at least two valid methods.
        if len(methods) < 2:
            continue

        values = np.array([
            m["fair_value"] for m in methods
        ])

        fair_value = float(np.median(values))

        # Conservative envelope of available method ranges.
        low = min(m["low"] for m in methods)
        high = max(m["high"] for m in methods)

        spread = float(values.max() / values.min())

        price = prices.get(ticker)

        if (
            price is None
            or not np.isfinite(price)
            or price <= 0
        ):
            continue

        if not (
            np.isfinite(low)
            and np.isfinite(high)
            and low > 0
            and high >= fair_value >= low
        ):
            continue

        results.append({
            "ticker": ticker,
            "fair_value": fair_value,
            "low": float(low),
            "high": float(high),
            "mispricing": float(fair_value / price - 1),
            "method_spread": spread,
        })

    return pd.DataFrame(results, columns=COLUMNS)




def filter_candidates(as_of, candidates, data):
    """Reliability-aware multi-method valuation filter.

    A stock is removed only when:
    1. At least two valuation methods are available.
    2. At least two methods independently signal
       more than 30% overvaluation.
    3. All available method estimates agree within 2x.

    Missing or conflicting evidence means KEEP.
    """

    candidates = tuple(candidates)

    growth = sorted(
        (
            p for p in candidates
            if p.bucket == "growth_tech"
        ),
        key=lambda p: p.rank_score,
        reverse=True,
    )

    top_candidates = growth[:30]

    if not top_candidates:
        return candidates

    tickers = [p.symbol for p in top_candidates]

    # Evaluate methods separately.
    peer = peer_multiple_value(as_of, tickers, data)
    historical = historical_ev_ebitda_value(
        as_of, tickers, data
    )
    dcf = dcf_value(as_of, tickers, data)

    estimates = {}

    for frame in [peer, historical, dcf]:
        for _, row in frame.iterrows():

            ticker = row["ticker"]
            fair_value = float(row["fair_value"])
            mispricing = float(row["mispricing"])

            if not (
                np.isfinite(fair_value)
                and fair_value > 0
                and np.isfinite(mispricing)
            ):
                continue

            estimates.setdefault(ticker, []).append(
                (fair_value, mispricing)
            )

    removed = set()

    for ticker, methods in estimates.items():

        # Require at least two independent methods.
        if len(methods) < 2:
            continue

        fair_values = np.array(
            [m[0] for m in methods],
            dtype=float,
        )

        mispricings = np.array(
            [m[1] for m in methods],
            dtype=float,
        )

        # Method disagreement.
        spread = (
            fair_values.max() / fair_values.min()
        )

        if spread > 2.0:
            continue

        # Require at least two strongly bearish methods.
        bearish_votes = int(
            np.sum(mispricings < -0.30)
        )

        if bearish_votes >= 2:
            removed.add(ticker)

    # Preserve the original Selection predictions.
    return tuple(
        p for p in candidates
        if p.symbol not in removed
    )
