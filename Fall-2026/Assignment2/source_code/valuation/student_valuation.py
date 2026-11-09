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
import pandas as pd

COLUMNS = ["ticker", "fair_value", "low", "high", "mispricing", "method_spread"]


def value(as_of, tickers, data) -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def filter_candidates(as_of, candidates, data):
    return candidates
