"""Timing module: decide how much of the portfolio is invested.

The timing group edits only this file.

exposure(as_of, data) is called once a week at the close of ``as_of`` (the
last session before the rebalance). Orders fill at the next open:

    total invested = 0.95 (cash buffer) x exposure(as_of, data)

Rules
  * Return a float in [0.0, 1.0]. 0 = all cash, 1 = fully invested.
  * Read data only through ``data`` (shared.data_view.DataView). It returns
    nothing dated after as_of. Reading files or the network inside this
    function raises an error.
  * Same inputs, same output: fix the seed if you use randomness.

Useful calls
    data.close(["SPY"], lookback=252)   SPY daily closes, last 252 sessions
    data.market("^VIX")                 VIX closes
    data.market("XLK")                  sector ETF closes (XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY)
    data.macro("T10Y2Y")                10-year minus 2-year Treasury spread
    data.macro("BAA10Y")                BAA corporate minus 10-year Treasury spread, 1986 onward
    data.macro("BAMLH0A0HYM2")          high-yield spread, only from 2023-09
    data.extra("news_sentiment")        your own student_data/news_sentiment.parquet

Baselines to beat (timing/baselines.py): full, sma200, engine.

Run
    python evaluate_timing.py --period dev                 on SPY alone, against the baselines
    python run_portfolio.py --period dev --timing student   on the stock portfolio
    python check_submission.py --timing
"""


def exposure(as_of, data) -> float:
    return 1.0
