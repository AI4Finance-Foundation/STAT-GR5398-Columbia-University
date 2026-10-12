# When a trading backtest becomes a data audit

STAT GR5398 Fall 2026 Assignment 1 - FinRL-Trading

This draft was prepared with computational assistance and needs the student's review and personal interpretation before publication.

I started with the course's stock-selection pipeline and a completed 11-sector run. Its 174,571 selections looked substantial, but a completed model run was not enough to establish a valid trading result. Sixty-two selected tickers were absent from the entire price database. Treating their returns as zero would silently change what the experiment meant.

The timing audit found a second issue: models could be chosen using validation returns that had not finished at the first date assigned to their predictions. A filing date on a prediction does not prove the model itself could have existed then.

I preserved the original results and built a separate experiment. Each quarter, training and validation require both published features and completed return labels. Accounting features become usable only after the latest of quarter end, filing date and accepted date, plus one day. A fixed Random Forest/Extra Trees comparison selects a model by validation MSE; there is no tuning against portfolio returns. I rank current point-in-time S&P 500 members, take the top quarter, and hold equal target allocations between quarterly executions.

A missing entry quote leaves its intended allocation in cash. Other stocks are not enlarged to hide the missing allocation. Any missing quote on an already-held position stops valuation. This is an explicit restricted-price experiment, not a fix for missing historical securities.

From 2020-06-30 through 2025-09-30, the executed net strategy returned 109.10%; SPY returned 132.66% and QQQ 150.00%. Strategy CAGR was 15.08%, Sharpe 0.96, annualized volatility 15.98%, and maximum drawdown -22.22%. These numbers include 10 bp per gross traded dollar. They do not meet the assignment's requirement to beat the S&P 500.

The main lesson is that universe coverage, label maturity, execution time and valuation conventions are part of the strategy. Lower volatility does not compensate for an invalid universe claim. My next step would be historical security prices with corporate actions and stable identifiers, then evaluation of frozen rules before further tuning.

Sources: [Assignment requirements](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment1/README.md), [FinRL track](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment1/FinRL-Trading/README.md), [FinRL project](https://github.com/AI4Finance-Foundation/FinRL-Trading). Full numerical results and caveats are in the accompanying research report.
