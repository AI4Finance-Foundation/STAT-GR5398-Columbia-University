# Track A FinRobot Integration

This folder contains our wrapper around the original FinRobot repository in
`external/FinRobot`. The external repository is treated as vendor code.

## Input

Primary FinRL input:

```text
finrl2/finrl_stock_selection.csv
```

It contains the weekly selected FinRL portfolio from `2026-01-02` through
`2026-04-24`.

## Run Full FinRobot

```bash
.venv_finrobot/bin/python finrobot_tracka/run_full_finrobot.py \
  --finrl-csv finrl2/finrl_stock_selection.csv \
  --skip-existing
```

The script runs real FinRobot equity analysis for every unique selected ticker,
using:

- FMP for financials, news, peer comparisons, and market metrics
- OpenAI for equity research text sections

## Build Multipliers

First build point-in-time historical news signals from FMP. The script uses a
rolling 30-day window ending at each rebalance date and keeps only articles with
`publishedDate <= date`. It also applies a conservative relevance filter using
ticker, company-name, and product aliases, with explicit exclusions for ticker
ambiguity such as `Wet AMD`.

```bash
.venv_finrobot/bin/python finrobot_tracka/build_historical_news_signals.py \
  --finrl-csv finrl2/finrl_stock_selection.csv \
  --output-csv finrobot_tracka/historical_news_signals.csv \
  --window-days 30
```

Then build FinRobot multipliers:

```bash
.venv_finrobot/bin/python finrobot_tracka/build_finrobot_multipliers.py \
  --finrl-csv finrl2/finrl_stock_selection.csv \
  --result-csv finrobot_tracka/finrobot2_multipliers.csv \
  --panel-csv finrobot_tracka/finrobot2_rebalanced_panel.csv \
  --news-signals-csv finrobot_tracka/historical_news_signals.csv
```

Outputs:

```text
finrobot_tracka/finrobot2_multipliers.csv
finrobot_tracka/finrobot2_rebalanced_panel.csv
finrobot_tracka/historical_news_signals.csv
```

## Scoring Methodology

FinRobot is used as a fundamental confirmation layer after FinRL. It does not
replace the FinRL stock selection model. For each FinRL-selected ticker, the
wrapper reads the real FinRobot analysis outputs and converts them into a
point-in-time research score for each `date, ticker` pair:

```text
finrobot_score =
0.45 * business_quality_score
+ 0.25 * valuation_score
+ 0.20 * catalyst_score
+ 0.10 * risk_score
```

To avoid look-ahead bias, financial metrics are filtered by filing
`acceptedDate`. For each rebalance date, the script uses only the latest fiscal
year whose filing was accepted on or before that rebalance date. If the 2025
filing was not yet available, the score falls back to 2024 data.

The business quality score combines absolute financial quality with
peer-relative ranking inside the weekly FinRL-selected universe. It uses revenue
growth, EBITDA margin, contribution margin, EPS, free cash flow quality,
operating cash flow margin, debt-to-equity, current ratio, and ROE.

The valuation score is peer-relative, using PE ratio, EV/EBITDA, and free cash
flow yield. Lower PE and EV/EBITDA are better; higher free cash flow yield is
better.

The catalyst and risk scores use real historical FMP news. For each rebalance
date, the script builds a rolling 30-day window and keeps only news articles
with `publishedDate` on or before that rebalance date. The catalyst score is a
smoothed positive-versus-negative news score. The risk score is a
severity-weighted penalty normalized by the relevant-news count, so high-news
volume firms are not penalized purely for having more articles. Risk terms
include lawsuit, investigation, recall, downgrade, margin pressure, weak
guidance, and execution risk. Retail/social sentiment is intentionally excluded
from the FinRobot layer and left to the FinGPT layer, because the available
retail output did not provide reliable point-in-time timestamps.

The final score maps to a portfolio action:

```text
score >= 0.80  strong_keep  multiplier = 1.20
score >= 0.60  keep         multiplier = 1.00
score >= 0.45  reduce       multiplier = 0.70
score <  0.45  reject       multiplier = 0.00
```

The final pipeline can use:

```text
after_finrobot_weight = initial_weight * finrobot_multiplier
```

then normalize within each rebalance date before passing weights to FinGPT.

## Backtest Comparison

Run the before/after comparison against SPY:

```bash
.venv_finrobot/bin/python finrobot_tracka/backtest_finrobot_comparison.py
```

The comparison uses daily adjusted prices from Yahoo Finance for
`2026-01-01` through `2026-04-30` and evaluates:

```text
FinRL-only
FinRL + no-lookahead FinRobot
SPY
```

Outputs:

```text
finrobot_tracka/backtest_results/performance_summary.csv
finrobot_tracka/backtest_results/equity_curve.csv
finrobot_tracka/backtest_results/equity_curve.png
finrobot_tracka/backtest_results/finrl_vs_no_lookahead_finrobot_attribution.csv
```
