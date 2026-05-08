# Building a Modest AI-Assisted Portfolio Pipeline

**Collaborators:** Rui Zhao, Qingyue Wang, Ruping Zhang

## Project Overview

This repository captures a layered AI research pipeline that combines FinRL stock selection, FinRobot research gate scoring, and FinGPT news sentiment overlay into a weekly long-only portfolio.

The goal is to turn noisy price data, financial evidence, and news context into a traceable portfolio decision framework, with a fixed 60% QQQ market sleeve and a 40% stock basket.

## What the project is trying to solve

The project asks whether a structured pipeline can convert market data, research signals, and news overlay into an inspectable portfolio strategy. It avoids treating the LLM as a standalone picker and instead assigns each module a narrow role.

## Pipeline Overview

### 1. Price and feature inputs

Adjusted close prices are downloaded in selection and backtest scripts. Momentum, volatility, and drawdown features are built point-in-time and forward-filled where appropriate.

### 2. FinRL selection

A 20-stock universe is ranked weekly using hybrid machine learning and technical scores. The script then keeps the top 5 candidates as the initial selection set.

### 3. FinRobot research gate

Selected names receive business quality, valuation, catalyst, and risk scores. These scores map into keep, reduce, reject, or modest overweight decisions, reducing blind reliance on the quantitative ranking.

### 4. FinGPT news overlay

A no-lookahead news signal layer turns sentiment, confidence, and materiality into a composite score. This overlay adjusts conviction while keeping the evaluation path transparent.

### 5. Portfolio construction

The final portfolio blends a 60% QQQ sleeve with a 40% stock basket. Weights are normalized after each overlay, producing the final target portfolio weights.

### 6. Backtest evaluation

The strategy is compared with QQQ and SPY using return, volatility, Sharpe, Sortino, and drawdown metrics.

## What the current repository supports

### Universe

The FinRL selection script defines a 20-name large-cap universe: AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA, AVGO, AMD, NFLX, ADBE, COST, PEP, QCOM, CSCO, TXN, AMAT, AMGN, INTU, and ISRG.

### Time and frequency

The study window is 2026-01-01 to 2026-04-30. Selection is weekly, using Friday-style rebalance dates where available.

### Feature handling

Price features include 20-day momentum, 60-day momentum, 20-day volatility, and 60-day drawdown. The pipeline drops rows or columns with all missing values while preserving point-in-time feature logic.

### Available artifacts

The repository includes generated CSV outputs, backtest images, and module scripts. Raw external FinRobot dependencies, API keys, and a root narrative report are not present in this snapshot.

## Strategy Logic

### FinRL: candidate generation

FinRL is used as the first filter. It scores stocks with a hybrid ML and technical ranking process and selects the weekly top 5 candidates. The output includes dates, tickers, ranks, scores, and initial equal weights.

### FinRobot: research confirmation

FinRobot acts as an analyst-style gate, combining point-in-time financial and news evidence into a final score. The score becomes a multiplier indicating strong keep, keep, reduce, or reject.

### FinGPT: sentiment overlay

The FinGPT layer is applied conservatively. It produces sentiment, confidence, and materiality values that are combined into a composite signal for each ticker-date observation.

### Backtesting and weights

The backtest uses the layered FRG signals with a one-trading-day execution lag and a transaction cost parameter of 0.001. The final portfolio uses a 60% QQQ sleeve and compares to QQQ/SPY benchmarks.

## What the result files show

- FRG total return: 12.84%
- QQQ total return: 7.14%
- SPY total return: 4.06%
- FRG Sharpe: 1.79
- QQQ Sharpe: 1.32
- SPY Sharpe: 0.94
- FRG max drawdown: -14.35%
- FinGPT observations: 85 (59 positive, 26 negative)

This short test indicates the FRG portfolio beat both benchmarks on return and Sharpe, while also showing a larger drawdown.

## Interpretation

This README is best read as a pipeline demo and research prototype. The strongest evidence is that the project preserves a concrete, inspectable path from stock selection to research scoring, news scoring, target weights, portfolio values, and benchmark metrics.

The 60% QQQ sleeve stabilizes the portfolio and makes the system partly a market exposure strategy rather than a pure stock-selection strategy.

## Limitations

- The 2026-01-01 to 2026-04-30 window is too short to establish stable performance.
- The current snapshot includes outputs but not every raw data dependency or external FinRobot component needed for a clean rerun.
- Survivorship bias, data timing, and benchmark alignment require careful audit.
- No full sensitivity study, turnover report, or robustness grid is included in this export.
- News and sentiment signals depend on filtering and LLM-style calibration assumptions that can miss context.

## Reproducibility

### Where to inspect the project

- Scripts: inrl/scripts/, inrobot/scripts/, ingpt/scripts/
- Results: inrl/results/, inrobot/results/
- Static site assets: docs/assets/figures/

### Known missing or unavailable artifacts

A root narrative report and some raw external data dependencies are not included in this snapshot.

## Additional Links

- Website: https://zzzzzzzzrrrrrrrrrrrrrr.github.io/pipeline-finrl-finrobot-fingpt/?source=post_page-----4d2ee0e5e890---------------------------------------
