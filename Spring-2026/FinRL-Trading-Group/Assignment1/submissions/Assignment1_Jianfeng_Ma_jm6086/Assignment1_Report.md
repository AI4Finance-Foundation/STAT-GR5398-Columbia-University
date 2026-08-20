# Assignment 1 Report (Draft)

Course: GR5398 26 Spring - FinRL-Trading Quantitative Trading Strategy Track  
Assignment: Assignment 1  
Author: `Jianfeng Ma` (`jm6086`)  
Date: February 20, 2026

## 1. Objective

This project implements the Assignment 1 full pipeline:
- Fundamental data preprocessing
- ML-based stock selection
- Portfolio construction and backtesting
- Custom strategy design and benchmark comparison

The main evaluation window required in the assignment is **2018-01-01 to 2025-12-31**, and we report cumulative return, annualized return, annualized volatility, Sharpe ratio, and maximum drawdown.

## 2. Data and Setup

### 2.1 Data sources
- `data/fundamental_quarterly.csv` (Compustat quarterly fundamentals)
- `data/security_daily.csv` (daily prices and adjustment factors)

### 2.2 Universe and period
- Equity universe: tickers available in the provided files (primarily S&P 500 / NASDAQ constituents in this assignment setup)
- Full backtest period for final strategy reporting: **2018-01-01 to 2025-12-31**

## 3. Methodology

### 3.1 Baseline pipeline (provided notebook)
The notebook preprocesses fundamental features, trains sector models (RF / LightGBM / XGBoost), and outputs selected stocks and equal-weight holdings.

### 3.2 Custom strategy (implemented in cell 27)
I implemented a custom long-only strategy with the following rules:
1. Monthly rebalance (last trading day of each month)
2. Signal ranking: 6-month momentum normalized by 20-day realized volatility
3. Eligibility filter:
   - Positive momentum
   - Stock price above its 200-day moving average
4. Portfolio construction:
   - Top 40 eligible stocks
   - Inverse-volatility weighting
   - Single-stock cap at 5%
5. Regime filter:
   - If SPY is below its 200-day moving average at rebalance, reduce gross exposure to 20%
6. Transaction cost:
   - 10 bps (`0.1%`) proportional to daily turnover

## 4. Results (2018-01-01 to 2025-12-31)

Backtest aligned period in results: **2018-01-02 to 2025-12-31**.

| Portfolio | Cumulative Return | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Custom Strategy | 647.59% | 28.61% | 37.94% | 0.8553 | -50.24% |
| QQQ | 308.42% | 19.25% | 24.05% | 0.8540 | -35.12% |
| SPY | 187.48% | 14.12% | 19.45% | 0.7779 | -33.72% |

### Interpretation
- The custom strategy significantly outperformed SPY and QQQ in cumulative and annualized returns over this window.
- Sharpe ratio is slightly above QQQ and notably above SPY.
- Risk remains high: maximum drawdown is materially deeper than both benchmarks.

## 5. Discussion

### 5.1 What worked
- Trend and momentum filters improved directional capture over the long horizon.
- Inverse-volatility weighting and weight caps reduced single-name concentration risk.
- Regime filter helped partially de-risk during weak market phases.

### 5.2 Limitations
- Drawdown is still large, suggesting the strategy is return-seeking but risk-aggressive.
- Backtest excludes slippage and market impact beyond proportional turnover cost.
- Parameter choices are heuristic and may require out-of-sample robustness checks.

### 5.3 Potential improvements
- Add portfolio-level volatility targeting.
- Introduce stronger downside controls (e.g., stop-loss or faster regime trigger).
- Validate using walk-forward parameter selection and sector exposure constraints.

## 6. Reproducibility

Main code file:
- `source_code/FinRL-Trading-Full-Workload.ipynb`

Generated outputs used in this report:
- `outputs/step2/custom_strategy_metrics.csv`
- `outputs/step2/custom_strategy_comparison.png`
- `outputs/step2/stock_selected.csv`

Quick rerun (after kernel restart): run notebook cells `5, 16, 19, 21, 27`.

## 7. Conclusion

The Assignment 1 pipeline is completed end-to-end, and the custom strategy achieves higher return and Sharpe than SPY/QQQ in the 2018-2025 window, at the cost of higher drawdown. Future work should prioritize risk control while preserving the return edge.
