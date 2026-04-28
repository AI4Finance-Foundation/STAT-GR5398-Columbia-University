# Assignment 2 Report

Course: GR5398 26 Spring - FinRL-Trading Quantitative Trading Strategy Track  
Author: `Jianfeng Ma` (`jm6086`)  
Backtest Window: `2018-01-01` to `2025-12-31`

## 1. Objective

This submission implements the Adaptive Rotation Strategy described in the Assignment 2 prompt and compares:

- Original Adaptive Rotation
- Modified Adaptive Rotation with Assignment 1 QQQ rolling-selection logic
- `QQQ` benchmark

## 2. Strategy Setup

### 2.1 Original Adaptive Rotation

- Weekly rebalance using trailing 12-week Information Ratio versus `QQQ` to choose the strongest asset group
- Intra-group ranking by residual momentum
- Final allocation to the top 2 assets in the chosen group
- Slow regime overlay from `^GSPC` trend / drawdown / `^VIX`
- Fast risk-off and stop-loss overlays for daily risk control

### 2.2 Modified Version

The modified strategy keeps the same regime and risk framework, but changes the growth-tech stock-selection pool:

- When the Assignment 1 rolling selector is available, the growth-tech candidate pool is replaced by the latest available Assignment 1 picks
- Because the Assignment 1 fundamentals start in early 2018 and the original rolling model needs four years of history, the Assignment 1 selector only becomes available from `2023-02-15`
- Before `2023-02-15`, the modified strategy falls back to the original static growth-tech basket to preserve the required `2018-01-01` to `2025-12-31` evaluation window

## 3. Results

| Portfolio | Cumulative Return | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown | Calmar Ratio | Win Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original Adaptive Rotation | 267.79% | 17.69% | 23.86% | 0.80 | -33.65% | 0.53 | 53.38% |
| Modified Adaptive Rotation | 293.51% | 18.69% | 24.72% | 0.82 | -33.65% | 0.56 | 52.14% |
| QQQ | 308.42% | 19.25% | 24.05% | 0.85 | -35.12% | 0.55 | 56.32% |
| SPY | 187.48% | 14.12% | 19.46% | 0.78 | -33.72% | 0.42 | 55.42% |

Chart:

![Assignment 2 comparison](comparison_chart.png)

## 4. Discussion

- The original strategy provides the baseline adaptive rotation behavior from the assignment prompt
- The modified strategy tests whether Assignment 1 stock-selection signals improve the growth-tech sleeve without changing the broader regime logic
- The warm-up limitation from Assignment 1 is handled explicitly rather than backfilling nonexistent pre-2023 selector signals

## 5. Output Files

- `outputs/comparison_metrics.csv`
- `outputs/comparison_nav.csv`
- `outputs/comparison_chart.png`
- `outputs/original_weights.csv`
- `outputs/modified_weights.csv`
- `outputs/original_decisions.csv`
- `outputs/modified_decisions.csv`
