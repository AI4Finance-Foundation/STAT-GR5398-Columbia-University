# GR5398 Assignment 2: Adaptive Rotation Strategy Enhancement

## Goal
Reproduce Adaptive Rotation, enhance it with Rui's Assignment 1 QQQ rolling selection logic, and compare original, modified, QQQ, SPY, and equal-weight benchmarks.

## Folder Structure
`run_assignment2.py`, `src/`, `configs/`, `data/cache/`, `results/`, `figures/`, and `blog/`.

## Strategy Summary
Original: 12-week IR group selection, SPY moving-average regime filter, residual momentum top-2 equal weighting, and drawdown risk control. Modified: same framework, but Growth Tech is pre-filtered with Rui's Assignment 1 Top-N ranking idea.

## Assignment 1 Integration
Inspected `FinRL-Trading-Full-Workload.ipynb`, `ray_blog.md`, `outputs_step2/stock_selected.csv`, `stock_selected_analysis.py`, and `strategy_framework/strategies/equal_weight_topk.py`. Recovered logic: top 25 percent ML-predicted stock selection, sector-neutral predicted-return residual score, robust z-score ranking, dynamic Top-N, and hysteresis.

## Installation
`pip install -r requirements.txt`

## Run
`python run_assignment2.py --config configs/assignment2_config.yaml --run-all`
`python run_assignment2.py --mode original`
`python run_assignment2.py --mode modified`
`python run_assignment2.py --mode benchmarks`

## Outputs
CSV outputs are in `results/`, figures in `figures/`, and the report in `blog/`.

## Main Results
| strategy | cumulative_return | annualized_return | annualized_volatility | sharpe_ratio | max_drawdown | cagr | calmar_ratio | number_of_rebalances | average_turnover | hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 8.7445 | 0.2275 | 0.2570 | 0.8853 | -0.3644 | 0.2275 | 0.6242 | 580.0000 | 1.4118 | 0.3961 |
| modified | 8.0359 | 0.2192 | 0.2562 | 0.8556 | -0.3758 | 0.2192 | 0.5832 | 580.0000 | 1.4038 | 0.3950 |
| qqq | 2.8344 | 0.1286 | 0.2032 | 0.6331 | -0.3562 | 0.1286 | 0.3611 | 0.0000 | 0.0000 | 0.3982 |
| spy | 1.5476 | 0.0878 | 0.1647 | 0.5334 | -0.3410 | 0.0878 | 0.2576 | 0.0000 | 0.0000 | 0.3918 |
| equal_weight_optional | 21.1214 | 0.3215 | 0.2636 | 1.2195 | -0.4520 | 0.3215 | 0.7114 | 0.0000 | 0.0000 | 0.5585 |

## Assumptions
local Assignment 1 fallback prices used because yfinance was unavailable or rate-limited

Signals are history-only, weights are applied with a one-day lag, and transaction cost is configurable.

## Known Limitations
Local fallback data may have fewer tickers than the full yfinance universe when Yahoo is rate-limited; missing tickers are logged in `data/cache/missing_tickers.csv`.
