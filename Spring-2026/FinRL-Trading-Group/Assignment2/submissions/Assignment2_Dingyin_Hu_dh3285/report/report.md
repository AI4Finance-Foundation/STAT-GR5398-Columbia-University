# Adaptive Rotation Strategy Backtest (README)

## Project Summary

Medium post:https://medium.com/p/2a5e28907979?postPublishedType=initial

This project implements and backtests an **Adaptive Rotation Strategy** for GR5398 Assignment 2. The strategy rotates across different asset groups under different market regimes and compares performance against **QQQ** and **SPY / S&P 500 proxy**.

I evaluate the following strategy variants:

1. **Original Adaptive Rotation**: baseline adaptive rotation strategy.
2. **Modified Adaptive Rotation**: adaptive rotation strategy with the growth stock selection part replaced by the QQQ rolling selection idea from Assignment 1.
3. **QQQ Benchmark**: buy-and-hold QQQ benchmark.
4. **S&P 500 Proxy**: broad-market benchmark.

The main improvement in the modified strategy is that the **Growth Tech** asset pool is no longer treated as a fully static list. Instead, it can use the rolling stock selection result from Assignment 1, so the strategy adapts both across asset groups and within the growth universe.

## Key Data

### Data Inputs

`data/fmp_daily/*.csv`  
Contains daily OHLCV price data for individual assets and benchmarks used in the backtest.

`data/a1_growth_pool.csv`  
Contains the Assignment 1 rolling selection result used to dynamically update the growth technology pool.

`submission_final/code/AdaptiveRotationConf_v1.2.2.yaml`  
Contains the strategy configuration, including asset groups, regime rules, ranking parameters, fallback assets, and stop-loss settings.

### Asset Groups

The strategy uses multiple asset groups to adapt to different market regimes:

1. **Growth Tech**: large-cap technology / growth stocks, improved with QQQ rolling selection.
2. **Cyclical / Real Assets**: energy, materials, commodities, and cyclical exposure.
3. **Defensive Assets**: defensive equity sectors and lower-risk assets.
4. **Fallback Assets**: SPY, QQQ, IAU, XLU, XLV, used when normal selection is weak or risk conditions are unfavorable.

### Return Definition

Daily asset return is computed from close-to-close price changes:

```text
r_i,t = P_i,t / P_i,t-1 - 1
```

Portfolio NAV is calculated by compounding portfolio returns:

```text
NAV_t = NAV_t-1 * (1 + r_portfolio,t)
NAV_0 = 1
```

## Strategy Details

### 1. Group Selection

The strategy reselects assets weekly. For each rebalance date, it evaluates recent group strength using a 12-week risk-adjusted return / information-ratio style signal.

The strongest group or groups are selected subject to the current market regime and portfolio constraints.

### 2. Intra-Group Asset Ranking

Inside the selected group, assets are ranked using **Residual Momentum**:

```text
Residual Momentum = Asset Return - Group Return
```

The strategy ranks assets by robust z-score of residual momentum and selects the top assets in the group. This helps avoid holding the whole group passively and instead focuses on the strongest assets relative to their peers.

### 3. QQQ Rolling Selection Modification

The key modification is applied to the Growth Tech group. Instead of relying only on the static list:

```text
AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA
```

I use the Assignment 1 QQQ rolling selection result to update the growth candidate pool over time. If the rolling selection file is unavailable or has insufficient candidates, the strategy falls back to the static Growth Tech list.

This modification is designed to make the strategy more adaptive to changes in market leadership.

## Risk Management and Timing

### Slow Regime Layer

The slow regime layer uses market trend, drawdown, and volatility information to classify the market into risk states such as:

1. **Risk-on**
2. **Neutral**
3. **Risk-off**

The regime state controls total exposure, group caps, and cash floor settings.

### Fast Risk-Off Layer

The fast risk-off layer monitors sudden market stress using short-term price shocks and volatility shocks. When triggered, it reduces portfolio exposure more quickly than the normal weekly rebalance schedule.

### Stop-Loss Rule

The strategy also includes daily stop-loss monitoring. If selected positions suffer losses beyond the configured threshold, the risk manager can reduce or remove exposure.

## Backtest Mechanics

The final backtest period is:

```text
2018-01-01 to 2025-12-31
```

The strategy uses:

1. **Weekly rebalancing** for regular portfolio selection.
2. **Daily monitoring** for fast risk-off and stop-loss adjustments.
3. **Equal weighting** among selected assets.
4. **Fallback allocation** when normal group selection is not favorable.

The final equity curve compares:

1. Modified Adaptive Rotation
2. Original Adaptive Rotation
3. QQQ
4. S&P 500 Proxy

## Results (Performance Metrics)

Metrics reported:

1. Final NAV
2. Total Return
3. Annualized Return
4. Annualized Volatility
5. Sharpe Ratio, with risk-free rate assumed to be 0
6. Max Drawdown
7. Win Rate

| Strategy | Final NAV | Total Return | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Modified Adaptive Rotation | 4.9920 | 399.20% | 22.34% | 22.53% | 0.99 | -30.05% | 58.81% |
| Original Adaptive Rotation | 3.7172 | 271.72% | 17.90% | 16.26% | 1.10 | -20.11% | 59.06% |
| QQQ | 3.8531 | 285.31% | 18.43% | 22.71% | 0.81 | -35.46% | 57.82% |
| S&P 500 Proxy | 2.5263 | 152.63% | 12.33% | 19.20% | 0.64 | -31.81% | 58.06% |

## Key Takeaway

The **Modified Adaptive Rotation** strategy achieves the highest final NAV and annualized return among the compared strategies. It also outperforms QQQ on final NAV over the full backtest period.

However, the improvement comes with a tradeoff. The modified strategy has a higher max drawdown than the original Adaptive Rotation strategy, and the original strategy has a slightly higher Sharpe ratio. This means the modified version is more return-seeking, while the original version is more conservative and smoother.

Overall, the QQQ rolling selection modification improves return potential, but risk management remains important because stronger growth exposure can increase drawdown during market corrections.

## How to Run

Verify that the required files exist:

```text
submission_final/code/AdaptiveRotationConf_v1.2.2.yaml
submission_final/code/intra_group_ranking.py
submission_final/code/run_adaptive_rotation_strategy.py
data/fmp_daily/*.csv
data/a1_growth_pool.csv
```

Run the strategy script to:

1. Load and preprocess price data.
2. Apply the adaptive regime detection logic.
3. Select asset groups based on recent group strength.
4. Rank assets inside selected groups using residual momentum.
5. Generate portfolio weights.
6. Run the backtest and save output files.

Example command:

```bash
python submission_final/code/run_adaptive_rotation_strategy.py --backtest --start 2018-01-01 --end 2025-12-31 --config submission_final/code/AdaptiveRotationConf_v1.2.2.yaml
```

## Outputs

Final outputs are stored in `submission_final/outputs/`:

```text
assignment2_line_chart.png
backtest_2018-01-01_to_2025-12-31.final.png
backtest_2018-01-01_to_2025-12-31.final.csv
ars_portfolio_weights_2018-01-01_to_2025-12-31.final.csv
comparison_metrics.csv
combined_metrics_comparison_table.png
```

### Main Figures

`assignment2_line_chart.png`  
Shows the equity curve comparison between Modified Adaptive Rotation, Original Adaptive Rotation, QQQ, and the S&P 500 proxy.

`combined_metrics_comparison_table.png`  
Shows the summary performance metric comparison.

`backtest_2018-01-01_to_2025-12-31.final.png`  
Shows the final backtest result generated by the strategy runner.


## Implemented Optional Improvement: QQQ Trend Exposure Overlay

In addition to replacing the static Growth Tech pool with the QQQ rolling selection result, I implemented one extra optional improvement: a **no-lookahead QQQ trend exposure overlay**.

The motivation is that the modified strategy improves return, but it still experiences meaningful drawdown during market stress. Instead of changing the stock selection engine again, I added a risk-management overlay that scales the modified strategy's exposure based on the broad market trend.

The overlay uses QQQ and an 8-week moving average. To avoid look-ahead bias, the signal is shifted by one rebalance period before it is applied:

```text
Risk-on if QQQ > 8-week moving average
Risk-off otherwise
```

During risk-on periods, the modified strategy keeps full exposure:

```text
Exposure = 1.00
```

During risk-off periods, the strategy keeps partial exposure instead of going fully to cash:

```text
Exposure = 0.60
```

The improved return is calculated as:

```text
r_improved,t = Exposure_t * r_modified,t
```

This improvement is implemented in:

```text
submission_final/code/apply_trend_exposure_overlay.py
```

The key point is that this improvement is designed to improve **risk-adjusted performance**, not simply maximize final return.

### Improvement Results

| Strategy | Final NAV | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown | Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| Modified + QQQ Trend Exposure Overlay | 4.3168 | 20.13% | 20.12% | 1.00 | -24.53% | 58.81% |
| Modified Adaptive Rotation | 4.9920 | 22.34% | 22.53% | 0.99 | -30.05% | 58.81% |
| QQQ | 3.8531 | 18.43% | 22.71% | 0.81 | -35.46% | 57.82% |

### Improvement vs Base Modified Strategy

| Metric | Base Modified | QQQ Trend Overlay | Change |
|---|---:|---:|---:|
| Final NAV | 4.9920 | 4.3168 | -0.6752 |
| Annual Return | 22.34% | 20.13% | -2.21 percentage points |
| Annual Volatility | 22.53% | 20.12% | -2.41 percentage points |
| Sharpe Ratio | 0.99 | 1.00 | +0.01 |
| Max Drawdown | -30.05% | -24.53% | +5.52 percentage points |
| Final NAV vs QQQ | +1.1390 | +0.4638 | Still above QQQ |

The QQQ trend overlay reduces total return compared with the unscaled modified strategy, but it meaningfully reduces maximum drawdown and annual volatility while keeping final NAV above the QQQ benchmark. This is a useful tradeoff because a real trading strategy should not only maximize return, but also control downside risk and portfolio volatility.

My interpretation is that the QQQ rolling selection improves the alpha side of the strategy, while the QQQ trend exposure overlay improves the portfolio risk-control side. Together, they make the strategy more complete: one layer selects stronger assets, and the other layer controls exposure when the broad market trend weakens.

## Conclusion

This project shows that combining an Adaptive Rotation Strategy with QQQ rolling stock selection can improve long-term return. The modified strategy performs better than the original strategy and QQQ in terms of final NAV and annualized return.

The main limitation is that the modified strategy takes more risk than the original Adaptive Rotation strategy. Future improvements could include volatility targeting, transaction cost modeling, and combining momentum with fundamental or quality signals.
