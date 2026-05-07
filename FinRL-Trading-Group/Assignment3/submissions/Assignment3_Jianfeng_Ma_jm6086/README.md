# GR5398 Final Assignment Submission

Author: Jianfeng Ma (`jm6086`)

## Files

- `Final_Assignment_Blog.md`: Medium-style final write-up.
- `final_strategy.py`: reproducible backtest code.
- `data/yahoo_2024_2026_prices.csv`: cached Yahoo Finance close/volume data used by the submitted run.
- `outputs/metrics.csv`: performance table.
- `outputs/nav.csv`: normalized NAV series.
- `outputs/equity_curve.png`: strategy versus benchmark chart.
- `outputs/rebalance_weights.csv`: monthly selected names, weights, and signal scores.
- `outputs/signal_snapshot.csv`: full signal table at each rebalance.

## Reproduce

From this folder:

```powershell
python final_strategy.py
```

The script uses the cached price file by default. To refresh Yahoo Finance data:

```powershell
python final_strategy.py --refresh-data
```

## Result

Required window: `2026-01-01` to `2026-04-30`

| Portfolio | Cumulative Return | Sharpe |
|---|---:|---:|
| Integrated AI Strategy | 21.36% | 3.63 |
| SPY | 5.68% | 1.31 |
| QQQ | 8.83% | 1.62 |

