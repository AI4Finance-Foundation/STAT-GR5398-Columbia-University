# A longer validation history for the monthly ensemble

A separate [price-path feature experiment](path_efficiency/README.md) changes the selection model inputs and retrains the six monthly models on the original calibration window. Its half-year return was 14.71% with Sharpe 1.299 and maximum drawdown −10.84%, versus 10.93%, 1.007 and −11.37% for the shared same-environment control. Annualized turnover increased from 9.45× to 10.52×.

## Paired experiment

The stock selector chooses the three most accurate base regressors for its stack and weights model predictions by inverse validation mean squared error. A short validation window may favor models that happen to fit a single reporting cycle. Combining more reporting dates may stabilize those choices, while reducing the initial training sample and responding more slowly to a change in market conditions. The final refit includes both the initial training and validation observations.

The control uses `selection/configs/growth.json`. The alternative changes only `model.val_quarters` from 3 to 6. The implementation collects the last *distinct labeled financial reporting dates* within each training bucket up to the monthly cutoff. The parameter does not determine the separate assignment-level validation period. Both arms keep the course features, seed, models, stacking procedure, universe, allocation limits, weekly execution and transaction costs.

The experiment fixes six monthly refits, January through June 2018, within the development period. The identical selection-only replay runs January 2 through June 29. This half-year offers six successive monthly decisions while keeping both complete refits reproducible on a laptop. No selection of the window or model setting uses bonus returns. Both score files must contain all six months. The paired growth-tech ranking requires matching security, bucket and prediction cutoff within that sleeve. The all-bucket coverage table separately counts names present in either arm. Each monthly cutoff must precede the first day of its score month. The mandatory assignment ablation uses the original course scores.

## Reading the outputs

`results/portfolio_comparison.csv` contains the course portfolio statistics; turnover is total buy plus sell notional divided by average equity and years. `results/monthly_rank_comparison.csv` reports Spearman correlation between the two rank-score series within each month's growth-tech candidates and the number of common names in the raw top tens. Two further columns report matched names and rank correlation between the same-environment control and the released course scores in those six development months. The subsequent investability screen and weight scaling can make realized positions differ from these diagnostic lists. A short window limits inference about differences in annualized Sharpe or drawdown.

`results/manifest.json` records versions, run times, config and score hashes, and the official scores' verified checksum before and after training. Each arm also has an input manifest, the actual monthly scores, execution ledger, daily equity and official summary statistics.

## Reproduction

With the assignment Python environment active, run from this bonus directory. Set `SOURCE` to the official `source_code` directory containing the unpacked data. Both model libraries in `requirements.txt`, including XGBoost and LightGBM, are required to reproduce these two arms.

```sh
python retrain.py --source "$SOURCE" --out ./reproduced
```

The experiment uses two workers per arm. On macOS both optional boosting libraries require a usable OpenMP runtime. A fresh staging directory prevents changes in configuration or library versions from silently reusing older training checkpoints.

## Results for January–June 2018

| Course replay metric | Three dates, control | Six dates |
| --- | ---: | ---: |
| Total return | 10.93% | 11.90% |
| CAGR | 23.46% | 25.67% |
| Annualized volatility | 23.73% | 23.95% |
| Sharpe | 1.007 | 1.074 |
| Maximum drawdown | −11.37% | −11.57% |
| Calmar | 2.062 | 2.218 |
| Annualized two-way turnover | 9.45× | 9.62× |
| Trades | 78 | 80 |

The longer calibration window gained 0.97 percentage points of total return with slightly higher turnover and a drawdown 0.20 points deeper. The two rankings preserved 7–9 of the same raw top ten each month. Growth-tech rank correlations ranged from 0.957 to 0.983. Comparing the control with the course's released scores gave monthly correlations from 0.983 to 1.000. The six-month results support reporting modest changes in holdings and implementation cost. They provide limited evidence about full-cycle stability.

The longer internal validation window also removed 459 real-assets security-months from the emitted score file across these six months. The course selector skips a bucket when its initial training set has fewer than 20 labeled rows or its inference set is empty. The input universe and inference date are fixed in this paired comparison, while a longer validation slice can leave too few initial training rows. All 523 monthly growth-tech observations were available in both arms, so these missing real-assets scores do not affect the traded sleeve. This coverage loss would matter for a multi-bucket portfolio.
