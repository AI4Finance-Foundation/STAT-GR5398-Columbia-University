# Four-quarter price-path efficiency

Two stocks with the same four-quarter gain can arrive there through a steady climb or repeated reversals. The original selector observes the latest one- and four-quarter changes and a measure of acceleration. This experiment adds a single model input capturing the direction and consistency of all four intervening moves. A positive result would suggest the *route* of a price move contains information beyond its endpoints.

For a ticker with five consecutive quarterly observations, compute `rᵢ = log(Pᵢ / Pᵢ₋₁)` and `efficiency = sum(rᵢ) / sum(abs(rᵢ))` across the four intervals. A steady rise gives +1, a steady decline −1, and a complete round trip zero. A flat path or an incomplete window receives zero, including gaps, nonpositive prices and missing observations. Four quarters match the original model's annual momentum horizon. The bounded feature needs no fitted clipping threshold or additional hyperparameter. The patch registers it in the momentum inputs of each fitted bucket and the unified model.

`P` is the course `trade_price` for a historical financial row at its modeled tradable date, typically about two months after quarter-end. The unchanged selector first filters by filing visibility and replaces any price dated after the monthly cutoff with the last available daily close. The new feature reads that capped column within ticker. It has no dependency on realized forward returns or current news. The original course model's other predictors, including its existing full-visible-sample imputation and winsorization, are shared by both arms. Formula checks cover rising, falling, flat, round-trip, missing-quarter and nonpositive-price paths, short histories and invariance to an appended future observation.

The baseline and treatment use the same original three-date calibration, asset universe, six regressors and stacking procedure. Each model produces six monthly score vintages from January to June 2018. Each selection-only replay uses the course portfolio simulator and identical cost and allocation rules from January 2 through June 29. The code for this additional experiment evaluates only the 2018 development window. The primary valuation holdout had already been run when this extension was requested. Neither the primary valuation configuration nor its test ledger changed. The feature definition was fixed before the new selection treatment was fit. The feature and the six-date calibration experiment are independent changes against the original control.

## Reproduce

From the submission directory, use the course environment with both optional boosting libraries installed. Supply a populated copy of the course `source_code` folder at revision `43eeda7b011270814fa28cc38e9081e36128cf8a` and a separate output directory. This sequentially retrains both six-month arms in new temporary directories with two monthly workers per arm.

```sh
python bonus/path_efficiency/run.py --source "$SOURCE" --out ./replayed_path --refit-control --workers 2
```

When the original calibration experiment has already run in the current workflow, pass `--control-results /path/to/retrained/control` to use its result directly. The script requires all five score, metadata and execution files to match the archived control byte for byte and checks the enclosing run manifest. Omitting both control options reuses the original bonus control's six completed fits only after matching its package versions, Python version, course-source hashes, control configuration, score checksum and data checksums. The 23 files listed in `course_files.sha256` were matched against the pinned course Git tree. The model patch is visible in `selector.patch`. It is applied only to a private source copy. All controls and altered scores are written through the course trainer's separate `--out` destination.

`results/path_efficiency/` contains all six altered monthly score sets in one parquet, training input hashes, daily equity, transaction ledger and official summary statistics. The two comparison CSVs pair returns, risk and turnover with growth-tech rank agreement by month. `feature_distribution.csv` separately reconstructs each monthly inference cross-section through the course filing, price-cap, membership and latest-report routines. It records 152 negative, 88 zero and 283 positive feature readings across 523 stock-months. Zero can indicate a round trip or an unusable window. `results/manifest.json` records the two model hashes, input and runtime checks, signs of the new feature on known paths, decision dates, fit and replay times. The official course score file remains the basis of the formal valuation A/B/C/D experiment.

## Paired findings

| Selection-only metric, January–June 2018 | Original control | With price-path efficiency |
| --- | ---: | ---: |
| Total return | 10.93% | 14.71% |
| CAGR | 23.46% | 32.18% |
| Annualized volatility | 23.73% | 23.63% |
| Sharpe | 1.007 | 1.299 |
| Maximum drawdown | −11.37% | −10.84% |
| Calmar | 2.062 | 2.968 |
| Annualized two-way turnover | 9.45× | 10.52× |
| Trades | 78 | 82 |

The course computes total return from the first recorded closing equity to the last. Adding the price-path feature increased the half-year return by 3.78 percentage points and reduced maximum drawdown by 0.53 points. Annualized turnover rose by 1.07 times average portfolio equity. Growth-tech ranks were strongly correlated (monthly Spearman 0.979–0.995), yet two or three raw top-ten picks differed in four months. In April and June the top-ten membership was identical. The six-month figures motivated a longer comparison. Across 84 months with the same feature, the improvement was larger in development and smaller in validation, where it won 13 of 24 months. See page 15 of the research report and the [seven-year paired refits](long_run/README.md).

The independent paired repeat refit both arms from empty monthly checkpoint directories. All 13 checked files matched byte for byte: two sets of scores, training metadata, daily equity, transactions and official metrics, plus the portfolio, rank and feature diagnostics. Each arm scored 2,975 names across six monthly vintages and generated 125 equity closes. The full control had 78 trades and the changed model had 82. Recorded versions, file hashes and actual fit times are in [the repeat record](results/replay_verification.json). After running the paired command, verify another replay with `python bonus/path_efficiency/verify_replay.py --replay "$REPLAY" --record ./verified_path.json`.
