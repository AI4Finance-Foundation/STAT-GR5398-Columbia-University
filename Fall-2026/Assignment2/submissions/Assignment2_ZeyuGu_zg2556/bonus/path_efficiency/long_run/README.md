# Seven years of monthly selection refits

This bonus follows up the January–June 2018 experiment on the four-quarter price-path efficiency feature. Both arms retrain the original six-model selector once per calendar month from January 2017 to December 2023. They use the same course `selection/configs/growth.json` with the original three-report-date calibration. The feature arm applies the existing [selector patch](../selector.patch). Every monthly fit uses the previous month's last equity session as cutoff. The course's published stock scores remain the inputs to the graded valuation study.

The selection portfolios reset to $100,000 at the start of development (2017–21) and validation (2022–23). Both arms use a 5% cash buffer, ten target growth stocks, weekly rebalancing, whole shares and 6 basis points of cost per dollar traded. Prices and membership inputs end on December 29, 2023. Turnover includes buys and sells and is annualized against mean portfolio equity. The return basis is the first closing net asset value in each period.

| Period | Model | Net return | Maximum drawdown magnitude | Annual turnover |
|---|---|---:|---:|---:|
| Development | Retrained control | 203.34% | 33.31% | 10.13× |
| Development | Price-path feature | 242.78% | 32.93% | 10.18× |
| Validation | Retrained control | −5.11% | 41.38% | 9.60× |
| Validation | Price-path feature | −3.25% | 40.76% | 9.26× |

The feature led in 36 of 60 development months, four of five development years and 13 of 24 validation months. In validation, it lost less than the control in 2022 and trailed by 1.30 percentage points in 2023. The feature was fixed before this longer replay. Its initial six-month interval and the validation period had already been examined. The original-vs-modified comparison is a retrospective stability check on these dates.

The course engine carries DISCK at its last quote after April 8, 2022. A separate event replay exchanges held DISCA or DISCK shares one-for-one into WBD before regular trading on April 11 and excludes the retired tickers from future orders. Net validation return after conversion is −3.01% for the control and +0.60% with the feature, with drawdown magnitudes of 43.70% and 42.46%. This reconstruction covers this corporate action only. [Issuer's SEC filing, items 3.01 and 3.03](https://www.sec.gov/Archives/edgar/data/1437107/000119312522103051/d328161d8k.htm)

## Reproduce from the submitted files

Use the course environment and a populated course `source_code` tree at revision `43eeda7b011270814fa28cc38e9081e36128cf8a`. This experiment used Python 3.13.2 on macOS with the packages in [requirements-lock.txt](../../../requirements-lock.txt). The runner uses POSIX file locking; macOS and Linux support it. Git is required for the feature patch. The two locally generated 84-month score tables are included under `scores/` (about 1.1 MiB together). Training checkpoints can be resumed by repeating the command with the same output directory and input hashes.

From the submission directory, replace `/path/to/source_code` with the course directory holding the populated `data` folder.

```sh
python bonus/path_efficiency/long_run/run.py --source /path/to/source_code --out ./replayed_long_run --workers 4
```

The runner builds a temporary, date-bounded data snapshot, verifies the course files, fits both model arms in the same environment and checks the resulting score files against the [training manifest](results/training_manifest.json). Four workers per arm allow up to eight simultaneous fits. The new score files and checkpoints go under `replayed_long_run/scores/`. The runner then replays both portfolios, the original six-month anchor and the separate conversion sensitivity. All period, year and month tables, daily NAVs, fills and weekly targets are written beneath `replayed_long_run/results/`. The final `verification.json` compares the generated figures against the archived tables.

To replay the entire portfolio experiment immediately from the included predictions, run the command below. Both score files are checked against the recorded 84-month hashes. You may also supply the `scores/` directory from a completed refit. Choose a separate output directory for the replay.

```sh
python bonus/path_efficiency/long_run/run.py --source /path/to/source_code --out ./verified_long_run --scores-from bonus/path_efficiency/long_run/scores
```

[Period levels](results/period_metrics.csv) · [Period differences](results/period_differences.csv) · [Annual levels](results/year_metrics.csv) · [Annual differences](results/year_differences.csv) · [Monthly differences](results/month_differences.csv) · [Win counts and stability](results/stability.csv) · [Conversion results](results/corporate_action/period_metrics.csv)
