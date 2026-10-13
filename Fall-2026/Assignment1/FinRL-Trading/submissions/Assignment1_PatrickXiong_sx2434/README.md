# Assignment 1 - Patrick Xiong (sx2434)

Columbia STAT5398, Fall 2026. Final submission package, October 12, 2026.

## Read first

- `report.pdf`: four-page research report, latest WRDS results only.
- `Bias_Avoidance_Evidence.pdf`: two-page executable proof and its scope.
- `Review_Results.ipynb`: executed result review; no WRDS account is needed to inspect it.
- `results/three_strategy_comparison.csv` and `results/three_strategy_paths.png`: common-source comparison.
- `medium_draft.txt`: published article source text.

The main, validation-selected headline ordering is QQQ enhancement204.45% > QQQ-target active selection192.04% > SPY enhancement159.39%. SPY returned154.91%; QQQ191.86%. RF-only active selection is a separate declared comparator:214.29%, not a validation-selected winner adopted by final performance. It has higher volatility and turnover. All figures use May1,2020-September30,2025 and the same CRSP inputs.

## Publication and submission status

The author approved public publication and course submission on October 12, 2026. The Medium article is published: [FinRL stock selection: a WRDS replay of index enhancement and active portfolios](https://medium.com/@sx2434/finrl-stock-selection-a-wrds-replay-of-index-enhancement-and-active-portfolios-c38a724bf557). This final package is prepared for the course folder `Fall-2026/Assignment1/FinRL-Trading/submissions/Assignment1_PatrickXiong_sx2434/` in `AI4Finance-Foundation/STAT-GR5398-Columbia-University`. The source submission is proposed through a pull request from `pattybepatient/STAT-GR5398-Fall2026-Submission`; acceptance/merge is controlled by course maintainers.

## Install from a clean clone

Use Python3.9-3.11 (recorded estimator results used3.9); native wheels and package availability can differ by platform.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/review_results.py
```

The notebook reads compact result CSVs and verifies the displayed comparisons. It does not fit models or access raw data.

## Licensed source inputs

Download/export the four source files through an authorized WRDS/CRSP account. Exact queries, fields, dates, batch PERMNO lists, SQL and hashes are under `data_access/`; see `WRDS_query_instructions.txt` (the prior SPY source/access protocol, retained for its query details). Do not commit raw exports, prepared panels or fitted models.

Required frozen files in `/private/wrds/inputs`:

1. `crsp_daily_raw.parquet`
2. `crsp_names_all.csv`
3. `crsp_sp500_members.csv.gz`
4. `crsp_delisting.csv`
5. `frozen_manifest.json` containing the hashes of those four files.

To construct these normalized frozen inputs from the original WRDS CSV/gzip exports:

```bash
python code/prepare_data.py   --daily /private/wrds/daily_batch1.csv.gz /private/wrds/daily_batch2.csv.gz /private/wrds/daily_batch3.csv.gz   --membership /private/wrds/members.csv.gz   --names /private/wrds/names.csv   --delisting /private/wrds/delisting.csv   --output-dir /private/wrds/inputs
```

Fresh mutable vendor downloads may not reproduce the original frozen bytes. The exact locally retained frozen input set is checked by hash. This package includes no licensed data.

## Full source-to-model replay

```bash
python code/reproduce_all.py   --input-dir /private/wrds/inputs   --output-dir /private/replay
```

Use a new empty output directory. It creates common CRSP preparation and the SPY target first, then QQQ and absolute-target experiments. Add `--skip-absolute` to replay only the two benchmark-relative targets. Each target fits286estimators across22forecast windows; total runtime is typically tens of minutes on a laptop. Existing snapshots are preserved. Fresh outputs and audit logs are under `/private/replay/{spy,qqq,absolute}`. Small saved snapshots in this folder document the reviewed run; floating-point/threading differences can produce negligible numerical changes.

The SPY pipeline performs raw preparation, six-feature prefix checks, fitting, cost-net backtest, saved-estimator checks, independent raw-return/dollar-ledger checks, historical-source checks and bias witnesses. The QQQ and absolute pipelines inherit the numerically identical feature rows and independently refit their targets; they rerun clocks, saved-estimator audits, rankings and raw-source ledgers. QQQ relative targets are independently computed from raw stock and ETF returns.

**Executed replay scope.** `reproduce_all.py` has not been executed end to end from this packaged tree. The SPY pipeline was fully replayed once before packaging (zero NAV difference). For QQQ and absolute targets the packaged trainers were run in `--validate-inputs-only` mode (22 windows, inputs identical); fits were not repeated. The final Claude audit freshly rebuilt preparation and reran all three packaged backtests with saved predictions, and reran the packaged non-training audit scripts.

## Construction and proof

- Official dated S&P500 membership;705historical stock PERMNOs,41decision quarters,20,675cases. No end-period survivor screen.
- Six strictly preceding-session price features; train-only medians and missing indicators.
- Eligible completed-horizon labels are required, rather than silently filtering away missing failure outcomes.20,169observed outcomes;503censored, two unknown CVR cases and one non-executable INFO start remain forecasts.
- Past-only purged validation MSE chooses RF/XGB/LightGBM. RF averages all11seeds. The nominal validation span covers four decision quarters; strictly prior observed labels retain three completed quarters in each of22windows (1,508-1,516rows). The latest quarter ends on the forecast day and is excluded. Counts are independently matched to actual fit-manifest row hashes in `evidence/actual_validation_quarter_counts.json`.
- Quarterly decision schedules use the first market day of each March/June/September/December; labels span identical decision endpoints. Trade at that day's close after using only prior-session inputs.
-10bps per traded dollar including funding and event liquidation; stress20bps plus one session. Core portfolios hold the ETF before first forecastJune1; active portfolios hold zero-yield cash. No shorts or leverage.
- CIZ terminal return is counted once, with delayed quote/payment recognition and frozen held dollars. Unknown held consideration or unresolved held returns are fatal.
- Evidence folders contain actual role hashes, failure witnesses, future deletion/poisoning controls and independent reconstruction receipts. Vendor original-release vintages and index-announcement timestamps remain outside the finite proof. No untouched holdout/significant-alpha claim.

Point-in-time preparation excludes 110 of 815 exported stock PERMNOs that never belong at the 41 decisions. Among 27,643 positive-close, active-status stock-quarter candidates, membership removes 6,969 non-member rows and keeps 20,674. INFO's one no-start-quote member case is retained separately, giving 20,675 total cases. See `evidence/pit_filter_removal_check.json` for the candidate definition and the one no-quote case.

## What is included

Root `code/`, `results/`, `evidence/`: SPY source pipeline and combined report outputs. `experiments/qqq/` and `experiments/absolute/`: corresponding runnable code, protocols, compact results and receipts. The RF-only comparator and all seed sensitivities were declared before the latest replay; no winning seed is promoted. The latest QQQ results supersede older free-source values204.27%/198.67%.

Earlier Claude audit transcripts are retained with their exact scope. A read-only Claude Opus 5.5 audit (October 12, 2026) rebuilt the universe, labels, features, fit partitions and NAVs from the frozen inputs and re-ran the packaged non-training scripts. It did not refit models or certify CRSP release vintages. The final verdict was READY FOR AUTHOR APPROVAL; see evidence/claude_opus_final_review.txt and evidence/claude_opus_final_scope.json.

Original workload: the28-cell notebook workflow was run on prepared inputs;20code cells executed with zero errors, and one redundant raw-preprocessing cell was skipped. It reused completed sector-model outputs. See `evidence/original_notebook_execution_check.json`. The handout's31.00% baseline and this project's27.16% archived intraday-proxy reconstruction are documented separately; they are not comparable close-to-close total-return strategies.

## Regenerate the reviewed PDFs

```bash
python code/build_submission.py
```

This reads the included reviewed result snapshots and rewrites the two PDF artifacts. For fresh numerical results, inspect the replay comparison/audit outputs before replacing reviewed snapshots.

To regenerate selected-model feature importance after a new QQQ replay:

```bash
python code/summarize_model_importance.py --run-dir /private/replay/qqq/run --output-dir /private/replay/qqq/summary
```
