# Assignment 1 - FinRL-Trading

JIAO XU / jx2687 | Columbia STAT GR5398 Fall 2026

**Status:** executed restricted-price-universe experiment and bias audit. The strategy **did not beat SPY**. Complete historical security-price coverage and proof of a fully survivorship-free tradable universe remain incomplete. Do not present this package as satisfying every graded requirement.

**Medium blog link:** [Read the article on Medium](https://medium.com/@xujiao0224/when-a-trading-backtest-becomes-a-data-audit-1dd32eb2140c)

Computational assistance was used for code audit, implementation, execution and drafting. The student must review the work and comply with the instructor's assistance/disclosure rules; no policy approval is inferred.

## Files

- `Research_Report.pdf`: brief research report with numerical results, method and limitations.
- `Assignment1_FinRL.ipynb`: executed result-validation companion (reads saved results; does not rerun the old training).
- `run_analysis.py`: deterministic replacement model fit, holdings simulation, metrics and figures.
- `results/`: NAV, gross/net returns, turnover, costs, target weights, selections, timing audit, missing-price counts, model scores, feature importance and figures.
- `cache/`: original completed 104-minute stock-selection run, including all 11 sectors' model result files and `stock_selected.csv`. Original cached signals fail the as-of audit; they are preserved evidence, not a valid unbiased strategy.
- `original_code/`: unchanged course preparation and model scripts.
- `environment.json`, `requirements.txt`, `validation_results.txt`: environment and verification evidence.
- `requirements_checklist.md`: Chinese requirement/status/source checklist.

## Results

Common initial close 2020-06-30 to 2025-09-30, 1,319 daily return observations:

| Series | Net cumulative return | CAGR | Sharpe (rf=0) | Volatility | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Purged ML, restricted prices | 109.10% | 15.08% | 0.96 | 15.98% | -22.22% |
| SPY, frictionless adjusted buy-and-hold | 132.66% | 17.45% | 1.02 | 17.24% | -24.50% |
| QQQ, frictionless adjusted buy-and-hold | 150.00% | 19.06% | 0.88 | 23.03% | -35.12% |

Mean daily turnover 0.010297 (sum of absolute dollars bought/sold / pretrade NAV). Fee 0.001 per dollar of gross traded notional. Daily cost fractions sum to 1.36%; this is not the terminal compounded wealth drag. First purchase cost included, no final liquidation. Missing entry allocations stay in zero-interest cash; no stock quote forward-fill or fabricated returns. Benchmark fees are excluded. See PDF for the full scope.

## Reproduce in an isolated environment

The local validated interpreter was `/opt/miniconda3/envs/finrl5398/bin/python` (Python 3.11.17). Exact core package versions are in `environment.json`. Do not install over another project's environment. For a new environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the official data separately. It is not included in this submission:

```bash
git clone https://github.com/AI4Finance-Foundation/FinRL-Trading.git finrl_data_repo
mkdir -p input/source_code/data
7z x finrl_data_repo/data/finrl_trading.7z -oinput/source_code/data
cp finrl_data_repo/data/fundamental_data_full.csv input/source_code/data/
cp finrl_data_repo/data/sp500_historical_constituents.csv input/source_code/data/
```

Then regenerate the preparation files (never commit `input/`, `prepared/`, or downloaded benchmark prices):

```bash
.venv/bin/python original_code/prepare_data.py \
  --db input/source_code/data/finrl_trading.db \
  --fundamentals input/source_code/data/fundamental_data_full.csv \
  --constituents input/source_code/data/sp500_historical_constituents.csv \
  --output-dir input/source_code/prepared \
  --start 2015-01-01 --end 2025-09-30 --trade-start 2020-06-30

.venv/bin/python fetch_benchmarks.py --output benchmark_data
.venv/bin/python run_analysis.py \
  --source input/source_code --output rerun_results \
  --benchmark-dir benchmark_data
.venv/bin/python extra_audit_local.py --source input/source_code --output rerun_results
```

`run_analysis.py` uses the supplied `cache/` when the original dated run is absent from `--source`. It refits the new experiment only (52 seconds in this local final run). The legacy 104-minute run is not needed. An independent directory rerun reproduced portfolio returns, selections, timing and metrics; numerical prediction differences were at most 1.1e-16 from parallel floating-point sums. File hashes in `audit_summary.json` identify the original inputs; an updated upstream dataset or revised Yahoo prices can change results and should be disclosed. The database price pool/selection counts are audited, not substituted with current component prices.

Open `Assignment1_FinRL.ipynb` from this folder in an environment containing pandas/numpy/ipykernel. Run all cells to independently validate the saved output metrics, dates, costs, weights and information timing. For a new model run, use the explicit script command above and compare `rerun_results/` to `results/`; the notebook does not claim to refit models by reading saved files.

The final local run was:

```bash
MPLCONFIGDIR="$PWD/work/matplotlib" /opt/miniconda3/envs/finrl5398/bin/python outputs/run_analysis.py \
  --source /Users/xujiao/Desktop/5398/STAT-GR5398-Columbia-University-master/Fall-2026/Assignment1/FinRL-Trading/source_code \
  --output outputs/results --benchmark-dir work
```

## Method and important limitations

Quarterly PIT-member ranking with accounting-only features; availability is max(quarter end, filing, accepted date) + one day. Labels must have completed by cutoff in both training and validation. Up to 16 training quarters + four validation quarters choose a fixed RF/Extra Trees model by MSE; refit uses only known labels. Latest eligible rows used for inference are not filtered by future label presence. Top ceil(25%) is equal weighted at the next trading close, with share holdings between quarterly rebalances.

Original output: 174,571 records, 418 tickers. Sixty-two selected tickers have no price history in the supplied database; 18,300 original records are affected (10.48%). Current database constituents exactly cover 503 members plus BRK/SPY/QQQ, consistent with incomplete historical-price coverage. No ticker mapping was applied. In the replacement experiment, 282 of 2,622 targets lack entry quotes; mean cash-reserved allocation at rebalances is 10.76%. This is material survivorship/coverage selection, even though the fundamental PIT filter is used.

Raw vendor fundamentals may be revised; filing metadata, labels, stable security IDs, dividend/split/merger adjustments and the historical membership reconstruction are not independently certified. All these prevent a blanket claim of zero bias. Benchmark downloads fill the actual database benchmark start-date gap (database SPY/QQQ begin 2020-12-01). The benchmark is SPY for the S&P 500; QQQ is not the S&P 500.

## Manual submission

Course originals: [Assignment 1](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1), [FinRL track](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRL-Trading).

1. Review the report, results, limitations and assistance disclosure; verify that the published Medium link above opens correctly.
2. Put this folder at `Fall-2026/Assignment1/FinRL-Trading/submissions/Assignment1_JIAOXU_jx2687/` in your course-repository fork.
3. Inspect the staged files (no raw datasets or secrets), then personally open the required PR against `master`.

The requirements state October 12, 2026 but do not specify the hour or timezone. Canvas course timezone is America/New_York, but no due-time record was available. Exact deadline remains unconfirmed; do not assume 23:59. No external submission, push, publication, email or live order was made.

Publication update: the Medium link above has now been supplied. Statements about pending publication in the report/notebook describe the earlier material-preparation stage.
