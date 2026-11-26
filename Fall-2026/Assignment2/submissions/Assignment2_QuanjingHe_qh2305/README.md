# Assignment 2 — Recovery-Aware Market Timing

**Course:** STAT GR5398, Fall 2026  
**Track:** FinRL-Trading (Timing Module)  
**Student:** Quanjing He  
**UNI:** `qh2305`  
**Submission:** Individual

**Research report:** [Recovery-Aware Market Timing for a Growth and Technology Equity Portfolio](GR5398_Assignment2_Research_Report.pdf)  
**Published Medium article:** [I Tested a Simple Timing Rule on a Growth-Stock Portfolio](https://medium.com/@qh2305/i-tested-a-simple-timing-rule-on-a-growth-stock-portfolio-06d517cbef90)

## 1. Research question and individual contribution

I tested whether a transparent market-timing overlay can reduce drawdowns in the course-supplied growth/technology stock portfolio without changing stock selection.

- **My contribution:** designed, implemented, evaluated, and documented the timing module in [`student_timing.py`](student_timing.py). I developed candidate rules E1–E8, compared them across the course-defined development and validation periods, selected E5, and performed sensitivity analysis, crisis analysis, A/B/C/D ablations, and look-ahead checks. I also examined E9–E11 as **exploratory** variants after reviewing validation diagnostics; I do not treat those later experiments as independent validation evidence.
- **Course-provided components:** selection scores, growth/technology stock selection, valuation baseline, portfolio backtester, and historical data. I did not retrain the selection model or implement a new valuation module.
- **Deliverables:** the final timing module, research PDF, checker output, supporting experiment records, and this README. I completed the assignment individually.

## 2. Final timing rule (E5)

E5 adjusts portfolio exposure using point-in-time SPY closing prices. The rule starts at 40% exposure, adds 30 percentage points when SPY is above its 50-session moving average, and adds another 30 points when it is above its 200-session moving average. If SPY remains below its 200-session average but is above both its 20-session average and its closing price 20 sessions earlier, E5 adds a 15-point early-recovery adjustment. The result is clipped to `[0.10, 1.00]` (the implemented rule normally stays at or above 0.40). If fewer than 200 valid SPY observations are available, the function returns `1.0`.

The course portfolio invests `0.95 × exposure` in selected stocks and keeps the remainder in cash. All inputs are accessed through the course-provided `DataView`; the function does not read local files, make network requests, or call an LLM. Decisions use information available by the previous trading session's close; weekly trades execute at the next open.

## 3. Research periods and main results

| Period | Dates | Selection-only CAGR | E5 CAGR | Selection-only Sharpe | E5 Sharpe | Selection-only max drawdown | E5 max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 2017-01-03 to 2021-12-31 | 26.71% | 21.71% | 0.980 | 0.981 | -32.99% | -23.07% |
| Val | 2022-01-03 to 2023-12-29 | -3.20% | 2.50% | 0.028 | 0.223 | -42.52% | -36.66% |
| Locked Test | 2024-01-02 to 2026-10-09 | 42.84% | 44.45% | 1.301 | 1.348 | -30.53% | -27.08% |

Values come from the archived [`output/ablation_dev.csv`](output/ablation_dev.csv), [`output/ablation_val.csv`](output/ablation_val.csv), and [`output/ablation_test.csv`](output/ablation_test.csv). For this timing-track submission, the valuation module is unchanged, so ablations **A = B** and **C = D**; **C − A** isolates the timing contribution.

The Test-period improvement is concentrated in the 2026 decline. The two portfolios had identical annual returns to reported precision in 2024 and 2025; I do **not** claim that E5 beat every official timing baseline on the locked Test, because SMA200 and Engine were not independently run on Test.

## 4. Submitted files and evidence

| Path | Purpose |
| --- | --- |
| [`student_timing.py`](student_timing.py) | Final E5 timing function; copy into the course's `source_code/timing/` directory. |
| [`GR5398_Assignment2_Research_Report.pdf`](GR5398_Assignment2_Research_Report.pdf) | Five-page research report, including audits, baselines, ablations, limitations, and judgment. |
| [`check_submission_output.txt`](check_submission_output.txt) | Archived official timing checker output: **0 FAIL, 0 WARN, 4 PASS**. |
| [`output/ablation_dev.csv`](output/ablation_dev.csv), [`output/ablation_val.csv`](output/ablation_val.csv), [`output/ablation_test.csv`](output/ablation_test.csv) | Required A/B/C/D marginal-contribution results. |
| [`output/test_runs.jsonl`](output/test_runs.jsonl) | Record of the **one** formal Test run. |
| [`output/final_E5_lock.json`](output/final_E5_lock.json) | E5 configuration, local Git-lock commit, SHA256, and run-count-at-lock record. |
| [`output/`](output/) | E5 and baseline timing evaluations, portfolio statistics/equity, sensitivity, crisis diagnostics, and Test attribution. |
| [`experiments/`](experiments/) | Archived E1–E11 sources and selected Dev/Val results; E9–E11 are post-validation exploratory variants. |

No course data bundle, credentials, local Git repository, or environment files are included. E5 does not require any `student_data/*.parquet` file.

## 5. Reproducibility

### 5.1 Setup and install the submitted module

Use the **instructor's original** `Fall-2026/Assignment2/source_code/` project, not this submission directory, to execute the course scripts. Python 3.11+ and the course dependencies are required. Download/unzip the instructor-provided data bundle into `source_code/data/` as described in the course README; **do not commit the data directory to GitHub**.

From `Fall-2026/Assignment2/source_code/`, on Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
Copy-Item ..\submissions\Assignment2_QuanjingHe_qh2305\student_timing.py .\timing\student_timing.py -Force
$env:PYTHONHASHSEED = "0"
```

On macOS/Linux, the equivalent setup is:

```bash
python -m pip install -r requirements.txt
cp ../submissions/Assignment2_QuanjingHe_qh2305/student_timing.py timing/student_timing.py
export PYTHONHASHSEED=0
```

These commands assume the submission folder is located next to `source_code/` under `Assignment2/`. Run subsequent course commands from `source_code/`. Set `PYTHONHASHSEED` **before** starting Python to match the archived fixed-seed comparisons.

### 5.2 Course checks and safe Dev/Val reproduction

```bash
python check_submission.py --timing

# SPY-only timing diagnostics and comparisons against official baselines
python evaluate_timing.py --period dev --timing student full sma200 engine
python evaluate_timing.py --period val --timing student full sma200 engine

# Stock-portfolio A/B/C/D ablations
python run_portfolio.py --period dev --ablation --timing student
python run_portfolio.py --period val --ablation --timing student
```

The SPY-only diagnostics and stock-portfolio ablations are **different evaluations**; their metrics should not be interchanged. The archived timing outputs are in `output/timing_eval_dev_E5.csv`, `output/timing_eval_val_E5.csv`, and the baseline/portfolio files under `output/` in this submission. The archived validation A/C comparison uses `PYTHONHASHSEED=0`. A standalone historical Full run with a different execution context gave a slightly different valuation-period number and is not substituted for the fixed-seed A/C ablation.

In a clean course checkout, the checker's Test-log count depends on the local `source_code/output/test_runs.jsonl`; refer to the **archived checker output and Test log in this submission** for the original one-Test evidence. Do not generate a new Test simply to reproduce its results.

### 5.3 Verify the archived report numbers without rerunning Test

From `Fall-2026/Assignment2/submissions/Assignment2_QuanjingHe_qh2305/`:

```bash
# Dev, Val, Test portfolio results (A/B/C/D)
python -c "import pandas as pd; periods=['dev','val','test']; x=pd.concat([pd.read_csv(f'output/ablation_{p}.csv').assign(period=p) for p in periods]); print(x[['period','run','cagr_pct','volatility_pct','sharpe','max_drawdown_pct','turnover_per_year']].to_string(index=False))"

# Development candidate and official baseline comparison
python -c "import pandas as pd; print(pd.read_csv('output/dev_phase2_portfolio_summary.csv').to_string(index=False))"

# Recovery-boost sensitivity (E2 and E5 with boosts 0.10 / 0.15 / 0.20)
python -c "import pandas as pd; print(pd.read_csv('output/dev_E5_parameter_sensitivity_full.csv').to_string(index=False))"

# Annual attribution from the single archived Test
python -c "import pandas as pd; print(pd.read_csv('output/final_test_E5/annual_return_attribution.csv',index_col=0).to_string())"

# Number of formal Test runs in the submitted log
python -c "from pathlib import Path; p=Path('output/test_runs.jsonl'); print(sum(bool(s.strip()) for s in p.read_text().splitlines()))"
```

Additional reported results can be checked directly against the corresponding archived files: `output/val_portfolio_comparison.csv`, `output/val_spy_comparison.csv`, `output/timing_crises_dev_E5.csv`, `output/timing_crises_val_E5.csv`, `output/dev_E10_crisis_comparison.csv`, `output/final_test_E5/`, and the per-experiment `experiments/*/stats.json`. The E1–E11 source files are available for rerunning development variants in a **separate scratch copy** of the course project; they must not replace the final submitted E5 module.

### 5.4 Git lock and one-time formal Test

- **Selected strategy:** E5 (`experiments/E5_recovery.py`), identical to the submitted `student_timing.py`.
- **Original local Git lock commit (before Test):** `03a995ab3616a3b8376a3b5b8fbd87428c15dc05`.
- **SHA256 of frozen `student_timing.py`:** `9bc611c8e62cd8ffffb42b8c72e45d74f43bd1bbbfc1bf1a8c0fd3e18abf78cb`.
- **Fixed Python hash seed:** `PYTHONHASHSEED=0`.
- **Formal Test runs:** **1**, confirmed by [`output/test_runs.jsonl`](output/test_runs.jsonl). There were **0 Test runs at the Git lock**.

The historical command was `python run_portfolio.py --period test --ablation --timing student`, executed **once** after the Git lock. **Do not rerun this command**: the submitted Test tables, equity curves and trade logs are archived under `output/` and are sufficient to inspect the original result. Uploading this submission to another GitHub repository creates a new Git commit; it does not transfer the original local lock commit into that repository's history.

## 6. Limitations and research judgment

E5 improved maximum drawdown against selection alone across Dev, Val, and the locked Test, but it traded away some CAGR during Dev. Its Test benefit was concentrated in one 2026 market episode, and its validation maximum drawdown remained substantial. The results are historical backtests under the course's fixed selection scores, costs, and execution assumptions; they are not evidence of live-trading profitability.

I would not invest a substantial amount of real money based on these backtests alone. I would first paper-trade on new data, examine higher trading costs, and test whether the overlay behaves similarly with other stock portfolios. See the linked [research report](GR5398_Assignment2_Research_Report.pdf) for full discussion.
