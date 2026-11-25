# Assignment 2 Point in Time Valuation Filter

## Student

- **Name:** Bruce Qi
- **UNI:** jq2445
- **Module:** Valuation
- **Final configuration commit:** `e24e447cfb1fac723191830714349cf2a39ab3f7`
- **Published Medium article:** https://medium.com/@jiqibruce/can-valuation-improve-a-growth-stock-selection-strategy-32357083aba4

## Contribution

I implemented `student_valuation.py`. The module combines three point in time valuation methods:

1. trailing peer P/E using historical index members in the same GICS sector;
2. the stock's own historical EV/EBITDA range; and
3. a per share FinRobot DCF using financial information and interest rates available at each decision date.

The filter removes a candidate only when at least two valid methods value it below 70% of its current price and the ratio between the highest and lowest point estimates is no greater than 3.0. It removes no more than three top ten candidates per week, and the next ranked candidates fill the vacancies. Missing or invalid accounting inputs are not replaced with estimated values; a stock without at least two valid estimates remains eligible.

## Main Results

The valuation filter did not produce consistent effects across periods. It reduced development period Sharpe from 0.980 to 0.956, increased validation Sharpe from 0.028 to 0.035, and increased test Sharpe from 1.301 to 1.374. Test maximum drawdown improved from -30.53% to -28.08%, while annual turnover increased from 12.88 to 14.79. Rank IC was negative in development and validation and only modestly positive in testing, so the module should be interpreted as an experimental risk screen rather than a proven valuation ranking signal.

## Environment and Data

Use Python 3.11 or later. Obtain the Assignment 2 course data bundle separately and extract it as `source_code/data/`. The course data bundle is not included in this submission.

From `source_code/`, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the submitted `student_valuation.py` in `source_code/valuation/` before running the commands below.

## Reproduction Commands

Run the automated submission check:

```bash
python check_submission.py --valuation
```

Development period:

```bash
python evaluate_valuation.py --period dev
python run_portfolio.py --period dev --ablation --timing full
```

Validation period:

```bash
python evaluate_valuation.py --period val
python run_portfolio.py --period val --ablation --timing full
```

Test period commands used to produce the submitted results:

```bash
python evaluate_valuation.py --period test
python run_portfolio.py --period test --ablation --timing full
```

The test ledger contains two runs: one valuation evaluation and one portfolio ablation. Both used commit `e24e447cfb1fac723191830714349cf2a39ab3f7` with no uncommitted changes to the student file. The saved test results should be reviewed instead of rerunning the test period when checking this submission.

## Submitted Files

- `student_valuation.py`: final valuation implementation
- `research_report.pdf`: research design, evaluation, ablation results and judgment
- `check_submission_output.txt`: automated check with 0 FAIL and 0 WARN
- `output/valuation_eval_dev.csv`
- `output/valuation_eval_val.csv`
- `output/valuation_eval_test.csv`
- `output/ablation_dev.csv`
- `output/ablation_val.csv`
- `output/ablation_test.csv`
- `output/test_runs.jsonl`

The course data bundle, API keys, cache files and generated system files are excluded.
