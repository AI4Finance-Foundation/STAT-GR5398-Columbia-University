# Assignment 2 — FinRobot Valuation Track

**Course:** STAT-GR5398, Columbia University (Fall 2026)
**Student:** Xiaoyu Zhang
**UNI:** xz3423
**Project:** Point-in-Time Multi-Method Valuation for Machine-Learning Growth Stock Selection

## Overview

This individual project implements a point-in-time valuation module for the course-provided machine-learning growth-stock portfolio. The module combines (1) same-sector peer trailing P/E, (2) company-specific historical EV/EBITDA, and (3) a FinRobot-based DCF using historically available Treasury yields and financial statements. It reports median fair value and an uncertainty envelope, and separately filters a candidate only when at least two methods indicate more than 30% overvaluation and the ratio of the highest to lowest valid method estimate is at most 2.0. Otherwise, the candidate is retained. The Selection model, candidate ranking, and portfolio engine are unchanged.

The principal research result is **not** a consistent improvement in growth-stock returns. Relative to Selection alone, the valuation filter changes cumulative return by **−4.43 percentage points (Dev), +1.51 pp (Val), and −2.14 pp (Test)**. On Test, the Sharpe ratio improves only slightly (1.301 to 1.315), while annual turnover increases (12.88 to 14.71). See `Research_Report.pdf` for methods, data and look-ahead audit, benchmark comparisons, robustness analysis, and personal investment judgment.

## Author and contribution

**Xiaoyu Zhang (xz3423)** — Sole author. Designed and implemented the three point-in-time valuation methods, the multi-method aggregation and conservative filtering rule; audited data availability and model limitations; conducted valuation evaluation, portfolio ablations, annual and replacement attribution, and development-period threshold sensitivity; authored the research report and Medium article.

## Final model and integrity

- **Pre-test frozen model commit:** `f88ada700dd8fa2ba46a4fe762b081d364d4616c`
- **Module to grade:** `student_valuation.py`, a copy of `source_code/valuation/student_valuation.py` at the frozen commit.
- **Test-period executions:** **2**, both after the freeze, recorded in `output/test_runs.jsonl`:
  1. `PYTHONHASHSEED=0 python run_portfolio.py --period test --ablation --timing engine`
  2. `PYTHONHASHSEED=0 python evaluate_valuation.py --period test`
- The same model commit was used in both executions; no post-test model adjustment was made. The five-case threshold sensitivity was conducted retrospectively on **Dev only**, with the original Dev portfolio equity reproduced exactly. It is **not** a newly selected test-period parameter configuration.
- The submission checker previously passed all valuation-module checks without a FAIL. See the submitted `check_submission.txt` for the actual final checker output. **Note:** the course checker intentionally reports a `WARN` for more than one logged Test execution; with 2 disclosed runs, this bookkeeping warning is expected and is not a module FAIL.

## Reproduction

All commands below are run from `Fall-2026/Assignment2/source_code/` with Python 3.11+ and the course-provided data bundle unpacked in `source_code/data/`. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The course data bundle and any private credentials are **not** included in this submission. Restore the supplied bundle per the [Assignment 2 setup instructions](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment2/README.md).

To reproduce the valuation-module evaluations:

```bash
python evaluate_valuation.py --period dev
python evaluate_valuation.py --period val
# Test command shown for provenance only; this project's Test was already executed twice.
PYTHONHASHSEED=0 python evaluate_valuation.py --period test
```

To reproduce the portfolio A/B/C/D tables and the SPY/QQQ benchmarks in the report:

```bash
PYTHONHASHSEED=0 python run_portfolio.py --period dev --ablation --timing engine
PYTHONHASHSEED=0 python run_portfolio.py --period val --ablation --timing engine
# Historical frozen Test invocation, NOT a request to run Test again:
PYTHONHASHSEED=0 python run_portfolio.py --period test --ablation --timing engine
```

The two distinct **full-exposure** outcomes are A (Selection) and B (Selection + Valuation). These are emitted by the above ablation runs regardless of `--timing engine`, which affects only C and D. Under the course's full-exposure timing baseline, C=A and D=B by construction; the original final Test run was not repeated with `--timing full`. The Engine C/D outcomes are a supplementary comparison.

To validate the module and document the checker output (without rerunning Test):

```bash
python check_submission.py --valuation | tee ../submissions/Assignment2_XiaoyuZhang_xz3423/check_submission.txt
```

Reproduction of the **supplementary analyses** requires their accompanying analysis scripts in `source_code/`: `sensitivity_dev.py` for the five fixed Dev-only filter variants and `analyze_replacements.py` for target-level replacement attribution. The annual portfolio comparisons were computed from the saved A/B equity curves, using the initial capital and preceding year-end NAV to calculate annual returns. The corresponding CSVs are provided with this submission. For sensitivity, run `PYTHONHASHSEED=0 python sensitivity_dev.py` from `source_code/` after confirming that the preserved frozen Dev B equity series is present. The replacement-attribution script's exact invocation and the annual-aggregation reproducibility command should match the included source scripts; do not infer a test-period execution from either diagnostic.

**Important:** Source-code paths and historical command lines are documented for reproducibility. The listed Test commands describe the *already recorded* runs, not additional experiments to execute after observing the held-out results.

## Submitted files and supporting results

The submission directory is intended to contain:

```text
Assignment2_XiaoyuZhang_xz3423/
├── README.md
├── student_valuation.py
├── Research_Report.pdf
├── check_submission.txt
└── output/
    ├── valuation_eval_dev.csv
    ├── valuation_eval_val.csv
    ├── valuation_eval_test.csv
    ├── ablation_dev.csv
    ├── ablation_val.csv
    ├── ablation_test.csv
    ├── sensitivity_dev.csv
    ├── annual_robustness_ab_final.csv
    ├── annual_replacement_attribution_dev_current.csv
    ├── annual_replacement_attribution_val_current.csv
    └── test_runs.jsonl
```

The two annual replacement-attribution CSVs are supplementary diagnostics. Reproduction scripts used for supplementary figures or tables should be retained alongside the deliverables or referenced with their relative repository paths.

**Integrity check before submission:** The final `ablation_dev.csv` must report **221.59%** for `dev_B_valuation`, and `ablation_val.csv` must report **−4.73%** for `val_B_valuation`; earlier development CSVs with different values are not the frozen final model and must not be submitted as supporting results. The final Test `test_B_valuation` cumulative return is **165.19%**. Keep the recorded two-line Test log unchanged.

## Research report and Medium publication

- **Research report:** `Research_Report.pdf`
- **Medium article:** [Can Fundamental Valuation Improve AI Stock Selection?](https://medium.com/@xz3423/can-fundamental-valuation-improve-ai-stock-selection-c8ec2d9f2752)
- **Course assignment:** [STAT-GR5398 Assignment 2](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment2/README.md)
- **FinRobot framework:** [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot)

This project does not include the restricted course data bundle, network/API credentials, or a trained replacement Selection model. Results are educational research and are not investment recommendations.
