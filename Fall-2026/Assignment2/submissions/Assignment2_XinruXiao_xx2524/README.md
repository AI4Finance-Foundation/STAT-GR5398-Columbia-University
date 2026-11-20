# Assignment 2 - Valuation and selection retraining

Xinru Xiao, UNI xx2524. Individual submission. I conducted the valuation study and interpreted the results. I used AI assistance for coding, documentation, and editing the wording of my investment judgment. The study uses the valuation track with full timing exposure, long-only holdings and no leverage.

Course version: f5df2feab002b4d463fdd3bf30c34f6d91c0e49a.
Original frozen study: e749cf8304b561fe4eb01d47b917206f87c9c0ee.
Expanded frozen sources: 1ba67615924078d9680702870352496583672341.
These are local commits, not a claim that code was pushed to GitHub. Checksum manifests supplement the course logger's incomplete dirty-file check.
The expanded freeze covers valuation logic, selectors and configurations. Later staging, equivalent runtime optimizations, analyses and report formatting are separately documented and hashed; they do not represent a new financial configuration chosen on the reported test results.

## Primary submission
Copy student_valuation.py to valuation/student_valuation.py in a clean checkout of the fixed course source_code. Install requirements_reproduction.txt, download the official data bundle, and place it at source_code/data. The module needs no extra student_data, API key or network access. Do not commit the data bundle. Timing remains full. The grader should use its original course scores; bonus scores never replace them.

All weekly growth_tech candidates are checked, subject to the course universe and score coverage. At least two methods must be valid; missing or disputed estimates retain a candidate. Monthly official top20 diagnostics are retained alongside all_growth diagnostics. Methods: historical-input DCF, own-history EV/EBITDA and same-sector trailing P/E. Constants and limitations are in the report.

## Reproduce the main results
Run from source_code (create output first). Each command touching test creates an additional log entry; do not run repeatedly to choose a result. The original test had already been seen before the expanded study. All new test/forward runs are disclosed retrospective extensions, not pristine holdouts.

```sh
mkdir -p output
python evaluate_valuation.py --period dev --example
cp output/valuation_eval_dev.csv output/valuation_example_dev.csv
python evaluate_valuation.py --period dev
python run_portfolio.py --period dev --ablation --timing full
python evaluate_valuation.py --period val
python run_portfolio.py --period val --ablation --timing full
python evaluate_valuation.py --period test
python run_portfolio.py --period test --ablation --timing full
python evaluate_valuation.py --period forward
python run_portfolio.py --period forward --ablation --timing full
python check_submission.py --valuation
```

Those standard commands reproduce the portfolio and official top20 signal calculations. To reproduce the exact saved expanded directory names, all_growth diagnostics and input audit capture, use reproduction/outputs/retraining/run_evaluation.py, as shown below. It wraps the unchanged official evaluator and backtester. A per-period memo only reuses identical filter inputs between B and D; it does not change decisions. Do not run both sets unless deliberately reporting the additional test invocations.

## Exact expanded and bonus reproduction
The reproduction folder uses the same relative layout as the experiment. Install the pinned dependencies in your own Python environment. On macOS XGBoost/LightGBM also require OpenMP; the original run used conda-forge llvm-openmp 15.0.5 extracted into reproduction/work/openmp. Other platforms can use their supported OpenMP runtime. Source metadata and hashes are preserved. The dependencies of the original FinRobot environment were not modified during this study.

From reproduction/, place or link the official data bundle at work/course/Fall-2026/Assignment2/source_code/data and link that data directory into work/retraining/baseline/data, work/retraining/improved/data and work/retraining/evaluation/course/data. These links are deliberately omitted from the submission. Use absolute symlink targets on macOS. Then run:

For the original Apple Silicon runtime only, the exact OpenMP archive was:

```sh
mkdir -p work/openmp
curl --fail -L https://conda.anaconda.org/conda-forge/osx-arm64/llvm-openmp-15.0.5-h7cfbb63_0.tar.bz2 -o work/llvm-openmp.tar.bz2
tar -xjf work/llvm-openmp.tar.bz2 -C work/openmp
export DYLD_LIBRARY_PATH="$PWD/work/openmp/lib"
```

Archive SHA256: df8d545338b3a895d5d010aa1d672d484006b4f3f406ff5146b18f51f3acd0ea, matching the release metadata in openmp_runtime_provenance.json. Do not install the arm64 binary on Intel/Linux. Continue with:

```sh
python outputs/retraining/run_evaluation.py --source work/retraining/evaluation/course --scores work/course/Fall-2026/Assignment2/source_code/data/selection/monthly_scores.parquet --variant course
python outputs/retraining/run_training.py --task-root "$PWD" --python "$(command -v python)" --workers 6
python outputs/retraining/finish_retrained_evaluation.py --task-root "$PWD" --python "$(command -v python)"
python outputs/retraining/analyze_expanded.py --task-root "$PWD"
python outputs/retraining/analyze_scores.py --task-root "$PWD"
python outputs/retraining/analyze_years.py --task-root "$PWD"
python outputs/retraining/analyze_filter_robustness.py --task-root "$PWD"
python outputs/retraining/analyze_holdings.py --task-root "$PWD"
python outputs/retraining/analyze_benchmarks.py --task-root "$PWD"
python outputs/retraining/analyze_price_quality.py --task-root "$PWD"
python outputs/retraining/write_assumption_inventory.py
python outputs/retraining/check_training_preprocessing.py
```

The last preprocessing check is run from reproduction/. The provided checkpoints are merged scores and metadata; monthly parts are omitted to avoid duplicate tables. The training script resumes when monthly parts exist locally, otherwise retrains all 119 months. If changing any source, config, dependency or input data, use a fresh parts directory rather than mixing old checkpoints. It never overwrites course scores. Independent official commands for comparison are `python selection/build_scores.py --config selection/configs/growth.json --out output/retrained_baseline/scores.parquet --workers 4`, and, in the improved source tree, the same command with `selection/configs/improved.json` and a distinct --out. These official commands also fit the zero-weight unified model; the bounded driver skips only that unused fit. The 2019-06 equality check is included. Seeds, six base learners, four buckets, stacking and parameters are preserved.

The additional `bonus/feature_ablation_v2` experiment starts from the preprocessing-improved selector and removes `pe`, `ps`, `pb`, `peg`, and `ev_multiple`. Its configuration, complete selector, patch, frozen-before-test record, merged 119-month scores, logs, raw portfolio outputs and integrity report are included. The version was admitted to one test run only after dev/val results were frozen. It remains a bonus appendix and does not replace the course scores or required valuation module.

The subsequent `bonus/rank_target_v3` experiment changes only the matured-return target to a within-date, within-bucket percentile rank. Its test-admission criteria were written before training. It completed all 119 months and improved validation, but failed the preregistered development-return floor. Therefore it has dev/val evidence only, zero new test invocations, and v2 remains the selected bonus candidate.

## Evidence map
- output: required course-score A/B/C/D tables, `ablation_*.csv`, `test_runs.jsonl`, and top20/all_growth valuation diagnostics for dev, val, test, and forward.
- analysis: same-environment retraining comparisons and the required portfolio, coverage, robustness, and yearly summary tables.
- bonus: the first two 119-month score files, metadata, selector patch and preprocessing tests; `bonus/feature_ablation_v2` contains the feature-ablation experiment and `bonus/rank_target_v3` contains the preregistered rank-target experiment rejected before test.
- reproduction: source copies, configs, drivers, training logs (including initial failures), hashes and protocol.
- test_run_ledger.jsonl: complete source-preserving ledger across the original, expanded and feature-ablation experiments. The count in test_run_ledger_summary.json is authoritative; the individual course check sees only its local three invocations.

All numerical summary tables are computed from raw output files. Course Sharpe uses rf=0; turnover is two-sided. Forward data end on 2026-11-06; its short-period annualization is not a forecast. No mature 63/126/252-day forward labels exist. Data revisions, adjusted-price/per-share consistency, stale DISCK holdings and terminal reinvestment reset remain limitations. Preprocessing improvements do not prove a leak-free upstream database or time-series-safe stacking CV.

The final research report is research_report.pdf. The commands above reproduce the financial calculations. The earlier report-generation script documents how the numerical exhibits were assembled.

## Published Medium article

https://medium.com/@xx2524/does-valuation-improve-a-growth-stock-portfolio-b41963156218

The published article reports the required course-score experiment separately from the optional retraining work and preserves the same conclusions as this report.
