# Assignment 2 · Point-in-time valuation of a growth portfolio

**Zeyu Gu (zg2556), solo FinRobot track**  
**Published Medium article:** [When a valuation filter tells a growth portfolio to sell](https://medium.com/@zeyu.gu0302/when-a-valuation-filter-tells-a-growth-portfolio-to-sell-b30a4a16ac64)  
**Frozen configuration** `5af60c20f1e93da8f6350aec9001ebfbafea57cf`  
**Official course base** `43eeda7b011270814fa28cc38e9081e36128cf8a`

The submission evaluates three ways to value the stocks that the course selector ranks most highly. The selected rule requires at least two usable methods, a largest-to-smallest ratio at most 2.5 and a fair value more than 15% below the decision close before excluding a name. Review [the research report](Research_Report.pdf) for its rationale, coverage, annual results, comparative rank IC and uncertainty.

## Results available for review

| Period | Selection / filtered Sharpe | Selection / filtered drawdown | Cumulative return change | Annual turnover change |
|---|---:|---:|---:|---:|
| Development, 2017–21 | 0.980 / 0.928 | −32.99% / −33.93% | −24.02 pp | +2.89× |
| Validation, 2022–23 | 0.028 / 0.063 | −42.52% / −38.99% | +2.42 pp | +1.34× |
| Holdout, Jan 2024–Oct 9, 2026 | 1.301 / 1.434 | −30.53% / −28.54% | +22.78 pp | +1.88× |

In the formal FinRobot experiment timing is full exposure. The A/C and B/D daily NAV series agree exactly within each period. Annual turnover is gross traded notional divided by mean equity and annualized. All three periods include 6 bp per traded dollar. The official checker reports 10 PASS, 0 WARN and 0 FAIL. A fresh official source checkout reproduced all 53 archived development/validation output files byte for byte. [Independent reproduction hashes](reproduction_verification.json) [Development table](output/ablation_dev.csv) · [Validation table](output/ablation_val.csv) · [Holdout table](output/ablation_test.csv) · [Checker output](check_submission.txt)

**Release status.** The ledger contains one run of the January 2, 2024 to October 9, 2026 holdout. The configuration was committed locally before the run. Zeyu Gu subsequently disclosed the hash in the public course repository and asked whether the late disclosure is acceptable. A TA response is pending. [Course issue #21](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/issues/21) · [Test ledger](output/test_runs.jsonl) · [Article source](Medium_draft.md)

**Contributions.** Zeyu Gu is the sole team member and supplied the investment judgment. AI assistance supported implementation, experiment execution, checking and preparation of technical material. The personal view was supplied in Chinese and translated for the report. [Original judgment](personal_original_zh.txt) · [English translation](MY_JUDGMENT.txt)

## Reproduce development and validation

Use a clean course checkout at the official base above and place this submission folder under `Fall-2026/Assignment2/submissions/`. Download the course data bundle linked in the [official assignment](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/43eeda7b011270814fa28cc38e9081e36128cf8a/Fall-2026/Assignment2/README.md) and unpack it as `Fall-2026/Assignment2/source_code/data/`. The bundle ZIP SHA-256 is `e76edf7eca96abd374b330f45ff144786a3de88d279864749446bd1aaf4bdedc`. The supplied score Parquet SHA-256 is `f32b737d5a6bc4178fd272d99e4ee291f0e2f671ca3eb4e2e6572d908d8edc32`.

The tested interpreter is Python 3.13.2. In the submission directory, install `python -m pip install -r requirements-lock.txt`. With the data and that environment in place, the following command reproduces the development and validation valuation study, the official checker, both bounded course evaluator runs and the two January–June 2018 selection comparisons. It trains the original selector once and shares that freshly generated control across the two comparisons. Use a new destination on repeat runs.

```sh
python reproduce.py --out reproduced --with-bonus
```

On macOS, the boosting wheels used here require a discoverable `libomp.dylib`. The actual run used the library supplied by the installed scikit-learn wheel. If needed, set `DYLD_LIBRARY_PATH` to the environment’s `site-packages/sklearn/.dylibs` before launching the command. Omit `--with-bonus` to reproduce only the graded core. The wrapper copies the course source into a temporary directory, installs the submitted module there, links the course data read-only by convention, fixes `PYTHONHASHSEED=0` and writes results beneath the requested destination. No model training is needed for the main study.

The seven-year selection bonus on page 15 has its own entry point. From this submission directory, run the following with the same pinned Python environment and course data. Both arms use the original three-reporting-date calibration window. The command reproduces the comparison for January 2017 through December 2023.

```sh
python bonus/path_efficiency/long_run/run.py --source ../../source_code --out ./replayed_long_run --workers 4
```

For a quicker check, add `--scores-from bonus/path_efficiency/long_run/scores` and choose a fresh output directory. This verifies both submitted score files before replaying the portfolios and comparing the results with the archived tables.

## Evidence and evaluation conventions

- `student_valuation.py` is the single file for the grader to copy into `source_code/valuation/`. All model decisions read the supplied `DataView`. The unchanged course selector and full timing baseline complete the portfolio.
- `output/valuation_{dev,val,test}_observations.csv` includes every monthly candidate, available component estimates and exact 63/126/252-session endpoints within its own split. Missing and unmatured outcomes remain empty. The `summary`, `yearly`, `ic_monthly`, `inputs` and `audit` files explain the denominators and identify the matching rows.
- `official_reference.py` also runs the original course `evaluate_valuation.py` twice per development or validation split in an isolated copy whose price history ends on that split’s final session. `output/valuation_eval_*` and `output/valuation_example_*` preserve the official student and example CSVs and printed output separately. The course evaluator accepts the last available price in a horizon window. The research report uses exact endpoints and counts missing and right-censored outcomes separately.
- `output/weekly_marks_*` and `output/sensitivity_*` compare the three predeclared thresholds. The frozen variant’s live portfolio must reproduce the replay metrics. Each group directory contains its equity, trades and `stats.json`. [Selection protocol](EXPERIMENT.md)
- Page 12 compares two internal validation windows in the selector. Page 14 holds that window at three reporting dates and adds four-quarter price-path efficiency as the only new feature. Both studies use six monthly refits in January–June 2018. The feature arm returned 14.71% against 10.93% for its control, with higher turnover. The model patch, scores, trades and comparisons are included. [Validation-window experiment](bonus/DESIGN.md) · [Price-feature experiment](bonus/path_efficiency/README.md)
- Page 15 follows the same feature through 84 monthly refits per arm. The comparison uses the original model refit in the same environment. The feature raised development CAGR from 24.89% to 27.99%. In the previously examined validation period, cumulative net return went from −5.11% to −3.25%, maximum drawdown depth from 41.38% to 40.76%, and annual two-way turnover from 9.60× to 9.26×. The feature won 13 of 24 months and lagged in 2023. The formal valuation study uses the supplied course scores throughout. [Long-run method and reproduction](bonus/path_efficiency/long_run/README.md) · [Paired period results](bonus/path_efficiency/long_run/results/period_metrics.csv)
- `template_dev.log` records the initial four equal cells before adding valuation. `build_report.py` generates the PDF from the saved data, original judgments and frozen commit. For the PDF, install `python -m pip install reportlab==5.1.0` alongside the locked dependencies, then run `python build_report.py --out output --bonus bonus/results`. See `--help` to choose other locations.

## Supplemental news timing experiments

Page 16 pairs a neutral news input with a locally fine-tuned GPT-2 small answer head on dated Federal Reserve headlines. Page 17 checks the coverage of scheduled policy statements and compares two weekly averaging rules on development dates. Page 18 compares the same 38 eligible scheduled meetings using titles, full statements and neutral input. Full text improved development stock return by 0.537 percentage points versus titles. In validation, return was 0.010 points lower and maximum drawdown 0.005 points deeper. All FinRobot holdout results belong to the original valuation comparison.

Page 19 pairs each statement with its preceding decision and scores the direction of the target-rate change and the change in forward guidance. Relative to the original full-text model, the rule lost 0.466 percentage points of stock return and deepened maximum drawdown by 0.284 points in validation. Its ten hawkish and six neutral signals provide limited evidence about easing cycles.

Page 20 applies a single development-calibrated scale of 0.3388 to each target change from neutral while keeping the policy signal’s direction. Scaling lifted validation return from 3.298% to 3.675% and narrowed maximum drawdown from 27.981% to 27.712%. The full-text model returned 3.765%, with a 27.697% drawdown. Of nine actionable cuts, four preceded a decline in the neutral stock basket. The validation stock results include trading costs.

Page 21 uses the full-text model’s own target magnitude at each meeting and replaces its direction with the policy signal. Relative to full text, validation return rose by 0.02032 percentage points and maximum drawdown deepened by 0.00096 points. Comparing portfolios that both skip neutral calls isolates the six actual sign reversals, which raised return by 0.01587 points.

The [supplemental timing archive](timing_extensions.zip) contains the news adapter, scored headlines, official statements, paired policy scores, corpus audit, daily sentiment series and portfolio evidence. It also contains the fixed-rule timing baseline and its valuation comparison. The PDF generator reads the saved tables directly from this archive. To replay the news studies in a **fresh course checkout**, run the following from its `Fall-2026/Assignment2` directory with the course data and locked packages installed.

```sh
python -m zipfile -e submissions/Assignment2_ZeyuGu_zg2556/timing_extensions.zip .
PYTHONHASHSEED=0 python extensions/fingpt/evaluate.py --source source_code --results extensions/fingpt/results
python extensions/fingpt/corpus_diagnostic.py
PYTHONHASHSEED=0 python extensions/fingpt/release_day_pilot/run.py
python extensions/fingpt/statement_comparison/fetch_statements.py
PYTHONHASHSEED=0 python extensions/fingpt/statement_comparison/backtest.py
python extensions/fingpt/stance_comparison/build_pairs.py
python extensions/fingpt/stance_comparison/score_policy.py
PYTHONHASHSEED=0 python extensions/fingpt/stance_comparison/backtest_stance.py
python extensions/fingpt/stance_comparison/validation_detail.py
python extensions/fingpt/stance_amplitude/calibrate.py
PYTHONHASHSEED=0 python extensions/fingpt/stance_amplitude/backtest.py
python extensions/fingpt/stance_amplitude/direction.py
PYTHONHASHSEED=0 python extensions/fingpt/direction_swap/backtest.py
python extensions/fingpt/direction_swap/audit_events.py
```

The extracted FinRL and FinGPT READMEs give the commands for replaying the fixed-rule baselines, verifying the weekly signal diagnostics and repeating the language-model fine-tune. A replay from the packaged news series reproduced the stock and SPY metrics and daily NAVs exactly. The same stock fills appeared on the same dates, with different row ordering within some dates.

## Holdout run record

The 15% threshold, module and evaluators were frozen locally before the test. The course design brief calls for submission of the commit hash before opening the holdout. The hash first appeared in the course issue after run 1, and the threshold stayed fixed. The command below records the invocation used for the archived results. Its `--registered-commit` option verifies the local Git identity and file contents. [Dated disclosure in the course repository](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/issues/21)

```sh
# Historical invocation from Fall-2026/Assignment2/source_code
cp ../submissions/Assignment2_ZeyuGu_zg2556/student_valuation.py valuation/student_valuation.py
PYTHONHASHSEED=0 python ../submissions/Assignment2_ZeyuGu_zg2556/research.py holdout test --registered-commit 5af60c20f1e93da8f6350aec9001ebfbafea57cf
```

The single run evaluated the student and trailing-P/E reference on split-local exact labels, then executed the four portfolio cells. Use the [run ledger](output/test_runs.jsonl), [portfolio table](output/ablation_test.csv), [valuation statistics](output/valuation_test_summary.csv) and each portfolio’s daily NAV and fills to inspect the outcome. The independent reproduction manifest records the development/validation archive at the time of the earlier check, including the then-empty ledger. The final archive preserves the actual single test entry.
