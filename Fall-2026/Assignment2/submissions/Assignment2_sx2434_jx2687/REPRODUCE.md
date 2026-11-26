# Reproducing the valuation study

## Grader installation

The frozen configuration is `0c8eeec6683633120f2ed63a375463356c90ee80`. Copy the submission's `student_valuation.py` to the course's `source_code/valuation/student_valuation.py` and `student_data/history_monthly.parquet` to `source_code/student_data/history_monthly.parquet`. These are the two files needed at grading time. The self-contained module includes the team's EV/EBITDA implementation and uses the FinRobot library supplied with the course.

The formal test used two source files at the frozen research commit. To deliver one importable grader file, we placed the complete `historical_ev_ebitda.py` first, followed by the original `student_valuation.py`, replacing its module docstring with a comment and removing its import of the now-included helper. We compared the source against the freeze: all 11 function definitions and the model constants match. The SHA-256 values of the two original files are `41ab3276b9bb2cf65f7198b36b2ac68d69a8b3b65b0f571000103293e27a8f7e` (history) and `43e13b72c514a7afe86dbd6077e4671292fdb1226a83f052621b04397b16d0ec` (valuation). The table below identifies the files the grader receives.

The parquet is the only student-built data input. It contains 45,366 dated stock/month observations from June 2016 through September 2026. Each decision reads earlier month ends through the course DataView. To reconstruct the original input, the production build used `python build_history_inputs.py --through-month 2026-09` with the supplied builder in the course source directory. That command needs the full dated data bundle. The reproduction below uses only the development and validation dates.

| Submitted grader artifact | SHA-256 |
| --- | --- |
| `student_valuation.py` | `343a5cd28b30b8351d9fe159af8d4822faec243a5710dc8cd570dc46b95dda40` |
| `student_data/history_monthly.parquet` | `7e62ae671d2a4036bc0b393081c32a5b69c91e095f9523a628b341e14a23a08d` |
| `build_history_inputs.py` | `96cf126ce483036b089e603bcd7cba691092a10bb70f10c95bc40e26e509ac3c` |

## Running development and validation

Use the distributed Assignment 2 course source, the course data bundle and a Python 3.11+ environment. The team used Python 3.12.14, NumPy 2.5.5, pandas 3.1.1, PyArrow 26.0.1, SciPy 1.18.1, scikit-learn 1.9.1, Pydantic 2.14.3 and Matplotlib 3.11.2. The portfolio uses the course's published monthly selection scores. Install the course requirements plus SciPy and Matplotlib if using a new environment.

Run from the submitted UNI folder with the course data installed. In the course repository the source directory is two levels up. Set `A2_PYTHON` to the interpreter for the course environment if it differs from `python3`. Set `COURSE_SRC` to an absolute source directory when working from a standalone copy.

```sh
COURSE_SRC="${COURSE_SRC:-../../source_code}"
A2_PYTHON="${A2_PYTHON:-python3}"
WORK="$(mktemp -d)/bounded-replication"
"$A2_PYTHON" reproduction/prepare.py "$COURSE_SRC" "$WORK" && \
    "$A2_PYTHON" reproduction/run.py "$WORK" checker && \
    "$A2_PYTHON" reproduction/run.py "$WORK" formal && \
    "$A2_PYTHON" reproduction/run.py "$WORK" extension
```

`prepare.py` builds an independent scratch directory with market data, financial filings, membership histories, precomputed scores and the team's monthly multiples restricted to dates through December 29, 2023. It preserves the filing row IDs used to break ties and masks unneeded later training labels. All three stages write to that scratch directory. The runner fixes `PYTHONHASHSEED=0` to reproduce course order execution. The extension stage uses the formal-stage portfolios as cross-checks.

| Material | Command stage and regenerated files under the scratch directory |
| --- | --- |
| Course valuation checker | `checker`, which runs `check_submission.py --valuation`. The package includes the original preflight in `checker/preflight.txt` and a subsequent clean dev/val copy check in `checker/valuation.txt`. Their logs report the test count at the moment of each check. |
| Formal A–D portfolio results | `formal` runs `run_portfolio.py --period dev --ablation --timing full` and the analogous validation invocation. See `valuation_candidate/source_code/output/ablation_*.csv` and the per-arm equity and trade files there. |
| Standalone signals | `formal` runs `evaluate_valuation.py --period dev` for the team and appends `--example` for the course P/E comparator. It saves both evaluation files in `valuation_candidate/results/pretest/`. The course evaluator allows returns from development decisions to mature into validation. The matched analysis separately bounds outcomes to each split in `final_report_material/signal_comparison/matched_signals.csv`. |
| Within-rule tests, years and stock contributions | `formal` runs `validate_rules.py` for each split, `review_inputs.py`, and `attribute_valuation.py`. See `valuation_candidate/results/{dev,val}_rules.csv` and `results/pretest/{annual_returns,stock_attribution}.csv`. The attribution script consumes the official B column generated in the same run. |
| Development threshold curve and peak attribution | `extension` rebuilds weekly rankings and valuations, constructs weighted signals, enumerates the candidate boundaries and runs both NVDA universes. See `vote_bottleneck_review/threshold/dev_full_steps.csv`, `ex_nvda/dev_full_steps.csv` and `ex_nvda_attribution/dev_peak_triggers.csv` under the same directory. |
| Fixed gates, ANET, placebo and stock cases | `extension` writes `vote_bottleneck_review/gate_robustness/` (including `placebo/`), `vote_bottleneck_review/frozen_1p983/` (including `anet_execution/`), and `vote_bottleneck_review/nvda_contrast/`. The companion plots are written to `final_report_material/`. |

The original `output/evaluation/val_matured_signals.csv` predates the final integration of the EV/EBITDA implementation. The formal-stage runner rebuilds this archive using the exact historical valuation file at commit `bc98428` in an isolated Python process. The comparison uses the frozen module and checks its candidate membership against the earlier file. For validation signal metrics in the paper, use `output/evaluation/matched_signals.csv` and its explicit date conventions.

## Historical-multiple data audit

After preparing the bounded workspace above, run this check from the submission folder. It recreates the June 2021 Microsoft example, the August 2019 candidate pilot, all five CSVs and the 35-check log. The input hashes in its generated run summary describe the local bounded replica.

```sh
"$A2_PYTHON" analysis/history_audit/run_delivery.py --source-root "$WORK/valuation_candidate/source_code" --out "$WORK/history_audit"
```

The original observations and 35 reproduced checks are in [output/formal/history_audit](output/formal/history_audit/).

## Formal test archive

The single official invocation used `PYTHONHASHSEED=0 python -u run_portfolio.py --period test --ablation --timing full` on the frozen configuration listed above. The submission contains its original `output/ablation_test.csv` and one-entry `output/test_runs.jsonl`. The reproduction runner reads that ledger to check the count. Its executable stages accept only the development/validation workspace. The 1.983 extension and the alternative cash-allocation rules have development or validation results only.

## Selection-model retraining bonus

The [bonus reproduction guide](bonus/README.md) documents Jiao Xu’s same-environment V1 and V2 comparison and the four selection-only runs. Its scores and configuration changes apply to this bonus.
