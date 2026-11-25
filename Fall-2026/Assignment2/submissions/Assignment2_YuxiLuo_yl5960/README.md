# Assignment 2 — Valuation module (FinRobot track)

| | |
|---|---|
| Member | Yuxi Luo (yl5960), working alone |
| Contribution | Module design and implementation, data and look-ahead audit, evaluation and ablation, robustness analysis, report and blog |
| AI assistance | Claude Code (Anthropic) assisted with writing and debugging the module code, writing the analysis scripts, drafting and editing the report text, and drafting this README. Section 8 of the report (Judgment) is my own writing. |
| Final configuration | commit `8288c743f89b7ab897686b02b059d9231fbb5581`: `valuation/student_valuation.py` with `FILTER_MIN_METHODS = 2`, `FILTER_PREMIUM = 0.0`, chosen on val and committed before the test run. `student_valuation.py` in this folder is that file, unchanged. |
| Test-period runs | **2**, both at that commit, both with `PYTHONHASHSEED=0` (`output/test_runs.jsonl`): `evaluate_valuation.py --period test`, then `run_portfolio.py --period test --ablation --timing full` |
| Report | `Assignment2_YuxiLuo_yl5960_report.pdf` |
| Medium blog | [Too Expensive to Own? Testing a Valuation Filter on a Growth-Stock Portfolio](https://medium.com/@yl5960/too-expensive-to-own-testing-a-valuation-filter-on-a-growth-stock-portfolio-3cb26a26def3) |

## Reproduce every dev/val number with one command

1. Put the course data bundle in `Fall-2026/Assignment2/source_code/data/` and install the course requirements
   (`source_code/requirements.txt`; the numbers here were produced with scikit-learn 1.9.0, xgboost 3.4.0,
   lightgbm 4.7.0, numpy 2.5.2, pandas 2.3.3, Python 3.13).
2. Run, from anywhere:

```bash
Fall-2026/Assignment2/submissions/Assignment2_YuxiLuo_yl5960/reproduce.sh
```

It copies `student_valuation.py` into `source_code/valuation/` (as the grader does), sets `PYTHONHASHSEED=0`, runs
the evaluation (this module and the course's trailing-P/E example) and the A/B/C/D ablation for dev and val, runs
`check_submission.py --valuation`, and compares every regenerated file with the copy in `output/` (MATCH / DIFF).
About 5 minutes. `reproduce.sh --all` also reruns every analysis in the report (about 30–40 minutes) into
`reproduced/analysis/` and compares it file by file with `analysis/`. Set `PYTHON=/path/to/python` to choose the
interpreter and `JOBS=n` for the number of parallel jobs.

**Neither mode runs the test period.** The test outputs in `output/` (`valuation_eval_test.csv`, `ablation_test.csv`,
`test_*/`, `test_runs.jsonl`) come from the two logged runs and are shipped as they are. In a clean clone
`check_submission.py` finds no `output/test_runs.jsonl` under `source_code/`; `check_submission.txt` here is the run
in the original working copy (0 FAIL, 1 WARN for the 2 test runs, 6 PASS).

### Individual commands

From `Fall-2026/Assignment2/source_code/`, with `export PYTHONHASHSEED=0` and `A=../submissions/Assignment2_YuxiLuo_yl5960/analysis`:

| command | report section |
|---|---|
| `python evaluate_valuation.py --period dev --example`, then rename `output/valuation_eval_dev.csv` to `valuation_eval_dev_example.csv`; same for `val` | §4.1 course example |
| `python evaluate_valuation.py --period dev` (and `val`) | §4.1, §5.1 |
| `python run_portfolio.py --period dev --ablation --timing full` (and `val`) | §4.2, §4.3 |
| `python check_submission.py --valuation` | `check_submission.txt` |
| `python $A/audit_fundamentals.py` | §2 |
| `python $A/tune_filter.py --period dev --tag m2_p0.0 --set FILTER_MIN_METHODS=2 --set FILTER_PREMIUM=0.0 --log $A/tuning_seeded.jsonl` (each grid point) | §1.4 |
| `python $A/run_sensitivity.py`, then `cd $A && python summarize_sensitivity.py` | §5.2 |
| `python $A/filter_study.py --period dev --variant committed` (8 rules × dev, val) | §4.4, §5.3 |
| `PYTHONHASHSEED=<s> python $A/seed_check.py --period dev` (s = 0–4; and val) | §5.4 |
| `python $A/spot_check_basis.py` | §2 |
| `python $A/lookahead_test.py`; `python $A/constant_binding.py` | §3.2, §3.4 |
| `python $A/dcf_diagnostic.py` | §5.5 |
| `python $A/v2_eval.py --period dev --variant v2c`; `python $A/filter_study.py --period dev --variant committed --version v2c` (v2_0, v2a, v2b, v2c × dev, val); `python $A/lookahead_test.py --version v2c` | §5.6 |

`analysis/report/build_report.py` rebuilds the report's tables and figures from these files (needs matplotlib and
Google Chrome, which are not course requirements); `analysis/plot_sensitivity.py` also needs matplotlib.

## What is in this folder

| path | contents |
|---|---|
| `student_valuation.py` | the module (commit `8288c74`) |
| `output/` | evaluation CSVs (module and course example), `ablation_{dev,val,test}.csv`, `test_runs.jsonl`, equity curves and trades of A/B/C/D for every period, and the two v2c portfolio runs used in §5.6 |
| `analysis/` | every analysis script and the result files the report reads: `sens/` (sensitivity), `study/` (rule ablation, swaps, spot check, look-ahead tests, seeds, DCF diagnostic, `v2/`), `tuning_log.jsonl`, `logs/` (including the two test-run logs) |
| `analysis/report/` | the report source and its build script |
| `reproduce.sh` | the one-command reproduction above |
| `check_submission.txt` | `check_submission.py --valuation` output |

## Notes

- The course backtester's results depend on Python's hash seed (`portfolio/backtester.py:148` iterates a set), so
  every number uses `PYTHONHASHSEED=0`; report §5.4 shows the effect.
- `shared/periods.py` `record_test_run` checks uncommitted student files with a pathspec that does not match
  `valuation/student_valuation.py`, so `student_files_uncommitted` is always false; report §6.
- `analysis/tuning_log.jsonl` is the filter-tuning log as it was when I chose the configuration; those runs did not
  fix the hash seed. `analysis/tuning_seeded.jsonl` reruns the same grid with `PYTHONHASHSEED=0` (and B under seeds
  1–4); the report uses it. Only two val rows differ, and the choice is the same (report §1.4).
- `analysis/valuation_v2.py` (report §5.6) is a post-test exploration on dev and val only. It is not the submission
  and was never run on test.
