# Assignment 2 · Valuation group

## Members

| Name | UNI | Contribution |
| --- | --- | --- |
| Jiayi (Alisa) He | jh5111 | DCF (`dcf_value`), combining the methods in `value()`, `filter_candidates()`, A/B/C/D ablation runs, robustness experiments, merging the code |
| Xinyu Zhang | xz3490 | Historical EV/EBITDA (`ev_ebitda_value`, `historical_ev_estimates`), peer P/E (`peer_pe_value`, `peer_pe_estimates`) |

## Contents

| Path | What it is |
| --- | --- |
| `student_valuation.py` | The module file |
| `experiments/` | Scripts behind the robustness and calibration numbers in the report |
| `check_submission.txt` | `check_submission.py --valuation` output |
| `output/` | Evaluation CSVs, `ablation_*.csv`, `experiments/` tables |
| `output/experiments/earlier_code/` | Results from earlier code versions (before the 11/23 method change, the 11/24 data fixes and the 11/25 review fixes), cited in report section 4.2; `pre_review_1125/` holds the outputs just before the review fixes |
| `report.pdf` | Research report |

## Final configuration

Git commit: `32a02b5a004dfc50a968538c0a28b426cfa48337`

Medium blog: TODO link

## Reproduce

Copy `student_valuation.py` into `source_code/valuation/` and `experiments/` into `source_code/`, set up
the environment and data bundle as in the course README (section 2), then run from `source_code/`.
The course backtester places orders in Python set order, so every command is run with
`PYTHONHASHSEED=0`; without it the val numbers move by a few tenths of a percent.

```bash
export PYTHONHASHSEED=0
```

Checks:

```bash
python check_submission.py --valuation
```

Valuation evaluation (rank IC, range coverage, coverage):

```bash
python evaluate_valuation.py --period dev
python evaluate_valuation.py --period val
python evaluate_valuation.py --period test
python experiments/eval_example.py --period dev
python experiments/eval_example.py --period val
python experiments/eval_example.py --period test
python experiments/eval_method.py dcf --period dev
python experiments/eval_method.py ev --period dev
python experiments/eval_method.py peer_pe --period dev
```

Range calibration and filter rules (dev to tune, val to choose once):

```bash
python experiments/calibrate_range.py
python experiments/filter_rules.py --period dev
python experiments/filter_rules.py --period val
```

Ablation (A/B/C/D, timing held at `full`):

```bash
python run_portfolio.py --period dev --ablation --timing full
python run_portfolio.py --period val --ablation --timing full
python run_portfolio.py --period test --ablation --timing full
```

Robustness:

```bash
python experiments/robustness_dev.py
python experiments/robustness_dev.py --period val
python experiments/yearly_ab.py
python experiments/baseline_filter.py --period dev
python experiments/baseline_filter.py --period val
python experiments/variants.py --period dev --eval
python experiments/variants.py --period val --only current F3
python experiments/rate_lag_check.py --period dev
python experiments/rate_lag_check.py --period val
python experiments/cost_sensitivity.py
python experiments/bootstrap_dsr.py
python experiments/diagnostics_dev.py placebo
python experiments/diagnostics_dev.py controls
python experiments/diagnostics_dev.py methods
python experiments/diagnostics_dev.py constants
python experiments/diagnostics_dev.py flip
python experiments/diagnostics_dev.py flipcheck
python experiments/val_choice.py
python experiments/engine_ablation.py --period dev
python experiments/engine_ablation.py --period val
python experiments/report_tables.py dev val test
```

`engine_ablation.py` is a supplementary A/B/C/D table with the course's engine timing (the required table uses `--timing full`). `report_tables.py` writes the SPY/QQQ benchmark rows and the rank IC with Newey-West t-statistics, both as the course evaluator computes it and with labels that end after the period dropped.

## Bonus: retrained selection model

`experiments/configs/growth_blend.json` is `selection/configs/growth.json` with one change: `bucket_alpha.growth_tech` 0.0 -> 0.5 (growth_tech ranks blended 50/50 with the all-stock model). No code changes. Both retrains ran in the same environment (scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, numpy 2.5.5, pandas 3.1.1). Results: `output/experiments/bonus_selection.csv`.

```bash
python selection/build_scores.py --config selection/configs/growth.json --out output/retrained/unchanged/scores.parquet --workers 4
python selection/build_scores.py --config experiments/configs/growth_blend.json --out output/retrained/blend/scores.parquet --workers 4
cp output/retrained/unchanged/scores.parquet output/retrained/scores_unchanged.parquet
cp output/retrained/blend/scores.parquet output/retrained/scores_blend.parquet
python run_portfolio.py --period dev --scores output/retrained/scores_unchanged.parquet
python run_portfolio.py --period dev --scores output/retrained/scores_unchanged.parquet --valuation
python run_portfolio.py --period dev --scores output/retrained/scores_blend.parquet
python run_portfolio.py --period dev --scores output/retrained/scores_blend.parquet --valuation
```

The same four backtests with `--period val`. The copies only give the two score files different names, so their output folders do not overwrite each other.

