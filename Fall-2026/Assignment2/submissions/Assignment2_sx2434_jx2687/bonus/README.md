# Selection retraining bonus

Jiao Xu compared the course selector's three-date internal validation window (V1) with a four-date window (V2). Each month the selector uses this window to assess and combine its regressors. The two arms shared the training environment, input extracts, features, portfolio rules and costs. Each produced 84 monthly scores from January 2017 through December 2023. The saved cutoff for the last month was November 30, 2023. All four portfolios used growth-tech selection alone.

The report uses V1 and V2 as version numbers. In the archived results and reproduction commands, V1 (three dates) is labeled `Q3` or `base_q3`, and V2 (four dates) is labeled `Q4` or `variant_q4`.

## Paired results

| Period and statistic | V1 (3 dates) | V2 (4 dates) | V2 minus V1 |
| --- | ---: | ---: | ---: |
| Development CAGR | 22.67% | 20.97% | −1.70 pp |
| Development Sharpe | 0.863 | 0.819 | −0.044 |
| Development maximum drawdown | −34.56% | −37.00% | −2.44 pp |
| Development annual turnover | 10.05× | 10.24× | +0.19× |
| Validation CAGR | −2.30% | −1.19% | +1.11 pp |
| Validation Sharpe | 0.056 | 0.097 | +0.041 |
| Validation maximum drawdown | −40.09% | −40.27% | −0.18 pp |
| Validation annual turnover | 9.85× | 9.76× | −0.09× |

Development covers 2017–2021 and validation covers 2022–2023. Differences use the displayed precision of each course statistics file. Turnover is annual traded dollars divided by mean portfolio value. The preselected outcome was the difference in validation CAGR. The extra internal date helped in validation and hurt in development. In 2023 V1 gained 51.07% and V2 gained 55.59%, measured from the last close of 2022. Whole-share execution and 6 bp transaction costs apply to both arms.

The two configurations scored the same growth-tech candidates every month. The raw growth-tech top tens shared 9.31 names on average and were identical in 37 of 84 months; the minimum overlap was seven. These monthly rankings precede the course's weekly incumbent and execution rules. Three months of real-assets coverage account for the 230-row difference in the files for the entire universe. [Paired statistics](results/comparison.csv) · [Monthly scores comparison](results/monthly_score_check.csv) · [Other-bucket coverage](verification/non_growth_row_differences.csv)

## Files and checks

- [Frozen configurations](configs/) and [both full score files](scores/) document the change to `model.val_quarters`. Each JSON sidecar beside a Parquet records the input hashes. The two runs used identical price and fundamentals inputs. Rewritten Parquet and SQLite files can differ in byte representation across software versions. [Historical input audit](verification/input_audit.json) · [Reference comparison](verification/reference_input_comparison.csv)
- [Four sets of daily NAVs, trades and statistics](results/) support the portfolio comparison. An independent replay of the delivered scores with the course backtester on an extract ending December 29, 2023 reproduced all four statistics files and daily NAV series. Trade identities matched, with prices and notionals within $0.00000001.
- [Python dependency pins](reproduction/training-requirements.txt) describe the training environment: Python 3.13.12 on macOS arm64. Both runs used the course fallback for missing optional LightGBM. Selection-only scores in this experiment come from the same-environment V1 and V2 fits.
- [Shared selector patch](reproduction/selector.diff) skips the unified fit when every bucket's blend coefficient is zero. Both arms first generated monthly scores with the original course selector, V1 through June 2021 and V2 through July 2021, then resumed with this patch. The optimization changes available unified-model diagnostics. For both V1 and V2, checks in January and February 2017 and May 2021 found matching score keys and growth-tech top-ten order, and maximum absolute numeric differences below 1e−8. A separate January–February V1 repeat checked run-to-run variation. [Original cache months and code hashes](verification/optimization_manifest.json) · [Optimization comparisons](verification/optimization_equivalence.json) · [V1 repeat results](verification/pilot_check.log)

Run the following from the bonus directory in an environment with pandas, NumPy and PyArrow. It checks the packaged files, recalculates portfolio statistics, 2023 returns and monthly top-ten overlaps, and compares the saved repeat and optimization probes with the full score panels.

```sh
shasum -a 256 -c verification/package.sha256
python3 verify.py
```

## Replaying the supplied scores

From this directory, set `COURSE_SRC` to the course's original Assignment 2 `source_code` directory. Use the Python environment above for numerical reproduction. The preparation script creates a fresh workspace with the original course code and data bounded to the development and validation periods. All following commands operate in that workspace.

```sh
BONUS="$(pwd -P)"
WORK="$(mktemp -d)/bonus_work"
python3 "$BONUS/reproduction/prepare_bonus.py" "$COURSE_SRC" "$WORK"
cd "$WORK"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
shasum -a 256 -c "$BONUS/verification/frozen_configs.sha256"
mkdir -p output/bonus
cp "$BONUS"/scores/* output/bonus/
cp "$BONUS"/verification/repeat_q3.* output/bonus/
for period in dev val; do
    for arm in base_q3 variant_q4; do
        python3 run_portfolio.py --period "$period" --scores "output/bonus/$arm.parquet"
    done
done
python3 "$BONUS/reproduction/collect_results.py" --pilot
python3 "$BONUS/reproduction/collect_results.py"
```

## Retraining the scores from scratch

Start a separate fresh workspace using the same preparation and environment settings. Execute the following there. Each run has its own output and matching monthly cache. The split in June/July 2021 reproduces where the original selector stopped and the shared optimization resumed. The checkout's original selector hash is recorded in [the optimization manifest](verification/optimization_manifest.json).

```sh
mkdir -p output/bonus
python3 selection/build_scores.py --config selection/configs/growth.json --start 2017-01 --end 2017-02 --workers 2 --out output/bonus/base_q3.parquet
python3 selection/build_scores.py --config selection/configs/growth_q4.json --start 2017-01 --end 2017-02 --workers 2 --out output/bonus/variant_q4.parquet
python3 selection/build_scores.py --config selection/configs/growth.json --start 2017-01 --end 2017-02 --workers 2 --out output/bonus/repeat_q3.parquet
python3 "$BONUS/reproduction/collect_results.py" --pilot
python3 selection/build_scores.py --config selection/configs/growth.json --start 2017-01 --end 2021-06 --workers 2 --out output/bonus/base_q3.parquet
python3 selection/build_scores.py --config selection/configs/growth_q4.json --start 2017-01 --end 2021-07 --workers 2 --out output/bonus/variant_q4.parquet
patch -p1 < "$BONUS/reproduction/selector.diff"
python3 selection/build_scores.py --config selection/configs/growth.json --start 2017-01 --end 2023-12 --workers 2 --out output/bonus/base_q3.parquet
python3 selection/build_scores.py --config selection/configs/growth_q4.json --start 2017-01 --end 2023-12 --workers 2 --out output/bonus/variant_q4.parquet
```

With these regenerated scores, run the four `run_portfolio.py` commands in the loop above and then `collect_results.py`. The formal valuation study retains its original course-supplied scores and its one archived test run. The bonus covers development and validation.
