# From Machine Learning Stock Selection to Portfolio Construction

**GR5398 — FinRL-Trading Assignment 1 | October 2026**  
**Author:** Kevin Ma (UNI: km4189)

## Overview

This project evaluates how stock-selection predictions from the course-provided FinRL-Trading pipeline translate into portfolio performance. The upstream pipeline uses Random Forest, XGBoost, and LightGBM across 11 GICS sectors. The independent research extension compares nine portfolio construction strategies, transaction costs, rebalancing schedules, historical membership, signal timing, sector attribution, and robustness.

**Research report:** [`5398_Assignment_1.pdf`](5398_Assignment_1.pdf)  
**Research notebook:** [`Assignment1_Portfolio_Research.ipynb`](Assignment1_Portfolio_Research.ipynb)  
**Medium article:** [Do Better Stock Predictions Lead to Better Portfolios? Lessons from Nine Machine Learning Strategies](https://medium.com/@km4189/do-better-stock-predictions-lead-to-better-portfolios-c2f6c363e8d4)

## Main findings

| Experiment | Result |
| --- | --- |
| Nine-strategy comparison (Sep 2020–Sep 2025) | Monthly equal weight NAV **2.1718** vs. SPY **2.0347**; 8 of 9 strategies exceeded SPY in terminal NAV. |
| Highest terminal NAV among nine strategies | ML Top-50% filter **2.1774**. |
| Rebalancing robustness | Quarterly equal weight NAV **2.1713**, with **27.8% less** cumulative turnover than monthly equal weight. |
| Full window (May 2020–Sep 2025) | Exploratory SPY fallback + ML equal weight NAV **2.7003**, vs. SPY **2.5482**. |
| Historical membership audit | **19,007** of **21,299** records retained; **2,292** removed; **0** membership violations among retained records. |
| Risk adjustment | Common-window annualized alpha approximately **0.34%**, not statistically significant (**p ≈ 0.90**). |

The benchmark-fallback rule was developed after inspecting the cash-initialized backtest, so its improvement is **exploratory and in-sample**, not independently validated out-of-sample performance. Historical price coverage is incomplete, and upstream ML information-timing flags remain unresolved; this is **not** a fully certified look-ahead-free trading study. See the report for details.

## Submission contents

```text
Assignment1_KevinMa_km4189/
├── README.md
├── requirements.txt
├── prepare_data.py
├── ml_model.py
├── fundamental_run_model.py
├── Assignment1_Portfolio_Research.ipynb
├── 5398_Assignment_1.pdf
├── figures/                  # Five final report figures
└── results/                  # Selected CSV outputs from the notebook
```

The notebook preserves the **course-provided workflow above its original separator** and adds independent research below it. Cell output and intermediate results from the completed research run have been retained for inspection.

### Figures

| Report figure | File |
| --- | --- |
| 1 — ML diagnostics | `figures/report_fig1_ml_diagnostics.pdf` |
| 2 — Nine strategy terminal NAVs | `figures/report_fig2_strategy_comparison.pdf` |
| 3 — Full-window NAV and relative wealth | `figures/spy_fallback_full_window.png` |
| 4 — Rebalancing and subperiod robustness | `figures/report_fig4_robustness.pdf` |
| 5 — Sector exposure and attribution | `figures/report_fig5_sector_attribution.pdf` |

### Key result files

The selected outputs in `results/` include:

- `final_strategy_leaderboard.csv` and `final_subperiod_comparison.csv`
- `spy_fallback_full_window_nav.csv` and `spy_fallback_full_window_metrics.csv`
- `pit_membership_reproduction_audit.csv` and `original_model_timing_audit.csv`
- `model_win_summary.csv` and `rf_sector45_feature_importance.csv`
- `sector_attribution_evaluation_2024_2025.csv`
- `ml_universe_coverage_audit.csv`, `market_adjusted_alpha_metrics.csv`, and `ml_prediction_rank_ic.csv`

The optional intermediate `results/stock_selected.csv` is an input to the portfolio research extension. The full notebook generates additional diagnostic CSVs that are not all included in this compact submission.

## Viewing and Reproducing the Results

### Review the completed research

The submitted notebook includes the code and saved outputs from the completed experiments. The PDF report summarizes the methodology, portfolio performance, robustness tests, and limitations.

The `figures/` and `results/` directories contain selected visualizations and machine-readable research outputs.

### Reproduction from a clean clone

**Environment:** Python 3.11 is recommended.

The notebook was developed and executed from the repository's `source_code/` directory. To reproduce the experiments, the submitted notebook and helper scripts should be copied into that directory, preserving the original working-directory structure.

**Step 1 — Clone the repository and install dependencies.**

From the repository root:

```powershell
python -m pip install -r submissions/Assignment1_KevinMa_km4189/requirements.txt
```

**Step 2 — Prepare the input data.**

Use the original dataset files in the repository's `data/` directory. Extract `finrl_trading.7z` to obtain `finrl_trading.db` if it has not already been extracted.

Copy the submitted notebook and helper scripts into `source_code/`:

```powershell
Copy-Item submissions/Assignment1_KevinMa_km4189/Assignment1_Portfolio_Research.ipynb source_code/

Copy-Item submissions/Assignment1_KevinMa_km4189/*.py source_code/
```

**Step 3 — Run the data preparation pipeline.**

From the repository root:

```powershell
cd source_code

python prepare_data.py --db ../data/finrl_trading.db --fundamentals ../data/fundamental_data_full.csv --constituents ../data/sp500_historical_constituents.csv --output-dir ./prepared --start 2015-01-01 --end 2025-09-30 --trade-start 2020-09-30
```

This generates `prepared/final_ratios.csv`, eleven sector-specific Excel files, and `prepared/daily.csv`.

**Step 4 — Execute the research notebook.**

Remain inside `source_code/` and launch:

```powershell
jupyter notebook Assignment1_Portfolio_Research.ipynb
```

Run the notebook's initial imports, but skip the original Section 1 preprocessing execution, which contains empty path placeholders for the alternative WRDS workflow. Its preprocessing functionality is replaced by `prepare_data.py`.

Run Section 2 (stock selection) to regenerate the sector-level model predictions and `results/stock_selected.csv`. Then execute the portfolio construction, backtesting, and independent research cells in order.

The original model-training stage may take substantial time. Some benchmark downloads use `yfinance` and require internet access.

**Important:** Relative paths are based on executing the notebook from `source_code/`. In particular, the point-in-time audit expects the original dataset under `../data/`.

The raw dataset and generated `prepared/` files are intentionally excluded from the submission because they can be regenerated. The submitted `results/` files provide reference outputs from the completed research run.


## Methodological limitations

- **Membership audit:** the PIT filter matches historical membership snapshots at fiscal-quarter dates; this does not itself establish complete coverage of delisted securities.
- **Timing:** 124 of 238 model-evaluation windows had predictions earlier than nominal quarter-end dates, and 96 showed disclosure-date overlap flags. These are **diagnostic flags, not established leakage**.
- **Coverage:** approximately 89.2% of selected stocks had usable prices in the monthly portfolio process, potentially introducing selection bias.
- **Cost model:** portfolios assume 10 basis points per unit of turnover; ETF benchmark NAVs are adjusted-price series without additional execution costs.
- **Inference:** results are historical and exploratory. The ML strategy's estimated market-adjusted alpha is not statistically significant.

## References

- [AI4Finance Foundation — FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)
- GR5398 course-provided FinRL-Trading Assignment 1 materials (Fall 2026)
- Ledoit, O., and Wolf, M. (2004). *Honey, I Shrunk the Sample Covariance Matrix*. *Journal of Portfolio Management*, 30(4), 110–119.
