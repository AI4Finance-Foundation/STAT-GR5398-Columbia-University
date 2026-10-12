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

## Viewing and reproduction

### Review the completed work (no data download required)

1. Open the PDF for methods, summary tables, and conclusions.
2. Open the notebook to inspect code and its **saved outputs**. The independent research begins after the original separator labeled “Your own strategy and visulization can be built based on the above code.”
3. See `figures/` for publication figures and `results/` for machine-readable evidence.

### Re-running the analysis (additional inputs required)

**Environment:** Python 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
jupyter notebook Assignment1_Portfolio_Research.ipynb
```

Open Jupyter **from the directory containing the notebook**. Several cells use relative paths and assume that this directory is the working directory.

This compact submission does **not** contain all raw data and helper scripts needed for a clean end-to-end run. The original workflow uses:

- `prepared/daily.csv` — adjusted daily stock prices;
- `prepared/final_ratios.csv` and `prepared/sector{N}.xlsx` — processed fundamental data;
- `results/stock_selected.csv` — historical ML-selected stock observations;
- `results/sector{N}/` — original sector-by-sector prediction and model-selection CSVs;
- `prepare_data.py`, `fundamental_run_model.py`, and `ml_model.py` — original data preparation/model helpers;
- adjacent raw `data/` files for independent PIT reconstruction, including `fundamental_data_full.csv` and `sp500_historical_constituents.csv`.

The original first-stage notebook contains **empty path placeholders** for its WRDS-style preprocessing route. To reproduce the reported study from source data, obtain the source files/scripts from the course's [FinRL-Trading repository](https://github.com/AI4Finance-Foundation/FinRL-Trading), prepare the inputs, complete the upstream model-selection run, and then execute the portfolio research cells **in notebook order**. Do not select **Run All** in an empty checkout and expect a clean run. Some benchmark-price retrieval uses `yfinance` and requires network access.

The five report figures were saved from the completed local research session. Figures 1 and 2 use verified numeric summaries rather than querying raw CSVs again in their plotting cells; the preceding diagnostic and strategy-comparison cells provide their underlying analyses. Figure 2's final formatting cell depends on the preceding Figure 2 plotting cell, while Figure 3's save step depends on the previously-created `fig` object.

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
