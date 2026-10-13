# Machine-Learning Equity Selection and Portfolio Backtesting

**Columbia University | STAT GR5398 | Fall 2026 — Assignment 1 (FinRL-Trading)**

This project builds a fundamental-data-driven equity selection workflow using Random Forest, XGBoost and LightGBM models across 11 market sectors. It converts model predictions into a concentrated, equally weighted portfolio and compares an exploratory historical backtest against the SPY ETF.

> **Research status:** Historical simulation and course research, **not** a validated live-trading strategy. The final reported portfolio uses a price-covered universe because some predicted securities lack prices in the available dataset. Point-in-time feature availability, historical universe coverage and out-of-sample validity have not been fully established.

## Research workflow

1. Prepare and clean stock-price and financial-ratio datasets.
2. Construct financial features and sector-level stock-selection models.
3. Combine sector predictions and rank eligible stocks by `predicted_return`.
4. Form an **equal-weight Top 20** portfolio from securities represented in the supplied historical price panel.
5. Apply a **one-trading-day signal lag** and calculate **adjusted close-to-close** portfolio returns, with simplified transaction costs.
6. Compare the strategy with dividend-adjusted **SPY** over matched trading dates; report cumulative/annualised returns, volatility, Sharpe ratio, drawdown and annual performance.

## Key results

**Evaluation period:** 2020-05-01 through 2025-09-30 (aligned trading days).

| Metric | Price-Covered Top 20 | SPY |
|:--|--:|--:|
| Cumulative return | **194.76%** | 148.07% |
| Annualised return | **22.16%** | 18.32% |
| Annualised volatility | **24.74%** | 17.55% |
| Sharpe ratio (risk-free rate = 0) | 0.93 | **1.05** |
| Maximum drawdown | **−38.95%** | −24.50% |

The portfolio generated higher *historical cumulative returns*, but its **Sharpe ratio was lower and its maximum drawdown was substantially larger** than SPY's. These are exploratory results obtained after portfolio design decisions were assessed on the same evaluation sample; they must not be interpreted as independent out-of-sample evidence.

The stock-selection output contains **171,013 prediction records for 480 distinct tickers**. Of those tickers, **67 do not have a corresponding price column** in the supplied `prepared/daily.csv`. The final strategy therefore ranks candidates **within the price-covered universe before** selecting Top 20. This is different from the original, unrestricted Top 20 experiment, whose reported backtest should not be treated as validated because of unpriced positions.

## Submission files

The course submission directory is intended to contain the following **small, reviewable** files (the full training data are not bundled):

```text
Assignment1_Name_UNI/
├── FinRL-Trading.ipynb
├── FinRL_Trading_Research_Report.pdf
├── README.md
├── requirements.txt
├── prepare_data.py
├── fundamental_run_model.py
├── ml_model.py
├── results/
│   ├── price_covered_top20.csv
│   └── price_covered_top20_vs_spy.csv
└── figures/
    ├── price_covered_top20_vs_spy_nav.png
    └── price_covered_annual_vs_spy.png
```

**Quick links:** [Medium article](https://medium.com/@jiqibruce/can-machine-learning-beat-the-s-p-500-building-a-multi-sector-stock-selection-strategy-8b86691db368) · [Research report](FinRL_Trading_Research_Report.pdf) · [NAV chart](figures/price_covered_top20_vs_spy_nav.png) · [Annual comparison](figures/price_covered_annual_vs_spy.png) · [Performance summary](results/price_covered_top20_vs_spy.csv).

The submission package is a **reviewable snapshot**; rerunning the complete training/backtest requires the original upstream data and stored predictions from the `source_code/` working directory. Avoid committing `.venv/`, a 403-MB SQLite database, the 116-MB prepared daily panel, or other large intermediate data.

## Environment

The completed research was run in Python **3.11** with a `finrl` environment. Key Python packages include:

- `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`
- `xgboost`, `lightgbm`, `openpyxl`
- `yfinance` (used previously to download SPY data; **not required when `spy_prices.csv` is present**)
- `ipykernel` / Jupyter for the notebook

Install the dependencies in your own Python 3.11 environment as appropriate:

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn xgboost lightgbm openpyxl yfinance ipykernel
```

## Running and reproducibility

1. Create a Python 3.11 environment and install the packages in `requirements.txt` (or the dependencies listed above).
2. Open `FinRL-Trading.ipynb` in VS Code/Jupyter. Its outputs include the executed final backtest tables and charts for inspection without any additional training.
3. **To rerun** the research, first restore the upstream data under `source_code/prepared/`, the model predictions under `source_code/outputs/stock_selection/stock_selected.csv`, and adjusted SPY prices under `source_code/outputs/backtest/spy_prices.csv`.
4. Start Jupyter from `source_code/` with **`RUN_PREPROCESSING = False`** and **`RUN_STOCK_SELECTION = False`**. Run the remaining cells in order; the final results are written to `outputs/backtest/` and `outputs/figures/`.
5. Rerunning full model training additionally requires the original raw fundamental/price inputs. Those large source datasets are not included in the submission. The code and results are provided for review, but this is **not a self-contained, from-scratch reproducibility bundle**.

## Methodology details

**Models:** Sector-based selection using Random Forest, XGBoost and LightGBM. Hyperparameters and selection details reside in `ml_model.py` and the notebook; this README does not substitute for the implementation.

**Selection:** Rank model-selected securities by `predicted_return` for each signal date, limiting eligibility to tickers for which a historical adjusted-close price column exists. Hold up to 20 selected stocks with equal target weights.

**Return calculation:** Use adjusted close-to-close daily stock returns, with portfolio exposures derived from the **prior trading day's** target weights. The adjusted stock-close calculation in this dataset is `prccd / ajexdi`; SPY uses Yahoo Finance's adjusted-price history (`auto_adjust=True`).

**Costs:** Deduct `0.001 × sum(abs(change in target weights))` per day; the model does not explicitly capture bid–ask spreads, market impact, drift-adjusted trades or all possible initial-entry costs.

**Benchmark:** Dividend-adjusted SPY ETF closing prices aligned to the portfolio evaluation dates. The Sharpe ratios reported here use a zero risk-free rate for comparability with the notebook output.

## Important limitations

- **Price-coverage and survivorship risk:** 67 predicted tickers lacked corresponding price columns in the prepared price panel. Restricting eligibility to covered names changes the investment universe and may create selection bias. It **does not** prove the absence of survivorship bias or recover delisting cash flows.
- **Point-in-time fundamentals:** Some `reportdate` values precede the associated quarter-end dates. Feature publication/availability timing has **not** been fully verified. A one-day signal lag alone does not eliminate look-ahead bias if the features were not available at the prediction date.
- **Sample selection:** The Top 20 specification was evaluated using the same historical sample. Results are exploratory and not independently validated out-of-sample.
- **Trading realism:** Turnover-based transaction costs are approximate, and liquidity, spreads, market impact and price drift are not fully simulated.
- **Model and data replicability:** Large upstream files are stored locally; exact replication requires the original prepared files, training code and model outputs. Results depend on the provided coverage and adjustment conventions.

## Deliverables and article

- **Executed notebook:** [`FinRL-Trading.ipynb`](FinRL-Trading.ipynb)
- **Research report (PDF):** [`FinRL_Trading_Research_Report.pdf`](FinRL_Trading_Research_Report.pdf)
- **Final chart — NAV:** [`figures/price_covered_top20_vs_spy_nav.png`](figures/price_covered_top20_vs_spy_nav.png)
- **Final chart — annual returns:** [`figures/price_covered_annual_vs_spy.png`](figures/price_covered_annual_vs_spy.png)
- **Daily strategy output:** [`results/price_covered_top20.csv`](results/price_covered_top20.csv)
- **SPY comparison:** [`results/price_covered_top20_vs_spy.csv`](results/price_covered_top20_vs_spy.csv)
- **Medium article:** [Can Machine Learning Beat the S&P 500? Building a Multi-Sector Stock Selection Strategy](https://medium.com/@jiqibruce/can-machine-learning-beat-the-s-p-500-building-a-multi-sector-stock-selection-strategy-8b86691db368) (published on Medium).

## Conclusion

Within the available historical price-covered universe, the machine-learning-based Top 20 portfolio exceeded SPY's cumulative return during the study interval. The strategy also incurred materially higher downside risk and a lower Sharpe ratio. Robust point-in-time data construction, historical security coverage and walk-forward evaluation remain necessary before making credible investability claims.
