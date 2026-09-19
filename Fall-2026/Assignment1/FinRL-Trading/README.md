# GR5398 26 Fall: FinRL-Trading Quantitative Trading Strategy Track
## Assignment 1

## 0. Targets

In this assignment 1, we want you to:

+ Run `source_code/FinRL-Trading-Full-Workload.ipynb`, which is a simplified FinRL-Trading whole process, and have a basic knowledge of what we will do in this semester
+ Design a portfolio using the selected stocks, and learn some fundamental information of quantitative trading (especially stock selection part)
+ Implement a full backtest process to verify your strategy's performance using real historical data
+ Summarize your result in a very brief research report, and write a `Medium Blog`. Submit your code files onto GitHub repo in a new folder called `Assignment1_Name_UNI` in `/submissions` (not a new branch!)
    + An example of medium blog: [Applying new LLMs on FinGPT: Fine-tune DeepSeek and Llama3](https://medium.com/p/6ac9198d88b2)

For the full `FinRL-Trading` project, please refer to [AI4Finance/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master_backup).

Assignment 1 Report Submission Due Day: **Oct 12, 2026**.

### Pipeline at a glance

| # | Step | Input | Output |
| --- | --- | --- | --- |
| 0 | Get data | Stock pool (`SP500_components.txt` / `NASDAQ_components.txt`) + WRDS account | `fundamental.csv` (quarterly), `daily.csv` (daily OHLCV) |
| 1 | Data preprocessing | The two CSVs above | `final_ratios.csv` + one `sector{N}.xlsx` per GICS sector |
| 2 | Stock selection | `final_ratios.csv` + `sector{N}.xlsx` | `results/sector{N}/df_predict_*.csv`, `stock_selected.csv` |
| 3 | Portfolio construction | `stock_selected.csv` + `daily.csv` | Daily weight matrix (dates × tickers) |
| 4 | Backtest | Weight matrix + daily returns | NAV curve, net returns, SPY/QQQ comparison |

---

## 1. Prerequisites

### 1.1 Choose your stock pool

Pick your universe from the **S&P 500** or **NASDAQ 100**. Both component lists are provided:

+ `source_code/SP500_components.txt`
+ `source_code/NASDAQ_components.txt`

### 1.2 Download the data

Download from [WRDS](https://wrds-www.wharton.upenn.edu/):

+ **Daily price data** — [WRDS Security Daily](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/security-daily/)
+ **Quarterly fundamental data** — [WRDS Fundamental Quarterly](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/security-daily/)

Columbia provides every master's student with a free WRDS account — follow
[this guide](https://guides.library.columbia.edu/wrds) to register.

**Period to download: Jan 1, 2018 → Dec 31, 2025** (this is the backtest window).

To keep the files small and fast to load, query only the columns you need:

| File | Required columns | Meaning |
| --- | --- | --- |
| Daily price (`daily.csv`) | `tic` | Ticker |
| | `gvkey` | Compustat unique company id |
| | `datadate` | Trading date |
| | `prcod` | Price — Open — Daily |
| | `prccd` | Price — Close — Daily |
| | `ajexdi` | Cumulative adjustment factor (for splits/dividends) |
| Fundamental (`fundamental.csv`) | `gvkey`, `tic`, `gsector`, `datadate`, `rdq` | Ids, GICS sector, quarter end, report date |
| | `prccq`, `adjex`, `cshoq`, `epspxq` | Quarterly price, adjustment factor, shares outstanding, EPS |
| | `revtq`, `oiadpq`, `niq`, `cogsq` | Revenue, operating income, net income, COGS |
| | `atq`, `ltq`, `teqq`, `ceqq` | Total assets, total liabilities, equity |
| | `actq`, `lctq`, `cheq`, `rectq`, `invtq`, `apq` | Current assets/liabilities, cash, receivables, inventory, payables |
| | `dlttq`, `dlcq`, `dvpspq`, `epspiy` | Long/short-term debt, dividend per share, EPS incl. extraordinary |

> [!TIP]
> `rdq` (the report date) matters: fundamentals are only knowable **after** they are reported.
> Using the quarter-end date as if it were tradable is look-ahead bias.

---

## 2. Step 1 — Data Preprocessing

Reference implementation: [FinRL-Trading/data_processor/Step2_preprocess_fundmental_data.py](https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master_backup/data_processor/Step2_preprocess_fundmental_data.py)

Set these three variables in the notebook before running:

```python
Stock_Index_fundation_file = ""  # path to your quarterly fundamental CSV
Stock_Index_price_file     = ""  # path to your daily price CSV
output_dir                 = ""  # where results are written
```

**Input**
+ `fundamental.csv` — raw quarterly fundamentals
+ `daily.csv` — raw daily prices

**What happens**

| Function | What it does |
| --- | --- |
| `load_data` | Reads both files |
| `adjust_trade_dates` | Maps each quarter to its quarter-end trading date, keeps `rdq` as `reportdate` |
| `calculate_adjusted_close` | `adj_close_q = prccq / adjex` |
| `match_tickers_and_gvkey` | Keeps only tickers present in both files |
| `calculate_next_quarter_returns` | **The label**: `y_return = log(adj_close_q[t+1] / adj_close_q[t])` |
| `calculate_basic_ratios` | PE, PS, PB |
| `calculate_financial_ratios` | 18 features: OPM, NPM, ROA, ROE, EPS, BPS, DPS, cur_ratio, quick_ratio, cash_ratio, inv_turnover, acc_rec_turnover, acc_pay_turnover, debt_ratio, debt_to_equity, pe, ps, pb |
| `handle_missing_values` | Fills NA/inf with 0, drops rows with zero adjusted close |
| `save_results` | Writes the output files |

**Output** (in `output_dir`)
+ `final_ratios.csv` — one row per (ticker, quarter), with all 18 ratios, `y_return`, `adj_close_q`, `reportdate`, `gsector`
+ `sector10.xlsx`, `sector15.xlsx`, … `sector60.xlsx` — the same table split by GICS sector (sector 0 = missing sector info, skipped by default)

---

## 3. Step 2 — Stock Selection

**Input**
+ `final_ratios.csv` and the `sector{N}.xlsx` files from Step 1

```python
data_path         = output_dir  # must match Step 1
output_path_step2 = ""          # where the selection result goes
```

**What happens**

For each sector, `run_stock_selection` shells out to
`fundamental_run_model.py`, which calls `ml_model.run_4model`:

+ **Rolling window**: train on 16 quarters (4 years), test on the next 4 quarters (1 year),
  first trade at quarter index 20, then roll forward. This is what keeps the evaluation
  out-of-sample.
+ **Models**: Random Forest, LightGBM and XGBoost are each trained per window; the one with the
  lowest validation MSE becomes the "best" prediction for that window.
+ **Features**: every numeric column in the sector file except ids, dates and the label.
+ **Label**: `y_return` — next quarter's log return.
+ **Dynamic refresh**: predictions are forward-filled from each stock's `reportdate`, so a stock's
  forecast updates the moment a new report lands, instead of on a fixed calendar date.
+ **Selection rule**: on each date, keep the stocks in the **top 25%** of predicted return
  (`quantile(0.75)`). This threshold is yours to tune.

**Output**
+ `results/sector{N}/df_predict_rf.csv`, `df_predict_gbm.csv`, `df_predict_xgb.csv` — per-model predictions (rows = dates, columns = tickers)
+ `results/sector{N}/df_predict_best.csv` — the selected-model prediction
+ `results/sector{N}/df_best_model_name.csv` — which model won in each window
+ `results/sector{N}/df_model_score.csv` — per-model scores
+ **`{output_path_step2}/stock_selected.csv`** — the file Step 3 consumes:

  | Column | Meaning |
  | --- | --- |
  | `tic` | Ticker held |
  | `predicted_return` | Predicted next-quarter log return |
  | `trade_date` | Date this holding is valid for |

You are also encouraged to try DRL-based selection and reallocation —
[fundamental_portfolio_drl.py](https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master_backup/fundamental_portfolio_drl.py).

---

## 4. Step 3 & 4 — Portfolio Construction and Backtest

**Input**
+ `stock_selected.csv` from Step 2
+ `daily.csv` (needs `tic`, `datadate`, `prcod`, `prccd`, `ajexdi`)

**Baseline strategy provided**: equal weight, buy & hold.

+ On each date, every selected stock gets weight `1 / n_stocks`, everything else 0 → a
  (dates × tickers) weight matrix whose rows sum to 1.
+ Prices are adjusted: `open_adj = prcod / ajexdi`, `close_adj = prccd / ajexdi`.
+ Per-stock return: `ret = close_adj / open_adj - 1`.
+ Portfolio gross return: `(weights * ret).sum(axis=1)`.
+ **Transaction cost**: `turnover = |Δweights|.sum(axis=1)`, charged at `fee_rate = 0.001` (0.1%).
+ Net return: `gross_ret - cost`; NAV is the cumulative product.

**Output**
+ Daily weight matrix
+ Daily gross return, turnover, cost and net return series
+ NAV curve, normalized to 1.0 at the start
+ A comparison plot against **SPY** and **QQQ** (downloaded via `yfinance`)

> [!IMPORTANT]
> **Your portfolio must beat the S&P 500 over the full backtest window.** The last cell of the
> notebook is where your own strategy goes — equal weight buy & hold is only the starting point.

---

## 5. Research Report for Assignment 1

Your report should cover:

+ Which universe you chose (S&P 500 or NASDAQ 100) and why
+ Data: source, period, coverage, how you handled missing values and delisted names
+ Stock selection: which model won most often, feature importance, how you picked the top-k%
+ Strategy: weighting scheme, rebalancing frequency, transaction-cost assumption
+ **Performance**: cumulative return, annualized return, Sharpe, max drawdown, volatility,
  turnover — versus SPY and QQQ. Never report return alone.
+ What did *not* work, and what you would do differently

## 6. What to Submit

Create `submissions/Assignment1_Name_UNI/` and include:

1. Your notebook / scripts (runnable from a clean clone)
2. `README.md` — how to reproduce your result, and the link to your Medium blog
3. Your research report (PDF)
4. Result files small enough to commit (e.g. `stock_selected.csv`, NAV series, plots)

> [!WARNING]
> ⚠️ **Do not commit raw WRDS data** — it is licensed. Commit your code and your results, and
> describe exactly which WRDS query reproduces your input files.
