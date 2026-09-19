# GR5398 26 Fall: FinRL-Trading Quantitative Trading Strategy Track
## Assignment 1

## 0. Targets

In this assignment 1, we want you to:

+ Run `source_code/FinRL-Trading-Full-Workload.ipynb`, which is a simplified FinRL-Trading whole process, and have a basic knowledge of what we will do in this semester
+ Design a portfolio using the selected stocks, and learn some fundamental information of quantitative trading (especially stock selection part)
+ Implement a full backtest process to verify your strategy's performance using real historical data
+ Build a sample that is free of **survivorship bias** and **look-ahead bias** — and be able to prove it
+ Summarize your result in a very brief research report, and write a `Medium Blog`. Submit your code files onto GitHub repo in a new folder called `Assignment1_Name_UNI` in `/submissions` (not a new branch!)
    + An example of medium blog: [Applying new LLMs on FinGPT: Fine-tune DeepSeek and Llama3](https://medium.com/p/6ac9198d88b2)

For the full `FinRL-Trading` project, please refer to [AI4Finance/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master_backup).

Assignment 1 Report Submission Due Day: **Oct 12, 2026**.

### Pipeline at a glance

| # | Step | Input | Output |
| --- | --- | --- | --- |
| 0 | Get data | [`FinRL-Trading/data`](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/data) — no account needed | `finrl_trading.db`, `fundamental_data_full.csv`, `sp500_historical_constituents.csv` |
| 1 | Prepare data | The three files above | `final_ratios.csv`, `sector{N}.xlsx`, `daily.csv` |
| 2 | Stock selection | `final_ratios.csv` + `sector{N}.xlsx` | `results/sector{N}/df_predict_*.csv`, `stock_selected.csv` |
| 3 | Portfolio construction | `stock_selected.csv` + `daily.csv` | Daily weight matrix (dates × tickers) |
| 4 | Backtest | Weight matrix + daily returns | NAV curve, net returns, SPY/QQQ comparison |

---

## 1. Data

### 1.1 The dataset (default path)

Everything you need is published in the FinRL-Trading repo — **no WRDS account, no license
restriction, no registration**:

https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/data

| File | Size | What is inside |
| --- | --- | --- |
| `finrl_trading.7z` | 42 MB → **403 MB** `finrl_trading.db` | SQLite. `price_data`: 1,275,978 daily bars, 506 tickers, 2015-06-15 → 2025-10-16, with `open/high/low/close/adj_close/volume`. `raw_payloads`: 77,364 raw FMP statement JSONs with `fillingDate` / `acceptedDate` |
| `fundamental_data_full.csv` | 23 MB | 22,909 quarterly rows × 64 columns, 715 tickers, 2015-06-30 → 2026-03-31. **Ratios are already computed** (52 usable features), plus the `y_return` label, `adj_close_q`, and `filing_date` / `accepted_date` / `actual_tradedate` |
| `sp500_historical_constituents.csv` | 5.5 MB | Point-in-time S&P 500 membership, daily, back to **1996-01-02** — **including delisted tickers** (AAMRQ, ENRNQ, EKDKQ …) |

```bash
git clone https://github.com/AI4Finance-Foundation/FinRL-Trading.git
cd FinRL-Trading/data
7z x finrl_trading.7z          # or: bsdtar -xf finrl_trading.7z
```

> [!IMPORTANT]
> `sp500_historical_constituents.csv` is **not optional**. Backtesting today's S&P 500 over 2020
> means you only ever trade companies that survived to 2026 — that is **survivorship bias**, and it
> inflates every performance number you will report. Use the point-in-time membership so a company
> stays in your universe exactly as long as it was actually in the index. Your report must state
> how many tickers and rows this removed from your sample.

### 1.2 Prepare it

`source_code/prepare_data.py` translates the dataset into what the models expect. Run it once:

```bash
python prepare_data.py \
    --db        /path/to/finrl_trading.db \
    --fundamentals /path/to/fundamental_data_full.csv \
    --constituents /path/to/sp500_historical_constituents.csv \
    --output-dir ./prepared \
    --start 2015-01-01 --end 2025-09-30 --trade-start 2020-09-30
```

What it does, and why each step is needed:

| Translation | Reason |
| --- | --- |
| `ticker` → `tic`, `datadate` → `date` | `ml_model.py` is written against those names |
| `filing_date` → `reportdate` | A quarter's numbers are only tradable once the filing is out — this is what prevents **look-ahead bias** |
| Sector **names** (`Technology`) → **GICS numbers** (`45`) | `run_stock_selection` iterates `range(10, 65, 5)` looking for `sector{N}.xlsx` |
| Drops `id`, `created_at`, `tradedate`, `actual_tradedate`, `trade_price` | Identifiers and prices would otherwise leak into the feature set |
| Drops rows with no `y_return` | The most recent quarter of each ticker has no next-quarter return yet |
| Point-in-time membership filter | Kills survivorship bias (see above) |
| `adj_close / close` → adjusted open | The dataset gives `adj_close`, not an `ajexdi` factor |

**Output** (in `./prepared`):

| File | Contents |
| --- | --- |
| `final_ratios.csv` | 19,007 rows × 59 columns — the full cross-section |
| `sector10.xlsx` … `sector60.xlsx` | The same table split into 11 sectors |
| `daily.csv` | `tic, date, open, high, low, close, adj_close, volume, open_adj, close_adj` |

A reference run (defaults, full history) produces:

```
Quarters available: 42  (2015-06-30 .. 2025-09-30)
kept 19,007 of 21,299 rows (674 tickers, 692 before)   <- survivorship filter
--trade-start 2020-09-30  ->  -first_trade_index 21
```

**Your backtest window is therefore 2020-06-30 → 2025-09-30** (about 21 quarters) with
2015-06-30 → 2020-06-30 used as training history. Two things constrain this and you should
understand both:

+ The **price table ends 2025-10-16**, so the backtest cannot run past then. If you want a longer
  window, top up recent prices with `yfinance` (the notebook already imports it) and say so.
+ The rolling window needs 16 quarters of training + 4 of testing before it can trade, which is
  what `-first_trade_index` encodes. The script prints the right value for your `--trade-start`.

> [!TIP]
> Don't commit `prepared/` — `daily.csv` alone is ~73 MB. One command regenerates it, which is
> better reproducibility than a committed file anyway. Commit the command, not the output.

### 1.3 Does the existing notebook still work with this data?

Yes — this was tested end to end, not assumed. Here is exactly what changes:

| Part of `source_code/` | Status with the prepared data |
| --- | --- |
| Notebook **Section 1** (preprocessing cells) | **Skip it.** `fundamental_data_full.csv` already contains computed ratios, so `prepare_data.py` replaces these cells. They remain for the WRDS path in 1.4 |
| Notebook **Section 2** (`run_stock_selection`) | Works unchanged — point `data_path` at `./prepared` |
| `fundamental_run_model.py`, `ml_model.py` | Work unchanged. Verified: sector15 trains RF / XGBoost / LightGBM per rolling window and writes all six result files |
| Notebook **Sections 3–4** (portfolio + backtest) | Work unchanged — `daily.csv` deliberately uses the WRDS column names (`datadate`, `prcod`, `prccd`, `ajexdi`), so `prccd / ajexdi` returns the adjusted close exactly (verified to 2e-13) |

Three practical notes:

1. **Run from the `source_code/` directory.** `run_stock_selection` shells out with
   `python fundamental_run_model.py ...` and writes to `./results/sector{N}/`, both relative to
   your working directory.
2. **`pip install -r requirements.txt` first.** pandas is pinned below 3.0 — `ml_model.py` and the
   notebook are written against the pandas 2 API. (One pandas-3 incompatibility in
   `fundamental_run_model.py` is already fixed; the rest is untested on pandas 3, and making it
   work would be a reasonable upstream contribution.)
3. **The last quarter's signals are not tradable.** Predictions are indexed by `reportdate`, so the
   quarter ending 2025-09-30 gets filing dates in Oct–Nov 2025 — after the price table ends on
   2025-10-16. Those rows will silently drop out when weights are joined to returns. That is
   correct behavior, not a bug: you cannot trade on a filing you have prices for.

Runtime: roughly 2–4 seconds per sector per quarter on a laptop (XGBoost dominates), so a full
11-sector run over ~22 trade dates is on the order of half an hour.

### 1.4 WRDS (optional, advanced)

The dataset above starts in 2015. If you want a longer history — a backtest across the 2008 crisis,
say — download it yourself from [WRDS](https://wrds-www.wharton.upenn.edu/), which Columbia
provides free to master's students ([registration guide](https://guides.library.columbia.edu/wrds)):

+ **Daily** — [Security Daily](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/security-daily/): `tic`, `gvkey`, `datadate`, `prcod`, `prccd`, `ajexdi`
+ **Quarterly** — [Fundamental Quarterly](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/security-daily/): `gvkey`, `tic`, `gsector`, `datadate`, `rdq`, `prccq`, `adjex`, `cshoq`, `epspxq`, `revtq`, `oiadpq`, `niq`, `cogsq`, `atq`, `ltq`, `teqq`, `ceqq`, `actq`, `lctq`, `cheq`, `rectq`, `invtq`, `apq`, `dlttq`, `dlcq`, `dvpspq`, `epspiy`

Going this route, you build `final_ratios.csv` yourself with the preprocessing cells in the
notebook (section 1), which compute 18 ratios from the raw Compustat fields — see
[Step2_preprocess_fundmental_data.py](https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master_backup/data_processor/Step2_preprocess_fundmental_data.py).
You will also need a point-in-time universe of your own.

Stock pools for reference: `source_code/SP500_components.txt`, `source_code/NASDAQ_components.txt`.

> [!WARNING]
> WRDS data is **licensed** — do not commit it. Commit your code and results, and state the exact
> query that reproduces your inputs.

---

## 2. Step 2 — Stock Selection

**Input**: `final_ratios.csv` and the `sector{N}.xlsx` files from `prepare_data.py`.

```python
data_path         = "./prepared"  # where prepare_data.py wrote its output
output_path_step2 = ""            # where the selection result goes
```

**What happens**

For each sector, `run_stock_selection` shells out to `fundamental_run_model.py`, which calls
`ml_model.run_4model`:

+ **Rolling window**: train on 16 quarters (4 years), test on the next 4 quarters (1 year), start
  trading at quarter `-first_trade_index + 1`, then roll forward. This is what keeps the evaluation
  out-of-sample.
+ **Models**: Random Forest, LightGBM and XGBoost are each trained per window; the one with the
  lowest validation MSE becomes the "best" prediction for that window.
+ **Features**: every numeric column in the sector file except ids, dates and the label — 52 of
  them in the prepared dataset.
+ **Label**: `y_return` — next quarter's return.
+ **Dynamic refresh**: predictions are forward-filled from each stock's `reportdate`, so a stock's
  forecast updates the moment a new filing lands, instead of on a fixed calendar date.
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
  | `predicted_return` | Predicted next-quarter return |
  | `trade_date` | Date this holding is valid for |

You are also encouraged to try DRL-based selection and reallocation —
[fundamental_portfolio_drl.py](https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/master_backup/fundamental_portfolio_drl.py).

---

## 3. Step 3 & 4 — Portfolio Construction and Backtest

**Input**
+ `stock_selected.csv` from Step 2
+ `daily.csv` from `prepare_data.py`

**Baseline strategy provided**: equal weight, buy & hold.

+ On each date, every selected stock gets weight `1 / n_stocks`, everything else 0 → a
  (dates × tickers) weight matrix whose rows sum to 1.
+ Prices are already adjusted by `prepare_data.py`: `close_adj = adj_close` and
  `open_adj = open * (adj_close / close)`. (The notebook's original `prccd / ajexdi` formula is for
  the WRDS path — this dataset ships `adj_close` directly, so use the adjusted columns as they are.)
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

### Reference run — what the baseline actually produces

The notebook was executed end to end on the prepared dataset (all 11 sectors, ~43 minutes). This
is the equal-weight buy & hold baseline, with no changes:

| | Final NAV | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- |
| **Baseline portfolio** | 1.31 | 5.07% | 0.37 | −19.76% |
| SPY | 2.54 | 18.77% | 1.07 | −24.50% |
| QQQ | 2.91 | 21.78% | 0.98 | −35.12% |

Backtest 2020-05-01 → 2025-09-30 (5.4 years, 1,361 trading days), 419 stocks selected across
174,001 (date, ticker) records, average daily turnover 0.032, total transaction-cost drag 4.38%.

**The baseline loses to the index by a wide margin.** That is the assignment: the fundamental
signal, the top-25% rule, the equal weighting and the quarterly-ish rebalance are all yours to
improve. Reproduce this number first, then beat it — and when you report your own result, report
it next to this one.

> [!NOTE]
> Selection dates run to 2025-11-25 (filing dates for the 2025-09-30 quarter) but the backtest
> stops at 2025-09-30, where the price data ends. Those trailing signals are silently dropped when
> weights are joined to returns — expected, as described in 1.3.

---

## 4. Research Report for Assignment 1

Your report should cover:

+ Data: which window you used, how many tickers and quarters survived preparation, and **what the
  point-in-time filter removed** — name a few delisted companies that stayed in your sample
+ Bias audit: how your setup avoids look-ahead bias (`reportdate`, not quarter end) and
  survivorship bias (point-in-time membership). Be specific; this is a graded section
+ Stock selection: which model won most often, feature importance, how you picked the top-k%
+ Strategy: weighting scheme, rebalancing frequency, transaction-cost assumption
+ **Performance**: cumulative return, annualized return, Sharpe, max drawdown, volatility,
  turnover — versus SPY and QQQ. Never report return alone
+ What did *not* work, and what you would do differently

## 5. What to Submit

Create `submissions/Assignment1_Name_UNI/` and include:

1. Your notebook / scripts (runnable from a clean clone)
2. `README.md` — the exact commands to reproduce your result, including your `prepare_data.py`
   invocation, and the link to your Medium blog
3. Your research report (PDF)
4. Result files small enough to commit (e.g. `stock_selected.csv`, NAV series, plots)

Do **not** commit the dataset or the prepared files — they are regenerable, and `daily.csv` alone
is ~73 MB.
