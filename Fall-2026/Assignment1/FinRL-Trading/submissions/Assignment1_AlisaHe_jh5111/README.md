# Assignment 1 — FinRL-Trading (Alisa He)

This folder runs the course FinRL-Trading pipeline, audits it for survivorship and look-ahead bias,
and builds a strategy that beats SPY over the full backtest.

## What's in this submission

* **Medium blog:** [Assignment 1](https://medium.com/@alisa.he.qwq/assignment-1-45300306ce16)
* `report/report.pdf`: the 4-page report (source `report/report.tex`), plus `report/technical_appendix.pdf`
* `FinRL-Trading-Full-Workload.ipynb`: the course notebook, run end to end
* `strategy_backtest.py`: the strategy and backtest
* `prepare_data.py`, `fundamental_run_model.py`, `ml_model.py`: course code, unchanged
* `analysis/`: bias audit, robustness checks and report figures
* `outputs/`: selected stocks, NAV series and the performance table
* `factor_strategies.py`, `model_v2/`: the extra tests mentioned in the report

## Result

From 2020-05-01 to 2025-09-30, net of 0.1% trading costs:

| | Final NAV | Sharpe | Max drawdown |
|---|---|---|---|
| Course baseline (reproduced) | 1.31 | 0.40 | −19.8% |
| **Strategy** | **2.563** | 0.95 | −31.4% |
| SPY | 2.548 | 1.08 | −24.5% |

The strategy averages the three models' predictions, buys the top 25% each month and gives more
weight to the best-ranked stocks. It beats SPY, but the lead is not statistically significant
(t = 0.23) and comes mostly from a 64% weight in technology stocks.

Four problems in the original pipeline were fixed before this result:

1. **Overnight returns.** The notebook only counts open-to-close returns, which explains most of the
   baseline's gap to SPY.
2. **Look-ahead.** Some ratios use prices from after the filing date, and the model choice uses
   returns that are not known yet. The strategy waits until every price exists, averages the three
   models and trades one day later.
3. **Survivorship.** The price table misses 173 of the 674 historical members. Missing prices come
   from Yahoo Finance where available, and SVB and First Republic are booked at −100%.
4. **Warm-up.** The strategy holds SPY until half the index has a live prediction (May–July 2020).

## How to reproduce

No API keys are needed.

```bash
# 1. Get the course data
git clone --depth 1 https://github.com/AI4Finance-Foundation/FinRL-Trading.git
cd FinRL-Trading/data && 7z x finrl_trading.7z && cd ../..
DATA=$PWD/FinRL-Trading/data

# 2. From this folder, prepare the data
pip install -r requirements.txt
python prepare_data.py \
    --db $DATA/finrl_trading.db \
    --fundamentals $DATA/fundamental_data_full.csv \
    --constituents $DATA/sp500_historical_constituents.csv \
    --output-dir ./prepared \
    --start 2015-01-01 --end 2025-09-30 --trade-start 2020-09-30

# 3. Train the models and run the notebook (~70 min; the last cell runs the strategy)
mkdir -p outputs
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    FinRL-Trading-Full-Workload.ipynb

# 4. Bias audit, robustness checks and report figures
python analysis/data_audit.py $DATA
python analysis/model_analysis.py
python analysis/robustness.py
python analysis/report_figures.py
(cd report && latexmk -pdf report.tex && latexmk -c)
```
