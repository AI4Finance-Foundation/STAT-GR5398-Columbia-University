# GR5398 26 Fall — Assignment 2: Selection, Valuation, Timing

Assignment 1 had you run one pipeline end to end. Assignment 2 builds one portfolio out of three
modules and asks a single research question: **how much does each module add?**

For the full design with diagrams, see the [Design Brief (PDF)](./GR5398_Assignment2_Design_EN.pdf).

| Module | What it decides | Who builds it |
| --- | --- | --- |
| **Selection** (选股) | Which stocks to hold: the 10 best-ranked growth_tech names each week | Provided by the course; **bonus**: retrain it ([section 8](#bonus-retrain-the-selection-model)) |
| **Valuation** (估值) | Is a candidate overpriced or underpriced? Drop the ones you would not buy | Valuation group |
| **Timing** (择时) | How much of the portfolio is invested, from 0% to 100% | Timing group |

You build **one** module. The other one stays at its baseline, so every result you report is the
marginal contribution of your own work.

| Track | Module | Why |
| --- | --- | --- |
| FinRobot | Valuation | Turn FinRobot's DCF engine into a point-in-time valuation you can backtest |
| FinRL-Trading | Timing | Position sizing is a textbook control problem; rule-based or RL |
| FinGPT | Timing (sentiment factor) | The bundled risk engine's 12% news factor is a constant 0.5 historically; build the real one |

**Due: Wednesday, Nov 25, 2026.**

**Teams of up to 3.** You may work alone or in a team of 2 or 3. A team builds **one** module
together and hands in one submission (see [What to submit](#10-what-to-submit)). "Valuation group"
and "timing group" below mean everyone building that module, not your team.

---

## 1. How the pieces fit

```
data/ (course bundle)
  fundamentals.db · daily prices · VIX & sector ETFs · FRED macro · monthly selection scores
        │
        ▼
selection/   this month's scores for every stock
        │
        ▼
valuation/   filter_candidates(as_of, candidates, data)   removes names you think are overpriced
        │                                                 (the next-ranked name takes the slot)
        ▼
selection/   target_weights(): top 10 growth_tech, weights 5%–15% by score
        │
        ▼
timing/      exposure(as_of, data) -> 0.0 ... 1.0         total invested = 0.95 × exposure
        │
        ▼
portfolio/   backtester: rebalance on the first session of each week at the open,
             6 bp per trade, whole shares, no leverage or shorting
        │
        ▼
output/      equity curve · trades · stats · A/B/C/D ablation
```

Every decision is made at the **close of the previous session** and filled at the next open.

---

## 2. Setup

1. Python **3.11 or newer**.
2. Install the dependencies (from `source_code/`):
   ```bash
   cd Fall-2026/Assignment2/source_code
   pip install -r requirements.txt
   ```
3. Download the **course data bundle**, [GR5398_A2_data.zip](https://www.dropbox.com/scl/fi/lsr76a9p76rwjzs1970gv/GR5398_A2_data.zip?rlkey=me93zy7c49hfa3yot6n8k2hpx&st=tmrxrmlv&dl=0)
   (29 MB, not in git), and unzip it inside `source_code/` so the folder is `source_code/data/`:
   ```bash
   curl -L -o GR5398_A2_data.zip "https://www.dropbox.com/scl/fi/lsr76a9p76rwjzs1970gv/GR5398_A2_data.zip?rlkey=me93zy7c49hfa3yot6n8k2hpx&st=tmrxrmlv&dl=1"
   unzip GR5398_A2_data.zip
   ```
   ```
   source_code/data/
   ├── fundamentals.db
   ├── prices/daily_adjusted.parquet
   ├── prices/market_series.parquet
   ├── macro/DGS10.parquet  T10Y2Y.parquet  BAA10Y.parquet  BAMLH0A0HYM2.parquet
   ├── selection/monthly_scores.parquet
   └── DATA_MANIFEST.json
   ```
4. Run the selection-only portfolio once. If this works, your setup is complete:
   ```bash
   python run_portfolio.py --period dev
   ```

Run every command from `source_code/`.

---

## 3. Code map

```
source_code/
├── selection/                 course selection model, full source; retrain it for a bonus
│   ├── selector.py              features, six regressors + stacking, monthly training
│   ├── strategy.py              monthly predictions -> ranked picks -> target_weights()
│   ├── allocation.py  ranking.py  config.py  configs/
│   ├── scores.py                replays the precomputed monthly scores (what backtests use)
│   ├── build_scores.py          retrains every month (~1.5 hours); bonus, see section 8
│   └── run_picks.py             prints the target portfolio for one date
├── valuation/
│   ├── student_valuation.py     ← the valuation group edits this
│   ├── examples.py              trailing-P/E method and per-share DCF, to copy from
│   └── finrobot/                FinRobot's DCF engine (Apache-2.0, do not edit)
├── timing/
│   ├── student_timing.py        ← the timing group edits this
│   ├── baselines.py             full, sma200, engine
│   └── engine.py                the 6-factor risk engine behind the engine baseline
├── portfolio/
│   ├── backtester.py            weekly backtest; valuation and timing plug in at lines 137–142
│   └── hooks.py                 connects your functions to the backtester and checks their output
├── shared/
│   ├── data_view.py             DataView: the only way your functions read data
│   ├── guard.py                 file and network access raise an error inside your functions
│   ├── store.py  prices.py  universe.py   fundamentals, prices, index membership
│   └── metrics.py  periods.py
├── run_portfolio.py             portfolio backtest, A/B/C/D ablation
├── evaluate_valuation.py        rank IC and range coverage of your valuation
├── evaluate_timing.py           your timing on SPY alone, against the three baselines
├── check_submission.py          automatic checks; run before you submit
└── student_data/                date-indexed files you build yourself (e.g. news sentiment)
```

**Start here**: read `portfolio/backtester.py` lines 128–145 (the weekly loop), then
`shared/data_view.py`, then your own module file. The template in your module file does nothing,
so `python run_portfolio.py --period dev --ablation --timing student` must print four identical
columns before you change anything.

---

## 4. The data, and the look-ahead rules

Your functions receive `data`, a `DataView` frozen at `as_of`. Every method returns only rows dated
on or before `as_of`:

| Call | Returns | Coverage |
| --- | --- | --- |
| `data.close(["SPY", "NVDA"], lookback=252)` | Split- and dividend-adjusted daily closes | 632 stocks + SPY, QQQ, 2016-06 → 2026-11-06 |
| `data.last_close(tickers)` | Latest close on or before `as_of` | |
| `data.market("^VIX")` | VIX, SPY and 11 sector ETFs (XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY) | XLC from 2018-06 |
| `data.macro("DGS10")` | FRED series: `DGS10`, `T10Y2Y`, `BAA10Y`, `BAMLH0A0HYM2` | `BAMLH0A0HYM2` only from 2023-09 |
| `data.fundamentals(tickers)` | One row per (ticker, quarter), filings known by `as_of` | 2006 → 2026-11-06 |
| `data.members()` | S&P 500 and Nasdaq-100 members on `as_of` | |
| `data.extra("news_sentiment")` | Your own `student_data/news_sentiment.parquet` | must have a `date` column |

Rules, enforced by the code and by `check_submission.py`:

- **Read data only through `data`.** Reading files or calling the network inside `exposure()`,
  `value()` or `filter_candidates()` raises `LookAheadGuardError`. Load a model you trained offline
  at import time, at the top of your file.
- **Your own data goes in `student_data/<name>.parquet`** with a `date` column meaning *the day
  that row became known*, and is read with `data.extra(name)`. A sentiment score for 2020-03-16
  built from news published on 2020-03-17 is a look-ahead, and the grader will check.
- **Three fundamentals columns are removed**: `trade_price` (a close two months after quarter end,
  often after `as_of`), `adj_close_q` (the leak found in Assignment 1) and `y_return` (next
  quarter's return, a training label).
- **Do not use the `pe` and `ev_multiple` columns.** Their basis is not verified: for NVDA's
  2026-09-30 row, `pe` is 19.9 while price over trailing-four-quarter EPS is about 28.7. Compute
  multiples yourself from `EPS`, `revenue_per_share` and prices.
- **Per-share and ratio columns are single-quarter values.** Add the last four quarters for TTM.
- **No LLM-generated numbers for past dates.** The model was trained on what happened next.

---

## 5. Evaluation periods

| Period | Dates | What it is for |
| --- | --- | --- |
| `dev` | 2017-01-03 → 2021-12-31 | Build and tune freely |
| `val` | 2022-01-03 → 2023-12-29 | Choose your final configuration here, then commit it |
| `test` | 2024-01-02 → 2026-10-09 | Run **once**, after the commit |
| `forward` | 2026-10-12 → | Live period; the course re-runs every submission on new data |

Every run that touches the test period is appended to `output/test_runs.jsonl` with the time and
your git commit. Report the number of test runs. A configuration chosen after looking at test
results is not a result.

All scripts take `--period dev|val|test|forward`, or `--start YYYY-MM-DD --end YYYY-MM-DD`.

---

## 6. Valuation group

### Task

Implement in `valuation/student_valuation.py`:

```python
def value(as_of, tickers, data) -> pd.DataFrame:
    # columns: ticker, fair_value, low, high, mispricing, method_spread
    # mispricing = fair_value / price - 1   (positive = undervalued)
    # method_spread = highest / lowest fair value across your methods

def filter_candidates(as_of, candidates, data):
    # candidates: tuple of selection.Prediction(symbol, bucket, rank_score, weight_score)
    # return the ones you keep; removed names are replaced by the next-ranked ones
```

Use **at least three methods**, all point-in-time:

| Method | Starting point | What you must change |
| --- | --- | --- |
| DCF | `valuation/finrobot/dcf.py` `calculate_dcf`, `calculate_sensitivity` | Risk-free rate from `data.macro("DGS10")`, not FinRobot's hard-coded 2026 value; a growth path from history or your own model, not analyst consensus; a range from the sensitivity grid, not a fixed ±20% |
| EV/EBITDA, own history | FinRobot [`valuation_aggregator.py:999`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/finrobot_desktop/finrobot/engine/compute/operators/valuation_aggregator.py#L999), [`historical_valuation.py:85`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/finrobot_desktop/finrobot/engine/primitives/historical_valuation.py#L85) | Past multiples from past prices and past filings only |
| Peer multiple | `valuation/examples.py` `trailing_pe_value` | Trailing, not forward, earnings; peers = same-sector index members on `as_of` |

`valuation/examples.py` has a working trailing-P/E method and a per-share DCF. The database has
per-share figures but no totals, so value everything per share (`shares_outstanding = 1`):

| `DCFInputs` field | Source |
| --- | --- |
| `revenue_base` | `revenue_per_share`, last four quarters summed |
| `revenue_growth_rates` | Your estimate. The most important assumption; justify it |
| `ebitda_margin` | `ebitda_margin`, median of the last four quarters |
| `da_pct_revenue` | `ebitda_margin − operating_margin` (D&A is not stored) |
| `capex_pct_revenue` | `capex_per_share / revenue_per_share`, four quarters |
| `nwc_pct_revenue` | Not in the database; set and state an assumption |
| `tax_rate` | `effective_tax_rate`, clipped to 0–35% |
| `risk_free_rate` | `data.macro("DGS10")` / 100 |
| `beta` | Weekly returns against SPY over 5 years, then `adjust_beta_blume` |
| `equity_risk_premium`, `terminal_growth_rate` | Stated constants |
| `cost_of_debt` | DGS10 plus a stated spread |
| `debt_ratio` | `debt_to_capital` |
| `net_debt` | `interest_debt_per_share − cash_per_share` |

FinRobot code you may **not** use as-is: the default rates in
[`dcf_seed.py:43–59`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/finrobot_desktop/finrobot/engine/compute/operators/dcf_seed.py#L43),
the consensus growth in [`pipelines/dcf.py:33`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/finrobot_desktop/finrobot/engine/pipelines/dcf.py#L33),
and the forward-P/E peer method in [`valuation_aggregator.py:514`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/finrobot_desktop/finrobot/engine/compute/operators/valuation_aggregator.py#L514).
All three only know about today.

### Evaluate

```bash
python evaluate_valuation.py --period dev              # your value()
python evaluate_valuation.py --period dev --example    # the trailing-P/E example, for reference
python run_portfolio.py --period dev --valuation       # effect on the portfolio
```

`evaluate_valuation.py` values the top 20 growth_tech candidates on the first session of every
month and reports the **rank IC** of `mispricing` against the next 63 / 126 / 252 session returns,
the share of prices that end inside `[low, high]` a year later, and your coverage.

For reference, the trailing-P/E example has a 63-session rank IC of **−0.17 (t = −3.2)** over
2017–2019: the growth stocks that look cheapest on P/E did *worse*. Expect valuation filters to hurt
a growth portfolio. An honest negative result, explained, earns full credit.

---

## 7. Timing group

### Task

Implement in `timing/student_timing.py`:

```python
def exposure(as_of, data) -> float:
    # 0.0 = all cash, 1.0 = fully invested; called once a week
```

Optional: the backtester's two-layer stop (15% below the 10-session high or the entry price) is
switched on with `--stops`; its parameters are in `portfolio/backtester.py` lines 38–53.

### Baselines

| Baseline | Rule | dev period on SPY: Sharpe / max drawdown |
| --- | --- | --- |
| `full` | Always 1.0 | 0.98 / −33.7% |
| `sma200` | 1.0 when SPY is above its 200-session average, else 0 | 1.07 / −15.9% |
| `engine` | `timing/engine.py`: 6-factor risk score × 16% volatility target, clamped to [0.10, 1.0] | 1.18 / −13.7% |

The engine has two dead factors in history: FRED publishes only about three years of
`BAMLH0A0HYM2`, so the high-yield factor is neutral before 2023-09, and the news factor is fixed at
0.5. `BAA10Y` covers the whole period and is the obvious substitute for the first; the FinGPT
group builds the second.

A warning from the baselines: on the **stock portfolio** over 2017–2019, `sma200` made the maximum
drawdown *worse* (−39% vs −28%). It switched five times between late October and early December
2018 and re-entered only in February 2019, after most of the rebound. Look at your whipsaws, not just your averages.

### Evaluate

```bash
python evaluate_timing.py --period dev                       # SPY only, against all baselines
python run_portfolio.py --period dev --timing student        # on the stock portfolio
```

`evaluate_timing.py` reports Sharpe, drawdown, Calmar, average exposure, number of changes, and a
crisis table for 2018 Q4, March 2020 and 2022: how much of each fall you avoided and how much of
the rebound you missed.

### FinGPT group: the sentiment factor

Build a daily market sentiment series from historical news with your fine-tuned FinGPT model, save
it as `student_data/news_sentiment.parquet` (`date`, `value` in [0, 1]), and use it inside
`exposure()` through `data.extra("news_sentiment")`. Show that it improves timing over the same
rule without it. `date` must be the day the news was available, not the day it describes.

---

## 8. Ablation (required)

```bash
python run_portfolio.py --period dev --ablation --timing student
```

| Run | Selection | Valuation | Timing | Answers |
| --- | --- | --- | --- | --- |
| A | ✓ | — | full | The selection model alone |
| B | ✓ | ✓ | full | What valuation adds |
| C | ✓ | — | yours | What timing adds |
| D | ✓ | ✓ | yours | Whether they add up or cancel |

The valuation group runs it with `--timing full`; the timing group's B equals A. The table is saved
to `output/ablation_<period>.csv`. Report it for `dev`, `val` and (once) `test`.

### Bonus: retrain the selection model

The selection model is provided, and the course scores stay the same for everyone so that every
team measures against the same run A. For a bonus you may retrain it and try to improve it, as an
extra section of your report. It never replaces the required work, and your module file must
still run on the course scores, which is what the grader uses.

1. Copy `selection/configs/growth.json` to your own file and change the `model` section, or change
   the features and models in `selection/selector.py`. Changing only `parameters` (number of picks,
   weight limits) needs no retraining: it already applies in every backtest.
2. Retrain into your own file. **Always pass `--out`**; without it the course scores are
   overwritten. One month takes 1–4 minutes, so all 119 months take hours on a laptop: use
   `--start/--end` for a shorter range first. An interrupted run resumes where it stopped.
   ```bash
   python selection/build_scores.py --config my_growth.json --out output/retrained/scores.parquet --workers 4
   python run_portfolio.py --period dev --scores output/retrained/scores.parquet
   ```
3. **Retrain the unchanged `growth.json` too**, in the same environment, and compare against that.
   Scores depend on library versions: with newer scikit-learn and xgboost than the course build
   used, an unchanged model already gives different scores (rank correlation 0.88–0.96 with the
   course file). The course scores were built with scikit-learn 1.9.0, xgboost 3.4.0,
   lightgbm 4.7.0, numpy 2.5.2 and pandas 2.3.3.
4. Report both sets of results, and submit your config and the code changes.

---

## 9. Research report

Cover:

+ **Design**: what your module does and why, every constant and where it came from
+ **Data audit**: what you checked about the inputs you use, and what you found
+ **Look-ahead audit**: why nothing in your module knows the future, specifically
+ **Results**: your evaluation script's output and the ablation table for dev, val and test,
  against every baseline, with Sharpe, CAGR, max drawdown, volatility, turnover
+ **Robustness**: parameter sensitivity; how results change across the three crises (timing) or
  across years (valuation)
+ **The number of test-period runs**, from `output/test_runs.jsonl`
+ **What did not work**, and what you would do next
+ **Judgment**, written by you: would you put money behind this module? Why or why not?

---

## 10. What to submit

One submission per team. Name the folder after every member's UNI:
`submissions/Assignment2_UNI1_UNI2_UNI3/` (alone: `Assignment2_Name_UNI`). Include:

1. Your module file: `student_valuation.py` **or** `student_timing.py`. The grader copies it into
   `source_code/` and runs it against the course data; it must run there unchanged.
2. Any `student_data/*.parquet` it reads, and the code that built them.
3. `check_submission.py` output showing no FAIL.
4. The `output/` files behind your report: the evaluation CSV, `ablation_*.csv`, `test_runs.jsonl`.
5. Your research report (PDF).
6. `README.md`: every member's name and UNI and what each person did, the exact commands to
   reproduce every number, your git commit for the final configuration, and the link to your
   **published** Medium blog.

Do **not** commit the data bundle.

> [!WARNING]
> ⚠️ **DO NOT SUBMIT YOUR OWN API KEY ONTO GITHUB!!!**

---

## 11. Grading (100)

| Item | Points |
| --- | --- |
| Module design and correctness | 25 |
| Evaluation against every baseline; numbers match your output files | 20 |
| Data and look-ahead audit (including LLM look-ahead) | 15 |
| Ablation and marginal-contribution analysis | 15 |
| Test-period result: stability relative to baselines, not raw return | 5 |
| Report quality and your own judgment | 10 |
| Reproducibility and `check_submission.py` | 10 |
| **Bonus**: retrain the selection model ([section 8](#bonus-retrain-the-selection-model)) | extra, on top of 100 |

Test-period return is deliberately worth little. A careful study of a module that does not beat
its baseline scores higher than a lucky one that does.

**AI tools** are allowed for code. The *Judgment* section must be your own writing, and
submissions containing unedited agent text (e.g. "the user asked…", unfilled placeholders) lose
points. Every team member may be asked to walk through the module in a short meeting.
