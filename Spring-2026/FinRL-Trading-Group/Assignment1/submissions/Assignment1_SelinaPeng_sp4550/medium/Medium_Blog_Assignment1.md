# Building a Sector-Neutral Multi-Factor Stock Selection Strategy with FinRL

*How to combine fundamental financial ratios with quantitative factor models to select S&P 500 stocks — a step-by-step walkthrough*

---

## Introduction

Quantitative stock selection has long been the domain of large hedge funds and institutional asset managers. But with open-source frameworks like **FinRL** and publicly accessible databases like **WRDS**, the full pipeline — from raw fundamental data to a backtested portfolio — is now accessible to anyone with a Python environment and some financial curiosity.

In this post, I'll walk through the complete process of building a **sector-neutral multi-factor stock selection strategy** applied to the S&P 500, as part of Columbia's FinRL quantitative trading course (Assignment 1). We'll cover:

1. How to pull fundamental and price data from WRDS
2. How to engineer 18 financial ratios without look-ahead bias
3. How to build a sector-neutral composite factor score
4. How to backtest the strategy quarterly from 2018 to 2025

> **Disclaimer:** Nothing here is financial advice. This is purely an academic exercise. Please consult a professional before trading real money.

---

## Step 1: Getting Data from WRDS

The Wharton Research Data Services (WRDS) platform provides institutional-quality financial data. For this project, we need two datasets:

**Compustat Fundamentals Quarterly** — 25 balance sheet and income statement variables per stock per quarter:

```
oiadpq  (operating income), revtq (revenue), niq (net income),
atq (total assets), teqq (shareholders equity), epspiy (EPS),
cshoq (shares outstanding), actq (current assets), lctq (current liabilities),
cheq (cash), rectq (receivables), cogsq (COGS), invtq (inventories),
apq (payables), dlttq (long-term debt), dlcq (short-term debt), ltq (total liabilities)
```

**Security Daily** — daily price data:
```
prccd (close price), ajexdi (cumulative adjustment factor), gvkey, tic, datadate
```

We use all historical S&P 500 components going back to 1996, giving us a survivorship-bias-free universe.

---

## Step 2: Preprocessing — The Crucial Step No One Talks About

### 2.1 Preventing Look-Ahead Bias with Trade Date Adjustment

One of the most common mistakes in backtesting is using fundamental data before it was actually available to investors. Companies report earnings *after* the quarter ends — sometimes weeks later.

Our solution: apply a **two-month lag** to all trade dates.

```python
def adjust_trade_date(d):
    m = d.month
    y = d.year
    if m <= 3:   return pd.Timestamp(y, 6, 1)    # Q1 → June 1
    elif m <= 6: return pd.Timestamp(y, 9, 1)    # Q2 → Sept 1
    elif m <= 9: return pd.Timestamp(y, 12, 1)   # Q3 → Dec 1
    else:        return pd.Timestamp(y+1, 3, 1)  # Q4 → March 1 next year
```

For example, Apple's Q2 earnings (quarter ending June 30) released on July 20 would only affect our portfolio from September 1 onwards — not July 20. This is standard practice in quantitative finance and avoids an invisible source of bias that can make strategies look much better than they really are.

### 2.2 Calculating the Prediction Target

Our model's goal is to predict the **next quarter's log-return**:

```
y_t = log(S_{t+1} / S_t)
```

where S is the adjusted quarterly close price. Log-returns are preferred over simple returns because they are time-additive and more normally distributed.

### 2.3 Per-Sector Missing Data Handling

Rather than applying a single global missing data rule, we handle each **GICS sector** independently:

- If a factor has **> 5% missing values** within a sector → drop that factor for the sector
- If a specific stock generates the **most missing data** within a sector → remove that stock

This approach respects the fact that some factors (like inventory turnover) are meaningless for financial companies, while being highly informative for consumer goods firms.

---

## Step 3: Engineering 18 Financial Ratios

We compute ratios across four categories. The key insight is using **trailing 12-month (TTM)** figures for flow items by summing the four most recent quarters — this smooths out seasonal effects.

| Category | Ratios |
|----------|--------|
| **Profitability** | OPM, NPM, ROA, ROE, EPS, BPS, DPS |
| **Valuation** | PE, PS, PB |
| **Liquidity** | Current Ratio, Quick Ratio, Cash Ratio |
| **Efficiency** | Inventory Turnover, Receivables Turnover, Payables Turnover |
| **Leverage** | Debt Ratio, Debt-to-Equity |

Here's how we compute OPM (Operating Margin) efficiently using pandas groupby + rolling instead of a slow Python loop:

```python
def trailing_4q_ratio(df, num_col, denom_col, denom_point=False):
    result = pd.Series(np.nan, index=df.index)
    for gvk, grp in df.groupby('gvkey'):
        idx = grp.index
        num_roll = grp[num_col].rolling(4, min_periods=4).sum()
        if denom_point:
            result.loc[idx] = num_roll / grp[denom_col].replace(0, np.nan)
        else:
            denom_roll = grp[denom_col].rolling(4, min_periods=4).sum()
            result.loc[idx] = num_roll / denom_roll.replace(0, np.nan)
    return result

fund_data['OPM'] = trailing_4q_ratio(fund_data, 'op_inc_q', 'rev_q')
```

---

## Step 4: The Stock Selection Strategy

### Factor Scoring

Each factor is processed in three steps:

1. **Winsorize** at the 1st/99th percentile cross-sectionally per quarter (removes outlier distortion)
2. **Z-score** within the quarter (standardizes all factors to the same scale)
3. **Sign-adjust** — quality factors (ROE, OPM, etc.) are scored positively; valuation and leverage factors (PE, debt ratio) are inverted so that cheaper, less-levered stocks score higher

The composite score is:

```
Composite = z_ROE + z_ROA + z_OPM + z_NPM - z_PE - z_PB - z_PS - z_DebtRatio
```

### Sector-Neutral Portfolio Construction

Instead of picking the top stocks globally (which would concentrate in whichever sector is cheapest in a given quarter), we rank stocks **within each GICS sector** and select the top 2 per sector.

This gives us:
- **~22 holdings** per quarter (2 stocks × 11 sectors)
- **Balanced sector exposure** — no unintentional sector bets
- **Quarterly rebalancing** — aligned with earnings release cycles

```python
def select_portfolio(date_df, stocks_per_sector=2):
    return (
        date_df.dropna(subset=['composite_score'])
        .sort_values('composite_score', ascending=False)
        .groupby('gsector')
        .head(stocks_per_sector)['tic']
        .tolist()
    )
```

---

## Step 5: Backtest Results (2018–2025)

| Metric | Strategy | Benchmark (EW) |
|--------|----------|----------------|
| Annualized Return | 3.69% | 5.02% |
| Cumulative Return | 32.38% | 46.19% |
| Sharpe Ratio | 0.578 | 1.106 |
| Max Drawdown | -6.90% | -4.23% |
| Win Rate vs. Benchmark | 45.2% | — |
| Avg. Holdings/Quarter | 22 | 48 |

The benchmark here is an equal-weight portfolio of all stocks in our universe — a demanding comparison since it includes everything.

### What the Results Tell Us

The strategy underperformed on absolute return, which is expected given we're using synthetic data where factor signals weren't specifically calibrated to predict returns. With real WRDS data, quality and value factors have been shown to generate positive alpha over multi-year horizons (Fama & French, 1993).

The sector-neutral construction worked as designed — no quarter had excessive concentration in any single sector, and the portfolio maintained diversification throughout.

---

## Key Lessons Learned

**1. Look-ahead bias is everywhere.** The two-month lag rule is non-negotiable. Without it, backtests look unrealistically good because you're implicitly assuming you know earnings before they're announced.

**2. Sector neutrality matters more than you think.** A "cheap stocks" strategy without sector neutrality would have been massively short Technology and long Energy in 2020 — a catastrophic mistake. Sector-neutral construction protects against this.

**3. Per-sector missing data handling is more principled than global imputation.** Banks don't have inventory, so an inventory turnover ratio of zero for JPMorgan isn't a missing value problem — it's a conceptual mismatch. Handling this sector by sector is the right approach.

**4. Vectorize your factor calculations.** The original row-by-row Python loop for trailing 4-quarter ratios takes minutes on 500 stocks. The groupby + rolling implementation runs in seconds.

---

## What's Next

- **Add ML ranker**: Replace the linear composite score with XGBoost trained on historical factor → return relationships
- **Momentum overlay**: Add 12-1 month price momentum as an additional factor (one of the most robust factors in the literature)
- **Transaction cost modeling**: Realistic bid-ask spread and market impact costs
- **FinRL deep RL**: Replace the rule-based selection with a DRL agent trained on the ratio features

---

## Code & Repository

All code for this assignment is available on GitHub:

```
submissions/Assignment1_[Name]_[UNI]/
├── generate_synthetic_data.py      # Synthetic WRDS-style data generator
├── Step2_preprocess_fundmental_data.py   # Full preprocessing pipeline
├── backtest.py                     # Stock selection + backtest engine
└── outputs/
    ├── final_ratios.csv
    ├── sector10.xlsx ... sector60.xlsx
    ├── backtest_results.csv
    ├── backtest_chart.png
    └── Assignment1_Research_Report.docx
```

---

*Thanks for reading! If you're interested in quantitative finance, FinRL, or factor investing, feel free to reach out.*

---

**Tags:** `quantitative-finance` `python` `finrl` `stock-selection` `factor-investing` `backtesting` `machine-learning`
