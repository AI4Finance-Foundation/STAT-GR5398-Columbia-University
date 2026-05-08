# A Multi-Stage AI Equity Strategy Combining FinRL Stock Selection and Point-in-Time FinRobot Reweighting

## Abstract

This report presents a multi-stage AI-driven equity strategy that combines **FinRL** and **FinRobot** in a sequential portfolio construction framework. FinRL is used as the primary stock selection engine, while FinRobot is used as a point-in-time research overlay to reweight selected stocks according to business quality, valuation, recent catalysts, and risk. The central hypothesis is that machine-learning-based selection can be improved by a second-stage research filter that reallocates capital toward higher-conviction positions and away from weaker names. Using weekly rebalancing from January 2026 through April 2026, the combined strategy achieved a **total return of 18.91%**, outperforming both **FinRL-only at 14.20%** and **SPY at 5.48%**. It also delivered the highest **Sharpe ratio (2.36)** among the evaluated strategies. The results suggest that modular AI systems with distinct functional roles may outperform single-layer approaches in short-horizon portfolio management.

## 1. Strategy Overview

The strategy was designed as a two-stage decision system. The first stage uses FinRL to rank and select a weekly equity portfolio from a restricted universe of large-cap stocks. The second stage uses FinRobot as a research and validation module that does not alter membership directly, but instead adjusts weights assigned to the stocks already chosen by FinRL.

This design was motivated by the observation that machine-learning selection models can identify promising candidates without necessarily distinguishing which candidates deserve the highest position sizes. A stock may exhibit favorable technical or predictive characteristics while also suffering from valuation excess, weak short-term catalysts, or deteriorating fundamentals. Therefore, a second-stage filter based on point-in-time research evidence may improve portfolio quality without replacing the original selection process.

The strategy uses the following modules:

- **FinRL** for stock selection
- **FinRobot** for post-selection research scoring and portfolio reweighting
- **Yahoo Finance** for daily adjusted backtest pricing
- **FMP-derived FinRobot outputs** for fundamentals, valuation, and news signals

## 2. System Architecture

The system architecture is sequential and modular. FinRL receives the stock universe and produces weekly selected names. These selected names are then passed into a FinRobot wrapper that extracts point-in-time research information from historical financial statements and historical news. FinRobot then produces a composite score that maps to a portfolio weight multiplier. The adjusted portfolio is then normalized and backtested against SPY.

The pipeline can be summarized as follows:

1. Universe and market features are processed by FinRL.
2. FinRL selects five stocks per rebalance date.
3. FinRobot evaluates each selected stock using only information available as of that date.
4. The FinRobot score is translated into a portfolio action.
5. Position weights are rescaled and the portfolio is backtested.

This architecture preserves interpretability because each module has a clearly defined role. FinRL determines inclusion, while FinRobot determines conviction.

## 3. Methodology

### 3.1 Stock Universe and Rebalancing

The backtest used a weekly selected portfolio over **17 rebalance dates** from **2026-01-02 to 2026-04-24**. Each rebalance date contained **5 selected holdings**, resulting in **85 total selected rows** across the sample.

The strategy rotated among **11 large-cap stocks**:

`AMAT, AMD, AMZN, COST, CSCO, GOOGL, ISRG, META, NFLX, TSLA, TXN`

### 3.2 FinRL Selection Layer

FinRL provided the base portfolio. The selected names were initially equally weighted within each rebalance date. This equal-weight baseline serves as the reference case for comparison against the reweighted strategy.

### 3.3 FinRobot Research Overlay

The FinRobot layer converted real FinRobot outputs into a structured, point-in-time score. The score consisted of four components:

- **Business quality score**
- **Valuation score**
- **Catalyst score**
- **Risk score**

The final score was defined as:

```text
FinRobot Score =
0.45 * Business Quality
+ 0.25 * Valuation
+ 0.20 * Catalyst
+ 0.10 * Risk
```

#### Business Quality

This component included revenue growth, EBITDA margin, contribution margin, EPS, free cash flow quality, operating cash flow margin, leverage, liquidity, and ROE. The score blended absolute company quality with peer-relative ranking among the selected names.

#### Valuation

This component used peer-relative PE ratio, EV/EBITDA, and free cash flow yield. Lower valuation multiples and higher free cash flow yield improved the score.

#### Catalyst and Risk

Catalyst and risk were computed using a rolling **30-day news window** ending at the rebalance date. Only articles published on or before the decision date were included. This eliminated look-ahead bias in the news layer.

Risk terms included items such as lawsuits, investigations, recalls, downgrades, margin pressure, weak guidance, and execution risk. The risk score was normalized to avoid mechanically penalizing firms solely for having higher news volume.

### 3.4 Portfolio Construction

The FinRobot score was mapped into four portfolio actions:

- `strong_keep` with multiplier `1.20`
- `keep` with multiplier `1.00`
- `reduce` with multiplier `0.70`
- `reject` with multiplier `0.00`

These multipliers were applied to the equal-weight FinRL portfolio, after which weights were renormalized within each rebalance date.

### 3.5 Risk Rules

The strategy incorporated the following controls:

- long-only implementation
- weekly rebalancing
- no look-ahead in financial statements via filing `acceptedDate`
- no look-ahead in news via publication timestamp filtering
- bounded sizing through multiplier-based reweighting

## 4. Backtest Results

The backtest compared three strategies:

- FinRL-only
- FinRL + no-lookahead FinRobot
- SPY

The performance results are reported below.

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Volatility |
|---|---:|---:|---:|---:|
| FinRL-only | 14.20% | 1.91 | -13.11% | 23.04% |
| FinRL + no-lookahead FinRobot | 18.91% | 2.36 | -13.66% | 24.12% |
| SPY | 5.48% | 1.24 | -8.89% | 14.23% |

The combined strategy achieved the highest return and highest Sharpe ratio. While its volatility and maximum drawdown were slightly higher than those of FinRL-only, the increase in return was sufficient to improve risk-adjusted performance overall.

By the end of the backtest, cumulative wealth grew to:

- **1.189** for FinRL + FinRobot
- **1.142** for FinRL-only
- **1.055** for SPY

Daily directional consistency also improved. The combined strategy achieved a daily win rate of **63.75%** excluding flat days, compared with **57.50%** for FinRL-only and **51.85%** for SPY.

## 5. Module Contribution

The contribution analysis suggests that FinRobot improved results primarily through better weight allocation rather than through wholesale changes in portfolio composition.

The most positive active contributions came from the following adjustments:

- overweighting **AMD**
- underweighting or rejecting **TSLA**
- overweighting **AMAT**
- overweighting **TXN**
- modest overweighting of **GOOGL** and **AMZN**

The most important finding is that TSLA was rejected **7 times**, which was beneficial because TSLA underperformed during the sample. This indicates that FinRobot added value by cutting exposure to weak names, not merely by increasing exposure to winners.

Therefore, the combined system appears superior to either component alone in this sample:

- **FinRL alone** selected viable candidates but treated them too uniformly
- **FinRobot alone** would not have provided the initial candidate generation logic
- **FinRL + FinRobot together** produced better sizing and better overall performance

## 6. Failure Analysis

Despite the favorable results, several limitations remain.

First, the sample period is short. The backtest spans only **81 trading days** and **17 rebalance periods**, so the observed outperformance may not generalize to other market conditions.

Second, the stock universe is narrow and concentrated. The strategy trades only 11 large-cap names and holds 5 at a time, which increases sensitivity to stock-specific outcomes.

Third, the overlay can still make timing errors. For example, higher weight in **META** did not translate into positive active contribution, indicating that strong research characteristics do not always produce favorable short-term price behavior.

Fourth, concentration risk remains present. The top holding weight occasionally exceeded **30%**, which is reasonable for a concentrated portfolio but still creates fragility.

Finally, the current study evaluates a fully implemented FinRL + FinRobot pipeline. A FinGPT-based sentiment layer was considered as a possible extension, but it is not included in the final audited results of this report.

## 7. Conclusion

This project demonstrates that a modular AI portfolio architecture can improve performance over both a benchmark index and a simpler single-layer baseline. By combining FinRL stock selection with a point-in-time FinRobot reweighting overlay, the strategy achieved **18.91% total return**, outperforming **FinRL-only (14.20%)** and **SPY (5.48%)** during the backtest period.

The main reason for the improvement was that the system separated two distinct tasks: candidate generation and conviction sizing. FinRL identified promising stocks, while FinRobot added a structured research perspective that rewarded stronger business quality, better valuation, and more favorable recent catalysts while reducing exposure to weaker names.

Future work should include a longer backtest horizon, a larger stock universe, more formal risk controls, and a fully audited FinGPT sentiment layer. Nonetheless, within the tested sample, the combined system provided a clear and interpretable improvement over the baseline approaches.
