# GR5398 26 Spring: FinRL-Trading Quantitative Trading Strategy Track

## Assignment 2

### 0. Target

In this assignment 2, we would like you to implement our latest strategy: **[Adaptive Rotation Strategy](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/src/strategies),** and use what you did in Assignment 1 (QQQ rolling selection) to improve this strategy. You can also find our paper for our latest FinRL-X in [ArXiv](https://arxiv.org/abs/2603.21330).

To be specific, you need to:

+ Implement this strategy by yourself
+ Replace stock selection part with QQQ rolling selection strategy
+ Try to find some signal/alpha that can improve this strategy (Optional)

You should write a medium blog with your backtest equity curve (and QQQ curve for comparison) and your analysis of this strategy's performance and mechanism, like you did in Assignment 1.

Assignment Due Day: April 20, 2026

### 1. Adaptive Rotation Strategy

This strategy's methodology is based on selecting different assets in different market regimes. Also, this strategy has strict asset-selection logic with several risk management layers, which makes this strategy performs well on paper-trading.

#### 1.1 Asset Groups

**Adaptive Rotation Strategy** has 4 different asset pools for different regimes:

+ Growth Tech: AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA (You may need to change to rolling selection logic)
+ Real Assets: XOM, CVX, FCX, BHP, GLD, SLV
+ Defensive Assets: TLT, IEF, XLU, XLV, IAU, SHY, UUP
+ Default Assets: SPY, QQQ, IAU, XLU, XLV

By dividing these assets into different groups, we can select different industry/field in different regimes to earn money and avoid loss.

#### 1.2 Assets Selection

First, this strategy rebalances/reselects assets per week, using **Information Ratio** in past 12 weeks as indicator to choose the highest IR asset group. Also, inside this group, this strategy uses **Residual Momentum** to select specific 2 assets in this group.

#### 1.3 Risk Management & Timing

Adaptive Rotation Strategy combines slow regime and fast risk-off layers as the indicator of market regime. For loss control, this strategy uses the simplest logic that sets up a threshold of loss.

### 2. What should you do?

First, you need to fully understand this pipeline and this strategy. This is a good start of quantitative strategy design, and we expect you to learn some fundamental ideas on how to design a strategy by yourself. Also, you need to combine 2 different strategies to find a way to improve.

Then, if you have enough capability, you should try to modify some parts of this strategy to perform better than the original one.

### 3. Your report

Your report/medium blog should includes the following parts:

+ Your understanding of this strategy and backtest curves
+ If the new combination performs better
+ Optional: your modification and improvement
+ Optional: new ideas
