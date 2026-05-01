# Assignment 2 Report  
Adaptive Rotation Strategy with QQQ Rolling Selection

## 1. Strategy Overview

In this assignment, I implemented the Adaptive Rotation Strategy and replaced the original fixed Growth Tech asset pool with a dynamic stock selection from Assignment 1 (QQQ rolling selection).

The idea is that instead of always using a fixed set of technology stocks (e.g. AAPL, MSFT, NVDA), the strategy dynamically selects stocks based on predicted returns, which allows the growth component to adapt to changing market conditions.

The strategy works as follows:

- Rebalance weekly
- Use past 12 weeks return to calculate Information Ratio (IR)
- Select the asset group with the highest IR
- Within that group, select top 2 assets using residual momentum
- Hold equally weighted positions until next rebalance

The four asset groups are:

- Growth (QQQ rolling selection)
- Real Assets
- Defensive Assets
- Default Assets

---

## 2. Backtest Result

### Equity Curve

The following figure shows the strategy equity curve compared to QQQ:

- Blue: Adaptive Rotation + QQQ Rolling Selection
- Orange: QQQ benchmark

Overall, the strategy performs slightly better than QQQ towards the end of the sample period, although for a long period it underperforms.

The strategy shows a more volatile path compared to QQQ, but it is able to capture strong upside in certain periods.

---

## 3. Drawdown Analysis

From the drawdown chart:

- The strategy generally experiences deeper drawdowns than QQQ
- However, in some periods (e.g. 2025), it avoids large losses compared to QQQ
- The drawdown recovery speed is relatively fast

This suggests that while the strategy takes more risk, it also has some ability to rotate into safer assets.

---

## 4. Asset Group Selection Behavior

From the group distribution chart:

- Growth_QQQ_Rolling is selected most frequently
- Default and Defensive groups are also used regularly
- Real Assets are selected less often

This indicates that:

- The model tends to favor growth stocks when conditions are favorable
- Defensive and default assets help stabilize the portfolio during weaker periods

---

## 5. Discussion

### Is the result reasonable?

Yes, the result is generally reasonable:

- The strategy does not unrealistically outperform the benchmark (which is good)
- It shows periods of both underperformance and outperformance
- The equity curve is noisy and not overly smooth, which is typical for a realistic strategy

Compared to the reference implementation (shown in the provided example), my strategy:

- Has lower overall return
- Has more fluctuations
- But still follows a similar pattern of regime switching

### Key limitation

One limitation is that the growth signal comes from fundamental data, which is relatively low frequency. Therefore, the predictions are forward-filled and do not change daily.

This may cause the strategy to hold similar stocks for extended periods.

---

## 6. Conclusion

In this assignment, I successfully combined QQQ rolling stock selection with Adaptive Rotation Strategy.

The modified strategy is able to dynamically adjust its growth exposure and shows reasonable performance compared to the benchmark.

Future improvements may include:

- Using higher frequency signals
- Improving residual momentum calculation
- Adding stronger risk management rules
