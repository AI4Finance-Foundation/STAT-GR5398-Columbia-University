# Regime-Aware Portfolio Backtest (README)

## Project Summary
**Medium post:** https://medium.com/p/257e00507a06?postPublishedType=initial

This project builds and backtests a systematic equity strategy and compares performance against **QQQ** and **SPY**. I evaluate three strategy variants:

1. **Portfolio_Base (No Regime)**  
2. **Portfolio_Regime (Regime overlay using SPY MA200)**  
3. **Portfolio_Regime_off_0.2 (More defensive regime overlay with risk-off exposure = 0.2)**  

All three variants share the same alpha engine (**momentum selection + inverse-volatility weighting**). The regime-based variants differ only in how they **scale market exposure during risk-off periods**.


## Key Data

### Data Inputs
1. `output/step2/stock_selected.csv`  
   Contains the selected stock universe for each rebalance date (`trade_date`) and ticker (`tic`).

2. `daily.csv`  
   Contains daily stock prices and the adjustment factor `ajexdi`, which is used to compute split-adjusted close prices.

3. **Benchmarks:** `QQQ`, `SPY`  
   Downloaded via `yfinance` with `auto_adjust=True` (adjusted for splits and dividends).



## Return Definition

### Split-adjusted close
Let `close_adj` be the split-adjusted close price:

$$
P^{adj}_{i,t} = \frac{P_{i,t}}{ajexdi_{i,t}}
$$

### Close-to-close return
Daily close-to-close return is computed as:

$$
r_{i,t} = \frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}} - 1
$$



## Strategy Details

### 1) Portfolio Construction (Base)
On each rebalance date, portfolio weights are assigned using inverse-volatility sizing:

$$
w_{i,t} \propto \frac{1}{\sigma_{i,t}}, \qquad \sum_i w_{i,t} = 1
$$

where $$\sigma_{i,t}$$ is a rolling volatility estimate (e.g., 60 trading days). A weight cap is applied to avoid excessive concentration, and weights are renormalized to sum to one.


### 2) Backtest Mechanics (Industry Standard)

#### No look-ahead (execution lag)
Weights are executed with a one-day lag to avoid look-ahead bias:

$$
w^{exec}_{i,t} = w^{signal}_{i,t-1}
$$

#### Transaction costs (drift-adjusted turnover)
Turnover is computed using drift-adjusted weights:

$$
\text{Turnover}_t = \sum_i \left| w^{exec}_{i,t} - w^{drift}_{i,t-1} \right|
$$

Transaction costs are modeled as:

$$
\text{Cost}_t = c \cdot \text{Turnover}_t
$$

where $$c = 0.001$$ (10 bps).

#### NAV calculation
Portfolio NAV is computed by compounding net returns:

$$
\text{NAV}_t = \text{NAV}_{t-1}\left(1 + r^{net}_t\right), \qquad \text{NAV}_0 = 1
$$



## Regime Overlay (Portfolio_Regime / Portfolio_Regime_off_0.2)

### Regime signal (SPY vs MA200)
Risk-off is triggered when SPY is below its 200-day moving average:

$$
\text{MA200}_t = \frac{1}{200}\sum_{k=0}^{199} S_{t-k}
$$

Risk-off condition:

$$
S_t < \text{MA200}_t
$$

### Exposure scaling
The regime overlay scales the base strategy’s net return:

$$
r^{net,regime}_t = \alpha_t \cdot r^{net,base}_t
$$

where:
- $$\alpha_t = 1$$ in risk-on periods  
- $$\alpha_t = \alpha_{off}$$ in risk-off periods  

Settings:
- **Portfolio_Regime:** baseline (less defensive) regime setting  
- **Portfolio_Regime_off_0.2:** $$\alpha_{off} = 0.2$$ (more defensive)



## Results (Performance Metrics)

Metrics reported:
- Cumulative Return  
- Annual Return (CAGR)  
- Annual Volatility  
- Sharpe Ratio (rf = 0)  
- Max Drawdown  

| Strategy | Cumulative Return | Annual Return | Annual Volatility | Sharpe Ratio | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| Portfolio_Base | 4.4693 | 0.2112 | 0.2637 | 0.8589 | -0.4145 |
| Portfolio_Regime | 8.4385 | 0.2880 | 0.1956 | 1.3925 | -0.1867 |
| Portfolio_Regime_off_0.2 | **9.5034** | **0.3036** | **0.1879** | **1.5059** | **-0.1756** |
| QQQ | 4.1450 | 0.2028 | 0.2306 | 0.9166 | -0.3512 |
| SPY | 2.4158 | 0.1486 | 0.1858 | 0.8388 | -0.3372 |

**Key takeaway:** The regime overlay significantly reduces drawdown and volatility while improving annual return, leading to higher Sharpe ratios. The most defensive variant (`off = 0.2`) achieves the best overall performance.



## How to Run

1. Verify the required files exist:
   - `output/step2/stock_selected.csv`
   - `daily.csv`

2. Run the main notebook/script to:
   - build portfolio weights  
   - compute returns  
   - backtest with transaction costs  
   - download benchmarks (QQQ, SPY)  
   - generate metrics and performance plots  

### Outputs
- `output/step2/custom_strategy_metrics*.csv`  
- `output/step2/portfolio_comparison_chart*.png`  
