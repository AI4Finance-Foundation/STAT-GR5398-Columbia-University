# portfolio_strategy/config.py

# Data start is earlier because momentum/volatility signals need lookback history.
DATA_START_DATE = "2025-01-01"

# Actual backtest evaluation window.
BACKTEST_START_DATE = "2026-01-01"
BACKTEST_END_DATE = "2026-04-30"

BENCHMARKS = ["SPY", "QQQ", "XLK"]

TECH_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "AMD",
    "CRM", "ORCL", "ADBE", "NFLX", "TSLA", "COST", "NOW", "PANW",
    "CRWD", "SNOW", "INTU", "CSCO", "QCOM", "TXN", "MU", "AMAT",
    "LRCX", "KLAC", "ANET", "PLTR", "IBM", "INTC"
]

TOP_N = 15
REBALANCE_FREQ = "Q"

# 10 bps transaction cost
TRANSACTION_COST = 0.001