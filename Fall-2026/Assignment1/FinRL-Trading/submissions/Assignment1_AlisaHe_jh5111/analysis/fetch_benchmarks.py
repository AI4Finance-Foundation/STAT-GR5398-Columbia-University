import pandas as pd, yfinance as yf

px = yf.download(["SPY", "QQQ", "RSP"], start="2019-01-01", end="2025-10-01", auto_adjust=True, progress=False)["Close"]
px.to_csv("analysis/benchmark_prices.csv")
print(f"wrote analysis/benchmark_prices.csv: {px.shape[0]} days, {list(px.columns)}, "
      f"SPY NAV 2020-05-01..2025-09-30 = {px.SPY.loc['2025-09-30'] / px.SPY.loc['2020-05-01']:.6f}")
