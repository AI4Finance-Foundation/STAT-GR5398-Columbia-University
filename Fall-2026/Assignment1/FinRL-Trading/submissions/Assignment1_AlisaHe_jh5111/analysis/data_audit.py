import pandas as pd, numpy as np
import os, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINRL_DATA", "../FinRL-Trading/data")
fr = pd.read_csv("prepared/final_ratios.csv", parse_dates=["date", "reportdate"])
cons = pd.read_csv(f"{D}/sp500_historical_constituents.csv", parse_dates=["date"])
latest = set(cons.iloc[-1]["tickers"].split(","))
print("Constituents file last date:", cons["date"].iloc[-1].date())
gone = sorted(set(fr.tic) - latest)
print(f"Tickers in sample: {fr.tic.nunique()}, not in the index on the last date: {len(gone)}")
last_seen = fr[fr.tic.isin(gone)].groupby("tic").date.max().sort_values()
print("Examples of tickers that left the index but stay in sample (last quarter in sample):")
print(last_seen.head(15).to_string())

lag = (fr.reportdate - fr.date).dt.days
print(f"\nFiling lag (reportdate - quarter end): median {lag.median():.0f}d, min {lag.min()}d, max {lag.max()}d, share <=0: {(lag<=0).mean():.4f}")

d = pd.read_csv("prepared/daily.csv", usecols=["tic", "datadate", "open_adj", "close_adj"], parse_dates=["datadate"])
d = d[d.datadate >= "2020-05-01"].sort_values(["tic", "datadate"])
d["intraday"] = d.close_adj / d.open_adj - 1
d["c2c"] = d.groupby("tic").close_adj.pct_change()
ew = d.groupby("datadate")[["intraday", "c2c"]].mean().dropna()
for c in ew:
    print(f"Equal-weight universe, {c:9s}: cumulative NAV {(1+ew[c]).prod():.2f}")

import glob
pred_tics = set()
for f in glob.glob("results/sector*/df_predict_best.csv"):
    pred_tics |= set(pd.read_csv(f, index_col=0, nrows=1).columns)
price_tics = set(d.tic.unique())
nopx = sorted(pred_tics - price_tics)
print(f"\nTickers with predictions: {len(pred_tics)}; with price data: {len(pred_tics & price_tics)}; "
      f"without price data: {len(nopx)}")
w = pd.read_csv("outputs/portfolio_weights.csv", index_col=0, parse_dates=True)
w = w.loc[:"2025-09-30"]
share = w[[c for c in w.columns if c not in price_tics]].sum(axis=1)
print(f"Baseline weight on tickers without prices (return silently = 0): mean {share.mean():.2%}, max {share.max():.2%}")
print("Examples:", nopx[:20])
