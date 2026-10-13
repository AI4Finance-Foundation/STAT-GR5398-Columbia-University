import glob
import sys
import numpy as np
import pandas as pd

RENAMES = {"BRK.B": "BRK-B", "ABC": "COR", "ANTM": "ELV", "FLT": "CPAY", "PKI": "RVTY", "RE": "EG",
           "WLTW": "WTW", "FBHS": "FBIN", "CDAY": "DAY", "NLOK": "GEN", "GPS": "GAP"}

CSV = "analysis/yf_missing_prices.csv"


def validate(prices):
    fr = pd.read_csv("prepared/final_ratios.csv", usecols=["tic", "date", "adj_close_q"], parse_dates=["date"])
    keep, dropped = {}, {}
    for t in prices.columns:
        s = prices[t].dropna()
        q = fr[(fr.tic == t) & (fr.date >= s.index[0]) & (fr.date <= s.index[-1])].set_index("date").adj_close_q
        yq = [s.loc[d + pd.Timedelta(days=1):] for d in q.index]
        ratio = pd.Series([v.iloc[0] for v in yq if len(v)], dtype=float) / q.values[:sum(len(v) > 0 for v in yq)]
        if len(ratio) and not 0.5 <= float(np.median(ratio)) <= 2.0:
            dropped[t] = float(np.median(ratio))
        else:
            keep[t] = prices[t]
    for t, r in dropped.items():
        print(f"DROPPED {t}: Yahoo price is {r:.1f}x the dataset's own price, a different security")
    return pd.DataFrame(keep)


if "--validate-only" in sys.argv:
    got = pd.read_csv(CSV, index_col=0, parse_dates=True)
else:
    import yfinance as yf
    db = set(pd.read_csv("prepared/daily.csv", usecols=["tic"]).tic.unique())
    pred = set()
    for f in glob.glob("results/sector*/df_predict_best.csv"):
        pred |= set(pd.read_csv(f, index_col=0, nrows=1).columns)
    miss = sorted(pred - db)
    found = {}
    for attempt in range(3): 
        todo = [t for t in miss if t not in found]
        if not todo:
            break
        data = yf.download([RENAMES.get(t, t) for t in todo], start="2019-01-01", end="2025-10-01",
                           auto_adjust=True, progress=False, group_by="ticker")
        for t in todo:
            y = RENAMES.get(t, t)
            if y in data.columns.get_level_values(0) and data[y]["Close"].notna().any():
                found[t] = data[y]["Close"].dropna()
    got = pd.DataFrame(found).sort_index()
    print(f"{len(miss)} predicted tickers lack DB prices; yfinance returned {got.shape[1]}")
valid = validate(got)
valid.to_csv(CSV)
print(f"{valid.shape[1]} validated fill-in series written to {CSV}")
