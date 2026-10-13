import sys
import pandas as pd

cons = pd.read_csv(sys.argv[1], parse_dates=["date"])
counts = cons.assign(n_members=cons.tickers.str.count(",") + 1)[["date", "n_members"]]
counts = counts[counts.date >= "2015-01-01"]
counts.to_csv("analysis/index_member_counts.csv", index=False)
print(f"wrote analysis/index_member_counts.csv: {len(counts)} dates, members {counts.n_members.min()}-{counts.n_members.max()}")
