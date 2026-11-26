"""ANET's frozen validation signals, observed weekly fills, and open-to-open P&L."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
LOCK = OUT.parents[1]
ROOT = LOCK.parents[1]
cfg = json.loads((LOCK / "protocol.json").read_text())
start = pd.Timestamp(cfg["start"])
last = pd.Timestamp("2022-06-06")
end = pd.Timestamp("2022-06-13")
assert cfg["split"] == "val" and cfg["K"] == 1.983 and start <= last < end <= pd.Timestamp(cfg["end"])
positions = pd.read_csv(LOCK / "attribution/val_positions.csv", parse_dates=["day"])
weeks = pd.DatetimeIndex(positions.loc[positions.day.between(start, end), "day"].sort_values().unique())
pos = positions.loc[positions.ticker.eq("ANET")].set_index("day").reindex(weeks)
sig = pd.read_csv(ROOT / "weighted_valuation_research/val_signals.csv", parse_dates=["day","as_of"])
sig = sig.loc[sig.ticker.eq("ANET") & sig.day.isin(weeks)].set_index("day")
tally = pd.read_csv(ROOT / "fable_review/phase2_checks/tally_dev_val.csv",
                    usecols=["date","as_of","ticker","rank"],parse_dates=["date","as_of"])
rank = tally.loc[tally.ticker.eq("ANET") & tally.date.isin(weeks)].set_index("date")
np.testing.assert_array_equal(sig.loc[weeks].as_of,rank.loc[weeks].as_of)
prices = pd.read_parquet(ROOT/"valuation_candidate/source_code/data/prices/daily_adjusted.parquet",
                         columns=["date","open","close"],
                         filters=[("symbol","==","ANET"), ("date",">=",sig.as_of.min()),("date","<=",end)])
px = prices.set_index("date").sort_index()
df = pd.DataFrame(index=weeks)
df.index.name = "rebalance"
df["rank"] = rank["rank"]
for col in ["as_of","price","dcf","peer","hist_mid","sensitive","adaptive_value"]:
    df[col] = sig[col]
assert (df.as_of < df.index).all()
np.testing.assert_allclose(df.price, px.loc[df.as_of].close)
centers = df[["dcf","peer","hist_mid"]]
df["span"] = centers.max(axis=1)/centers.min(axis=1)
df["estimate_over_price"] = df.adaptive_value/df.price
df["veto"] = df.estimate_over_price.lt(.85) & df.span.le(cfg["K"])
df["open"] = px.open
df["next_open"] = df.open.shift(-1)
for arm in ["baseline","strategy"]:
    qty = pos[f"{arm}_shares"].fillna(0)
    df[f"{arm}_target_portion"] = pos[f"{arm}_target"].fillna(0)
    df[f"{arm}_shares"] = qty
    df[f"{arm}_net_fill"] = qty.diff().fillna(qty)
    df[f"{arm}_pnl_to_next_open"] = qty*(df.next_open-df.open) - .0006*df[f"{arm}_net_fill"].abs()*df.open
df["delta_usd"] = df.strategy_pnl_to_next_open - df.baseline_pnl_to_next_open
df = df.loc[:last]
assert df.index[df.veto].strftime("%Y-%m-%d").to_list() == ["2022-01-03","2022-03-28","2022-04-04","2022-04-11"]
df.to_csv(OUT/"weekly_timeline.csv")
j = df.loc[last]
print(j.to_string())
print(df[["rank","veto","open","baseline_net_fill","strategy_net_fill","baseline_shares","strategy_shares","delta_usd"]].round(3).to_string())
