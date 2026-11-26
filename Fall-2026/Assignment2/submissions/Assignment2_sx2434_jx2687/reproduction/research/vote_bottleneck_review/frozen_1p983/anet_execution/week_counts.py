"""Activity during the validation weeks where portfolios held the same names."""
from pathlib import Path
import pandas as pd
import numpy as np

here = Path(__file__).resolve().parent
att = here.parent / "attribution"
pos = pd.read_csv(att / "val_positions.csv", parse_dates=["day"])
v = pd.read_csv(att / "val_weekly.csv", parse_dates=["day"]).set_index("day")
assert len(v) == 104 and v.index.min().year == 2022 and v.index.max().year == 2023
panels = {}
for arm in ["baseline", "strategy"]:
    size = pos.pivot(index="day", columns="ticker", values=arm+"_shares").fillna(0)
    panels[arm] = size
    # The engine executes at most one order per name per weekly rebalance.
    delta = size - size.shift(1, fill_value=0)
    v[arm+"_trades"] = delta.ne(0).sum(axis=1)
    v[arm+"_added_or_closed"] = (size.gt(0) != size.shift(1,fill_value=0).gt(0)).sum(axis=1)

v["same_names"] = panels["baseline"].gt(0).eq(panels["strategy"].gt(0)).all(axis=1)
v["same_shares"] = panels["baseline"].eq(panels["strategy"]).all(axis=1)
assert (v.same_names == ~v.changed).all()
b = v.baseline_trades.gt(0)
s = v.strategy_trades.gt(0)
v["activity"] = np.select([b & s, b, s],["both", "baseline only", "strategy only"],default="neither")
v[["same_names","same_shares","baseline_trades","strategy_trades","baseline_added_or_closed","strategy_added_or_closed","activity","pp"]].to_csv(here/"weekly_activity.csv")
print(v[v.same_names].groupby("activity").agg(weeks=("pp","size"), cagr_pp=("pp","sum")).to_string())
print("same shares:",v.loc[v.same_names,"same_shares"].sum())
print("same-names within both arms churn",v[v.same_names][["baseline_added_or_closed","strategy_added_or_closed"]].gt(0).any(axis=1).sum())
print("June 6:",v.loc["2022-06-06", ["baseline_trades","strategy_trades"]].to_dict())

# Verify counts against the engine's fill ledger, including weeks with no rows.
import sys
import os
root = here.parents[2]
sys.path.insert(0,str(root/"weighted_valuation_research"))
from study import MonthlyScores, RemovalScores, backtest, period
from selection.config import load
assert os.environ["PYTHONHASHSEED"]=="0"
start,end=period("val")
_,_,params,prices,_,universe=load("selection/configs/growth.json")
prices.open=prices.open.loc[:str(end)]
prices.close=prices.close.loc[:str(end)]
sig=pd.read_csv(root/"weighted_valuation_research/val_signals.csv",parse_dates=["day"])
mid=sig[["dcf","peer","hist_mid"]]
removed=sig[sig.adaptive_veto & mid.max(axis=1).le(1.983*mid.min(axis=1))]
by_day=removed.groupby(removed.day.dt.date).ticker.agg(set).to_dict()
ref=pd.read_csv(here.parent/"val_equity.csv",index_col=0,parse_dates=True)
for arm,rule,ref_col in [("baseline",{},"baseline"),("strategy",by_day,"K1.983")]:
    r=backtest(prices,params,universe,RemovalScores(MonthlyScores(),rule),start,end)
    assert r.equity.index.equals(ref.index)
    np.testing.assert_allclose(r.equity,ref[ref_col],atol=1e-6,rtol=0)
    counts=r.trades.groupby(pd.to_datetime(r.trades.date)).size().reindex(v.index,fill_value=0)
    assert counts.equals(v[arm+"_trades"])
