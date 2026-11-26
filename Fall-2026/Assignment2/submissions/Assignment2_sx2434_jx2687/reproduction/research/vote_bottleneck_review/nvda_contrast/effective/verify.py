"""Development-period ASML contrast. Run with PYTHONHASHSEED=0 under the course Python."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "Opus_handoff/strategy_research"))
from common import CACHE, OUT, RemovalScores, backtest, period
from selection import BUCKET, MonthlyScores
from selection.config import load
from valuation.student_valuation import rejected
from portfolio.backtester import BacktestConfig

start, end = period("dev")
event_day = pd.Timestamp("2019-11-04")
name = "ASML"
quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
assert quotes.day.between(start,end).all() and (quotes.as_of < quotes.day).all()
ev = pd.read_csv(ROOT / "vote_bottleneck_review/dev_events.csv", parse_dates=["day"])
assert ev.day.dt.date.between(start,end).all()
# Select the first top-ten unanimous NVDA veto with a complete 63-session label.
nv = ev[(ev.ticker == "NVDA") & ev.category.eq("unanimous")].sort_values("day")
assert event_day == nv.day.iloc[0]
g = ev[ev.day.eq(event_day) & ev.category.eq("unanimous") & ev.spread.lt(0)]
assert g.ticker.tolist() == [name]
_, _, params, prices, _, universe = load("selection/configs/growth.json")
scores = MonthlyScores()
sessions = prices.sessions[(prices.sessions.date >= start) & (prices.sessions.date <= end)]
previous = sessions[sessions.get_loc(event_day)-1]
future = sessions[sessions.get_loc(previous)+63]
q = quotes[quotes.day.eq(event_day.date())].set_index("ticker")
assert q.as_of.eq(previous.date()).all()
high = ["dcf_high","peer_high","hist_high"]
assert q.loc[name,high].lt(q.loc[name,"price"]).all()
drops = {day:rejected(df, "two_of_three") for day,df in quotes.groupby("day")}
members = set(universe.members_at(previous.date()))
opens = prices.open.loc[event_day]
order = [r.symbol for r in sorted(scores.get(event_day.date()), key=lambda r:-r.rank_score)
         if r.bucket == BUCKET and r.symbol in members and np.isfinite(opens.get(r.symbol,np.nan))]
a_top = set(order[:10])
b_top = set([k for k in order if k not in drops[event_day.date()]][:10])
flags = q.assign(rank=[order.index(k)+1 if k in order else np.nan for k in q.index])
flags.loc[sorted(a_top - b_top), ["rank","price","dcf","dcf_high","peer","peer_high","hist_mid","hist_high"]].to_csv(HERE/"decision.csv")
ret = prices.close.loc[future] / prices.close.loc[previous] - 1
replacements = sorted(b_top - a_top)
curves = pd.read_csv(OUT / "experiment_dev_curves.csv",index_col=0,parse_dates=True)
archives = {}
for label in ["A_baseline","B_two_of_three"]:
    archives[label] = pd.read_csv(OUT/f"experiment_dev_{label}_trades.csv",parse_dates=["date"])
hp = {label:df[df.date.le(event_day)].groupby("symbol").qty.sum() for label,df in archives.items()}
actual = set(hp["B_two_of_three"][hp["B_two_of_three"]>0].index)-set(hp["A_baseline"][hp["A_baseline"]>0].index)
summary = pd.DataFrame([dict(ticker=t,rank=order.index(t)+1, reconstructed_replacement=t in replacements,
                  filter_only_actual=t in actual, rank_top10=t in a_top,
                  decision_close=previous.date(),end_close=future.date(),ret63=ret[t],
                  a_shares=hp["A_baseline"].get(t,0), b_shares=hp["B_two_of_three"].get(t,0))
                  for t in sorted(a_top | set(replacements) | actual)])
summary.to_csv(HERE/"same_week.csv",index=False)
np.testing.assert_allclose(ret[name] - ret[replacements].mean(),g.spread.item())
a = archives["A_baseline"]; b = archives["B_two_of_three"]
exit_day=pd.Timestamp("2019-12-02")
at = a[a.symbol.eq(name) & a.date.between(event_day,exit_day)]
bt = b[b.symbol.eq(name) & b.date.between(event_day,exit_day)]
assert at.qty.tolist()==[30,-30] and bt.empty
assert hp["B_two_of_three"].get(name,0)==0 and hp["A_baseline"][name]==30
cfg=BacktestConfig(start=start,end=end)
cost=cfg.commission+cfg.slippage
pnl=-at.notional.sum()-at.notional.abs().sum()*cost
at.assign(estimated_fees=at.notional.abs()*cost).to_csv(HERE/"teacher_trade.csv",index=False)

# Replay B from the same original state. A single switch changes the ASML
# veto while preserving the other candidates, trading rules, and costs.
extra = {}
for label,last in [("allow_only_first_week",event_day), ("allow_during_baseline_position",exit_day-pd.Timedelta(days=1))]:
    exempt = {d:removed - {name} if event_day.date()<=d<=last.date() else removed for d,removed in drops.items()}
    res=backtest(prices,params,universe,RemovalScores(MonthlyScores(),exempt),start,exit_day.date())
    curve=res.equity
    np.testing.assert_allclose(curve.loc[:previous],curves.loc[:previous,"B_two_of_three"],atol=1e-6,rtol=0)
    extra[label]=curve
    res.trades.loc[(res.trades.date>=event_day.date()) & (res.trades.date<=exit_day.date())].to_csv(HERE/f"{label}_trades.csv",index=False)
replay = backtest(prices,params,universe,RemovalScores(MonthlyScores(),drops),start,exit_day.date())
np.testing.assert_allclose(replay.equity,curves.loc[:exit_day,"B_two_of_three"],atol=1e-6,rtol=0)
result=pd.DataFrame(dict(frozen=replay.equity,**extra)).loc[previous:exit_day]
result.to_csv(HERE/"counterfactual_equity.csv")
print("Price return start/end",previous.date(),future.date())
print(summary.to_string(index=False))
print("teacher ASML pnl",pnl,"return",pnl/(at.notional.iloc[0]*(1+cost)))
print("actual B-only", sorted(actual),ret[sorted(actual)].mean(), "rebuilt",replacements,ret[replacements].mean())
print("equity at checkpoints",result.loc[pd.DatetimeIndex([previous,'2019-11-08','2019-11-11','2019-11-29',exit_day])])
