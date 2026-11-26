"""Validation factorial: ANET veto, other vetoes, and a June 6 cash-order sensitivity."""
import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOCK = HERE.parents[1]
ROOT = LOCK.parents[1]
sys.path.insert(0, str(ROOT / "weighted_valuation_research"))
from study import CACHE, MonthlyScores, RemovalScores, evaluate, period, rebalance_days, np, pd
from selection.strategy import target_weights
from selection.config import load
from portfolio import backtester as engine
from experiment import contributions

assert os.environ["PYTHONHASHSEED"] == "0"
cfg = json.loads((LOCK / "protocol.json").read_text())
assert hashlib.sha256((ROOT/"weighted_valuation_research/study.py").read_bytes()).hexdigest() == cfg["implementation_sha256"]
start,end = period("val")
assert (cfg["start"],cfg["end"],cfg["K"]) == (str(start),str(end),1.983)
_, _, params, prices, _, universe = load("selection/configs/growth.json")
prices.open, prices.close = prices.open.loc[:str(end)], prices.close.loc[:str(end)]
days = rebalance_days(prices,start,end)
v = pd.read_parquet(CACHE/"valuations_val.parquet")
assert set(v.day)==set(days) and (v.as_of<v.day).all()
rules = {}
for day,g in v.groupby("day"):
    x=evaluate(g)
    m=x[["dcf","peer","hist_mid"]]
    rules[day]=set(x.loc[x.adaptive_veto & m.max(axis=1).le(cfg["K"]*m.min(axis=1)), "ticker"])
arms={"baseline":{}, "ANET_only":{d:s & {"ANET"} for d,s in rules.items()},
      "other_only":{d:s-{"ANET"} for d,s in rules.items()}, "all":rules}
march_episode={pd.Timestamp(d).date() for d in ("2022-03-28","2022-04-04","2022-04-11")}
arms["exempt_ANET_first_week"]={d:s-{"ANET"} if d==start else s for d,s in rules.items()}
arms["exempt_ANET_Mar28_Apr11"]={d:s-{"ANET"} if d in march_episode else s for d,s in rules.items()}
archive=pd.read_csv(LOCK/"val_equity.csv",index_col=0,parse_dates=True)
focal=pd.Timestamp("2022-06-06")
nxt=pd.Timestamp("2022-06-13")
mid=pd.Timestamp("2022-06-03")

def sorted_for_sensitivity(items, *args, **kwargs):
    # Restrict the intervention to the actual order queue on this date. All sells
    # keep their priority, as do all other buys relative to each other.
    ordered = builtins.sorted(items,*args,**kwargs)
    frame=sys._getframe(1)
    if (frame.f_code is engine.run_backtest.__code__ and frame.f_locals.get("day")==focal
            and items is frame.f_locals.get("orders")):
        ordered.sort(key=lambda order: 0 if order[1]<0 else 1 if order[0]=="ANET" else 2)
    return ordered

summary=[]
logs=[]
for schedule in ("course","June6_ANET_first_buy"):
    for label,drop in arms.items():
        if schedule=="June6_ANET_first_buy" and label not in ("baseline","all"):
            continue
        sort_fn=sorted_for_sensitivity if schedule!="course" else builtins.sorted
        with patch.object(engine,"sorted",sort_fn,create=True):
            r=engine.run_backtest(prices,RemovalScores(MonthlyScores(),drop),params,universe,engine.BacktestConfig(start=start,end=end))
        eq=r.equity
        assert eq.index.equals(archive.index)
        if label in ("baseline","all"):
            reference=archive["baseline" if label=="baseline" else "K1.983"]
            coord=eq.index if schedule=="course" else eq.index[eq.index<focal]
            np.testing.assert_allclose(eq.loc[coord],reference.loc[coord],atol=1e-6,rtol=0)
        t=r.trades.copy()
        t["date"]=pd.to_datetime(t.date)
        q=t.pivot_table(index="date",columns="symbol",values="qty",aggfunc="sum").reindex(eq.index).fillna(0)
        shares=q.cumsum()
        cash=100000-(q*prices.open.reindex_like(q)).fillna(0).sum(axis=1).add((q.abs()*prices.open.reindex_like(q)).fillna(0).sum(axis=1)*0.0006).cumsum()
        opens=prices.open.reindex_like(q).fillna(prices.close[q.columns].ffill().reindex_like(q))
        pre=cash.shift()+ (shares.shift()*opens).fillna(0).sum(axis=1)
        pre.iloc[0]=100000.
        cost=abs(q.at[focal,"ANET"])*prices.open.at[focal,"ANET"]*0.0006
        week_name=shares.at[focal,"ANET"]*(prices.open.at[nxt,"ANET"]-prices.open.at[focal,"ANET"])-cost
        total_pnl=contributions(r,prices)
        summary.append(dict(arm=label,execution=schedule,cagr_pct=100*((eq.iloc[-1]/eq.iloc[0])**(252/(len(eq)-1))-1),
                          end_equity=eq.iloc[-1],anet_total_pnl=total_pnl.ANET,shares_june6=shares.at[focal,"ANET"],
                          june6_order_qty=q.at[focal,"ANET"],anet_week_pnl=week_name,portfolio_week_pnl=pre[nxt]-pre[focal],
                          june6_pre_equity=pre[focal],june6_pre_cash=cash[mid],june6_final_cash=cash[focal]))
        if schedule=="course":
            targets={v["date"]:v["weights"] for v in r.weekly_targets}
            raw_scores=RemovalScores(MonthlyScores(),drop)
            for d in days:
                date=pd.Timestamp(d)
                if date>nxt: break
                idx=eq.index.get_loc(date)
                prior=eq.index[idx-1].date() if idx else (date-pd.Timedelta(days=1)).date()
                opening=prices.open.loc[date]
                eligible={k for k in universe.members_at(prior) if np.isfinite(opening.get(k,np.nan))}
                old=shares.iloc[idx-1].to_dict() if idx else {}
                weights=target_weights(raw_scores.get(d),params,eligible,old)
                weight=weights.get("ANET",0)
                assert round(weight,4)==targets[d].get("ANET",0)
                logs.append(dict(day=d,arm=label,anet_veto="ANET" in drop.get(d,set()), weight=weight,
                                 shares=shares.at[date,"ANET"],trade=q.at[date,"ANET"],
                                 desired=int(weight*.95*pre[date]//prices.open.at[date,"ANET"])))
s=pd.DataFrame(summary)
s.to_csv(HERE/"val_counterfactuals.csv",index=False)
pd.DataFrame(logs).to_csv(HERE/"val_weekly_paths.csv",index=False)
print(s.round(4).to_string(index=False))
