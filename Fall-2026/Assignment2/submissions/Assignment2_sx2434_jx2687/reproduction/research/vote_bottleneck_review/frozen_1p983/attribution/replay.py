"""Validation-only trade and name attribution. Dollar P&L includes initial fills."""
from pathlib import Path
import os
import sys
import hashlib
import json

HERE = Path(__file__).resolve().parent
LOCK = HERE.parent
ROOT = LOCK.parents[1]
STUDY = ROOT / "weighted_valuation_research"
cfg = json.loads((LOCK / "protocol.json").read_text())
assert hashlib.sha256((STUDY / "study.py").read_bytes()).hexdigest() == cfg["implementation_sha256"]
sys.path.insert(0, str(STUDY))
from study import CACHE, MonthlyScores, RemovalScores, backtest, evaluate, np, pd, period, rebalance_days
from selection.config import load
from portfolio.backtester import BacktestConfig

assert os.environ.get("PYTHONHASHSEED") == "0"
start, end = period("val")
assert (cfg["split"], str(start), str(end), cfg["K"]) == ("val", cfg["start"], cfg["end"], 1.983)
_, _, params, prices, _, universe = load("selection/configs/growth.json")
prices.open = prices.open.loc[:str(end)]
prices.close = prices.close.loc[:str(end)]
weeks = pd.DatetimeIndex(rebalance_days(prices,start,end))
quotes = pd.read_parquet(CACHE/"valuations_val.parquet")
assert set(quotes.day) == set(weeks.date) and (quotes.as_of < quotes.day).all()
archived = pd.read_csv(LOCK/"val_equity.csv",index_col=0,parse_dates=True)
engine = BacktestConfig(start=start,end=end)

def run_pair(exclude_nvda=False):
    data = quotes.loc[quotes.ticker.ne("NVDA")] if exclude_nvda else quotes
    base = {d: {"NVDA"} if exclude_nvda else set() for d in weeks.date}
    filtered = {}
    for day, group in data.groupby("day"):
        x=evaluate(group)
        mids=x[["dcf","peer","hist_mid"]]
        veto=x.adaptive_veto & mids.max(axis=1).le(cfg["K"]*mids.min(axis=1))
        filtered[day]=base[day] | set(x.loc[veto,"ticker"])
    results=[]
    for label, rule in [("baseline",base), ("K1.983",filtered)]:
        r=backtest(prices,params,universe,RemovalScores(MonthlyScores(),rule),start,end)
        assert r.equity.index.equals(archived.index) and set(r.trades.date)<=set(weeks.date)
        if exclude_nvda:
            assert not r.trades.symbol.eq("NVDA").any()
        else:
            np.testing.assert_allclose(r.equity, archived[label], atol=1e-6,rtol=0)
        results.append(r)
    return *results, filtered

def ledger(r):
    t=r.trades.copy()
    t["date"]=pd.to_datetime(t.date)
    qty=t.pivot_table(index="date",columns="symbol",values="qty",aggfunc="sum").reindex(r.equity.index).fillna(0)
    gross=t.assign(abs_qty=t.qty.abs()).pivot_table(index="date",columns="symbol",values="abs_qty",aggfunc="sum").reindex_like(qty).fillna(0)
    opens=prices.open.reindex_like(qty)
    closes=prices.close[qty.columns].ffill().reindex_like(qty)
    flows=((qty + gross*(engine.commission+engine.slippage))*opens).fillna(0).cumsum()
    held=qty.cumsum()
    pnl=(held*closes).fillna(0)-flows
    np.testing.assert_allclose(engine.initial_cash + pnl.sum(axis=1), r.equity, atol=1e-6, rtol=0)
    # Mark before first fill on each rebalance day. Incoming trades and fees
    # belong to the new week, overnight gaps on outgoing positions to the old.
    entry=(held.shift(fill_value=0)*opens.fillna(closes)).fillna(0)-flows.shift(fill_value=0)
    entry=entry.loc[weeks]
    end_pnl=entry.shift(-1)
    end_pnl.iloc[-1]=pnl.iloc[-1]
    raw=end_pnl-entry
    np.testing.assert_allclose(raw.sum(),pnl.iloc[-1],atol=1e-6,rtol=0)
    daily_origin=raw.copy()
    daily_origin.iloc[0]-=pnl.iloc[0]
    normalized=daily_origin/r.equity.iloc[0]
    return held.loc[weeks],raw,normalized

def cagr(eq):
    return 100*((eq.iloc[-1]/eq.iloc[0])**(252/(len(eq)-1))-1)

b,s,drops=run_pair()
positions_b,raw_b,unit_b=ledger(b)
positions_s,raw_s,unit_s=ledger(s)
pos_b,pos_s=positions_b.align(positions_s,fill_value=0)
raw=raw_s.subtract(raw_b,fill_value=0)
unit=unit_s.subtract(unit_b,fill_value=0)
delta=cagr(s.equity)-cagr(b.equity)
# Map additive cumulative-return differences onto the CAGR scale by a common
# secant factor. This preserves relative contributions and the exact total.
total_return_gap=s.equity.iloc[-1]/s.equity.iloc[0]-b.equity.iloc[-1]/b.equity.iloc[0]
ann=unit*(delta/total_return_gap)
np.testing.assert_allclose(ann.to_numpy().sum(),delta,atol=1e-8)
np.testing.assert_allclose(raw.to_numpy().sum(),s.equity.iloc[-1]-b.equity.iloc[-1],atol=1e-6)
changed=(pos_b.gt(0)!=pos_s.gt(0)).any(axis=1)
assert changed.sum()==12 and len(weeks)==104
# Document target-vs-filled differences and trace later position-size effects.
targets = [{r["date"]: r["weights"] for r in run.weekly_targets} for run in (b,s)]
position_rows = []
for day in weeks:
    for ticker in pos_b.columns[(pos_b.loc[day] + pos_s.loc[day]) > 0]:
        position_rows.append(dict(day=day.date(), ticker=ticker, baseline_shares=pos_b.at[day,ticker],
                                  strategy_shares=pos_s.at[day,ticker], baseline_target=targets[0][day.date()].get(ticker,0),
                                  strategy_target=targets[1][day.date()].get(ticker,0),
                                  has_open=pd.notna(prices.open.at[day,ticker])))
pd.DataFrame(position_rows).to_csv(HERE/"val_positions.csv",index=False)
stock=pd.DataFrame(dict(all_pp=ann.sum(),changed12_pp=ann.loc[changed].sum(),other92_pp=ann.loc[~changed].sum(),
                        all_usd=raw.sum(),changed12_usd=raw.loc[changed].sum(),other92_usd=raw.loc[~changed].sum()))
stock.sort_values("all_pp",ascending=False).to_csv(HERE/"val_stock_attribution.csv")
rows=[]
for day in weeks:
    pb,ps=pos_b.loc[day],pos_s.loc[day]
    basket_b=pb.index[(pb>0)&(ps==0)].tolist()
    basket_s=ps.index[(ps>0)&(pb==0)].tolist()
    def display(xs): return ",".join(xs)
    rows.append(dict(day=day.date(),changed=bool(changed.at[day]),baseline_only=display(basket_b),filter_only=display(basket_s),
        baseline_only_vetoed_this_week=display(sorted(set(basket_b)&drops[day.date()])),
        count_base=int(pb.gt(0).sum()),count_filter=int(ps.gt(0).sum()),nvda_veto="NVDA" in drops[day.date()],nvda_qty_base=pb.NVDA,nvda_qty_filter=ps.NVDA,
        nvda_pp=ann.at[day,"NVDA"],pp=ann.loc[day].sum(),usd=raw.loc[day].sum()))
weekly=pd.DataFrame(rows)
weekly.to_csv(HERE/"val_weekly.csv",index=False)
ann.to_csv(HERE/"val_week_by_stock_pp.csv")
ex_b,ex_s,_=run_pair(True)
without=dict(baseline_cagr=cagr(ex_b.equity),filtered_cagr=cagr(ex_s.equity),delta_cagr_pp=cagr(ex_s.equity)-cagr(ex_b.equity))
(HERE/"val_without_nvda.json").write_text(json.dumps(without,indent=2)+"\n")
print(weekly[weekly.changed].round(4).to_string(index=False))
print(stock.sort_values("all_pp").iloc[np.r_[:6,-8:]].round(4))
print(stock.sum(),stock.loc["NVDA"],without)
print(weekly.groupby("changed").agg(n=("day","size"),nvda_veto=("nvda_veto","sum"),nvda_held=("nvda_qty_base",lambda x:(x>0).sum())))
