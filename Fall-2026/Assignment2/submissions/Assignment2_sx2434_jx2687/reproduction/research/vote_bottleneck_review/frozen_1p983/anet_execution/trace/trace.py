"""Observe the unmodified course engine's local variables with a line trace; validation only."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import sys

HERE = Path(__file__).resolve().parent
LOCK = HERE.parents[1]
ROOT = LOCK.parents[1]
cfg = json.loads((LOCK / "protocol.json").read_text())
STUDY = ROOT / "weighted_valuation_research"
assert hashlib.sha256((STUDY / "study.py").read_bytes()).hexdigest() == cfg["implementation_sha256"]
sys.path.insert(0, str(STUDY))
from study import CACHE, MonthlyScores, RemovalScores, evaluate, np, pd, period, rebalance_days
from portfolio.backtester import BacktestConfig, run_backtest
from selection.config import load

assert os.environ.get("PYTHONHASHSEED") == "0"
start, end = period("val")
assert (cfg["split"], cfg["start"], cfg["end"], cfg["K"]) == ("val", str(start), str(end), 1.983)
_, _, params, prices, _, universe = load("selection/configs/growth.json")
prices.open = prices.open.loc[:str(end)]
prices.close = prices.close.loc[:str(end)]
weeks = rebalance_days(prices, start, end)
quotes = pd.read_parquet(CACHE / "valuations_val.parquet")
assert set(quotes.day) == set(weeks) and (quotes.as_of < quotes.day).all()
drops, signal = {}, []
for day, g in quotes.groupby("day"):
    x = evaluate(g).copy()
    c = x[["dcf","peer","hist_mid"]]
    x["span"] = c.max(axis=1) / c.min(axis=1)
    veto = x.adaptive_veto & x.span.le(cfg["K"])
    drops[day] = set(x.loc[veto,"ticker"])
    a=x[x.ticker.eq("ANET")]
    if not a.empty:
        signal.append(a.assign(veto=a.index.isin(x.index[veto])))
pd.concat(signal).to_csv(HERE / "anet_signal.csv", index=False)

lines, offset = inspect.getsourcelines(run_backtest)
def anchor(prefix):
    ids = [offset+i for i,l in enumerate(lines) if l.strip().startswith(prefix)]
    assert len(ids)==1
    return ids[0]

at_targets=anchor("orders = []")
at_order=anchor("if delta > 0:")
at_trade=anchor('trade(day, i, sym, delta, px, "rebalance")')
at_done=anchor('closes = prices.close.loc[day]')
rows, orders = [], []
def capture(frame, event, arg):
    if event != "line":
        return capture
    line=frame.f_lineno
    if line not in (at_targets,at_order,at_trade,at_done):
        return capture
    v=frame.f_locals
    day=v["day"].date()
    if day not in weeks:
        return capture
    if line==at_targets:
        pos=v["pos"]
        raw=v["preds"]
        growth=sorted([p for p in raw if p.symbol in v["tradeable"] and p.bucket=="growth_tech"], key=lambda p:-p.rank_score)
        rank=next((idx for idx,p in enumerate(growth,1) if p.symbol=="ANET"),None)
        stale={s:q for s,q in pos.items() if not np.isfinite(v["opens"].get(s,np.nan))}
        stval=sum(q*v["close_ff"].at[v["day"],s] for s,q in stale.items())
        rows.append(dict(arm=arm,day=day,rank_available=rank,veto=arm=='filtered' and 'ANET' in drops[day],
              price=v['opens'].get('ANET'),target=v['weights'].get('ANET',0),equity=v["equity"],opening_cash=v["cash"],
              qty_before=pos.get("ANET",0),desired=v["desired"].get("ANET",0),
              stale_cash_unavailable=stval,untradeable_names=','.join(sorted(stale))))
    elif line==at_order:
        orders.append(dict(arm=arm,day=day,symbol=v["sym"],before=v["pos"].get(v["sym"],0),
                           cash_before=v["cash"],price=v["px"],order=v["delta"],fill=0))
    elif line==at_trade:
        assert orders[-1]["symbol"]==v["sym"] and orders[-1]["day"]==day
        orders[-1]["fill"]=v["delta"]
    elif line==at_done:
        assert rows[-1]["day"]==day
        rows[-1].update(qty_after=v["pos"].get("ANET",0),ending_cash=v["cash"])
    return capture

def setup(frame, event, arg):
    if frame.f_code is run_backtest.__code__ and event=="call":
        return capture

reference=pd.read_csv(LOCK/"val_equity.csv",index_col=0,parse_dates=True)
for arm,removals,ref in [("baseline",{},"baseline"),("filtered",drops,"K1.983")]:
    scores=RemovalScores(MonthlyScores(),removals)
    sys.settrace(setup)
    try:
        r=run_backtest(prices,scores,params,universe,BacktestConfig(start=start,end=end))
    finally:
        sys.settrace(None)
    np.testing.assert_allclose(r.equity,reference[ref],atol=1e-6,rtol=0)
    fills=pd.DataFrame([o for o in orders if o["arm"]==arm and o["fill"]]).reset_index(drop=True)
    assert fills.symbol.equals(r.trades.symbol) and fills.fill.equals(r.trades.qty)
    assert list(fills.day)==list(r.trades.date)
out=pd.DataFrame(rows)
orddf=pd.DataFrame(orders)
anet=orddf.query('symbol=="ANET"')[['arm','day','cash_before','order','fill']].rename(columns={'cash_before':'cash_when_anet_order'})
out=out.merge(anet,on=['arm','day'],how='left',validate='1:1')
out.to_csv(HERE/"weekly_anet.csv",index=False)
orddf.to_csv(HERE/"cash_and_orders.csv",index=False)
print(out.query("day <= @pd.Timestamp('2022-06-06').date()")[['arm','day','rank_available','veto','price','target','equity','opening_cash','qty_before','desired','qty_after','cash_when_anet_order','stale_cash_unavailable']].round(2).to_string(index=False))
