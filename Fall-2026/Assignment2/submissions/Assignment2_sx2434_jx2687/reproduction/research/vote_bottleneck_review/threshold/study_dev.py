"""Development-only sensitivity at the three specified cutoffs, no parameter search."""
from itertools import combinations
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "weighted_valuation_research"
sys.path.insert(0, str(PREVIOUS))
from study import (CACHE, MonthlyScores, RemovalScores, backtest, evaluate, np,
                   pd, performance, period, rebalance_days, turnover, weekly_holdings)
from experiment import contributions
from selection.config import load

thresholds = {"K2.0": 2.0, "K2.3": 2.3, "K2.5": 2.5}
start, end = period("dev")
_, _, params, prices, _, universe = load("selection/configs/growth.json")
weeks = rebalance_days(prices, start, end)
scores = MonthlyScores()
quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
assert set(quotes.day) <= set(weeks) and (quotes.as_of < quotes.day).all()
signals = pd.concat([evaluate(g) for _, g in quotes.groupby("day")])
eligible = signals[signals.adaptive_veto].copy()
centers = eligible[["dcf", "peer", "hist_mid"]]
eligible["span"] = centers.max(axis=1) / centers.min(axis=1)
curves = {}
holdings = {}
pnl = {}
stats = []
for label in ["baseline", *thresholds]:
    by_day = {}
    if label in thresholds:
        mask = centers.max(axis=1) <= thresholds[label] * centers.min(axis=1)
        assert (mask == (eligible.span <= thresholds[label])).all()
        by_day = eligible.loc[mask].groupby("day").ticker.agg(set).to_dict()
    res = backtest(prices, params, universe, RemovalScores(scores, by_day), start, end)
    curves[label] = res.equity
    holdings[label] = weekly_holdings(res.trades, weeks)
    assert set(res.trades.date).issubset(set(weeks))
    pnl[label] = contributions(res, prices)
    stats.append(dict(rule=label, veto_name_weeks=sum(map(len, by_day.values())),
                      **performance(res.equity), turnover=turnover(res), trades=len(res.trades)))
stats = pd.DataFrame(stats).set_index("rule")
former = pd.read_csv(PREVIOUS / "zeyu_overlay/summary.csv", nrows=9).set_index("rule")
for field in former.select_dtypes("number"):
    np.testing.assert_allclose(stats.loc["K2.5", field], former.loc["adaptive_spread", field], atol=1e-9, rtol=0)
curves = pd.DataFrame(curves)
archived = pd.read_csv(PREVIOUS / "dev_equity.csv", index_col=0, parse_dates=True)
assert curves.index.equals(archived.index)
np.testing.assert_allclose(curves.baseline, archived.A, rtol=0, atol=1e-6)
# Independent replay from the prior saved weekly estimate table.
logs = pd.read_csv(PREVIOUS/"dev_signals.csv", parse_dates=["day"])
old = logs[logs.adaptive_veto].copy()
cent = old[["dcf","peer","hist_mid"]]
old = old[cent.max(axis=1) <= 2.5*cent.min(axis=1)]
old["day"] = old.day.dt.date
legacy = backtest(prices, params, universe, RemovalScores(scores, old.groupby("day").ticker.agg(set).to_dict()), start, end)
np.testing.assert_allclose(legacy.equity, curves["K2.5"], rtol=0, atol=1e-6)
curves.to_csv(HERE/"dev_curves.csv")
stats.to_csv(HERE/"dev_thresholds.csv")
checks = []
for a,b in combinations(curves.columns, 2):
    l,r = holdings[a].align(holdings[b], fill_value=False)
    changes = l.ne(r).any(axis=1)
    checks.append(dict(a=a,b=b,different_weeks=changes.sum(),weeks=len(weeks)))
pd.DataFrame(checks).to_csv(HERE/"dev_holdings.csv", index=False)
annual = (1 + curves.pct_change().fillna(0)).groupby(curves.index.year).prod().sub(1).mul(100)
annual.to_csv(HERE/"dev_calendar_returns.csv")
asset_pnl = pd.DataFrame(pnl).fillna(0)
asset_pnl["K2.3_minus_K2.5"] = asset_pnl["K2.3"]-asset_pnl["K2.5"]
asset_pnl["K2.3_minus_baseline"] = asset_pnl["K2.3"]-asset_pnl["baseline"]
asset_pnl.to_csv(HERE/"dev_stock_pnl.csv")
edges = []
for label, k in thresholds.items():
    below = eligible[eligible.span<=k].nlargest(1,"span")
    above = eligible[eligible.span>k].nsmallest(1,"span")
    for side, grp in [("lower_inclusive",below),("upper_exclusive",above)]:
        row=grp.iloc[0]
        edges.append(dict(rule=label,edge=side,ratio=row.span,ticker=row.ticker,day=row.day))
pd.DataFrame(edges).to_csv(HERE/"dev_signal_edges.csv", index=False)
print(stats.to_string())
print(pd.DataFrame(checks).to_string(index=False))
print(annual.round(2).to_string())
print(pd.DataFrame(edges).to_string(index=False))
for field in ["K2.3_minus_K2.5","K2.3_minus_baseline"]:
    delta = asset_pnl[field]
    print(field, delta.sum(), delta.nlargest(3).to_dict(), delta.nsmallest(3).to_dict())
