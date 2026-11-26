"""Replay the locked candidate and the course selection-only strategy on validation."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
cfg = json.loads((HERE / "protocol.json").read_text())
study = ROOT / "weighted_valuation_research"
assert hashlib.sha256((study / "study.py").read_bytes()).hexdigest() == cfg["implementation_sha256"]
sys.path.insert(0, str(study))
from study import (CACHE, RemovalScores, MonthlyScores, backtest, evaluate, np, pd,
                   performance, period, rebalance_days, weekly_holdings)
from selection.config import load

assert os.environ.get("PYTHONHASHSEED") == "0"
start, end = period("val")
assert (cfg["split"], cfg["start"], cfg["end"], cfg["K"]) == ("val", str(start), str(end), 1.983)
_, _, params, prices, _, universe = load("selection/configs/growth.json")
prices.open = prices.open.loc[:str(end)]
prices.close = prices.close.loc[:str(end)]
weeks = rebalance_days(prices, start, end)
quotes = pd.read_parquet(CACHE / "valuations_val.parquet")
assert set(quotes.day) == set(weeks) and not quotes.duplicated(["day", "ticker"]).any()
drops = {}
scores = MonthlyScores()
for day, group in quotes.groupby("day"):
    prior = pd.Timestamp(prices.last_session_on_or_before(day - pd.Timedelta(days=1)))
    assert pd.to_datetime(group.as_of).eq(prior).all()
    scores.get(day)
    assert pd.Timestamp(scores.cutoff) <= prior
    x = evaluate(group)
    centers = x[["dcf", "peer", "hist_mid"]]
    flag = x.adaptive_veto & (centers.max(axis=1) <= cfg["K"] * centers.min(axis=1))
    drops[day] = set(x.loc[flag, "ticker"])

# Confirm all vetoes against the separately archived prior-close signals.
saved = pd.read_csv(study / "val_signals.csv", parse_dates=["day"])
centers = saved[["dcf", "peer", "hist_mid"]]
check = saved[saved.adaptive_veto & ((centers.max(axis=1) / centers.min(axis=1)) <= cfg["K"])]
assert {(d, t) for d, tickers in drops.items() for t in tickers} == set(zip(check.day.dt.date, check.ticker))

curves, holdings, metrics = {}, {}, {}
for label, removal in [("baseline", {}), ("K1.983", drops)]:
    result = backtest(prices, params, universe, RemovalScores(scores, removal), start, end)
    assert (result.equity.index.min().date(),result.equity.index.max().date()) == (start,end)
    curves[label] = result.equity
    holdings[label] = weekly_holdings(result.trades, weeks)
    metrics[label] = performance(result.equity)
eq = pd.DataFrame(curves)
ref = pd.read_csv(study / "val_equity.csv", index_col=0, parse_dates=True)
assert eq.index.equals(ref.index)
np.testing.assert_allclose(eq.baseline, ref.A, atol=1e-6, rtol=0)
eq.to_csv(HERE / "val_equity.csv")
pd.DataFrame(metrics).T.to_csv(HERE / "val_metrics.csv")
rets = np.log(eq).diff().dropna()
n = len(rets)
rng = np.random.default_rng(20261124)
draws = []
for _ in range(5000):
    blocks = rng.integers(n, size=(n+19)//20)
    idx = ((blocks[:,None] + np.arange(20)) % n).ravel()[:n]
    cagr = 100*np.expm1(252*rets.to_numpy()[idx].mean(axis=0))
    draws.append(cagr[1]-cagr[0])
yearly = 100*np.expm1(rets.groupby(rets.index.year).sum())
yearly.to_csv(HERE / "val_calendar_returns.csv")
a,b = holdings["baseline"].align(holdings["K1.983"],fill_value=False)
details = dict(weeks=len(weeks), different_holdings_weeks=int(a.ne(b).any(axis=1).sum()),
               delta_cagr_pp=float(100*np.diff(np.expm1(252*rets.mean().to_numpy()))[0]),
               low90_pp=float(np.quantile(draws,.05)), high90_pp=float(np.quantile(draws,.95)))
(HERE / "val_comparison.json").write_text(json.dumps(details,indent=2)+"\n")
print(pd.DataFrame(metrics).T.to_string(),yearly.round(3).to_string(),details,sep="\n")
