"""Locate in-sample parity crossings on [2, 2.5] using only development data."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "weighted_valuation_research"))
from study import CACHE, MonthlyScores, RemovalScores, backtest, evaluate, np, pd, period
from selection.config import load

start, end = period("dev")
quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
assert quotes.day.min() >= start and quotes.day.max() <= end and (quotes.as_of < quotes.day).all()
signals = pd.concat([evaluate(g) for _,g in quotes.groupby("day")], ignore_index=True)
events = signals.loc[signals.adaptive_veto].copy()
centers = events[["dcf","peer","hist_mid"]]
events["span"] = centers.max(axis=1)/centers.min(axis=1)
thresholds = sorted({2., 2.3, 2.5, *events.loc[events.span.between(2,2.5),"span"]})
_, _, params, prices, _, universe = load("selection/configs/growth.json")
scores = MonthlyScores()
cached = pd.read_csv(HERE/"dev_curves.csv", index_col=0, parse_dates=True)
baseline = cached.baseline
def annual(s):
    return 100*((s.iloc[-1]/s.iloc[0])**(252/(len(s)-1))-1)
baseline_cagr = annual(baseline)
rows = []
for k in thresholds:
    if k in (2.,2.3,2.5):
        curve = cached["K"+str(k)]
    else:
        drops = events.loc[events.span<=k].groupby("day").ticker.agg(set).to_dict()
        curve = backtest(prices,params,universe,RemovalScores(scores,drops),start,end).equity
    rows.append(dict(K=k,delta_cagr_pp=annual(curve)-baseline_cagr))
df = pd.DataFrame(rows)
df.to_csv(HERE/"dev_empirical_steps.csv",index=False)
sign = np.sign(df.delta_cagr_pp)
print(len(rows), df[sign.ne(sign.shift())].to_string(index=False))
