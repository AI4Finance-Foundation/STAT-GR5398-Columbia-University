"""Development-only first NVDA unanimous veto, same-day controls and execution."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
ROOT = REVIEW.parent
sys.path.insert(0, str(ROOT / "Opus_handoff/strategy_research"))
from common import CACHE, OUT, RemovalScores, backtest, load_course, period
from selection import MonthlyScores
from shared.data_view import DataView
from valuation.historical_ev_ebitda import cached_history_value
from valuation.student_valuation import valuations

start, end = period("dev")
params, px, universe, bundle = load_course()
dates = px.sessions[(px.sessions >= pd.Timestamp(start)) & (px.sessions <= pd.Timestamp(end))]
quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
ranks = pd.read_parquet(CACHE / "ranks_dev.parquet")
assert quotes.day.between(start, end).all() and ranks.day.between(start, end).all()
votes = quotes[["dcf_high", "peer_high", "hist_high"]].lt(quotes.price, axis=0)
quotes["veto"] = votes.sum(axis=1).ge(2)
quotes["unanimous"] = votes.all(axis=1)
cuts = quotes[quotes.veto].groupby("day").ticker.agg(set).to_dict()
three = quotes[quotes.unanimous].groupby("day").ticker.agg(set).to_dict()


def ranked_at(dt, as_of):
    tradeable = set(px.open.loc[pd.Timestamp(dt)].dropna().index) & universe.members_at(as_of)
    return ranks.loc[ranks.day.eq(dt) & ranks.ticker.isin(tradeable)].sort_values("rank_score", ascending=False, kind="stable").reset_index(drop=True)


# Choose the first top-ten event, without requiring a mature outcome.
for day in sorted(dt for dt, symbols in three.items() if "NVDA" in symbols):
    as_of = dates[dates.get_loc(pd.Timestamp(day))-1].date()
    ranking = ranked_at(day, as_of)
    if "NVDA" in set(ranking.ticker.head(10)):
        break
else:
    raise ValueError("No original-top-ten NVDA event")
assert quotes.loc[quotes.day.eq(day), "as_of"].eq(as_of).all()
ranking["rank"] = np.arange(len(ranking))+1
ranking["veto"] = ranking.ticker.isin(cuts[day])
base = set(ranking.ticker.head(10))
passed = set(ranking.loc[~ranking.veto,"ticker"].head(10))
repl = passed-base
targets = sorted(base & three[day])
assert targets == ["ASML", "NVDA"]
i = dates.get_loc(pd.Timestamp(as_of))
end63 = dates[i+63]
first_week_end, next_week_open = dates[i+5], dates[i+6]
assert first_week_end.dayofweek==4 and next_week_open.dayofweek==0
selected = ranking[ranking.ticker.isin(base | repl)].copy().set_index("ticker")
selected["base"] = selected.index.isin(base)
selected["replacement"] = selected.index.isin(repl)
selected["unanimous"] = selected.index.isin(targets)
selected["price_from"] = px.close.loc[pd.Timestamp(as_of),selected.index]
selected["price_to"] = px.close.loc[end63,selected.index]
selected["ret63"] = selected.price_to/selected.price_from - 1
selected["entry_open_to_end63"] = selected.price_to/px.open.loc[pd.Timestamp(day),selected.index] - 1
selected["ret63_minus_repl"] = selected.ret63-selected.loc[selected.replacement,"ret63"].mean()
selected["as_of"] = as_of
selected["end63"] = end63.date()
for label in ["A_baseline", "B_two_of_three"]:
    trades = pd.read_csv(OUT / f"experiment_dev_{label}_trades.csv", parse_dates=["date"])
    positions = trades[trades.date.le(pd.Timestamp(day))].groupby("symbol").qty.sum()
    actual = set(positions[positions.gt(0)].index)
    expected = base if label == "A_baseline" else passed
    assert actual == expected
    selected[f"{label}_pos"] = positions.reindex(selected.index, fill_value=0)
    today = trades[trades.date.eq(pd.Timestamp(day))].set_index("symbol")
    selected[f"{label}_trade_qty"] = today.qty.reindex(selected.index, fill_value=0)
    selected[f"{label}_trade_px"] = today.price.reindex(selected.index)
selected.to_csv(HERE / "ranked_holdings_and_outcomes.csv")
events = pd.read_csv(REVIEW / "dev_events.csv", parse_dates=["day"])
nv = events[events.ticker.eq("NVDA") & events.category.eq("unanimous")].copy()
assert nv.day.min().date() == day
np.testing.assert_allclose(nv.sort_values("day").spread.iloc[0],selected.loc["NVDA","ret63_minus_repl"])
nv.to_csv(HERE / "nvda_dev_events.csv", index=False)

view = DataView(bundle, as_of)
inputs = valuations(as_of, targets, view, reverse=True).set_index("ticker")
history = cached_history_value(as_of, targets, view).set_index("ticker")
fields = ["dcf","dcf_high","peer","peer_high","hist_mid","hist_high","price"]
np.testing.assert_allclose(inputs[fields], quotes[quotes.day.eq(day)].set_index("ticker").loc[inputs.index, fields])
funds = view.fundamentals(targets).sort_values(["ticker","datadate"]).groupby("ticker").tail(8).copy()
funds["rps_growth_yoy"] = funds.groupby("ticker").revenue_per_share.pct_change(4,fill_method=None)
funds[["ticker","datadate","accepted_date","filing_date","revenue_per_share","rps_growth_yoy","EPS","ebitda_margin"]].to_csv(HERE/"quarters.csv", index=False)
inputs["raw_growth"] = funds.groupby("ticker").rps_growth_yoy.median()
inputs["current_pe"] = inputs.price/inputs.eps
inputs = inputs.join(history[["multiple_p50","multiple_p75","history_first","history_last","ttm_ebitda_per_share","net_debt_per_share"]])
inputs["current_ev_ebitda"] = (inputs.price+inputs.net_debt_per_share)/inputs.ttm_ebitda_per_share
centers = inputs[["dcf","peer","hist_mid"]]
inputs["spread_ratio"] = centers.max(axis=1)/centers.min(axis=1)
inputs["as_of"] = as_of
inputs.to_csv(HERE / "valuation_inputs.csv")

# Change just one stock's eligibility on this one date. All portfolios start from
# the same original development start, with the same score files and dated cuts.
reference = pd.read_csv(OUT / "experiment_dev_curves.csv", index_col=0, parse_dates=True).B_two_of_three
scores = MonthlyScores()
cf, cf_trades = {}, []
for sym in ["frozen", *targets]:
    override = dict(cuts)
    if sym != "frozen":
        override[day] = cuts[day] - {sym}
    result = backtest(px, params, universe, RemovalScores(scores, override), start, end63.date())
    pre = result.equity.loc[:pd.Timestamp(as_of)]
    np.testing.assert_allclose(pre, reference.loc[pre.index], atol=1e-6, rtol=0)
    if sym == "frozen":
        np.testing.assert_allclose(result.equity, reference.loc[result.equity.index], atol=1e-6, rtol=0)
    cf[sym] = result.equity.loc[pd.Timestamp(as_of):]
    tr = result.trades[result.trades.date.between(day,next_week_open.date())].copy()
    tr["exempt_on_event_date"] = sym
    cf_trades.append(tr)
pd.concat(cf_trades, ignore_index=True).to_csv(HERE/"one_week_exemption_trades.csv", index=False)
curves = pd.DataFrame(cf)
curves.to_csv(HERE/"one_week_exemption_equity.csv")
for sym in targets:
    curves[sym+"_minus_frozen_pp"] = 100*(curves[sym]-curves.frozen)/curves.frozen.iloc[0]
print(selected[["rank","ret63","ret63_minus_repl","A_baseline_pos","B_two_of_three_pos"]].round(5))
print(inputs[["price","dcf","dcf_high","peer","peer_high","hist_mid","hist_high","raw_growth","growth","implied_start_growth","current_pe","peer_multiple","current_ev_ebitda","multiple_p50","WACC" if "WACC" in inputs else "wacc","latest_release"]].T)
print("last close / event / first-week end / next rebalance / evaluation end:", as_of, day, first_week_end, next_week_open,end63)
print(curves.loc[[first_week_end,next_week_open,end63]].round(4))

# Persist while both conditions first remain true: ORIGINAL rank <=10 and
# all three methods veto. Stop immediately on the first week either fails.
episode = []
for t in sorted(dt for dt in quotes.day.unique() if dt >= day):
    prior = dates[dates.get_loc(pd.Timestamp(t))-1].date()
    orig = ranked_at(t, prior)
    pos = orig.index[orig.ticker.eq("NVDA")]
    is_top = len(pos) > 0 and pos[0] < 10
    is_three = "NVDA" in three.get(t, set())
    if not (is_top and is_three):
        stop_reason = dict(first_week_after_episode=t, rank=int(pos[0]+1) if len(pos) else None, unanimous=is_three)
        break
    episode.append(t)
overrides = dict(cuts)
for t in episode:
    overrides[t] = cuts[t] - {"NVDA"}
multi = backtest(px, params, universe, RemovalScores(scores, overrides), start, end63.date())
np.testing.assert_allclose(multi.equity.loc[:pd.Timestamp(as_of)], reference.loc[:pd.Timestamp(as_of)], atol=1e-6, rtol=0)
path = pd.DataFrame({"frozen": cf["frozen"], "exempt_NVDA_first_episode": multi.equity.loc[pd.Timestamp(as_of):]})
path["net_dollar_difference"] = path.exempt_NVDA_first_episode-path.frozen
path["return_difference_pp_since_asof"] = path.net_dollar_difference/path.frozen.iloc[0]*100
path.to_csv(HERE/"nvda_episode_equity.csv")
mt = multi.trades[multi.trades.date.between(day,end63.date()) & multi.trades.symbol.eq("NVDA")]
mt.to_csv(HERE/"nvda_episode_trades.csv",index=False)
print("RUN", episode, stop_reason)
print(path.loc[[str(episode[-1]),"2019-12-31","2020-01-03",str(stop_reason["first_week_after_episode"]),end63.strftime("%Y-%m-%d")]].to_string())
print(mt.to_string(index=False))
