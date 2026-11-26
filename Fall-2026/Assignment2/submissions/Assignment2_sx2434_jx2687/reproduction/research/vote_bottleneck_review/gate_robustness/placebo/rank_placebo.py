"""Gross weekly target-weight placebo using 500 fixed random paths per split.

Conditional on the number of vetoes among overpriced stocks in ranks 1–10,
11–20 and 21+, reselect their identities uniformly without replacement. The
control tests whether the gate chose better names at the same veto intensity.

Each week buys the top ten remaining eligible names fractionally, with 95%
of wealth invested at the course score weights and 5% in cash. Returns run
from the week's first open to the next rebalance open, or the final close;
a missing next open is marked at the last available close. This target-return
proxy excludes costs, whole shares, order priority, drift after a failed fill,
and the course's minimum rebalance thresholds. Random paths draw independently
each week, so veto duration and turnover are not matched. Quantiles summarize
the simulated strategies conditional on the observed data.
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COURSE = ROOT / "valuation_candidate/source_code"
ARCHIVE = ROOT / "Opus_handoff/strategy_research"
sys.path.insert(0, str(COURSE))
from selection.allocation import ScoreAllocation
from shared.universe import TradeableUniverse

PERIODS = {"dev": ("2017-01-03", "2021-12-31"), "val": ("2022-01-03", "2023-12-29")}
N_RANDOM = 500
K = 1.983
ALLOCATION = ScoreAllocation(.05, .15)


def bounded_membership(db, end):
    with sqlite3.connect(db) as con:
        original = (COURSE / "data/fundamentals.db").as_uri() + "?mode=ro"
        con.execute("ATTACH DATABASE ? AS src", (original,))
        for table, field in (("universe_membership", "snapshot_date"), ("symbol_changes", "change_date")):
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM src.{table} WHERE {field} <= ?", (end,))
    return TradeableUniverse.from_db_path(str(db), ["sp500", "nasdaq100"])


def summarize(split, temp):
    start, end = PERIODS[split]
    universe = bounded_membership(temp / f"{split}.db", end)
    source = COURSE / "data/prices/daily_adjusted.parquet"
    frame = pd.read_parquet(source, columns=["date", "symbol", "open", "close"],
                            filters=[("date", ">=", pd.Timestamp(start)), ("date", "<=", pd.Timestamp(end))])
    assert frame.date.min() >= pd.Timestamp(start) and frame.date.max() <= pd.Timestamp(end)
    opens = frame.pivot_table(index="date", columns="symbol", values="open", aggfunc="last")
    closes = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    sessions = closes["SPY"].dropna().index
    closes = closes.reindex(sessions).ffill()
    rank = pd.read_parquet(ARCHIVE / f"cache/ranks_{split}.parquet")
    signal = pd.read_csv(ROOT / f"weighted_valuation_research/{split}_signals.csv", parse_dates=["day", "as_of"])
    assert (signal.as_of < signal.day).all()
    centers = signal[["dcf", "peer", "hist_mid"]]
    signal["gated"] = (centers.max(axis=1) <= K * centers.min(axis=1)) & signal.adaptive_veto
    overpriced = signal[signal.adaptive_veto].groupby("day")
    checks = pd.read_csv(ARCHIVE / f"results/diag_{split}_weekly_attribution.csv", parse_dates=["day"])
    checks = checks[checks.wA > 0].groupby("day")
    rank["day"] = pd.to_datetime(rank.day)
    by_day = rank.groupby("day")
    weeks = sorted(by_day.groups)
    assert weeks[0] == sessions[0] and weeks[-1] <= sessions[-1]
    rng = np.random.default_rng(20261125)
    all_returns = []
    weekly = []
    for i, day in enumerate(weeks):
        last = i + 1 == len(weeks)
        stop = sessions[-1] if last else weeks[i+1]
        # If the next open disappears, mark at the most recent prior close.
        exit_price = closes.loc[stop] if last else opens.loc[stop].fillna(closes.loc[sessions[sessions < stop][-1]])
        prior = sessions[sessions < day][-1] if i else day - pd.Timedelta(days=1)
        names = universe.members_at(prior.date())
        group = by_day.get_group(day).sort_values("rank_score", ascending=False, kind="stable")
        group = group[group.ticker.isin(names & set(opens.loc[day].dropna().index))].reset_index(drop=True)
        names = list(group.ticker)
        assert len(names) == len(set(names))
        model_scores = group.weight_score.to_numpy()
        returns = (exit_price[names] / opens.loc[day,names] - 1).to_numpy()
        active = overpriced.get_group(day) if day in overpriced.groups else signal.iloc[:0]
        pools, counts = [], []
        gated = set(active.loc[active.gated, "ticker"])
        possible = set(active.ticker)
        actual = {j for j,ticker in enumerate(names) if ticker in gated}
        for first, after in ((0,10), (10,20), (20,len(names))):
            pool = [j for j in range(first, min(after, len(names))) if names[j] in possible]
            pools.append(pool)
            counts.append(len(actual.intersection(pool)))
        assert sum(counts) == len(actual)
        future = {}

        def gross(drop):
            picks = tuple(j for j in range(len(names)) if j not in drop)[:10]
            if picks not in future:
                w = ALLOCATION.weights([(j,model_scores[j]) for j in picks], 1.0)
                assert np.isfinite(returns[list(w)]).all()
                future[picks] = .95 * sum(w[j] * returns[j] for j in w)
            return picks, future[picks]

        base, observed = gross(set()), gross(actual)
        saved = checks.get_group(day).set_index("ticker").wA
        weight = ALLOCATION.weights([(names[j],model_scores[j]) for j in base[0]], 1.0)
        assert set(weight) == set(saved.index)
        assert max(abs(weight[t]-saved[t]) for t in weight) < .000051
        fixed = {j for pool, count in zip(pools, counts) if count == len(pool) for j in pool}
        changed = 0
        row = [base[1], observed[1]]
        for _ in range(N_RANDOM):
            drop = fixed.copy()
            for pool,count in zip(pools,counts):
                if 0 < count < len(pool):
                    drop.update(rng.choice(pool,size=count,replace=False))
            chosen, result = gross(drop)
            changed += chosen != observed[0]
            row.append(result)
        all_returns.append(row)
        weekly.append(dict(day=day.date(), end=stop.date(), eligible_overpriced=sum(map(len,pools)),
                           matched_vetoes=sum(counts), top10_vetoes=counts[0], next10_vetoes=counts[1],
                           any_swaps_feasible=any(0<c<len(g) for g,c in zip(pools,counts)),
                           changed_path_fraction=changed/N_RANDOM, unique_target_lists=len(future),
                           baseline_gross=base[1], observed_gross=observed[1], placebo_mean_gross=np.mean(row[2:])))
    arr = np.array(all_returns)
    cagr = 100 * np.expm1(np.log1p(arr).sum(axis=0) * 252/(len(sessions)-1))
    base, gate, null = cagr[0], cagr[1], cagr[2:]
    w = pd.DataFrame(weekly)
    w.to_csv(HERE / f"{split}_weeks.csv", index=False)
    return dict(period=split, weeks=len(w), rank_band_swappable_weeks=w.any_swaps_feasible.sum(),
                observed_targets_changeable_weeks=(w.changed_path_fraction>0).sum(),
                actual_vs_base_target_return_changed_weeks=(w.baseline_gross!=w.observed_gross).sum(),
                paths=N_RANDOM, base_proxy_cagr=base, gate_proxy_cagr=gate,
                placebo_mean_cagr=null.mean(), placebo_median_cagr=np.median(null),
                placebo_q05_cagr=np.quantile(null,.05), placebo_q95_cagr=np.quantile(null,.95),
                empirical_share_placebo_ge_gate=np.mean(null>=gate),
                gate_minus_placebo_median_pp=gate-np.median(null))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        output = pd.DataFrame([summarize(period, Path(tmp)) for period in PERIODS])
    output.to_csv(HERE / "summary.csv", index=False)
    print(output.to_string(index=False))
