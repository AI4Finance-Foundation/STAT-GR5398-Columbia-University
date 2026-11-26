"""Exact development-only threshold curve. Run with PYTHONHASHSEED=0."""
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "weighted_valuation_research"))
from study import CACHE, MonthlyScores, RemovalScores, backtest, evaluate, period
from selection.config import load
from selection.strategy import target_weights


def cagr(equity):
    years = (len(equity) - 1) / 252
    return 100 * ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def holdings_before(trades, weeks):
    qty = trades.pivot_table(index="date", columns="symbol", values="qty", aggfunc="sum")
    return qty.reindex(weeks).fillna(0).cumsum().shift(1).fillna(0).to_dict("index")


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Use PYTHONHASHSEED=0 to reproduce the frozen run.")
    start, end = period("dev")
    quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
    assert quotes.day.min() >= start and quotes.day.max() <= end and (quotes.as_of < quotes.day).all()
    signals = pd.concat([evaluate(g) for _, g in quotes.groupby("day")], ignore_index=True)
    vetoes = signals.loc[signals.adaptive_veto].copy()
    centers = vetoes[["dcf", "peer", "hist_mid"]]
    vetoes["span"] = centers.max(axis=1) / centers.min(axis=1)
    assert vetoes.span.ge(1).all() and not vetoes.duplicated(["day", "ticker"]).any()
    triggers = {k: g.groupby("day").ticker.agg(set).to_dict()
                for k, g in vetoes[vetoes.span <= 2.5].groupby("span")}
    cuts = sorted({1., 2., 2.3, 2.5, *triggers})
    _, _, params, prices, _, universe = load("selection/configs/growth.json")
    prices.open = prices.open.loc[:str(end)]
    prices.close = prices.close.loc[:str(end)]
    dates = prices.sessions[(prices.sessions >= pd.Timestamp(start)) &
                            (prices.sessions <= pd.Timestamp(end))]
    scores = MonthlyScores()
    week_rows, available = {}, {}
    prev_week = None
    for i, dt in enumerate(dates):
        week = dt.isocalendar()[:2]
        if week == prev_week:
            continue
        prev_week = week
        day = dt.date()
        decision = dates[i - 1].date() if i else day - timedelta(days=1)
        week_rows[day] = scores.get(day)
        opens = prices.open.loc[dt]
        available[day] = {s for s in universe.members_at(decision) if np.isfinite(opens.get(s, np.nan))}
    weeks = list(week_rows)
    dropped = defaultdict(set)
    frozen = pd.read_csv(HERE / "dev_curves.csv", index_col=0, parse_dates=True)
    run = backtest(prices, params, universe, RemovalScores(scores, dropped), start, end)
    np.testing.assert_allclose(run.equity, frozen.baseline, rtol=0, atol=1e-6)
    base = cagr(run.equity)
    pos = holdings_before(run.trades, weeks)
    rows = []
    replays = 1
    for k in cuts:
        changes = triggers.get(k, {})
        dirty = False
        for day, added in changes.items():
            # Only these weeks receive new removals. If their exact target weights
            # stay identical at pretrade holdings, all trades and all later states
            # remain identical by induction. Otherwise replay the course engine.
            before = dropped[day]
            after = before | added
            if not dirty:
                def weights(exclude):
                    picks = tuple(x for x in week_rows[day] if x.symbol not in exclude)
                    return target_weights(picks, params, available[day], pos[day])
                dirty = weights(before) != weights(after)
            dropped[day] = after
        if dirty:
            run = backtest(prices, params, universe, RemovalScores(scores, dropped), start, end)
            pos = holdings_before(run.trades, weeks)
            replays += 1
        if k in (2., 2.3, 2.5):
            np.testing.assert_allclose(run.equity, frozen["K"+str(k)], rtol=0, atol=1e-6)
        perf = cagr(run.equity)
        rows.append(dict(K=k, delta_cagr_pp=perf-base, cagr_pct=perf,
                         veto_name_weeks=sum(map(len, dropped.values()))))
    grid = pd.DataFrame(rows)
    old = pd.read_csv(HERE / "dev_empirical_steps.csv", float_precision="round_trip")
    overlap = grid[grid.K >= 2]
    np.testing.assert_allclose(overlap.K, old.K, rtol=0, atol=1e-13)
    np.testing.assert_allclose(overlap.delta_cagr_pp, old.delta_cagr_pp, rtol=0, atol=1e-8)
    # Representative independent checks at interior values, including plateaus
    # early, midway through the new interval, and directly below the old range.
    old_inputs = pd.read_csv(HERE.parents[1]/"weighted_valuation_research/dev_signals.csv", parse_dates=["day"])
    old_inputs = old_inputs.loc[old_inputs.adaptive_veto].copy()
    old_inputs["day"] = old_inputs.day.dt.date
    assert old_inputs.day.min() >= start and old_inputs.day.max() <= end
    values = old_inputs[["dcf","peer","hist_mid"]]
    for p in [0.05, .5, .95]:
        idx = int(p * (grid.K < 2).sum())
        midpoint = grid.K.iloc[idx:idx+2].mean()
        mask = values.max(axis=1) <= midpoint * values.min(axis=1)
        drops = old_inputs[mask].groupby("day").ticker.agg(set).to_dict()
        direct = backtest(prices, params, universe, RemovalScores(scores, drops), start, end)
        np.testing.assert_allclose(cagr(direct.equity), grid.cagr_pct.iloc[idx], atol=1e-8, rtol=0)
    grid.to_csv(HERE / "dev_full_steps.csv", index=False)
    print(f"{len(grid)} points, {len(triggers)} distinct observed ratios, {replays} full portfolio runs, checks passed.")
    print(grid.iloc[[grid.delta_cagr_pp.idxmax(), grid.delta_cagr_pp.idxmin()]].to_string(index=False))


if __name__ == "__main__":
    main()
