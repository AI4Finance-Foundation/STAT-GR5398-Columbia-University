"""Development-only NVDA leave-one-name-out replay, excluding NVDA before allocation."""
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from full_curve_dev import (
    CACHE, MonthlyScores, RemovalScores, backtest, cagr, evaluate, holdings_before,
    load, period, target_weights,
)

def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 for a deterministic run.")
    start, end = period("dev")
    quotes = pd.read_parquet(CACHE / "valuations_dev.parquet")
    assert quotes.day.min() >= start and quotes.day.max() <= end and (quotes.as_of < quotes.day).all()
    quotes_ex = quotes.loc[quotes.ticker.ne("NVDA")]
    signals = pd.concat([evaluate(group) for _, group in quotes_ex.groupby("day")], ignore_index=True)
    vetoes = signals[signals.adaptive_veto].copy()
    centers = vetoes[["dcf", "peer", "hist_mid"]]
    vetoes["span"] = centers.max(axis=1) / centers.min(axis=1)
    assert vetoes.span.ge(1).all() and not vetoes.duplicated(["day","ticker"]).any()
    triggers = {span: group.groupby("day").ticker.agg(set).to_dict()
                for span, group in vetoes[vetoes.span <= 2.5].groupby("span")}
    fixed = [1.,1.983,2.,2.3,2.5]
    cuts = sorted({*fixed, *triggers})
    _, _, params, prices, _, universe = load("selection/configs/growth.json")
    prices.open = prices.open.loc[:str(end)]
    prices.close = prices.close.loc[:str(end)]
    dates = prices.sessions[(prices.sessions >= pd.Timestamp(start)) & (prices.sessions <= pd.Timestamp(end))]
    scores = MonthlyScores()
    weeks = []
    rankings = {}
    eligible = {}
    for i, dt in enumerate(dates):
        if i and dt.isocalendar()[:2] == dates[i-1].isocalendar()[:2]:
            continue
        day = dt.date()
        prior = dates[i-1].date() if i else day - timedelta(days=1)
        weeks.append(day)
        rankings[day] = tuple(row for row in scores.get(day) if row.symbol != "NVDA")
        assert pd.Timestamp(scores.cutoff) <= pd.Timestamp(prior)
        opening = prices.open.loc[dt]
        eligible[day] = {t for t in universe.members_at(prior) if np.isfinite(opening.get(t,np.nan))}
    ex_scores = RemovalScores(scores, {day: {"NVDA"} for day in weeks})
    dropped = defaultdict(set)

    def replay(removals):
        result = backtest(prices, params, universe, RemovalScores(ex_scores, removals), start, end)
        assert result.trades.symbol.ne("NVDA").all()
        assert set(result.trades.date) <= set(weeks)
        return result

    run = replay({})
    baseline = cagr(run.equity)
    pos = holdings_before(run.trades, weeks)
    equity = {"baseline": run.equity}
    grid = []
    full_runs = 1
    for k in cuts:
        changed_targets = False
        for day, additions in triggers.get(k,{}).items():
            before = dropped[day]
            after = before | additions
            if not changed_targets:
                def targets(excluded):
                    rows = tuple(row for row in rankings[day] if row.symbol not in excluded)
                    return target_weights(rows, params, eligible[day], pos[day])
                changed_targets = targets(before) != targets(after)
            dropped[day] = after
        if changed_targets:
            run = replay(dropped)
            pos = holdings_before(run.trades, weeks)
            full_runs += 1
        if k in fixed:
            equity[f"K{k}"] = run.equity
        growth = cagr(run.equity)
        grid.append(dict(K=k, cagr_pct=growth, delta_cagr_pp=growth-baseline,
                         veto_name_weeks=sum(map(len,dropped.values()))))
    grid = pd.DataFrame(grid)
    # Independent single-wrapper implementation for fixed K and interior plateaus.
    for k in fixed+[1.31, 1.715, 1.899, 1.984, 2.101, 2.422]:
        active = vetoes.loc[centers.max(axis=1).le(k * centers.min(axis=1))]
        weekly = active.groupby("day").ticker.agg(set).to_dict()
        both = {day: weekly.get(day,set()) | {"NVDA"} for day in weeks}
        check = backtest(prices,params,universe,RemovalScores(scores,both),start,end)
        assert check.trades.symbol.ne("NVDA").all()
        if k in fixed:
            np.testing.assert_allclose(equity[f"K{k}"],check.equity,atol=1e-6,rtol=0)
        nearest = grid[grid.K <= k].iloc[-1]
        np.testing.assert_allclose(cagr(check.equity),nearest.cagr_pct,atol=1e-8,rtol=0)
    # Report sensitivity to recalibrating the cross-section after removing NVDA.
    old = pd.concat([evaluate(g) for _, g in quotes.groupby("day")])
    comp = signals.merge(old[old.ticker.ne("NVDA")],on=["day","ticker"],suffixes=("_ex","_orig"),validate="1:1")
    vals = comp[["dcf_ex","peer_ex","hist_mid_ex"]]
    comp["ratio"] = vals.max(axis=1)/vals.min(axis=1)
    cand = comp.adaptive_veto_ex.ne(comp.adaptive_veto_orig)
    comp.loc[cand, ["day","ticker","ratio","adaptive_veto_orig","adaptive_veto_ex"]].to_csv(HERE/"recalibration_changes.csv",index=False)
    grid.to_csv(HERE/"dev_full_steps.csv",index=False)
    pd.DataFrame(equity).to_csv(HERE/"dev_fixed_curves.csv")
    print(f"{len(grid)} full-grid rows, {full_runs} course runs; 11 independent checks passed. "
          f"Cross-section recalibration switches {cand.sum()} pre-gate candidate-weeks, "
          f"{(cand & comp.ratio.le(2.5)).sum()} in the tested range.")
    peak = grid.delta_cagr_pp.max()
    print(grid[grid.K.isin(fixed) | np.isclose(grid.delta_cagr_pp,peak)].to_string(index=False))

if __name__ == "__main__":
    main()
