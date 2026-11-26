"""Direct development replays isolating the two edges of the full-universe peak."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "weighted_valuation_research"))
from study import CACHE, MonthlyScores, RemovalScores, evaluate, period, backtest, rebalance_days
from selection.config import load
from experiment import contributions

def cagr(equity):
    return 100 * ((equity.iloc[-1]/equity.iloc[0]) ** (252/(len(equity)-1))-1)

def main():
    assert os.environ.get("PYTHONHASHSEED") == "0"
    start, end = period("dev")
    prices_on_day = pd.read_parquet(CACHE / "valuations_dev.parquet")
    assert prices_on_day.day.between(start,end).all() and (prices_on_day.as_of < prices_on_day.day).all()
    grid = pd.read_csv(HERE.parent / "dev_full_steps.csv", float_precision="round_trip")
    plateau = grid[grid.delta_cagr_pp.eq(grid.delta_cagr_pp.max())]
    entry = plateau.K.iloc[0]
    exit_cut = grid.loc[plateau.index[-1]+1,"K"]
    cuts = {"baseline":1., "entry_left":np.nextafter(entry,-np.inf), "entry":entry,
            "K1.983":1.983, "exit_left":np.nextafter(exit_cut,-np.inf), "exit":exit_cut, "K2.0":2.}
    _, _, params, prices, _, universe = load("selection/configs/growth.json")
    prices.open = prices.open.loc[:str(end)]
    prices.close = prices.close.loc[:str(end)]
    weeks = rebalance_days(prices,start,end)
    summaries, deltas, triggers = [], [], []
    for arm in ["all","excluded_fixed","excluded_recomputed"]:
        keep_nvda = arm == "all"
        estimates = prices_on_day
        if arm == "excluded_recomputed":
            estimates = estimates[estimates.ticker.ne("NVDA")]
        signals = pd.concat([evaluate(g) for _,g in estimates.groupby("day")])
        eligible = signals[signals.adaptive_veto].copy()
        if not keep_nvda:
            eligible = eligible[eligible.ticker.ne("NVDA")]
        mids = eligible[["dcf","peer","hist_mid"]]
        eligible["span"] = mids.max(axis=1)/mids.min(axis=1)
        for boundary, k in [("entry",entry),("exit",exit_cut)]:
            r = eligible[eligible.span.eq(k)][["day","ticker","span"]].copy()
            r["arm"] = arm
            r["edge"] = boundary
            triggers.append(r)
        base_scores = MonthlyScores()
        run_pnl, run_equity = {}, {}
        for label,k in cuts.items():
            removed = eligible[eligible.span.le(k)].groupby("day").ticker.agg(set).to_dict()
            if not keep_nvda:
                removed = {day: removed.get(day,set()) | {"NVDA"} for day in weeks}
            direct = backtest(prices, params, universe, RemovalScores(base_scores,removed),start,end)
            assert direct.equity.index.max().date()<=end
            if not keep_nvda:
                assert not direct.trades.symbol.eq("NVDA").any()
            run_pnl[label] = contributions(direct,prices)
            run_equity[label] = direct.equity
            spread = cagr(direct.equity)-cagr(run_equity["baseline"])
            if keep_nvda:
                expected = grid.loc[grid.K<=k].iloc[-1]
                np.testing.assert_allclose(spread,expected.delta_cagr_pp,rtol=0,atol=1e-8)
            summaries.append(dict(arm=arm, label=label,cutoff=k,cagr_pct=cagr(direct.equity),
                                  alpha_pp=spread,final_value=direct.equity.iloc[-1]))
        attribution = pd.DataFrame(run_pnl).fillna(0)
        for before,after in [("baseline","K1.983"),("baseline","K2.0"),("entry_left","entry"),
                             ("exit_left","exit"),("K2.0","K1.983")]:
            pnl = attribution[after]-attribution[before]
            total = run_equity[after].iloc[-1]-run_equity[before].iloc[-1]
            np.testing.assert_allclose(pnl.sum(),total,atol=1e-6,rtol=0)
            for ticker, value in pnl.items():
                deltas.append(dict(arm=arm,before=before,after=after,ticker=ticker,delta_usd=value))
    results = pd.DataFrame(summaries)
    results.to_csv(HERE/"dev_peak_direct.csv", index=False)
    pd.DataFrame(deltas).to_csv(HERE/"dev_peak_pnl.csv", index=False)
    pd.concat(triggers).to_csv(HERE/"dev_peak_triggers.csv", index=False)
    print(results.round(5).to_string(index=False))
    print(pd.concat(triggers).to_string(index=False))

if __name__ == "__main__":
    main()
