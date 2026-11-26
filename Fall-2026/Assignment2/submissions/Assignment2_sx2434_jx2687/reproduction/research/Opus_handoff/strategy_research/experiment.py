"""Step 3: minimal three-arm comparison, rules fixed in PREREGISTRATION.md.

    A  baseline (course selection only)
    B  team frozen two_of_three veto
    C  protect_top3: same veto, but the 3 highest rank_score growth_tech
       candidates (index members with a valid close at the decision date)
       are exempt.

    python experiment.py dev
    python experiment.py val     (historical review only)
"""
import sys
from datetime import timedelta

from common import (CACHE, OUT, RemovalScores, backtest, load_course, np, pd, period,
                    rebalance_days, turnover)
from selection import MonthlyScores
from shared.metrics import performance
from valuation.student_valuation import rejected

PROTECT = 3


def contributions(res, prices, cfg_cost=0.0006, initial=100_000.0):
    """Per-stock cumulative after-cost P&L, reconciled to the equity curve."""
    t = res.trades.copy(); t["date"] = pd.to_datetime(t.date)
    eq = res.equity
    q = t.pivot_table(index="date", columns="symbol", values="qty", aggfunc="sum").reindex(eq.index).fillna(0)
    g = t.assign(a=t.qty.abs()).pivot_table(index="date", columns="symbol", values="a", aggfunc="sum").reindex_like(q).fillna(0)
    o = prices.open.reindex_like(q)
    c = prices.close.loc[:eq.index[-1], q.columns].ffill().reindex(eq.index)
    held = q.cumsum()
    cum = (held * c).fillna(0) - ((q * o).fillna(0) + (g * o).fillna(0) * cfg_cost).cumsum()
    np.testing.assert_allclose(cum.sum(axis=1) + initial, eq, rtol=0, atol=1e-6)
    return cum.iloc[-1]


def main(name):
    start, end = period(name)
    params, prices, univ, bundle = load_course()
    v = pd.read_parquet(CACHE / f"valuations_{name}.parquet")
    ranks = pd.read_parquet(CACHE / f"ranks_{name}.parquet")
    closes = prices.close
    rem_B, rem_C, protected_log = {}, {}, []
    for day in rebalance_days(prices, start, end):
        g = v[v.day == day]
        veto = rejected(g, "two_of_three") if len(g) else set()
        rem_B[day] = veto
        prior = prices.last_session_on_or_before(day - timedelta(days=1))
        members = univ.members_at(pd.Timestamp(prior).date())
        last = closes.loc[pd.Timestamp(prior)]
        r = ranks[(ranks.day == day) & ranks.ticker.isin(members)]
        r = r[r.ticker.map(lambda s: np.isfinite(float(last.get(s, np.nan))))]
        top = set(r.sort_values("rank_score", ascending=False).ticker.head(PROTECT))
        rem_C[day] = veto - top
        protected_log.append(dict(day=day, top3=",".join(sorted(top)), vetoed=len(veto),
                                  spared=",".join(sorted(veto & top))))
    base = MonthlyScores()
    runs = {"A_baseline": {}, "B_two_of_three": rem_B, "C_protect_top3": rem_C}
    res, rows, curves = {}, [], {}
    for k, rem in runs.items():
        res[k] = backtest(prices, params, univ, RemovalScores(base, rem), start, end)
        curves[k] = res[k].equity
        res[k].trades.to_csv(OUT / f"experiment_{name}_{k}_trades.csv", index=False)
        rows.append(dict(arm=k, **performance(res[k].equity), trades=len(res[k].trades),
                         turnover_per_year=round(turnover(res[k]), 2)))
    table = pd.DataFrame(rows).set_index("arm")
    table.to_csv(OUT / f"experiment_{name}.csv")
    pd.DataFrame(curves).to_csv(OUT / f"experiment_{name}_curves.csv")
    pl = pd.DataFrame(protected_log); pl.to_csv(OUT / f"experiment_{name}_protected.csv", index=False)

    contrib = pd.DataFrame({k: contributions(r, prices) for k, r in res.items()}).fillna(0)
    contrib["C_minus_B"] = contrib.C_protect_top3 - contrib.B_two_of_three
    contrib["C_minus_A"] = contrib.C_protect_top3 - contrib.A_baseline
    contrib["B_minus_A"] = contrib.B_two_of_three - contrib.A_baseline
    contrib.sort_values("C_minus_B").to_csv(OUT / f"experiment_{name}_stock_pnl.csv")

    # Annual returns for each arm.
    eq = pd.DataFrame(curves)
    yearly = {}
    prev = pd.Series(100_000.0, index=eq.columns)
    for y, g in eq.groupby(eq.index.year):
        yearly[y] = (g.iloc[-1] / prev - 1) * 100; prev = g.iloc[-1]
    yearly = pd.DataFrame(yearly).T.round(2)

    with pd.option_context("display.width", 200):
        print(table.to_string()); print(yearly.to_string())
        print("weeks with any veto spared by protection:", int(pl.spared.ne("").sum()), "of", len(pl))
        print("spared names:", pd.Series([s for x in pl.spared if x for s in x.split(",")]).value_counts().head(10).to_dict())
        cb = contrib.C_minus_B
        print("C-B net $:", round(cb.sum(), 0), " top +:", cb.sort_values(ascending=False).head(5).round(0).to_dict())
        print("C-B top -:", cb.sort_values().head(5).round(0).to_dict())
        top = cb.sort_values(ascending=False)
        print("largest single-stock share of C-B net gain:", round(top.iloc[0] / cb.sum(), 3) if cb.sum() > 0 else "n/a")
        print("C-A top -:", contrib.C_minus_A.sort_values().head(5).round(0).to_dict(),
              " top +:", contrib.C_minus_A.sort_values(ascending=False).head(5).round(0).to_dict())
    yearly.to_csv(OUT / f"experiment_{name}_annual.csv")


if __name__ == "__main__":
    main(sys.argv[1])
