"""Step 1 diagnosis: where does the two-of-three filter lose money in dev?

Replays baseline (A) and the frozen two_of_three rule (B) from the cached
valuations, checks both against the official course outputs, then attributes
the weekly target-weight difference to removed / added / reweighted names.

    python diagnose.py dev      (val only as a historical cross-check)
"""
import sys
from pathlib import Path

from common import (CACHE, OUT, SRC, RemovalScores, backtest, load_course, np, pd, period,
                    rebalance_days, turnover)
from selection import MonthlyScores
from valuation.student_valuation import rejected
from shared.metrics import performance


def flags_table(v):
    v = v.copy()
    v["v_dcf"] = v.price.gt(v.dcf_high)
    v["v_peer"] = v.price.gt(v.peer_high)
    v["v_hist"] = v.price.gt(v.hist_high)
    v["votes"] = v[["v_dcf", "v_peer", "v_hist"]].sum(axis=1)
    v["combo"] = (v.v_dcf.map({True: "D", False: ""}) + v.v_peer.map({True: "P", False: ""})
                  + v.v_hist.map({True: "H", False: ""}))
    return v


def removal_sets(v, rule):
    out = {}
    for day, g in v.groupby("day"):
        out[day] = rejected(g, rule)
    return out


def main(name):
    start, end = period(name)
    params, prices, univ, bundle = load_course()
    v = pd.read_parquet(CACHE / f"valuations_{name}.parquet")
    days = rebalance_days(prices, start, end)
    base = MonthlyScores()
    rem = removal_sets(v, "two_of_three")
    A = backtest(prices, params, univ, RemovalScores(base, {}), start, end)
    B = backtest(prices, params, univ, RemovalScores(base, rem), start, end)

    # Reconcile with the official course runs saved in the frozen tree.
    for tag, res in (("A_selection", A), ("valuation", B)):
        off = SRC / "output" / f"{name}_{tag}"
        if not off.exists():
            off = SRC / "output" / f"{name}_B_valuation"
        eq = pd.read_csv(off / "equity.csv", index_col=0, parse_dates=True).iloc[:, 0]
        np.testing.assert_allclose(res.equity.values, eq.values, rtol=0, atol=1e-6)
        assert len(res.trades) == len(pd.read_csv(off / "trades.csv"))
    print("replay matches official", name, "A and B runs")

    # Weekly open-to-next-rebalance-open returns.
    opens = prices.open
    idx = [pd.Timestamp(d) for d in days] + [pd.Timestamp(prices.sessions[prices.sessions > pd.Timestamp(end)][0]) if (prices.sessions > pd.Timestamp(end)).any() else prices.sessions[-1]]
    px = opens.reindex(idx).ffill()
    fwd = px.shift(-1) / px - 1
    closes = prices.close.ffill()

    wa = {r["date"]: r["weights"] for r in A.weekly_targets}
    wb = {r["date"]: r["weights"] for r in B.weekly_targets}
    vf = flags_table(v).set_index(["day", "ticker"])
    rows = []
    for d in days:
        a, b = wa.get(d, {}), wb.get(d, {})
        r = fwd.loc[pd.Timestamp(d)]
        prev_close = closes.loc[:pd.Timestamp(d) - pd.Timedelta(days=1)]
        for s in set(a) | set(b):
            ret = float(r.get(s, np.nan))
            kind = "removed" if s in a and s not in b else "added" if s in b and s not in a else "common"
            rec = dict(day=d, ticker=s, kind=kind, wA=a.get(s, 0.0), wB=b.get(s, 0.0), ret=ret,
                       contrib=(b.get(s, 0.0) - a.get(s, 0.0)) * (0 if np.isnan(ret) else ret))
            if (d, s) in vf.index:
                f = vf.loc[(d, s)]
                rec.update(votes=f.votes, combo=f.combo, price=f.price)
            hist = prev_close[s].dropna() if s in prev_close else pd.Series(dtype=float)
            if len(hist) >= 200:
                rec["above_sma200"] = bool(hist.iloc[-1] > hist.iloc[-200:].mean())
                rec["ret_126"] = float(hist.iloc[-1] / hist.iloc[-127] - 1)
            rows.append(rec)
    att = pd.DataFrame(rows)
    att["month"] = pd.to_datetime(att.day).dt.to_period("M")
    first = att.groupby("month").day.transform("min")
    att["month_start"] = att.day.eq(first)
    att.to_csv(OUT / f"diag_{name}_weekly_attribution.csv", index=False)

    lines = []
    def p(*x):
        s = " ".join(str(y) for y in x); print(s); lines.append(s)

    pa, pb = performance(A.equity), performance(B.equity)
    p(f"## {name}: baseline vs two_of_three")
    p("A", pa, "trades", len(A.trades), "turnover", round(turnover(A), 2))
    p("B", pb, "trades", len(B.trades), "turnover", round(turnover(B), 2))
    weekly = att.groupby("day").contrib.sum()
    p("sum of weekly target-weight contribution diffs (simple sum, B-A):", round(weekly.sum(), 4))
    p("by kind:", att.groupby("kind").contrib.sum().round(4).to_dict())
    p("weeks where B targets differ from A:", int((att.kind != "common").groupby(att.day).any().sum()), "of", len(days))

    # Trading split: month-start vs intra-month notional.
    for tag, res in (("A", A), ("B", B)):
        t = res.trades.copy(); t["date"] = pd.to_datetime(t.date)
        # month start = first rebalance day of the calendar month (new course scores)
        ds = pd.to_datetime(pd.Series(days))
        firsts = set(ds.groupby(ds.dt.to_period("M")).min())
        mstart = t.date.isin(firsts)
        p(f"{tag} notional at month-start rebalances {t.notional.abs()[mstart].sum():,.0f}; intra-month {t.notional.abs()[~mstart].sum():,.0f}; intra-month trades {int((~mstart).sum())}")

    rm = att[att.kind == "removed"]
    p("removed (stock-weeks):", len(rm), " month-start:", int(rm.month_start.sum()), " intra-month:", int((~rm.month_start).sum()))
    p("removed contribution, month-start vs intra-month:",
      round(rm[rm.month_start].contrib.sum(), 4), round(rm[~rm.month_start].contrib.sum(), 4))
    ad = att[att.kind == "added"]
    p("added contribution, month-start vs intra-month:",
      round(ad[ad.month_start].contrib.sum(), 4), round(ad[~ad.month_start].contrib.sum(), 4))
    p("removed contribution by trend state (above SMA200):", rm.groupby("above_sma200").contrib.agg(["count", "sum"]).round(4).to_dict())
    p("removed contribution by vote combo:", rm.groupby("combo").contrib.agg(["count", "sum"]).round(4).to_dict())
    p("removed mean weekly return:", round(rm.ret.mean(), 5), " added mean weekly return:", round(ad.ret.mean(), 5))
    by_stock = rm.groupby("ticker").agg(weeks=("contrib", "size"), contrib=("contrib", "sum"), mean_ret=("ret", "mean")).sort_values("contrib")
    p("top removed names by cost (contribution of removing them, before replacement gain):")
    p(by_stock.head(8).round(4).to_string())
    p(by_stock.tail(5).round(4).to_string())
    by_year = att.assign(year=pd.to_datetime(att.day).dt.year).groupby(["year", "kind"]).contrib.sum().unstack().round(4)
    p(by_year.to_string())

    # Flip frequency for names that were in the baseline top 10.
    top = att[att.wA > 0].sort_values(["ticker", "day"]).copy()
    top["removedB"] = top.kind.eq("removed")
    top["flip"] = top.groupby("ticker").removedB.diff().fillna(False).astype(bool) & top.groupby(["ticker", "month"]).day.transform("size").gt(1) & ~top.month_start
    p("intra-month flag flips on baseline top-10 names:", int(top.flip.sum()))
    (OUT / f"diag_{name}.txt").write_text("\n".join(lines))
    pd.DataFrame({"A": A.equity, "B": B.equity}).to_csv(OUT / f"diag_{name}_curves.csv")


if __name__ == "__main__":
    main(sys.argv[1])
