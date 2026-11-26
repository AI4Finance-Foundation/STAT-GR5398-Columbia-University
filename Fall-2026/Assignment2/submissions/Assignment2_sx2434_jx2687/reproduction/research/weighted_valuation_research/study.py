"""Run from any directory using assignment2_baseline/.venv/bin/python with PYTHONHASHSEED=0."""
from datetime import timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent / "Opus_handoff" / "strategy_research"
sys.path.insert(0, str(RESEARCH))
from common import CACHE, RemovalScores, backtest, load_course, np, pd, period, rebalance_days, turnover
from selection import MonthlyScores
from shared.metrics import performance
from valuation.student_valuation import rejected


def evaluate(g):
    centers = g[["dcf", "peer", "hist_mid"]]
    valid = (centers.gt(0) & np.isfinite(centers)).all(axis=1)
    valid &= np.isfinite(g[["dcf_high", "dcf_low", "price"]]).all(axis=1) & g.price.gt(0)
    valid &= g.dcf_high.ge(g.dcf) & g.dcf_low.le(g.dcf)
    x = g.loc[valid, ["day", "as_of", "ticker", "price", "dcf", "peer", "hist_mid"]].copy()
    x["width"] = (g.loc[valid, "dcf_high"] - g.loc[valid, "dcf_low"]) / x.dcf
    x["cutoff"] = x.width.quantile(.75)
    x["sensitive"] = x.width > x.cutoff
    x["equal_value"] = centers.loc[valid].mean(axis=1)
    x["adaptive_value"] = x.equal_value.where(~x.sensitive, (2*x.dcf + 5*x.peer + 5*x.hist_mid) / 12)
    x["equal_veto"] = x.equal_value < .85*x.price
    x["adaptive_veto"] = x.adaptive_value < .85*x.price
    assert not ((x.equal_veto != x.adaptive_veto) & ~x.sensitive).any()
    assert len(x) == int(valid.sum()) and x.ticker.notna().all()
    return x


def weekly_holdings(trades, weeks):
    quantities = trades.pivot_table(index="date", columns="symbol", values="qty", aggfunc="sum")
    return quantities.reindex(weeks).fillna(0).cumsum().gt(0)


def uncertainty(curves):
    log_returns = np.log(curves / curves.shift()).dropna()
    pairs = [("adaptive","A"), ("equal","A"), ("adaptive","equal")]
    rng = np.random.default_rng(20261124)
    n = len(log_returns)
    samples = []
    for _ in range(5000):
        starts = rng.integers(0, n, size=(n+19)//20)
        idx = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
        samples.append(np.expm1(252 * log_returns.to_numpy()[idx].mean(axis=0))*100)
    draws = pd.DataFrame(samples, columns=curves.columns)
    return [dict(rule=a, reference=b, lo90=(draws[a]-draws[b]).quantile(.05),
                 hi90=(draws[a]-draws[b]).quantile(.95)) for a,b in pairs]


def main():
    params, prices, universe, _ = load_course()
    scores = MonthlyScores()
    results, diagnostics, intervals = [], [], []
    for split in ("dev","val"):
        start, end = period(split)
        weeks = rebalance_days(prices, start, end)
        quotes = pd.read_parquet(CACHE / f"valuations_{split}.parquet")
        assert set(quotes.day) <= set(weeks) and not quotes.duplicated(["day","ticker"]).any()
        removals = {label:{} for label in ("A","team","equal","adaptive")}
        audit = []
        for day,g in quotes.groupby("day"):
            asof = pd.Timestamp(prices.last_session_on_or_before(day - timedelta(days=1)))
            candidates = scores.get(day)
            growth = {r.symbol for r in candidates if r.bucket == "growth_tech"}
            assert pd.to_datetime(g.as_of).eq(asof).all() and pd.Timestamp(scores.cutoff) <= asof
            assert set(g.ticker) <= growth.intersection(universe.members_at(asof.date()))
            removals["team"][day] = rejected(g, "two_of_three")
            x = evaluate(g)
            for label in ("equal","adaptive"):
                removals[label][day] = set(x.loc[x[label+"_veto"],"ticker"])
            audit.append(x)
        decisions = pd.concat(audit, ignore_index=True)
        decisions.to_csv(ROOT/f"{split}_signals.csv", index=False)
        reference = pd.read_csv(RESEARCH/"results"/f"experiment_{split}_curves.csv", index_col=0, parse_dates=True)
        curves, holdings = {}, {}
        for label,by_day in removals.items():
            r = backtest(prices, params, universe, RemovalScores(scores, by_day), start, end)
            if label in ("A","team"):
                expected = reference["A_baseline" if label=="A" else "B_two_of_three"]
                assert expected.index.equals(r.equity.index)
                np.testing.assert_allclose(r.equity,expected,atol=1e-6,rtol=0)
            curves[label] = r.equity
            holdings[label] = weekly_holdings(r.trades, weeks)
            results.append(dict(split=split,rule=label,**performance(r.equity),trades=len(r.trades),turnover=turnover(r)))
        eq = pd.DataFrame(curves)
        eq.to_csv(ROOT/f"{split}_equity.csv")
        left,right = holdings["equal"].align(holdings["adaptive"],fill_value=False)
        diagnostics.append(dict(split=split,weeks=len(weeks),complete=len(decisions),sensitive=int(decisions.sensitive.sum()),
          new_veto=int((decisions.adaptive_veto & ~decisions.equal_veto).sum()),rescinded=int((~decisions.adaptive_veto & decisions.equal_veto).sum()),
          different_holdings_weeks=int(left.ne(right).any(axis=1).sum())))
        intervals.extend(dict(split=split,**r) for r in uncertainty(eq))
    for filename,rows in [("summary",results),("diagnostics",diagnostics),("intervals",intervals)]:
        t=pd.DataFrame(rows)
        t.to_csv(ROOT/f"{filename}.csv",index=False)
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
