"""Development-only diagnostic using the frozen signals, course ranks and engine."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Opus_handoff/strategy_research"))
from common import CACHE, OUT, RemovalScores, backtest, load_course, np, pd, period
from selection import MonthlyScores
from shared.metrics import performance
from valuation.student_valuation import rejected

start, end = period("dev")
vals = pd.read_parquet(CACHE / "valuations_dev.parquet")
assert vals.day.min() >= start and vals.day.max() <= end and (vals.as_of < vals.day).all()
upper = vals[["dcf_high", "peer_high", "hist_high"]].set_axis(["DCF", "PE", "EV"], axis=1)
mid = vals[["dcf", "peer", "hist_mid"]].set_axis(upper.columns, axis=1)
votes = upper.lt(vals.price, axis=0)
exact = upper.notna().all(axis=1) & mid.notna().all(axis=1) & votes.sum(axis=1).eq(2)
third_mid = mid.where(~votes).max(axis=1)
vals["category"] = np.select([exact & third_mid.ge(vals.price), exact], ["center", "band"], default="other")
vals.loc[votes.sum(axis=1).eq(3), "category"] = "unanimous"
vals.loc[upper.notna().sum(axis=1).eq(2) & votes.sum(axis=1).eq(2), "category"] = "missing"
vals["dissent"] = (~votes).idxmax(axis=1).where(exact)
vals["frozen"] = votes.sum(axis=1).ge(2)
assert all(set(g.loc[g.frozen, "ticker"]) == rejected(g, "two_of_three") for _, g in vals.groupby("day"))
tally = pd.read_csv(ROOT / "fable_review/phase2_checks/tally_dev_val.csv", parse_dates=["date"])
tally = tally[tally.date.dt.date.between(start, end)].copy()
tally["day"] = tally.date.dt.date
tally = tally.merge(vals[["day","ticker","category","dissent","frozen"]], on=["day","ticker"], how="left", validate="one_to_one")
assert (tally.team_veto == tally.frozen.eq(True)).all()
rows = []
for day, g in tally.groupby("day"):
    g = g.sort_values("rank")
    base = g.head(10)
    veto = base[base.team_veto]
    repl = g[~g.team_veto].head(10).query("rank > 10")
    if veto.empty or len(repl) != len(veto) or pd.concat([veto.ret63, repl.ret63]).isna().any():
        continue
    for row in veto.itertuples():
        rows.append(dict(day=day, ticker=row.ticker, category=row.category, dissent=row.dissent,
                         spread=row.ret63-repl.ret63.mean(), nvda_repl="NVDA" in set(repl.ticker),
                         rank_control=row.ret63-base.ret63.mean() if base.ret63.notna().all() else np.nan))
events = pd.DataFrame(rows)
events.to_csv(HERE / "dev_events.csv", index=False)
for label, subset in [("all",events), ("ex_nvda_both",events[(events.ticker != "NVDA") & ~events.nvda_repl])]:
    print(label, subset.groupby("category").agg(n=("spread","size"), names=("ticker","nunique"), weeks=("day","nunique"),
              spread=("spread","mean"), median=("spread","median"), vs_top10=("rank_control","mean")).round(4))
    print(subset[subset.category!="other"].groupby(["category","dissent"]).spread.agg(["size","mean"]).round(4))
params, prices, universe, _ = load_course()
base_scores = MonthlyScores()
prior = pd.read_csv(OUT / "experiment_dev_curves.csv", index_col=0, parse_dates=True)
results = []
for label in ["A_baseline", "B_two_of_three", "keep_center", "keep_band"]:
    drops = {}
    for day, g in vals.groupby("day"):
        flag = g.frozen
        if label.startswith("keep_"):
            flag = flag & g.category.ne(label.removeprefix("keep_"))
        drops[day] = set(g.loc[flag, "ticker"])
    scores = RemovalScores(base_scores, {} if label == "A_baseline" else drops)
    r = backtest(prices, params, universe, scores, start, end)
    if label in prior:
        assert prior.index.equals(r.equity.index)
        np.testing.assert_allclose(r.equity, prior[label], atol=1e-6, rtol=0)
    results.append(dict(rule=label, **performance(r.equity)))
tbl = pd.DataFrame(results)
tbl.to_csv(HERE / "dev_portfolio.csv", index=False)
print(tbl.to_string(index=False))
