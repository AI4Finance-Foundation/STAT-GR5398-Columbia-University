"""Fixed development/validation comparison; use the parent experiment's frozen DCF trigger."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from study import (CACHE, RESEARCH, MonthlyScores, RemovalScores, backtest, evaluate,
                   load_course, np, pd, performance, period, rebalance_days, turnover, weekly_holdings)

def main():
    params, prices, universe, _ = load_course()
    scores = MonthlyScores()
    summaries, changes = [], []
    labels = ("A","equal","adaptive","adaptive_top20","adaptive_spread","equal_both","adaptive_both",
              "adaptive_both_two_ok","median_two_ok")
    for split in ("dev","val"):
        dates = period(split)
        weeks = rebalance_days(prices,*dates)
        drops = {name:{} for name in labels}
        quotes = pd.read_parquet(CACHE/f"valuations_{split}.parquet")
        for day, g in quotes.groupby("day"):
            asof = pd.Timestamp(g.as_of.iloc[0])
            assert g.as_of.nunique() == 1 and asof.date() < day
            members = universe.members_at(asof.date())
            growth = [r for r in scores.get(day) if r.bucket == "growth_tech" and r.symbol in members]
            assert pd.Timestamp(scores.cutoff) <= asof
            top = {r.symbol for r in sorted(growth,key=lambda r:-r.rank_score)[:20]}
            x = evaluate(g)
            centers = x[["dcf","peer","hist_mid"]]
            agree = centers.max(axis=1) <= 2.5*centers.min(axis=1)
            top_mask = x.ticker.isin(top)
            masks = {"equal":x.equal_veto, "adaptive":x.adaptive_veto,
                     "adaptive_top20":x.adaptive_veto & top_mask, "adaptive_spread":x.adaptive_veto & agree,
                     "equal_both":x.equal_veto & agree & top_mask, "adaptive_both":x.adaptive_veto & agree & top_mask}
            for label,mask in masks.items():
                drops[label][day] = set(x.loc[mask,"ticker"])
            all_centers = g[["dcf","peer","hist_mid"]].where(lambda v: np.isfinite(v) & v.gt(0))
            count = all_centers.count(axis=1)
            eligible = (count >= 2) & (all_centers.max(axis=1)<=2.5*all_centers.min(axis=1)) & g.ticker.isin(top)
            median_drop = eligible & (all_centers.median(axis=1)<.85*g.price)
            drops["median_two_ok"][day] = set(g.loc[median_drop,"ticker"])
            # With only two estimates use their midpoint; the sensitivity rule was defined for three.
            drops["adaptive_both_two_ok"][day] = drops["adaptive_both"][day] | set(g.loc[median_drop & count.eq(2),"ticker"])
        ref = pd.read_csv(HERE.parent/f"{split}_equity.csv",index_col=0,parse_dates=True)
        portfolios = {}
        for label in labels:
            r = backtest(prices,params,universe,RemovalScores(scores,drops[label]),*dates)
            if label in ("A","equal","adaptive"):
                assert ref.index.equals(r.equity.index)
                np.testing.assert_allclose(ref[label],r.equity,rtol=0,atol=1e-6)
            portfolios[label] = weekly_holdings(r.trades,weeks)
            summaries.append(dict(split=split,rule=label,**performance(r.equity),turnover=turnover(r),trades=len(r.trades)))
        for before,after in [("adaptive","adaptive_top20"),("adaptive","adaptive_spread"),("adaptive_spread","adaptive_both"),("adaptive","adaptive_both"),("equal_both","adaptive_both")]:
            a,b=portfolios[before].align(portfolios[after],fill_value=False)
            changes.append(dict(split=split,before=before,after=after,changed_weeks=int(a.ne(b).any(axis=1).sum()),weeks=len(weeks)))
    for filename,rows in [("summary",summaries),("holdings",changes)]:
        df=pd.DataFrame(rows)
        df.to_csv(HERE/f"{filename}.csv",index=False)
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
