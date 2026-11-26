"""Cache the team's weekly three-method valuations for every growth_tech
candidate, exactly as filter_candidates() sees them (as_of = previous session).

    python build_cache.py dev
    python build_cache.py val
"""
import sys
import time

from common import CACHE, load_course, period, rebalance_days, pd
from shared.data_view import DataView
from shared.guard import student_call
from selection import MonthlyScores
from valuation.student_valuation import valuations
from datetime import timedelta


def main(name):
    start, end = period(name)
    params, prices, univ, bundle = load_course()
    scores = MonthlyScores()
    rows, ranks = [], []
    t0 = time.time()
    for k, dt in enumerate(rebalance_days(prices, start, end)):
        prior = prices.last_session_on_or_before(dt - timedelta(days=1))
        preds = [p for p in scores.get(dt) if p.bucket == "growth_tech"]
        for p in preds:
            ranks.append(dict(day=dt, ticker=p.symbol, rank_score=p.rank_score, weight_score=p.weight_score))
        with student_call():
            df = valuations(prior, [p.symbol for p in preds], DataView(bundle, prior))
        if len(df):
            df = df.assign(day=dt, as_of=pd.Timestamp(prior).date())
            rows.append(df)
        if k % 25 == 0:
            print(name, dt, f"{time.time()-t0:.0f}s", flush=True)
    v = pd.concat(rows, ignore_index=True)
    for c in ("quarter_end", "latest_release"):
        v[c] = v[c].astype(str)
    v.to_parquet(CACHE / f"valuations_{name}.parquet")
    pd.DataFrame(ranks).to_parquet(CACHE / f"ranks_{name}.parquet")
    print("done", len(v), "rows", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
