"""Export the rank, frozen veto and matured return used by the NVDA and ASML case studies."""
from datetime import timedelta
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Opus_handoff/strategy_research"))
from common import CACHE, load_course, np, pd, period, rebalance_days
from selection import BUCKET, MonthlyScores
from valuation.student_valuation import rejected


def main():
    _, prices, universe, _ = load_course()
    scores = MonthlyScores()
    rows = []
    for split in ("dev", "val"):
        start, end = period(split)
        valuations = pd.read_parquet(CACHE / f"valuations_{split}.parquet")
        for day in rebalance_days(prices, start, end):
            stamp = pd.Timestamp(day)
            as_of = pd.Timestamp(prices.last_session_on_or_before(day - timedelta(days=1)))
            veto = rejected(valuations[valuations.day.eq(day)], "two_of_three")
            members = universe.members_at(as_of.date())
            opens = prices.open.loc[stamp]
            ranked = sorted(scores.get(day), key=lambda pick: -pick.rank_score)
            ranked = [pick.symbol for pick in ranked if pick.bucket == BUCKET and
                      pick.symbol in members and np.isfinite(opens.get(pick.symbol, np.nan))]
            pos = prices.sessions.get_loc(as_of)
            horizon = pos + 63
            for rank, ticker in enumerate(ranked[:30], start=1):
                ret63 = np.nan
                if split == "dev" and horizon < len(prices.sessions) and prices.sessions[horizon].date() <= end:
                    ret63 = prices.close[ticker].iloc[horizon] / prices.close.at[as_of, ticker] - 1
                rows.append((day, as_of.date(), ticker, rank, ticker in veto, ret63))
    fields = ("date", "as_of", "ticker", "rank", "team_veto", "ret63")
    pd.DataFrame(rows, columns=fields).to_csv(HERE / "tally_dev_val.csv", index=False)


if __name__ == "__main__":
    main()
