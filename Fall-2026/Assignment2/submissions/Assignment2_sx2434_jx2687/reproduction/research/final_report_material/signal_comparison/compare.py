"""Frozen team vs course trailing P/E for monthly top-20 growth candidates.

Run with the course venv. Coverage uses all candidate-months. Predictive tests
use the intersection of available valuations and identically dated returns.
'historical_course' uses the course's last available price inside the horizon.
'exact_close' additionally requires a print on the precise horizon session.
Neither uses an outcome after its split. Range width is (high-low)/as-of price.
"""
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATE = ROOT / "valuation_candidate"
SOURCE = CANDIDATE / "source_code"
PRETEST = CANDIDATE / "results/pretest"
sys.path[:0] = [str(PRETEST), str(SOURCE), str(ROOT / "Jiao_delivery/bonus_handoff")]
from prepare_bonus import copy_database, copy_market_data
from evaluate_valuation import check_frame, top_candidates
from selection import MonthlyScores
from shared.data_view import DataBundle, DataView
from shared.guard import student_call
from shared.prices import PriceStore
from shared.store import FundamentalStore
from shared.universe import TradeableUniverse
from student_valuation import value
from valuation.examples import trailing_pe_value

FIELDS = ["fair_value", "mispricing", "low", "high", "method_spread"]
KEYS = ["as_of", "ticker"]
PERIODS = {"dev": ("2017-01-03", "2021-12-31"), "val": ("2022-01-03", "2023-12-29")}


def read_inputs(temp):
    source = SOURCE / "data"
    copy_market_data(source, temp)
    db = temp / "fundamentals.db"
    copy_database(source / db.name, db)
    prices = PriceStore(temp / "prices/daily_adjusted.parquet")
    market = pd.read_parquet(temp / "prices/market_series.parquet")
    macro = {path.stem: pd.read_parquet(path).set_index("date").value.dropna().sort_index()
             for path in (temp / "macro").glob("*.parquet")}
    store = FundamentalStore(str(db))
    universe = TradeableUniverse(store, ["sp500", "nasdaq100"])
    bounded = [("date", "<=", pd.Timestamp(PERIODS["val"][1]))]
    history = pd.read_parquet(SOURCE / "student_data/history_monthly.parquet", filters=bounded)
    bundle = DataBundle(prices, market, macro, store, universe, {"history_monthly": history})
    score_path = temp / "scores.parquet"
    pd.read_parquet(source / "selection/monthly_scores.parquet",
                    filters=[("month", "<=", pd.Timestamp("2023-12-01"))]).to_parquet(score_path)
    return bundle, MonthlyScores(score_path)


def get_signals(period, bundle, scores):
    start, end = PERIODS[period]
    sessions = bundle.prices.sessions.to_series().loc[start:end]
    dates = sessions.groupby(sessions.index.to_period("M")).first()
    rows = {"team": [], "pe": []}
    asked = []
    for date in dates:
        prior = bundle.prices.last_session_on_or_before(date.date() - timedelta(days=1))
        view = DataView(bundle, prior)
        tickers = top_candidates(scores, date.date(), view.members(), 20)
        asked.extend((pd.Timestamp(prior), ticker) for ticker in tickers)
        if period == "val":
            for name, model in (("team", value), ("pe", trailing_pe_value)):
                with student_call():
                    observed = model(prior, tickers, view)
                observed = check_frame(observed, prior, tickers, view)
                rows[name].append(observed.assign(as_of=pd.Timestamp(prior)))
    inputs = pd.MultiIndex.from_tuples(asked, names=KEYS)
    assert inputs.is_unique
    if period == "dev":
        files = {"team": "team_eval_dev.csv", "pe": "course_pe_eval_dev.csv"}
        frames = {k: pd.read_csv(PRETEST / v, parse_dates=["as_of"]) for k, v in files.items()}
    else:
        frames = {k: pd.concat(v, ignore_index=True) for k, v in rows.items()}
    a, b = [frames[k].set_index(KEYS)[FIELDS].sort_index() for k in ("team", "pe")]
    assert a.index.is_unique and b.index.is_unique
    assert a.index.isin(inputs).all() and b.index.isin(inputs).all()
    if period == "val":
        saved = pd.read_csv(CANDIDATE / "results/val_matured_signals.csv", parse_dates=["as_of"])
        old = saved.set_index(KEYS)[FIELDS].sort_index()
        assert list(a.index) == list(old.index)
        # The November 23 pre-integration archive covers the same names.
        # Use the frozen export above for all analysis; report any value revisions.
        differences = ~np.isclose(a, old, atol=1e-8, rtol=1e-8, equal_nan=True)
        print("Validation historical archive vs frozen export:", dict(zip(FIELDS, differences.sum(axis=0))))
        print("Revised stock-months:", list(a.index[differences.any(axis=1)]))
    paired = a.add_suffix("_team").join(b.add_suffix("_pe"), how="inner")
    return paired.reset_index(), (len(inputs), len(a), len(b))


def matched_results(period, signals, counts, bundle):
    close = bundle.prices.close.loc[:PERIODS[period][1]]
    sessions = bundle.prices.sessions[bundle.prices.sessions <= close.index[-1]]
    close = close.reindex(sessions)
    rows = []
    for horizon in (63, 126, 252):
        months = []
        for prior, group in signals.groupby("as_of"):
            i = sessions.get_loc(prior)
            if i + horizon >= len(sessions):
                continue
            names = group.ticker.tolist()
            window = close[names].iloc[i+1:i+horizon+1]
            start = close.loc[prior, names].to_numpy()
            end = window.ffill().iloc[-1].to_numpy()
            months.append(group.assign(price=start, end_price=end, ret=end/start-1,
                                       exact=window.iloc[-1].notna().to_numpy()))
        paired = pd.concat(months).dropna(subset=["ret", "price"])
        for rule, subset in (("historical_course", paired), ("exact_close", paired[paired.exact])):
            ics = []
            for _, g in subset.groupby("as_of"):
                cols = ["mispricing_team", "mispricing_pe", "ret"]
                if len(g) >= 5 and g[cols].nunique().min() > 1:
                    ics.append(g[cols].rank().corr().ret.iloc[:2].to_numpy())
            team_ic, pe_ic = np.mean(ics, axis=0)
            asked, team_count, pe_count = counts
            row = dict(period=period, outcome=rule, sessions=horizon, asked=asked,
                       valued_team=team_count, valued_pe=pe_count, overlap=len(signals),
                       paired_outcomes=len(subset), ic_months=len(ics), ic_team=team_ic, ic_pe=pe_ic)
            if horizon == 252:
                for arm in ("team", "pe"):
                    low, high = subset["low_"+arm], subset["high_"+arm]
                    assert low.notna().all() and high.notna().all()
                    row["inside_"+arm] = subset.end_price.between(low, high).sum()
                    row["median_width_over_price_"+arm] = ((high-low)/subset.price).median()
            rows.append(row)
    return rows


def main():
    with tempfile.TemporaryDirectory() as scratch:
        bundle, scores = read_inputs(Path(scratch))
        output = []
        for period in PERIODS:
            signals, counts = get_signals(period, bundle, scores)
            output.extend(matched_results(period, signals, counts, bundle))
    out = pd.DataFrame(output)
    out.to_csv(HERE / "matched_signals.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
