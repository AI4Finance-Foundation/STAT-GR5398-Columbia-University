"""Fixed-gate comparison. Run with the course venv and PYTHONHASHSEED=0."""
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
ROOT = REVIEW.parent
STUDY = ROOT / "weighted_valuation_research"
sys.path.insert(0, str(ROOT / "Jiao_delivery" / "bonus_handoff"))
from prepare_bonus import copy_database

sys.path.insert(0, str(STUDY))
from study import (CACHE, MonthlyScores, RemovalScores, backtest, evaluate, np, pd,
                   period, rebalance_days, turnover)
from selection import config

GATES = {"K1.5": 1.5, "K1.75": 1.75, "K1.983": 1.983, "K2.0": 2.0,
         "K2.25": 2.25, "K2.5": 2.5, "K_inf": np.inf}


def load_bounded_inputs(directory):
    end = pd.Timestamp(period("val")[1])
    price_file = Path("prices/daily_adjusted.parquet")
    score_file = Path("selection/monthly_scores.parquet")
    for path, cutoff in ((price_file, "date"), (score_file, "month")):
        bounds = [(cutoff, "<=", end)]
        if path == score_file:
            bounds.append(("cutoff", "<=", end))
        destination = directory / path
        destination.parent.mkdir(parents=True)
        pq.write_table(pq.read_table(config.DATA / path, filters=bounds), destination)
    copy_database(config.DATA / "fundamentals.db", directory / "fundamentals.db")
    with patch.object(config, "DATA", directory):
        _, _, params, prices, _, universe = config.load("selection/configs/growth.json")
    scores = MonthlyScores(directory / score_file)
    assert prices.sessions.max() == end and max(scores.months).isoformat() == "2023-12-01"
    return params, prices, universe, scores


def simulate(split, params, prices, universe, scores):
    start, end = period(split)
    weeks = rebalance_days(prices, start, end)
    quotes = pd.read_parquet(CACHE / f"valuations_{split}.parquet")
    assert set(quotes.day) == set(weeks) and not quotes.duplicated(["day", "ticker"]).any()
    signals = []
    for day, group in quotes.groupby("day"):
        as_of = prices.last_session_on_or_before(day - pd.Timedelta(days=1))
        assert pd.to_datetime(group.as_of).eq(pd.Timestamp(as_of)).all()
        scores.get(day)
        assert scores.cutoff <= as_of < day
        signals.append(evaluate(group))
    eligible = pd.concat(signals).loc[lambda x: x.adaptive_veto].copy()
    centers = eligible[["dcf", "peer", "hist_mid"]]
    summary, curves = [], {}
    for label, threshold in {"A": None, **GATES}.items():
        removed = {}
        if threshold is not None:
            veto = centers.max(axis=1) <= threshold * centers.min(axis=1)
            removed = eligible.loc[veto].groupby("day").ticker.agg(set).to_dict()
        run = backtest(prices, params, universe, RemovalScores(scores, removed), start, end)
        curves[label] = equity = run.equity
        assert (equity.index[0].date(), equity.index[-1].date()) == (start, end)
        cagr = 100 * ((equity.iloc[-1] / equity.iloc[0]) ** (252 / (len(equity) - 1)) - 1)
        summary.append(dict(split=split, rule=label, cagr_pct=cagr,
                            delta_vs_A_pp=cagr - summary[0]["cagr_pct"] if summary else 0.,
                            veto_name_weeks=sum(map(len, removed.values())),
                            trades=len(run.trades), annual_turnover=turnover(run)))
    return pd.DataFrame(curves), pd.DataFrame(summary)


def check_archives(split, curves, summary):
    old = pd.read_csv(STUDY / f"{split}_equity.csv", index_col=0, parse_dates=True)
    assert curves.index.equals(old.index)
    np.testing.assert_allclose(curves[["A", "K_inf"]], old[["A", "adaptive"]], rtol=0, atol=1e-6)
    if split == "val":
        locked = pd.read_csv(REVIEW / "frozen_1p983/val_equity.csv", index_col=0, parse_dates=True)
        assert locked.index.equals(curves.index)
        np.testing.assert_allclose(curves["K1.983"], locked["K1.983"], rtol=0, atol=1e-6)
    else:
        steps = pd.read_csv(REVIEW / "threshold/dev_full_steps.csv").set_index("K")
        for label, gate in GATES.items():
            if np.isfinite(gate):
                reference = steps.loc[:gate].iloc[-1]["cagr_pct"]
                np.testing.assert_allclose(summary.set_index("rule").loc[label, "cagr_pct"],
                                           reference, rtol=0, atol=1e-9)
    former = pd.read_csv(STUDY / "zeyu_overlay/summary.csv").set_index(["split", "rule"])
    assert round(summary.set_index("rule").loc["K2.5", "cagr_pct"], 2) == former.loc[
        (split, "adaptive_spread"), "cagr_pct"]


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0 to reproduce course executions.")
    protocol = json.loads((REVIEW / "frozen_1p983/protocol.json").read_text())
    assert hashlib.sha256((STUDY / "study.py").read_bytes()).hexdigest() == protocol["implementation_sha256"]
    results = {}
    with TemporaryDirectory() as temp:
        inputs = load_bounded_inputs(Path(temp))
        for split in ("dev", "val"):
            curves, summary = simulate(split, *inputs)
            check_archives(split, curves, summary)
            results[split] = (curves, summary)
    for split, (curves, _) in results.items():
        curves.to_csv(HERE / f"{split}_equity.csv", index_label="date")
    table = pd.concat([summary for _, summary in results.values()], ignore_index=True)
    table.to_csv(HERE / "summary.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

