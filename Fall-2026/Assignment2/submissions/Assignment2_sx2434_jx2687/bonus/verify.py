"""Check the bonus results from the four saved portfolios and two score panels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARMS = {"Q3": "base_q3", "Q4": "variant_q4"}
KEYS = ["month", "bucket", "symbol"]


def load_json(path):
    return json.loads(path.read_text())


def check_portfolios():
    summary = pd.read_csv(ROOT / "results/comparison.csv").set_index(["period", "row"])
    columns = ["cagr_pct_or_difference_pp", "sharpe", "max_drawdown_pct_or_difference_pp",
               "turnover_times_per_year"]
    dates = {"dev": ("2017-01-03", "2021-12-31"), "val": ("2022-01-03", "2023-12-29")}
    for period, window in dates.items():
        recorded = {}
        for arm, stem in ARMS.items():
            path = ROOT / "results" / f"{period}_{stem}_selection"
            nav = pd.read_csv(path / "equity.csv", index_col=0, parse_dates=True)["equity"]
            trades = pd.read_csv(path / "trades.csv")
            stats = load_json(path / "stats.json")
            assert nav.index.is_monotonic_increasing and nav.index.is_unique
            assert tuple(nav.index[[0, -1]].strftime("%Y-%m-%d")) == window
            assert (stats["start"], stats["end"]) == window
            ret = nav.pct_change().dropna()
            vol = ret.std() * np.sqrt(252)
            change = nav.iloc[-1] / nav.iloc[0]
            cagr = change ** (252 / len(ret)) - 1
            drawdown = (nav / nav.cummax() - 1).min()
            turnover = trades.notional.abs().sum() / nav.mean() * 252 / len(nav)
            computed = {
                "cagr_pct": round(cagr * 100, 2),
                "sharpe": round(ret.mean() * 252 / vol, 3),
                "max_drawdown_pct": round(drawdown * 100, 2),
                "turnover_per_year": round(turnover, 2),
                "total_return_pct": round((change - 1) * 100, 2),
                "volatility_pct": round(vol * 100, 2),
                "calmar": round(cagr / abs(drawdown), 3),
                "trades": len(trades),
            }
            for name, value in computed.items():
                assert value == stats[name], (path.name, name, value, stats[name])
            values = [computed[k] for k in ("cagr_pct", "sharpe", "max_drawdown_pct", "turnover_per_year")]
            recorded[arm] = np.array(values)
            np.testing.assert_allclose(recorded[arm], summary.loc[(period, arm), columns].to_numpy(float))
            if period == "val":
                prior_year = nav.loc["2022"].iloc[-1]
                year_return = nav.loc["2023"].iloc[-1] / prior_year - 1
                print(f"{arm} 2023 calendar return: {year_return:.2%}")
        np.testing.assert_allclose(recorded["Q4"] - recorded["Q3"],
                                   summary.loc[(period, "Q4-Q3"), columns].to_numpy(float), atol=1e-10)


def check_scores():
    frames = {}
    configurations = []
    fingerprints = []
    top_sets = {}
    for arm, stem in ARMS.items():
        path = ROOT / "scores" / f"{stem}.parquet"
        scores = pd.read_parquet(path)
        meta = load_json(path.with_suffix(".json"))
        config = load_json(ROOT / "configs" / Path(meta["config"]).name)
        assert config["model"].pop("val_quarters") == int(arm[1])
        configurations.append(config)
        fingerprints.append((meta["prices_sha256"], meta["fundamentals_sha256"]))
        assert len(scores) == meta["rows"] and not scores.duplicated(["month", "symbol"]).any()
        assert pd.DatetimeIndex(scores.month.unique()).sort_values().equals(
            pd.date_range("2017-01-01", "2023-12-01", freq="MS"))
        assert scores.cutoff.lt(scores.month).all() and scores.groupby("month").cutoff.nunique().eq(1).all()
        frames[stem] = scores
        tech = scores.loc[scores.bucket.eq("growth_tech")].sort_values(
            ["month", "rank_score", "symbol"], ascending=[True, False, True])
        top_sets[arm] = tech.groupby("month").head(10).groupby("month").symbol.agg(set)
    assert configurations[0] == configurations[1] and fingerprints[0] == fingerprints[1]
    panels = [x.loc[x.bucket == "growth_tech"].set_index(KEYS).sort_index() for x in frames.values()]
    pd.testing.assert_series_equal(panels[0].cutoff, panels[1].cutoff)
    observed = pd.Series([len(a & b) for a, b in zip(top_sets["Q3"], top_sets["Q4"])],
                         index=top_sets["Q3"].index.strftime("%Y-%m"))
    published = pd.read_csv(ROOT / "results/monthly_score_check.csv").set_index("month")
    assert observed.index.equals(published.index) and np.array_equal(observed, published.growth_top10_common)
    print(f"Growth-tech top-ten overlap: {observed.mean():.2f}/10; "
          f"identical sets {sum(observed == 10)}/{len(observed)}; minimum {observed.min()}/10")
    checks = {"repeat_q3": "base_q3"}
    for stem in ARMS.values():
        for segment in ("pilot", "recent"):
            checks[f"optimized_{segment}_{stem}"] = stem
    for filename, stem in checks.items():
        probe = pd.read_parquet(ROOT / "verification" / f"{filename}.parquet")
        expected = frames[stem].loc[frames[stem].month.isin(probe.month.unique())]
        a, b = (x.set_index(KEYS).sort_index() for x in (probe, expected))
        pd.testing.assert_series_equal(a.cutoff, b.cutoff)
        np.testing.assert_allclose(a[["rank_score", "weight_score"]], b[["rank_score", "weight_score"]],
                                   rtol=0, atol=1e-8)


if __name__ == "__main__":
    check_portfolios()
    check_scores()
    print("Portfolio, score and pilot checks passed.")
