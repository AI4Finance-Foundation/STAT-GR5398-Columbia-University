"""Audit the two-month Q3 repeat (--pilot), or collect the full dev/val experiment."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OUTPUT = Path("output/bonus")
ARMS = {"Q3": ("base_q3", "growth.json"), "Q4": ("variant_q4", "growth_q4.json")}
PERIODS = {"dev": ("2017-01-03", "2021-12-31"), "val": ("2022-01-03", "2023-12-29")}
MONTHS = pd.date_range("2017-01-01", "2023-12-01", freq="MS")
METRICS = {"cagr_pct": "cagr_pct_or_difference_pp", "sharpe": "sharpe",
           "max_drawdown_pct": "max_drawdown_pct_or_difference_pp",
           "turnover_per_year": "turnover_times_per_year"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_configs():
    configs = []
    for expected, (_, name) in zip((3, 4), ARMS.values()):
        config = read_json(Path("selection/configs") / name)
        check(config["model"].pop("val_quarters") == expected, f"{name}: unexpected val_quarters")
        configs.append(config)
    check(configs[0] == configs[1], "Model configurations differ on another setting")


def previous_closes():
    prices = pd.read_parquet("data/prices/daily_adjusted.parquet", columns=["date", "close"],
                             filters=[("symbol", "=", "SPY"), ("date", "<=", pd.Timestamp("2023-12-29"))])
    sessions = pd.DatetimeIndex(prices.dropna(subset=["close"]).date).unique().sort_values()
    positions = sessions.searchsorted(MONTHS, side="left") - 1
    check((positions >= 0).all(), "Missing SPY history before the first scoring month")
    return pd.Series(sessions[positions], index=MONTHS)


def check_scores(stem, config_name, cutoffs, months=MONTHS, allow_full_base=False):
    path = OUTPUT / f"{stem}.parquet"
    meta = read_json(path.with_suffix(".json"))
    check(Path(meta["config"]).name == config_name, f"{stem}: unexpected training config")
    expected_window = [d.strftime("%Y-%m") for d in (months[0], months[-1])]
    larger = allow_full_base and meta["months"] == ["2017-01", "2023-12"]
    check(meta["months"] == expected_window or larger, f"{stem}: wrong scoring window")
    fingerprints = tuple(meta[key] for key in ("fundamentals_sha256", "prices_sha256"))
    check(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in fingerprints), f"{stem}: invalid input hashes")
    scores = pd.read_parquet(path, filters=[("month", ">=", months[0]), ("month", "<=", months[-1])])
    check(pq.read_metadata(path).num_rows == meta["rows"] and (larger or len(scores) == meta["rows"]),
          f"{stem}: score row count/window mismatch")
    check(not scores[["month", "symbol", "bucket"]].isna().any().any(), f"{stem}: missing score key")
    check(not scores.duplicated(["month", "symbol"]).any(), f"{stem}: duplicate symbol in a month")
    check(np.isfinite(scores[["rank_score", "weight_score"]].to_numpy(dtype=float)).all(),
          f"{stem}: missing or invalid score")
    check(pd.DatetimeIndex(scores.month.unique()).sort_values().equals(months), f"{stem}: missing score months")
    check(scores.cutoff.eq(scores.month.map(cutoffs)).all(), f"{stem}: wrong monthly decision cutoff")
    growth_months = pd.DatetimeIndex(scores.loc[scores.bucket == "growth_tech", "month"].unique()).sort_values()
    check(growth_months.equals(months), f"{stem}: missing growth_tech candidates in at least one month")
    return fingerprints, scores


def monthly_scores(left, right, months):
    # This diagnostic ranks all scored growth-tech candidates. Alphabetical
    # symbol order breaks equal scores; the portfolio still uses course rules.
    groups = [frame.loc[frame.bucket.eq("growth_tech")].groupby("month") for frame in (left, right)]
    rows = []
    for month in months:
        a, b = [g.get_group(month).sort_values(["rank_score", "symbol"], ascending=[False, True])
                .set_index("symbol") for g in groups]
        common = a.index.intersection(b.index)
        shared_top = len(a.head(10).index.intersection(b.head(10).index))
        rows.append({"month": month.strftime("%Y-%m"), "growth_candidates_left": len(a),
                     "growth_candidates_right": len(b), "growth_candidates_common": len(common),
                     "growth_top10_common": shared_top, "growth_top10_common_div_10": shared_top / 10,
                     "growth_common_max_abs_weight_score_diff":
                         (a.loc[common, "weight_score"] - b.loc[common, "weight_score"]).abs().max()})
    return pd.DataFrame(rows)


def pilot(cutoffs):
    months = MONTHS[:2]
    hashes, base = check_scores("base_q3", "growth.json", cutoffs, months, allow_full_base=True)
    repeat_hashes, repeat = check_scores("repeat_q3", "growth.json", cutoffs, months)
    check(hashes == repeat_hashes, "Q3 repeat used different data inputs")
    keys = ["month", "symbol", "bucket"]
    x, y = [f.set_index(keys).sort_index() for f in (base, repeat)]
    check(x.index.equals(y.index), "Q3 repeat scored different month/symbol/bucket keys")
    check(x.cutoff.equals(y.cutoff), "Q3 repeat used different cutoffs")
    scores = ["rank_score", "weight_score"]
    maxima = (x[scores] - y[scores]).abs().groupby("month").max()
    detail = monthly_scores(base, repeat, months)
    for column in scores:
        detail["all_max_abs_" + column + "_diff"] = maxima[column].to_numpy()
    print(detail.to_string(index=False))
    if maxima.to_numpy().max() > 1e-8:
        raise SystemExit("Q3 repeat exceeds atol=1e-8, rtol=0. Save these differences and investigate "
                         "run-to-run variation before interpreting a Q4 effect.")
    check(detail.growth_top10_common.eq(detail.growth_candidates_left.clip(upper=10)).all(),
          "Q3 scores agree within tolerance but its top-ten membership changed. Inspect near ties.")
    print("Q3 repeat agrees on keyed scores (atol=1e-8, rtol=0) and top-ten membership.")


def main(pilot_only):
    check_configs()
    cutoffs = previous_closes()
    if pilot_only:
        pilot(cutoffs)
        return
    checked = [check_scores(stem, name, cutoffs) for stem, name in ARMS.values()]
    (hashes3, scores3), (hashes4, scores4) = checked
    check(hashes3 == hashes4, "The two runs used different fundamentals or price inputs")
    detail = monthly_scores(scores3, scores4, MONTHS)
    different = ((detail.growth_candidates_common != detail.growth_candidates_left)
                 | (detail.growth_candidates_common != detail.growth_candidates_right))
    changed = detail.loc[different, "month"].tolist()
    print("Growth-tech candidate lists differ in", len(changed), "of 84 months.", " ".join(changed))

    rows = []
    for period, (start, end) in PERIODS.items():
        arms = {}
        for arm, (stem, _) in ARMS.items():
            run = f"{period}_{stem}_selection"
            stats = read_json(Path("output") / run / "stats.json")
            check((stats["run"], stats["start"], stats["end"]) == (run, start, end),
                  f"{run}: unexpected replay name or dates")
            values = {label: stats[key] for key, label in METRICS.items()}
            check(all(isinstance(v, (int, float)) and np.isfinite(v) for v in values.values()),
                  f"{run}: missing or invalid metric")
            arms[arm] = values
            rows.append({"period": period, "row": arm, **values})
        delta = {key: round(arms["Q4"][key] - arms["Q3"][key], 8) for key in arms["Q3"]}
        rows.append({"period": period, "row": "Q4-Q3", **delta})
    out = OUTPUT / "comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    detail.rename(columns={"growth_candidates_left": "growth_candidates_Q3",
                           "growth_candidates_right": "growth_candidates_Q4"}).to_csv(
                               OUTPUT / "monthly_score_check.csv", index=False)
    print(f"Saved {out}; differences use the precision in the course stats.json files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="compare original and repeated Q3 scores for Jan–Feb 2017")
    main(parser.parse_args().pilot)
