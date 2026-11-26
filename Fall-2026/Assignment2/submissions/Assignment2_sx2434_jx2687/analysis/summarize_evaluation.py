"""Recover each model's separate-sample predictive scores from the submitted evaluation rows."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "output/evaluation"


def monthly_rank_ic(group, horizon):
    sample = group[["mispricing", f"ret_{horizon}"]].dropna()
    if len(sample) < 5:
        return None
    ranks = sample.rank()
    return ranks.iloc[:, 0].corr(ranks.iloc[:, 1])


def summarize():
    sources = [("dev", "team", "team_eval_dev.csv", 1200),
               ("dev", "course_pe", "course_pe_eval_dev.csv", 1200),
               ("val", "team", "val_matured_signals.csv", 480)]
    rows = []
    for period, model, filename, asked in sources:
        data = pd.read_csv(HERE / filename)
        for horizon in (63, 126, 252):
            by_month = [monthly_rank_ic(group, horizon) for _, group in data.groupby("as_of")]
            ic = pd.Series(by_month, dtype=float).dropna()
            row = {"period": period, "model": model, "horizon": horizon, "asked": asked,
                   "valued": len(data), "coverage_pct": 100 * len(data) / asked,
                   "ic_months": len(ic), "rank_ic": ic.mean(),
                   "ic_t_stat": ic.mean() / ic.std() * len(ic) ** 0.5,
                   "interval_eligible": None, "interval_inside": None}
            if horizon == 252 and "price_252" in data:
                ranges = data.dropna(subset=["price_252", "low", "high"])
                row["interval_eligible"] = len(ranges)
                row["interval_inside"] = int(ranges.price_252.between(ranges.low, ranges.high).sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "separate_model_scores.csv", index=False)


if __name__ == "__main__":
    summarize()
