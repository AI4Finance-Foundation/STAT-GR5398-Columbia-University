"""Validate the rejected-before-test v3 experiment and write final evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
root = here.parents[2]
source = root / "work/retraining_v3/model"
scores_path = here / "scores_v3.parquet"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


scores = pd.read_parquet(scores_path)
scores["month_key"] = pd.to_datetime(scores.month).dt.strftime("%Y-%m")
months = sorted(scores.month_key.unique())
expected = pd.period_range("2017-01", "2026-11", freq="M").astype(str).tolist()
assert months == expected
assert len(scores) == 60_326
assert not scores.duplicated(["month", "symbol", "bucket"]).any()
assert np.isfinite(scores[["rank_score", "weight_score"]]).all().all()
assert (pd.to_datetime(scores.cutoff) < pd.to_datetime(scores.month)).all()

baseline = pd.read_parquet(here.parent / "baseline_scores.parquet")
keys = ["month", "cutoff", "symbol", "bucket"]
pd.testing.assert_frame_equal(
    scores[keys].sort_values(keys).reset_index(drop=True),
    baseline[keys].sort_values(keys).reset_index(drop=True),
)

v2 = pd.read_parquet(here.parent / "feature_ablation_v2/scores_v2.parquet")
joined = scores.merge(v2, on=keys, validate="one_to_one", suffixes=("_v3", "_v2"))
growth = joined[joined.bucket.eq("growth_tech")].copy()
monthly = []
for month, frame in growth.groupby(pd.to_datetime(growth.month).dt.strftime("%Y-%m")):
    corr = frame[["rank_score_v3", "rank_score_v2"]].rank().corr().iloc[0, 1]
    top3 = set(frame.nlargest(10, "rank_score_v3").symbol)
    top2 = set(frame.nlargest(10, "rank_score_v2").symbol)
    monthly.append({"month": month, "spearman_v3_vs_v2": float(corr), "top10_overlap": len(top3 & top2)})
comparison = pd.DataFrame(monthly)
comparison.to_csv(here / "monthly_score_comparison_v3_vs_v2.csv", index=False)

expected_course_hash = (here / "course_scores.sha256").read_text().split()[0]
course = source / "data/selection/monthly_scores.parquet"
assert sha(course) == expected_course_hash
decision = json.loads((here / "dev_val_decision.json").read_text())
assert decision["admit_to_one_final_test"] is False
output = source / "output"
assert not list(output.glob("test_scores_v3_*"))
assert not list(output.glob("forward_scores_v3_*"))
test_log = output / "test_runs.jsonl"
assert not test_log.exists() or not test_log.read_text().strip()

identity = json.loads((here / "run_identity.json").read_text())
for path in (here / "rank_target.json", here / "selector.py", here / "PRE_REGISTERED_PROTOCOL.md"):
    matches = [value for key, value in identity.items() if key.endswith(str(path.relative_to(here)))]
    assert matches and matches[0] == sha(path)

summary = {
    "status": "passed",
    "months": [months[0], months[-1]],
    "month_count": len(months),
    "rows": len(scores),
    "duplicate_score_keys": 0,
    "nonfinite_scores": 0,
    "all_cutoffs_before_score_month": True,
    "candidate_keys_exactly_match_same_environment_unchanged_retrain": True,
    "bucket_month_coverage": scores.groupby("bucket").month_key.nunique().sort_index().to_dict(),
    "course_scores_unchanged": True,
    "course_scores_sha256": expected_course_hash,
    "scores_v3_sha256": sha(scores_path),
    "selector_sha256": sha(here / "selector.py"),
    "config_sha256": sha(here / "rank_target.json"),
    "protocol_sha256": sha(here / "PRE_REGISTERED_PROTOCOL.md"),
    "growth_tech_mean_monthly_spearman_v3_vs_v2": round(float(comparison.spearman_v3_vs_v2.mean()), 6),
    "growth_tech_mean_top10_overlap_v3_vs_v2": round(float(comparison.top10_overlap.mean()), 3),
    "admitted_to_test": False,
    "v3_test_invocations": 0,
    "reason": "Failed pre-registered development return floor; v2 retained.",
}
(here / "final_integrity.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

