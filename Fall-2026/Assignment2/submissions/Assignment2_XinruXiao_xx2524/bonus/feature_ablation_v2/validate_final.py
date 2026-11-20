"""Validate the frozen bonus experiment and write a deterministic audit summary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
root = here.parents[2]
source = root / "work/retraining_v2/model"
scores_path = here / "scores_v2.parquet"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


scores = pd.read_parquet(scores_path)
scores["month_key"] = pd.to_datetime(scores["month"]).dt.strftime("%Y-%m")
months = sorted(scores["month_key"].unique())
expected = pd.period_range("2017-01", "2026-11", freq="M").astype(str).tolist()
assert months == expected
assert len(scores) == 60_326
assert not scores.duplicated(["month", "symbol", "bucket"]).any()
assert np.isfinite(scores[["rank_score", "weight_score"]]).all().all()
assert (pd.to_datetime(scores["cutoff"]) < pd.to_datetime(scores["month"])).all()

reference = pd.read_parquet(here.parent / "baseline_scores.parquet")
left = scores[["month", "cutoff", "symbol", "bucket"]].sort_values(
    ["month", "cutoff", "symbol", "bucket"]
).reset_index(drop=True)
right = reference[["month", "cutoff", "symbol", "bucket"]].sort_values(
    ["month", "cutoff", "symbol", "bucket"]
).reset_index(drop=True)
pd.testing.assert_frame_equal(left, right)

expected_course_hash = (here / "course_scores.sha256").read_text().split()[0]
course = source / "data/selection/monthly_scores.parquet"
assert sha256(course) == expected_course_hash

identity = json.loads((here / "run_identity.json").read_text())
for path in (here / "feature_ablation.json", here / "selector.py"):
    matching = [value for key, value in identity.items() if key.endswith(str(path.relative_to(here)))]
    assert matching and matching[0] == sha256(path)

equivalence = {}
for period in ("test", "forward"):
    output = source / "output"
    a = pd.read_csv(output / f"{period}_scores_v2_A_selection/equity.csv")
    c = pd.read_csv(output / f"{period}_scores_v2_C_timing_full/equity.csv")
    b = pd.read_csv(output / f"{period}_scores_v2_B_valuation/equity.csv")
    d = pd.read_csv(output / f"{period}_scores_v2_D_both_full/equity.csv")
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, d)
    equivalence[period] = {"A_equals_C": True, "B_equals_D": True}

test_runs = source / "output/test_runs.jsonl"
records = [json.loads(line) for line in test_runs.read_text().splitlines() if line.strip()]
assert len(records) == 1

summary = {
    "status": "passed",
    "months": [months[0], months[-1]],
    "month_count": len(months),
    "rows": len(scores),
    "duplicate_score_keys": 0,
    "nonfinite_scores": 0,
    "all_cutoffs_before_score_month": True,
    "candidate_keys_exactly_match_same_environment_unchanged_retrain": True,
    "bucket_month_coverage": scores.groupby("bucket")["month_key"].nunique().sort_index().to_dict(),
    "course_scores_sha256": expected_course_hash,
    "course_scores_unchanged": True,
    "scores_v2_sha256": sha256(scores_path),
    "frozen_config_sha256": sha256(here / "feature_ablation.json"),
    "frozen_selector_sha256": sha256(here / "selector.py"),
    "test_invocations_for_isolated_v2_experiment": len(records),
    "full_timing_equivalence": equivalence,
}
(here / "final_integrity.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
