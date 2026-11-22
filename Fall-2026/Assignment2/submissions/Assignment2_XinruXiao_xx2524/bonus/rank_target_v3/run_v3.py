"""Isolated, resumable 119-month rank-target retrain and dev/val evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = here.parents[2]
out = args.out.resolve()
if out != here / "scores_v3.parquet":
    parser.error("--out must be the isolated scores_v3.parquet next to this driver")

parts = here / "scores_v3_parts"
logs = here / "logs"
parts.mkdir(exist_ok=True)
logs.mkdir(exist_ok=True)
source = root / "work/retraining_v3/model"
driver = here.parent / "train_one_month.py"
course = source / "data/selection/monthly_scores.parquet"
reference = here.parent / "feature_ablation_v2/scores_v2.parquet"
baseline = here.parent / "baseline_scores.parquet"
config = source / "selection/configs/rank_target.json"
selector = source / "selection/selector.py"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


identity = {
    str(path.relative_to(root)) if path.is_relative_to(root) else str(path): sha(path)
    for path in (
        course.resolve(), reference, baseline, driver, config, selector,
        (source / "data/prices/daily_adjusted.parquet").resolve(),
        (source / "data/fundamentals.db").resolve(), Path(__file__).resolve(),
        here / "PRE_REGISTERED_PROTOCOL.md",
    )
}
identity.update(runtime=sys.version, pandas=pd.__version__, numpy=np.__version__)
expected_course_hash = (here.parent / "feature_ablation_v2/course_scores.sha256").read_text().split()[0]
assert sha(course) == expected_course_hash
manifest = here / "run_identity.json"
if manifest.exists():
    assert json.loads(manifest.read_text()) == identity, "Inputs changed: use a fresh experiment"
else:
    assert not list(parts.glob("*.parquet")), "Unidentified checkpoints"
    manifest.write_text(json.dumps(identity, indent=2) + "\n")

env = os.environ.copy()
env.update({
    key: "1" for key in (
        "LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
})
base = pd.read_parquet(baseline)
base["_month"] = pd.to_datetime(base.month).dt.strftime("%Y-%m")
keys = ["month", "cutoff", "bucket", "symbol"]
expected = {
    month: frame[keys].sort_values(keys).reset_index(drop=True)
    for month, frame in base.groupby("_month")
}


def validate(month: str) -> pd.DataFrame:
    frame = pd.read_parquet(parts / f"{month}.parquet")
    assert not frame.duplicated(["month", "symbol", "bucket"]).any(), month
    assert np.isfinite(frame[["rank_score", "weight_score"]]).all().all(), month
    pd.testing.assert_frame_equal(
        frame[keys].sort_values(keys).reset_index(drop=True), expected[month]
    )
    return frame


def train(month: str) -> tuple[str, int]:
    if not (parts / f"{month}.parquet").is_file():
        with (logs / f"{month}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable, str(driver), "--source", str(source),
                    "--month", month, "--parts", str(parts),
                    "--config", "selection/configs/rank_target.json",
                ],
                env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=1800,
            )
    return month, len(validate(month))


def stage(months: list[str]) -> None:
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(train, month) for month in months]):
            print(*future.result(), flush=True)


lock = (here / ".run.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
pilot = ["2019-05", "2019-06"]
stage(pilot)
print("PILOT PASSED; continuing to full 119 months", flush=True)
stage(sorted(set(expected) - set(pilot)))
all_scores = pd.concat([validate(month) for month in sorted(expected)], ignore_index=True)
tmp = out.with_suffix(".tmp")
all_scores.to_parquet(tmp, index=False)
tmp.replace(out)
assert sha(course) == expected_course_hash
print(f"TRAINING COMPLETE: {len(expected)} months, {len(all_scores)} rows", flush=True)

for period in ("dev", "val"):
    for variant, extra in (("A", []), ("B", ["--valuation"])):
        path = logs / f"backtest_{period}_{variant}.log"
        with path.open("w") as log:
            subprocess.run(
                [sys.executable, "run_portfolio.py", "--period", period,
                 "--scores", str(out), *extra],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        print(f"BACKTEST COMPLETE: {period} {variant}", flush=True)
assert sha(course) == expected_course_hash
subprocess.run([sys.executable, str(here / "decide_after_dev_val.py")], check=True)

