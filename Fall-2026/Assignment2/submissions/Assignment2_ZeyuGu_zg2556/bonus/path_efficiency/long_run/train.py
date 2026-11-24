"""Refit both full course selectors monthly over the development and validation periods."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import time

HERE = Path(__file__).resolve().parent
ARMS = ("control", "path_efficiency")
PACKAGES = (
    "numpy", "pandas", "scipy", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "joblib",
)
THREAD_LIMITS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
FEATURE = HERE.parent
BONUS = FEATURE.parent
CONFIG = Path("selection/configs/growth.json")
FIRST, LAST = "2017-01", "2023-12"
BOUND = "2023-12-29"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def bounded_data(source, destination):
    import pandas as pd

    price_file = Path("prices/daily_adjusted.parquet")
    (destination / price_file.parent).mkdir(parents=True)
    end = pd.Timestamp(BOUND)
    prices = pd.read_parquet(source / price_file, filters=[("date", "<=", end)])
    assert prices.date.max() == end
    prices.to_parquet(destination / price_file, index=False)

    db = source / "fundamentals.db"
    clauses = {
        "fundamental_data": (
            "datadate <= :end AND "
            "COALESCE(NULLIF(accepted_date,''), NULLIF(filing_date,''), datadate) <= :end"
        ),
        "universe_membership": "snapshot_date <= :end",
        "symbol_changes": "change_date <= :end",
    }
    with sqlite3.connect(destination / db.name, uri=True) as connection:
        connection.execute("ATTACH DATABASE ? AS original", (db.resolve().as_uri() + "?mode=ro",))
        schema = list(connection.execute(
            "SELECT type,name,sql FROM original.sqlite_master "
            "WHERE sql IS NOT NULL ORDER BY type DESC, name"
        ))
        for kind, name, definition in schema:
            connection.execute(definition)
            if kind == "table":
                # Preserve source order on ties between same-date filings.
                columns = ",".join(
                    row[1] for row in connection.execute(f"PRAGMA table_info({name})")
                )
                connection.execute(
                    f"INSERT INTO {name}(rowid,{columns}) SELECT rowid,* "
                    f"FROM original.{name} WHERE {clauses[name]} ORDER BY rowid",
                    {"end": BOUND},
                )
    return {"fundamentals": sha(destination / db.name), "prices": sha(destination / price_file)}


def worker(stage, arm_dir, months):
    sys.path.insert(0, str(stage))
    from selection import build_scores as builder
    from selection.selector import _build_models

    assert len(_build_models()) == 6, "Both course boosting libraries must be active"
    parts = arm_dir / "scores_parts"
    builder._init(str(stage / CONFIG), str(parts))
    for month in months:
        begin = time.monotonic()
        print(builder._run_month(month), flush=True)
        write_json(parts / f"{month}.json", {
            "month": month,
            "score_sha256": sha(parts / f"{month}.parquet"),
            "seconds": round(time.monotonic() - begin, 3),
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        })


def schedule_batches(months, workers):
    # Fit the newest date alongside the historical anchor early to surface environment issues.
    priority = [month for month in ("2018-01", LAST, FIRST) if month in months]
    ordered = priority + [month for month in months if month not in priority]
    return [ordered[i::workers] for i in range(min(workers, len(ordered)))]


def run_process(stage, arm_dir, months, number, env):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(stage),
               str(arm_dir), *months]
    with (arm_dir / f"worker-{number}.log").open("a", buffering=1) as log:
        subprocess.run(command, env=env, stdout=log, stderr=log, check=True)


def environment():
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_LIMITS}, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    if sys.platform == "darwin":
        omp = Path(sysconfig.get_paths()["purelib"]) / "sklearn/.dylibs"
        paths = env.get("DYLD_LIBRARY_PATH", "").split(os.pathsep)
        if str(omp) not in paths:
            env["DYLD_LIBRARY_PATH"] = os.pathsep.join([str(omp), *(p for p in paths if p)])
    return env


def create_stage(source, scratch, arm, reference):
    stage = scratch / arm
    stage.mkdir()
    for name in ("selection", "shared"):
        shutil.copytree(source / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (stage / "data").symlink_to(scratch / "data", target_is_directory=True)
    if arm == "path_efficiency":
        subprocess.run(["git", "apply", str(FEATURE / "selector.patch")], cwd=stage, check=True)
    key = "original_selector_sha256" if arm == "control" else "changed_selector_sha256"
    assert sha(stage / "selection/selector.py") == reference[key]
    return stage


def completed_month(parts, month):
    output = parts / f"{month}.parquet"
    receipt = parts / f"{month}.json"
    if not output.is_file() or not receipt.is_file():
        return False
    try:
        info = json.loads(receipt.read_text())
        return info["month"] == month and info["score_sha256"] == sha(output)
    except (KeyError, ValueError):
        return False


def verified_scores(arm_dir, months, arm):
    import pandas as pd

    parts = arm_dir / "scores_parts"
    frames = []
    for month in months:
        if not completed_month(parts, month):
            raise ValueError(f"Invalid checkpoint for {arm} {month}")
        frame = pd.read_parquet(parts / f"{month}.parquet")
        assert frame.cutoff.nunique() == 1 and frame.month.nunique() == 1
        assert frame.month.iloc[0].strftime("%Y-%m") == month
        frames.append(frame)

    scores = pd.concat(frames, ignore_index=True)
    assert scores.month.nunique() == len(months) and (scores.cutoff < scores.month).all()
    assert scores.loc[scores.bucket.eq("growth_tech"), "month"].nunique() == len(months)
    assert not scores.duplicated(["month", "bucket", "symbol"]).any()
    assert scores.notna().all().all()
    archive = BONUS / "results/control" if arm == "control" else FEATURE / "results/path_efficiency"
    anchor = scores.loc[scores.month.between("2018-01-01", "2018-06-01")].reset_index(drop=True)
    pd.testing.assert_frame_equal(anchor, pd.read_parquet(archive / "scores.parquet"), check_exact=True)
    return scores, len(anchor)


def compare_course_scores(source, control):
    """Cross-check fresh control fits against the course's published score table."""
    import pandas as pd

    keys = ["month", "symbol", "bucket"]
    values = ["cutoff", "rank_score", "weight_score"]
    reference = pd.read_parquet(
        source / "data/selection/monthly_scores.parquet",
        filters=[("month", ">=", control.month.min()), ("month", "<=", control.month.max())],
        columns=keys + values,
    )
    pairs = control.merge(
        reference, on=keys, how="outer", suffixes=("_fit", "_course"),
        indicator=True, validate="one_to_one",
    )
    monthly = []
    for month, rows in pairs.groupby("month", sort=True):
        common = rows.loc[rows["_merge"].eq("both")]
        counts = {
            "month": month.strftime("%Y-%m"),
            "common_keys": len(common),
            "fit_only": int(rows["_merge"].eq("left_only").sum()),
            "course_only": int(rows["_merge"].eq("right_only").sum()),
        }
        for field in values:
            counts[f"different_{field}"] = int(common[f"{field}_fit"].ne(common[f"{field}_course"]).sum())
        for field in ("rank_score", "weight_score"):
            error = (common[f"{field}_fit"] - common[f"{field}_course"]).abs()
            counts[f"{field}_beyond_1e-12"] = int(error.gt(1e-12).sum())
            counts[f"max_abs_{field}_diff"] = float(error.max()) if len(error) else None
        top_tens = []
        for scores in (control, reference):
            growth = scores.loc[scores.month.eq(month) & scores.bucket.eq("growth_tech")]
            top_tens.append(set(growth.nlargest(10, "rank_score").symbol))
        counts["growth_top10_shared"] = len(top_tens[0] & top_tens[1])
        monthly.append(counts)
    mismatches = ["fit_only", "course_only", *(f"different_{field}" for field in values)]
    totals = {name: sum(entry[name] for entry in monthly) for name in ["common_keys", *mismatches]}
    max_diffs = {}
    for field in ("rank_score", "weight_score"):
        key = f"max_abs_{field}_diff"
        max_diffs[key] = max((entry[key] for entry in monthly if entry[key] is not None), default=None)
        beyond = f"{field}_beyond_1e-12"
        totals[beyond] = sum(entry[beyond] for entry in monthly)
    return {
        "reference": "data/selection/monthly_scores.parquet",
        "fit_rows": len(control),
        "reference_rows": len(reference),
        "months_checked": control.month.nunique(),
        "months_exact": sum(all(entry[key] == 0 for key in mismatches) for entry in monthly),
        "months_with_identical_growth_top10": sum(entry["growth_top10_shared"] == 10 for entry in monthly),
        "totals": totals,
        "maxima": max_diffs,
        "monthly": monthly,
    }


def train(source, out, workers):
    import pandas as pd

    versions = {package: importlib.metadata.version(package) for package in PACKAGES}
    archive = json.loads((BONUS / "results/manifest.json").read_text())
    assert versions == archive["libraries"] and platform.python_version() == archive["python"]
    reference = json.loads((FEATURE / "results/manifest.json").read_text())
    for line in (FEATURE / "course_files.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(source / name) == expected, name

    months = [str(month) for month in pd.period_range(FIRST, LAST, freq="M")]
    assert len(months) == 84
    env = environment()
    with TemporaryDirectory(prefix="gr5398-full-selection-") as temporary:
        scratch = Path(temporary)
        hashes = bounded_data(source / "data", scratch / "data")
        binding = {
            "months": [FIRST, LAST],
            "max_date": BOUND,
            "libraries": versions,
            "python": platform.python_version(),
            "source_manifest_sha256": sha(FEATURE / "course_files.sha256"),
            "feature_patch_sha256": sha(FEATURE / "selector.patch"),
            "bounded_inputs_sha256": hashes,
            "config_sha256": sha(source / CONFIG),
            "thread_limits": 1,
            "PYTHONHASHSEED": 0,
        }
        bound = out / "training_inputs.json"
        if bound.exists() and json.loads(bound.read_text()) != binding:
            raise ValueError("Checkpoint provenance mismatch. Choose a new --out.")
        write_json(bound, binding)

        jobs = []
        for arm in ARMS:
            stage = create_stage(source, scratch, arm, reference)
            arm_dir = out / arm
            parts = arm_dir / "scores_parts"
            parts.mkdir(parents=True, exist_ok=True)
            pending = [month for month in months if not completed_month(parts, month)]
            for number, batch in enumerate(schedule_batches(pending, workers), 1):
                jobs.append((stage, arm_dir, batch, number, env))

        remaining = sum(len(job[2]) for job in jobs)
        print(f"Starting {len(jobs)} worker processes for {remaining} month-arm fits", flush=True)
        with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
            futures = [pool.submit(run_process, *job) for job in jobs]
            for future in futures:
                future.result()

    checked = {arm: verified_scores(out / arm, months, arm) for arm in ARMS}
    comparison = compare_course_scores(source, checked["control"][0])
    write_json(out / "course_control_comparison.json", comparison)
    arms = {}
    for arm, (scores, anchor_rows) in checked.items():
        output = out / arm / "scores.parquet"
        temporary = output.with_suffix(".tmp.parquet")
        scores.to_parquet(temporary, index=False)
        temporary.replace(output)
        arms[arm] = {
            "score_rows": len(scores),
            "growth_rows": int(scores.bucket.eq("growth_tech").sum()),
            "score_sha256": sha(output),
            "anchor_rows_exact": anchor_rows,
        }
    write_json(out / "training_manifest.json", {**binding, "arms": arms, "complete_refits": len(months) * len(ARMS)})
    print("Completed", arms, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] != ["--worker"]:
        raise SystemExit("Use run.py to launch the paired experiment.")
    worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
