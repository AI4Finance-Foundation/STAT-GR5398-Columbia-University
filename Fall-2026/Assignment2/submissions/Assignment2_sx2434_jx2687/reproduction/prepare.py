"""Create a disposable copy of the grader with inputs ending December 29, 2023."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import pyarrow.parquet as parquet

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
RESEARCH = HERE / "research"
sys.path.insert(0, str(RESEARCH / "Jiao_delivery/bonus_handoff"))
from prepare_bonus import END, copy_database, copy_market_data


def copy_through(source, destination, column, last_day):
    table = parquet.read_table(source, filters=[(column, "<=", datetime.fromisoformat(last_day))])
    if not table.num_rows:
        raise ValueError(f"No {column} observations through {last_day} in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, destination)


def prepare(course, destination):
    if destination.exists() or destination.is_relative_to(course) or destination.is_relative_to(PACKAGE):
        raise ValueError("Choose a new working directory outside the submission and course source tree")
    if not (course / "run_portfolio.py").is_file():
        raise ValueError("The course path must point directly to source_code")
    destination.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        work = Path(temporary) / "prepared"
        shutil.copytree(RESEARCH, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        source = work / "valuation_candidate/source_code"
        for group in ("selection", "portfolio", "shared", "timing", "valuation"):
            for path in (course / group).rglob("*"):
                if path.is_file() and (path.suffix in {".py", ".json"} or path.name == "LICENSE"):
                    target = source / path.relative_to(course)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        for name in ("run_portfolio.py", "evaluate_valuation.py", "check_submission.py", "requirements.txt"):
            shutil.copyfile(course / name, source / name)
        shutil.copyfile(PACKAGE / "student_valuation.py", source / "valuation/student_valuation.py")
        data = source / "data"
        copy_market_data(course / "data", data)
        copy_database(course / "data/fundamentals.db", data / "fundamentals.db", decision=END)
        copy_through(course / "data/selection/monthly_scores.parquet", data / "selection/monthly_scores.parquet",
                     "month", "2023-12-01")
        copy_through(PACKAGE / "student_data/history_monthly.parquet", source / "student_data/history_monthly.parquet",
                     "date", END)
        pretest = source.parent / "results/pretest"
        pretest.mkdir(parents=True)
        shutil.copyfile(PACKAGE / "student_valuation.py", pretest / "student_valuation.py")
        output = source / "output"
        output.mkdir()
        shutil.copyfile(PACKAGE / "output/test_runs.jsonl", output / "test_runs.jsonl")
        work.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course_source", type=Path)
    parser.add_argument("new_destination", type=Path)
    args = parser.parse_args()
    prepare(args.course_source.resolve(), args.new_destination.resolve())
