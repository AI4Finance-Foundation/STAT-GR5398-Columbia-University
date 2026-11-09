"""Fixed evaluation periods. Tune on dev, choose the final configuration on val,
run test once. forward is the live period after the frozen data cutoff."""
from datetime import date

PERIODS = {
    "dev": (date(2017, 1, 3), date(2021, 12, 31)),
    "val": (date(2022, 1, 3), date(2023, 12, 29)),
    "test": (date(2024, 1, 2), date(2026, 10, 9)),
    "forward": (date(2026, 10, 12), date(2099, 12, 31)),
}


def resolve(period: str | None, start: str | None, end: str | None):
    if period:
        return PERIODS[period]
    if not (start and end):
        raise SystemExit("Give --period dev|val|test|forward, or both --start and --end")
    return date.fromisoformat(start), date.fromisoformat(end)


def record_test_run(script: str, argv: list[str], start: date, end: date) -> int:
    """Log every run that touches the test period to output/test_runs.jsonl; return its number.

    Report this number. The test period is meant to be run once, after the
    final configuration was chosen on val and committed.
    """
    import json, subprocess
    from datetime import datetime
    from pathlib import Path

    t0, t1 = PERIODS["test"]
    if end < t0 or start > t1:
        return 0
    log = Path("output") / "test_runs.jsonl"
    log.parent.mkdir(exist_ok=True)
    n = (sum(1 for _ in log.open()) if log.exists() else 0) + 1
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "student_timing.py", "student_valuation.py"],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        head, dirty = "", None
    with log.open("a") as f:
        f.write(json.dumps({"n": n, "time": datetime.now().isoformat(timespec="seconds"), "script": script,
                            "argv": argv, "git_head": head, "student_files_uncommitted": dirty}) + "\n")
    print(f"*** Test-period run #{n} (logged in {log}). Report this number. ***")
    return n
