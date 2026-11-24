"""Reproduce the paired 84-month selection experiment using course data through 2023."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
from train import ARMS, environment, sha, train

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "results"


def run(args, source, out):
    import pandas as pd

    for line in (HERE.parent / "course_files.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(source / name) != expected:
            raise ValueError(f"Course file differs from the pinned experiment: {name}")

    manifest = json.loads((REFERENCE / "training_manifest.json").read_text())
    score_root = args.scores_from.resolve() if args.scores_from else out / "scores"
    if not args.scores_from:
        train(source, score_root, args.workers)
    scores = {arm: score_root / arm / "scores.parquet" for arm in ARMS}
    for arm, path in scores.items():
        if sha(path) != manifest["arms"][arm]["score_sha256"]:
            raise ValueError(f"The {arm} score file differs from the recorded 84 monthly fits: {path}")

    # Put the pinned course package first so its portfolio, selection and shared modules load.
    sys.path.insert(0, str(source))
    import replay
    import discovery

    results = out / "results"
    with TemporaryDirectory(prefix="gr5398-long-selection-replay-") as temporary:
        inputs = replay.make_inputs(source, Path(temporary))
        replay.run(inputs, scores, results)
        stability = replay.stability(results)
        stability.to_csv(results / "stability.csv", index=False)
        discovery.run(inputs, scores, results, results / "corporate_action")

    files = {}
    for archive in sorted(REFERENCE.rglob("*.csv")):
        relative = archive.relative_to(REFERENCE)
        produced = results / relative
        pd.testing.assert_frame_equal(pd.read_csv(produced), pd.read_csv(archive),
                                      check_exact=False, rtol=1e-10, atol=1e-10)
        files[str(relative)] = sha(produced)
    receipt = {
        "mode": "previously recorded score hashes verified" if args.scores_from else "trained or resumed 84 months per arm",
        "months_per_arm": 84,
        "last_price_date": manifest["max_date"],
        "scores_sha256": {arm: sha(path) for arm, path in scores.items()},
        "columns_rows_and_values_matched_to_archived_tables": files,
    }
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified both arms and {len(files)} result tables. See {out / 'verification.json'}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Populated official course source_code directory")
    parser.add_argument("--out", type=Path, required=True, help="Separate directory for fitted scores and replay outputs")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4, help="Training processes per arm")
    parser.add_argument("--scores-from", type=Path, help="Reuse both existing 84-month scores after matching their archived hashes")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if not (source / "selection/configs/growth.json").is_file():
        parser.error("--source must point to the populated course source_code directory")
    protected = [source, HERE]
    if args.scores_from is not None:
        protected.append(args.scores_from.resolve())
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in protected):
        parser.error("Keep --out separate from course inputs, the submitted study files and reused scores")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".selection-study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, f"Another reproduction is using {out}\n")
        run(args, source, out)


if __name__ == "__main__":
    env = environment()
    if any(os.environ.get(k) != v for k, v in env.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    main()
