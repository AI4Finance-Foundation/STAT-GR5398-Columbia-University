"""Reproduce the formal and exploratory analyses inside the date-bounded scratch directory."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent


def main(work, stage):
    source = work / "valuation_candidate/source_code"
    pretest = source.parent / "results/pretest"
    if not pretest.is_dir() or not (work / "fable_review/phase2_checks/case_inputs.py").is_file():
        raise ValueError("Create a new workspace using reproduction/prepare.py first")
    if (source / "data").is_symlink():
        raise ValueError("Scratch inputs must be local dated extracts")
    baseline = source / "output/test_runs.jsonl"
    ledger = (SUBMISSION / "output/test_runs.jsonl").read_bytes()
    if baseline.read_bytes() != ledger:
        raise ValueError("The scratch test ledger differs from the submission archive")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    def run(script, *args):
        print(f"Running {script} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, str(script), *args], cwd=source, env=env, check=True)

    if stage == "checker":
        run(source / "check_submission.py", "--valuation")
    if stage == "formal":
        for split in ("dev", "val"):
            run(source / "run_portfolio.py", "--period", split, "--ablation", "--timing", "full")
            run(source / "validate_rules.py", split)
        for name, example in [("team", []), ("course_pe", ["--example"])]:
            run(source / "evaluate_valuation.py", "--period", "dev", *example)
            shutil.copyfile(source / "output/valuation_eval_dev.csv", pretest / f"{name}_eval_dev.csv")
        run(work / "legacy_evaluation/review.py")
        run(source / "attribute_valuation.py")
        run(work / "final_report_material/signal_comparison/compare.py")
    if stage == "extension":
        study = work / "Opus_handoff/strategy_research"
        review = work / "vote_bottleneck_review"
        gate = review / "gate_robustness"
        threshold = review / "threshold"
        locked = review / "frozen_1p983"
        execution = locked / "anet_execution"
        weighted = work / "weighted_valuation_research"
        for split in ("dev", "val"):
            run(study / "build_cache.py", split)
            run(study / "experiment.py", split)
            run(study / "diagnose.py", split)
        for script in (
            weighted / "study.py", weighted / "zeyu_overlay/compare.py",
            threshold / "study_dev.py", threshold / "empirical_crossings.py", threshold / "full_curve_dev.py",
            threshold / "ex_nvda/replay.py", threshold / "ex_nvda_attribution/verify_peak.py",
            threshold / "plot_full_curve.py", threshold / "ex_nvda/plot_comparison.py",
            locked / "validate.py", locked / "attribution/replay.py",
            execution / "week_counts.py", execution / "trace/trace.py", execution / "counter/replay_val.py",
            execution / "pro_rata/replay_val.py", execution / "plot_path.py",
            gate / "replay.py", gate / "summarize.py", gate / "placebo/rank_placebo.py",
            work / "fable_review/phase2_checks/case_inputs.py", review / "dev_dissent.py",
            review / "nvda_contrast/nvda/audit_dev.py", review / "nvda_contrast/effective/verify.py",
            execution / "history/anet_timeline.py",
        ):
            run(script)
    if baseline.read_bytes() != ledger:
        raise RuntimeError("Test ledger changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("stage", choices=["checker", "formal", "extension"])
    args = parser.parse_args()
    main(args.workspace.resolve(), args.stage)
