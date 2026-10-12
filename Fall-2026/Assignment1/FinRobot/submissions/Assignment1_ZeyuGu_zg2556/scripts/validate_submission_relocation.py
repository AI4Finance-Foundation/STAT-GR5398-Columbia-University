"""Replay real saved V8.1 calls from a moved submission with its old root blocked.

No model or data API call is permitted. Copies the actual dependency closure (not
mock data), isolates all Python reads of the original submission in the worker
and renderer children, and uses a future execution clock. The source reports,
snapshots, paid responses and manifests are read-only comparison evidence.

Example (from submission):
  python -B scripts/validate_submission_relocation.py --upstream /path/to/clean-v8-2
"""
from __future__ import annotations
import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ("NVDA", "AMD", "INTC")
FUTURE_TIME = "2099-01-01T12:00:00+00:00"


def install_source_isolation(blocked_root):
    """Block old-root filesystem reads, including pathlib and native open calls.

    Stat/access probes report unavailable, while any attempted read fails loudly.
    The guard covers this Python process, not a machine-wide filesystem sandbox.
    The coordinator installs the same guard inside each renderer subprocess.
    """
    blocked = os.path.normcase(os.path.abspath(os.fspath(blocked_root)))
    events = []

    def denied(value, operation):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.path.normcase(os.path.abspath(os.fsdecode(value)))
        if path == blocked or path.startswith(blocked + os.sep):
            events.append({"operation":operation, "path":path})
            return True
        return False

    for name in ("stat", "lstat", "access"):
        original = getattr(os, name)
        def guarded(path, *args, _name=name, _original=original, **kwargs):
            if denied(path, _name):
                if _name == "access":
                    return False
                raise FileNotFoundError(errno.ENOENT, "Original submission isolated for migration test", os.fspath(path))
            return _original(path, *args, **kwargs)
        setattr(os, name, guarded)

    def audit(event, args):
        if event in {"open", "os.listdir", "os.scandir"} and args and denied(args[0], event):
            raise PermissionError("Original submission reads blocked during migration test")
    sys.addaudithook(audit)
    return events


def isolation_probe(blocked_root):
    try:
        (Path(blocked_root)/"README.md").read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("Original-root read isolation was not enforced")
    assert not (Path(blocked_root)/"snapshots").exists(), "Original-root stat isolation failed"


def run_worker(args):
    events = install_source_isolation(args.blocked_root)
    isolation_probe(args.blocked_root)
    probe_events = len(events)
    import experiment as e
    import revalidate_reasoning_offline as replay

    original_run = subprocess.run
    renderer_logs = []
    def guarded_subprocess(command, *positional, **kwargs):
        # setup() makes one read-only git rev-parse call; render uses guarded -c.
        if command[0] == sys.executable:
            if command[1] != "-c":
                raise AssertionError("Renderer did not use offline network guard")
            log_path = args.isolation_log.parent/f"renderer_isolation_{len(renderer_logs)+1}.json"
            renderer_logs.append(log_path)
            bootstrap = (
                "import runpy,json\n"
                f"_guard=runpy.run_path({str(Path(__file__).resolve())!r})\n"
                f"_events=_guard['install_source_isolation']({str(args.blocked_root)!r})\n"
                f"_guard['isolation_probe']({str(args.blocked_root)!r})\n"
                "_probe_events=len(_events)\n"
                "try:\n"
                f"    exec(compile({command[2]!r}, '<offline renderer>', 'exec'))\n"
                "finally:\n"
                f"    with open({str(log_path)!r}, 'w', encoding='utf-8') as _out:\n"
                "        json.dump({'probe_denials':_probe_events,'unexpected_original_root_attempts':"
                "_events[_probe_events:],'guard_events':_events},_out,indent=2)\n"
            )
            command = [command[0], "-B", "-c", bootstrap, *command[3:]]
        elif Path(command[0]).name.lower() not in {"git", "git.exe"} or command[-2:] != ["rev-parse", "HEAD"]:
            raise AssertionError("Unexpected subprocess during isolated replay")
        return original_run(command, *positional, **kwargs)

    sys.argv = [str(Path(replay.__file__)), "--upstream", str(args.upstream),
                "--output-root", str(args.output_root), "--verifier-version", args.verifier_version,
                "--disable-network", "--tickers", *TICKERS]
    try:
        with patch.object(e, "utc_now", lambda: FUTURE_TIME), patch.object(subprocess, "run", guarded_subprocess):
            replay.main()
    finally:
        args.isolation_log.write_text(json.dumps({"probe_denials":probe_events,
            "unexpected_original_root_attempts":events[probe_events:], "guard_events":events,
            "renderer_logs":[str(path) for path in renderer_logs]}, indent=2), encoding="utf-8")
    assert len(renderer_logs) == 3
    assert not events[probe_events:], "Replay attempted to access original submission"
    for path in renderer_logs:
        assert not json.loads(path.read_text())["unexpected_original_root_attempts"]


def long_path(path):
    path = Path(path).resolve()
    return Path("\\\\?\\" + str(path)) if os.name == "nt" and not str(path).startswith("\\\\?\\") else path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--verifier-version", choices=("v8_2", "v9", "v9_1"), default="v8_2",
                   help="Deterministic validator/report version to compare; all reuse preserved paid V8.1 reviews")
    p.add_argument("--work-root", type=Path, help="New empty short directory; retained for inspection")
    p.add_argument("--evidence-dir", type=Path, default=ROOT/"evidence/reproducibility/true_submission_relocation")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--blocked-root", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--isolation-log", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        run_worker(args)
        return

    import experiment as e
    from audit_io import write_json, utc_now, sha256
    source = ROOT.resolve()
    work = args.work_root.resolve() if args.work_root else Path(tempfile.mkdtemp(prefix="frrel-"))
    if args.work_root:
        if work.exists():
            raise ValueError("Use a new work-root; existing experiment evidence is never replaced")
        work.mkdir(parents=True)
    moved = work/"submission"
    output = work/"replay"
    moved.mkdir()
    dependencies = ["scripts", "config/budget.reasoning_followup.json", "evidence/costs.jsonl",
                    f"evidence/verifier_versions/verifier_agent_{args.verifier_version}.py"]
    for ticker in TICKERS:
        baseline = e.read_json(source/"runs"/f"{ticker}_baseline_1"/"manifest.json")
        dependencies.extend([f"runs/{ticker}_baseline_1", f"runs/{ticker}_final_reasoning_v8_1",
                             "snapshots/" + Path(baseline["source_analysis"]).parent.name])
    source_hashes = {}
    for relative in dependencies:
        origin = long_path(source/relative)
        target = moved/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
            source_hashes.update({relative+"/"+key:value for key, value in e.hashes(origin).items()})
        else:
            shutil.copy2(origin, target)
            source_hashes[relative] = sha256(origin.read_bytes())
    assert e.hashes(moved) == source_hashes, "Copied dependency bytes differ from original inputs"
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if (args.evidence_dir/"manifest.json").exists():
        raise ValueError("Use a new evidence-dir; previous validation manifests are preserved")
    isolation_log = work/"worker_isolation.json"
    command = [sys.executable, "-B", str(moved/"scripts"/Path(__file__).name), "--worker",
               "--blocked-root", str(source), "--upstream", str(args.upstream.resolve()),
               "--verifier-version", args.verifier_version,
               "--output-root", str(output), "--isolation-log", str(isolation_log)]
    completed = subprocess.run(command, cwd=work, text=True, capture_output=True, encoding="utf-8",
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    (args.evidence_dir/"replay.log").write_text(completed.stdout+completed.stderr, encoding="utf-8")
    for path in work.glob("*isolation*.json"):
        shutil.copy2(path, args.evidence_dir/path.name)
    if completed.returncode:
        raise RuntimeError("Isolated replay failed; inspect " + str(args.evidence_dir/"replay.log"))
    records = []
    for ticker in TICKERS:
        canonical = source/"runs"/f"{ticker}_final_reasoning_{args.verifier_version}"
        replayed = output/canonical.name
        original_manifest = e.read_json(canonical/"manifest.json")
        new_manifest = e.read_json(replayed/"manifest.json")
        analysis = replayed/"output"/ticker/"analysis"
        assert e.hashes(analysis) == original_manifest["analysis_hashes"]
        report = replayed/"output"/ticker/"report"
        reference_report = canonical/"output"/ticker/"report"
        html = f"Audited_Equity_Report_{ticker}.html"
        assert (report/html).read_bytes() == (reference_report/html).read_bytes(), "HTML differs: " + ticker
        asset_hashes = {key:value for key,value in e.hashes(report).items() if Path(key).suffix == ".png"}
        assert asset_hashes == {key:value for key,value in e.hashes(reference_report).items() if Path(key).suffix == ".png"}
        referenced_assets = {key:value for key,value in asset_hashes.items() if key.startswith("assets/")}
        assert len(referenced_assets) == 3 and len(asset_hashes) == 6
        sections = e.read_json(report/"rendered_section_texts.json")["sections"]
        assert set(sections) == set(e.FIELDS)
        assert all(sections[key]["sha256"] == sha256((analysis/(key+".txt")).read_text(encoding="utf-8")) for key in e.FIELDS)
        assert new_manifest["render_input_hashes"] == original_manifest["render_input_hashes"]
        assert e.hashes(replayed/"render_inputs") == original_manifest["render_input_hashes"]
        assert new_manifest["source_analysis"] == original_manifest["source_analysis"]
        assert new_manifest["started_at_utc"] == FUTURE_TIME
        assert new_manifest["report_date"] == original_manifest.get("report_date", original_manifest["started_at_utc"][:10])
        resolution = e.read_json(replayed/"render_input_resolution.json")
        assert Path(resolution["resolved_snapshot"]).is_relative_to(moved/"snapshots")
        market = e.read_json(replayed/"render_inputs/market_snapshot.json")
        engine = e.read_json(replayed/"render_inputs/valuation/engine_valuation.json")
        calls = []
        for section in e.FIELDS:
            reuse = e.read_json(replayed/"verifier"/section/"reused_review.json")
            assert reuse["new_model_generation"] is False
            for filename, digest in reuse["copied_file_hashes"].items():
                assert sha256((replayed/"verifier"/section/filename).read_bytes()) == digest
            calls.append(reuse["source_call_id"])
        records.append({"ticker":ticker, "exact_analysis_files":len(original_manifest["analysis_hashes"]),
            "exact_sections":len(sections), "html_sha256":sha256((report/html).read_bytes()),
            "exact_chart_files":asset_hashes, "exact_html_chart_assets":referenced_assets,
            "exact_render_input_hashes":new_manifest["render_input_hashes"],
            "market_metrics":market.get("metrics",market), "valuation_results":engine.get("results"),
            "source_paid_call_ids":calls, "relocation":resolution,
            "future_execution_time":FUTURE_TIME, "preserved_report_date":new_manifest["report_date"]})
    assert e.hashes(moved) == source_hashes, "Copied original inputs changed during replay"
    for relative, digest in source_hashes.items():
        assert sha256(long_path(source/relative).read_bytes()) == digest, "Original source changed: " + relative
    write_json(args.evidence_dir/"manifest.json", {"created_at_utc":utc_now(), "status":"passed",
        "verifier_version":args.verifier_version,
        "source_submission":str(source), "copied_submission":str(moved), "replay_root":str(output),
        "upstream":str(args.upstream.resolve()), "upstream_sha":e.UPSTREAM_SHA,
        "new_paid_calls":0, "new_paid_cost_usd":0, "network_disabled_during_verification_and_render":True,
        "isolation_scope":"Original submission Python filesystem access denied in worker and all three renderer children; explicit read/stat probes pass. Not a machine-wide OS sandbox.",
        "dependency_copy_hashes":source_hashes, "original_and_copied_inputs_unchanged":True,
        "script_sha256":sha256(Path(__file__).read_bytes()),
        "experiment_script_sha256":sha256((source/"scripts/experiment.py").read_bytes()),
        "replay_script_sha256":sha256((source/"scripts/revalidate_reasoning_offline.py").read_bytes()),
        "command":command, "records":records})
    print(json.dumps({"status":"passed", "copied_submission":str(moved), "exact_sections":24,
        "exact_html_reports":3, "exact_chart_assets":9, "evidence":str(args.evidence_dir)}, indent=2))


if __name__ == "__main__":
    main()
