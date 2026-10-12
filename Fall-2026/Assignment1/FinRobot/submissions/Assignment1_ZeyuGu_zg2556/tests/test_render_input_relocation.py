"""Synthetic migration regressions only; no fixtures are experiment evidence."""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import experiment as e


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    root = tmp_path/"moved_submission"
    source = root/"snapshots"/"TEST"
    analysis = source/"analysis"
    analysis.mkdir(parents=True)
    (analysis/"accounts.csv").write_text("year,revenue\n2025,100\n", encoding="utf-8")
    (source/"valuation").mkdir()
    for name, value in {"market_snapshot.json":{"price":20}, "valuation_inputs.json":{"shares":10},
                        "valuation/engine_valuation.json":{"target_price":25}}.items():
        e.write_json(source/name, value)
    inputs = e.hashes(analysis)
    snapshot = {"status":"frozen", "upstream_sha":e.UPSTREAM_SHA, "run_id":"TEST-frozen",
                "file_hashes":e.hashes(source)}
    e.write_json(source/"manifest.json", snapshot)
    parent = root/"runs"/"TEST_baseline_1"/"render_inputs"
    parent.mkdir(parents=True)
    supplements = e.render_supplement_hashes(snapshot["file_hashes"])
    historical = tmp_path/"unavailable_original"/"snapshots"/"TEST"
    provenance = {"source_snapshot":str(historical), "snapshot_run_id":snapshot["run_id"],
        "snapshot_manifest_sha256":e.sha256((source/"manifest.json").read_bytes()),
        "input_hashes":supplements, "scope":"test fixture provenance"}
    e.write_json(parent/"manifest.json", provenance)
    manifest = {"source_analysis":str(historical/"analysis"), "input_hashes":inputs,
        "parent_run_id":"TEST_baseline_1", "render_input_hashes":dict(supplements,
            **{"manifest.json":e.sha256((parent/"manifest.json").read_bytes())})}
    run = root/"replay"/"TEST_final"
    run.mkdir(parents=True)
    monkeypatch.setattr(e, "SUBMISSION_ROOT", root)
    return root, source, parent, run, manifest


def test_relocation_preserves_historical_manifest_and_all_supplements(relocated):
    root, source, parent, run, manifest = relocated
    original_manifest = copy.deepcopy(manifest)
    raw = (parent/"manifest.json").read_bytes()
    market = e.freeze_render_inputs(run, manifest)
    assert market.read_bytes() == (source/"market_snapshot.json").read_bytes()
    assert (run/"render_inputs"/"manifest.json").read_bytes() == raw
    assert manifest == original_manifest
    assert e.hashes(run/"render_inputs") == manifest["render_input_hashes"]
    resolution = e.read_json(run/"render_input_resolution.json")
    assert resolution["resolved_snapshot"] == str(source.resolve())
    assert resolution["historical_source_analysis"] == manifest["source_analysis"]


def test_completed_render_inputs_work_without_any_snapshot(relocated):
    root, source, parent, run, manifest = relocated
    e.freeze_render_inputs(run, manifest)
    shutil.rmtree(root/"snapshots")
    shutil.rmtree(root/"runs")
    assert e.freeze_render_inputs(run, manifest).is_file()
    assert e.read_json(run/"render_input_resolution.json")["method"] == "self-contained saved render inputs"


def test_changed_market_bytes_rejected_even_when_manifest_claims_same_hash(relocated):
    _, source, _, run, manifest = relocated
    (source/"market_snapshot.json").write_text('{"price":999}', encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen render supplement changed"):
        e.freeze_render_inputs(run, manifest)
    assert not (run/"render_inputs").exists()


def test_changed_analysis_bytes_rejected_before_copy(relocated):
    _, source, _, run, manifest = relocated
    (source/"analysis"/"accounts.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        e.freeze_render_inputs(run, manifest)


def test_same_analysis_with_different_market_snapshot_cannot_win(relocated):
    root, source, _, run, manifest = relocated
    wrong = root/"snapshots"/"AAA_same_accounts_new_price"
    shutil.copytree(source, wrong)
    e.write_json(wrong/"market_snapshot.json", {"price":999})
    different = e.read_json(wrong/"manifest.json")
    different["file_hashes"]["market_snapshot.json"] = e.sha256((wrong/"market_snapshot.json").read_bytes())
    e.write_json(wrong/"manifest.json", different)
    e.freeze_render_inputs(run, manifest)
    assert e.read_json(run/"render_input_resolution.json")["resolved_snapshot"] == str(source.resolve())


def test_missing_snapshot_fails_instead_of_omitting_market_and_valuation(relocated):
    root, _, _, run, manifest = relocated
    shutil.rmtree(root/"snapshots")
    with pytest.raises(ValueError, match="No frozen snapshot matches"):
        e.freeze_render_inputs(run, manifest)


def test_missing_historical_provenance_not_silently_regenerated(relocated):
    _, _, parent, run, manifest = relocated
    (parent/"manifest.json").unlink()
    with pytest.raises(ValueError, match="exact saved render provenance"):
        e.freeze_render_inputs(run, manifest)


def test_tampered_existing_copy_is_not_repaired_over_evidence(relocated):
    _, _, _, run, manifest = relocated
    e.freeze_render_inputs(run, manifest)
    (run/"render_inputs"/"valuation"/"engine_valuation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Copied render inputs changed"):
        e.freeze_render_inputs(run, manifest)


def test_partial_copy_can_resume_without_changing_provenance(relocated):
    _, source, parent, run, manifest = relocated
    partial = run/"render_inputs"
    partial.mkdir()
    shutil.copy2(source/"market_snapshot.json", partial/"market_snapshot.json")
    shutil.copy2(parent/"manifest.json", partial/"manifest.json")
    e.freeze_render_inputs(run, manifest)
    assert e.hashes(partial) == manifest["render_input_hashes"]


def test_missing_valuation_identity_fails_explicitly(relocated):
    _, _, _, run, manifest = relocated
    manifest["render_input_hashes"].pop("valuation/engine_valuation.json")
    with pytest.raises(ValueError, match="market/valuation inputs are required"):
        e.freeze_render_inputs(run, manifest)


def test_fresh_run_rejects_same_accounts_with_ambiguous_market_snapshots(relocated):
    root, source, _, run, manifest = relocated
    other = root/"snapshots"/"AAA_same_accounts_new_market"
    shutil.copytree(source, other)
    e.write_json(other/"market_snapshot.json", {"price":999})
    snapshot = e.read_json(other/"manifest.json")
    snapshot["file_hashes"]["market_snapshot.json"] = e.sha256((other/"market_snapshot.json").read_bytes())
    e.write_json(other/"manifest.json", snapshot)
    manifest.pop("render_input_hashes")
    assert e.hashes(other/"analysis") == manifest["input_hashes"] == e.hashes(source/"analysis")
    with pytest.raises(ValueError, match="Ambiguous frozen snapshots"):
        e.freeze_render_inputs(run, manifest)
    assert not (run/"render_inputs").exists()


def test_fresh_run_explicit_matching_snapshot_selects_its_own_market(relocated):
    root, source, _, run, manifest = relocated
    other = root/"snapshots"/"AAA_same_accounts_new_market"
    shutil.copytree(source, other)
    e.write_json(other/"market_snapshot.json", {"price":999})
    snapshot = e.read_json(other/"manifest.json")
    snapshot["file_hashes"]["market_snapshot.json"] = e.sha256((other/"market_snapshot.json").read_bytes())
    e.write_json(other/"manifest.json", snapshot)
    manifest.pop("render_input_hashes")
    # The first directory contains 999; the explicitly selected source contains 20.
    manifest["source_analysis"] = str(source/"analysis")
    assert e.hashes(other/"analysis") == manifest["input_hashes"]
    market = e.freeze_render_inputs(run, manifest)
    assert e.read_json(market) == {"price":20}
    assert e.read_json(run/"render_input_resolution.json")["resolved_snapshot"] == str(source.resolve())
    assert manifest["render_input_hashes"]["market_snapshot.json"] == e.sha256((source/"market_snapshot.json").read_bytes())
