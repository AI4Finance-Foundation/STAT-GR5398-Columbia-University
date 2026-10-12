"""Apply accepted machine measurement errata to NEW V8.2 audit directories.

No generation, model request, credential read, original-label edit or text edit.
Run from the submission root with the dedicated Python environment. Use
--validate-only to check all inputs without writing; normal execution refuses
every existing destination. Reproduction requires an unused --suffix.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import os
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True


def network_denied(*args, **kwargs):
    raise RuntimeError("Post-release errata are strictly offline")


socket.create_connection = network_denied
socket.socket.connect = network_denied
socket.socket.connect_ex = network_denied

from audit_io import sha256, utc_now, write_json
from grounding_audit import (
    FIELDS, compare, csv_write, load_audit, load_inputs, metrics, paragraphs,
    source_units, summarize as summarize_audits,
)
from analyze_verifier_annotations import key as fact_identity
from analyze_verifier_annotations import summarize as annotation_metrics

ROOT = Path(__file__).resolve().parents[1]
REVIEW = Path("evidence/post_release_review")
PINNED = {
    "comparison_proposals.json": "f10f8990e3012c16e24130db6f85f5a9699dceade41a5f47ca462d55ee5590b9",
    "cagr_proposals.json": "95409fbb3f79fce1c55e13c402633f81e77e774e23b21f368af894d9af0c2cbb",
}
PAIR_FILE = Path("config/grounding_reasoning_v8_2_pairs.json")
PAIR_HASH = "0cfc3fdc956f5a0e0cde3b49dc182aaaa1b7730eaeb86f84b2ab692ea7038807"
SPLIT_ID = "adjudicated.valuation_overview.p0.u1.c2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lexical_key(ticker, claim):
    return ticker.lower() + "|" + re.sub(r"\s+", " ", claim.lower()).strip()


def check_evidence(path, expected_hash, refs, normalized_text_hash=False):
    path = Path(path)
    content = path.read_text(encoding="utf-8") if normalized_text_hash else path.read_bytes()
    require(sha256(content) == expected_hash, "Changed original prompt: " + str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    for ref in refs:
        a, b = ref["line_start"], ref["line_end"]
        require(1 <= a <= b <= len(lines), "Invalid evidence range")
        require(ref["quote"] == "\n".join(lines[a-1:b]), "Altered evidence quote")


def root_decision(suffix):
    return {
        "decision_version": "post-release-errata-v1",
        "decision_maker": "Root Codex task agent /root; machine review, not human annotation",
        "human_review_performed": False,
        "decision_basis": "Root agent individually reviewed and accepted the 26 comparison proposals and five ambiguous after-CAGR scope proposals.",
        "accepted_comparison_proposals": 26,
        "accepted_ambiguous_cagr_proposals": 5,
        "exception": {
            "ticker": "AMD", "stage": "before", "claim_id": SPLIT_ID,
            "decision": "Split the original compound atom into Supported PE decline after2023 and Unsupported remaining-elevated judgment. Preserve the supported decline; do not discard it with the missing benchmark.",
            "supported_claim": "AMD PE declined after 2023.",
            "supported_evidence": "2023A278.1320754716981 > 2024A123.95049504950495 > 2025A80.52059925093633; original prompt lines4 and16.",
            "unsupported_claim": "AMD PE remained elevated.",
        },
        "cagr_instruction": "Keep each original table-scope wording. Do not invent a start/end statistical interval, promote ambiguous scope to Supported, or remove separately supported literal table-cell atoms.",
        "preservation_instruction": "Keep true AMD numerical-range contradictions and all other canonical judgments. New versioned output only; original text, labels, evidence and experiments remain immutable.",
        "measurement_warning": "All changes are corrections to measurement/atomization, not verifier improvements. No new model run or successful correction is inferred.",
        "proposal_hashes": PINNED,
        "parent_pair_sha256": PAIR_HASH,
        "output_suffix": suffix,
    }


def build_changes(comparison, cagr, pairs):
    changes = {}
    after_by_ticker = {Path(p["final_run"]).name.split("_")[0]:p["after"] for p in pairs}
    require(len(comparison["proposals"]) == 26, "Unexpected comparison decision scope")
    for p in comparison["proposals"]:
        audit = str(Path(p["audit_path"]).parent).replace("\\", "/")
        require(sha256(Path(p["audit_path"]).read_bytes()) == p["audit_sha256"], "Changed reviewed audit")
        check_evidence(p["prompt_path"], p["prompt_sha256"], p["exact_original_prompt_evidence"])
        k = (audit, p["claim_id"])
        require(k not in changes, "Duplicate decision")
        changes[k] = {
            "proposal_file": str(REVIEW/"comparison_proposals.json"),
            "proposal_case": p["case_id"], "source_quote": p["source_quote"],
            "old_claim": p["old_claim"], "old_classification": p["old_classification"],
            "old_numeric_status": p["old_numeric_status"],
            "claim": p["conservative_claim"], "classification": p["suggested_classification"],
            "numeric_status": p["suggested_numeric_status"], "reason": p["reason"],
            "evidence": p["exact_original_prompt_evidence"],
        }
    selected = [p for p in cagr["after_proposals"] if p["proposed_classification"] == "Unsupported"]
    require(len(selected) == 5, "Unexpected CAGR decision scope")
    for p in selected:
        require(p["proposed_numeric_status"] == "unverifiable", "Ambiguous CAGR is not a verified number")
        # Audit source hashes use decoded text with universal-newline normalization;
        # the run manifest separately checks original on-disk bytes in load_inputs.
        require(sha256(Path(p["source_file"]).read_text(encoding="utf-8")) == p["source_sha256"], "Changed CAGR source")
        check_evidence(p["writer_prompt"], p["writer_prompt_sha256"], p["prompt_evidence"], normalized_text_hash=True)
        k = (after_by_ticker[p["ticker"]], p["claim_id"])
        require(k not in changes, "Duplicate CAGR/comparison decision")
        require(p["faithful_claim"] == p["source_quote"], "CAGR scope must retain original wording")
        changes[k] = {
            "proposal_file": str(REVIEW/"cagr_proposals.json"),
            "proposal_case": p["scope_type"], "source_quote": p["source_quote"],
            "old_claim": p["original_claim"], "old_classification": p["original_classification"],
            "old_numeric_status": p["original_numeric_status"],
            "claim": p["source_quote"], "classification": "Unsupported", "numeric_status": "unverifiable",
            "reason": p["reason"] + " " + p["atomization_caution"], "evidence": p["prompt_evidence"],
        }
    require(len(changes) == 31, "Expected 31 unique reviewed original atoms")
    return changes


def verify_rows(info, rows, inputs, coverage):
    require(len({r["claim_id"] for r in rows}) == len(rows), "Duplicate claim ID")
    expected_units = {}
    for section, item in inputs.items():
        require(info["source_hashes"][section] == sha256(item["text"]), "Audit source hash mismatch")
        require(info["evidence_hashes"][section] == sha256(item["evidence"]), "Audit prompt hash mismatch")
        for unit in source_units(item["text"]):
            expected_units[(section, unit["unit_id"])] = unit
    supplied_units = {}
    for c in coverage:
        k = (c["section"], int(c["unit_id"]))
        require(k not in supplied_units, "Duplicate coverage unit")
        supplied_units[k] = c
        unit = expected_units[k]
        require(int(c["source_start"]) == unit["start"] and int(c["source_end"]) == unit["end"], "Coverage span changed")
        require(c["source_quote"] == inputs[k[0]]["text"][unit["start"]:unit["end"]], "Coverage text changed")
    require(set(expected_units) == set(supplied_units), "Incomplete eight-section textual coverage")
    counts = collections.Counter()
    for row in rows:
        section = row["section"]
        item = inputs[section]
        a, b = int(row["source_start"]), int(row["source_end"])
        require(item["text"][a:b] == row["source_quote"], "Claim no longer matches source span")
        require(row["source_sha256"] == sha256(item["text"]), "Claim source hash differs")
        require(row["evidence_prompt_sha256"] == sha256(item["evidence"]), "Claim prompt hash differs")
        unit_match = [k for k,u in expected_units.items() if k[0] == section and u["start"] == a and u["end"] == b]
        require(len(unit_match) == 1, "Claim has no unique original unit")
        counts[unit_match[0]] += 1
        refs = json.loads(row["evidence"])
        for ref in refs:
            lines = item["evidence"].splitlines()
            require(ref["quote"] == "\n".join(lines[ref["line_start"]-1:ref["line_end"]]), "Claim evidence altered")
        require(row["classification"] in {"Supported", "Unsupported", "Contradicted"}, "Invalid class")
        require(row["numeric_status"] in {"correct", "incorrect", "unverifiable", "not_numeric"}, "Invalid number status")
        require(row["classification"] == "Unsupported" or refs, "No original evidence for S/C")
        require(row["numeric_status"] != "incorrect" or row["classification"] == "Contradicted", "Incorrect number not Contradicted")
        require(row["classification"] != "Supported" or row["numeric_status"] not in {"incorrect", "unverifiable"}, "Unsupported number marked Supported")
    for k,c in supplied_units.items():
        require(int(c["claim_count"]) == counts[k], "Coverage claim count stale")
        require(counts[k] or c["nonfactual_reason"].strip(), "Empty unit without rationale")


def apply_audit(directory, destination, changes):
    info, old_rows = load_audit(directory)
    run = Path("runs")/info["run_id"]
    evidence_run = Path("runs")/old_rows[0]["evidence_run_id"]
    _, inputs = load_inputs(run, evidence_run)
    coverage = read_csv(Path(directory)/"text_coverage.csv")
    verify_rows(info, old_rows, inputs, coverage)
    rows, mappings, consumed = [], [], set()
    for old in old_rows:
        k = (directory, old["claim_id"])
        d = changes.get(k)
        replacements = [copy.deepcopy(old)]
        if d:
            for field in ("source_quote", "claim", "classification", "numeric_status"):
                expected = d["source_quote"] if field == "source_quote" else d["old_"+field]
                require(old[field] == expected, "Old atom changed since decision: " + repr(k) + " " + field)
            row = replacements[0]
            row.update({name:d[name] for name in ("claim", "classification", "numeric_status", "reason")})
            row["evidence"] = json.dumps(d["evidence"], ensure_ascii=False)
            row["fact_key"] = lexical_key(row["ticker"], row["claim"])
            if old["ticker"] == "AMD" and old["run_id"] == "AMD_baseline_1" and old["claim_id"] == SPLIT_ID:
                decline = copy.deepcopy(row)
                decline.update(claim_id=old["claim_id"]+".errata_decline", claim="AMD PE declined after 2023.",
                    classification="Supported", numeric_status="correct",
                    reason="The original says PE surged in2023 followed by partial normalization. Prompt line16 shows2023A278.1320754716981 >2024A123.95049504950495 >2025A80.52059925093633, with periods fixed by line4. This supported decline is separate from an unsupported elevated-level judgment.")
                lines = inputs[old["section"]]["evidence"].splitlines()
                decline["evidence"] = json.dumps([{"line_start":n,"line_end":n,"quote":lines[n-1]} for n in (4,16)])
                decline["fact_key"] = lexical_key("AMD", decline["claim"])
                row.update(claim_id=old["claim_id"]+".errata_elevated", claim="AMD PE remained elevated.", numeric_status="not_numeric")
                row["fact_key"] = lexical_key("AMD", row["claim"])
                replacements = [decline, row]
            for new in replacements:
                mappings.append({
                    "ticker":old["ticker"], "section":old["section"], "run_id":old["run_id"],
                    "old_audit":directory, "new_audit":destination,
                    "old_claim_id":old["claim_id"], "new_claim_id":new["claim_id"],
                    "old_claim":old["claim"], "new_claim":new["claim"],
                    "old_classification":old["classification"], "new_classification":new["classification"],
                    "old_numeric_status":old["numeric_status"], "new_numeric_status":new["numeric_status"],
                    "old_fact_key":old["fact_key"], "new_fact_key":new["fact_key"],
                    "source_quote":old["source_quote"], "source_sha256":old["source_sha256"],
                    "source_start":old["source_start"], "source_end":old["source_end"],
                    "evidence_prompt_sha256":old["evidence_prompt_sha256"],
                    "old_evidence":old["evidence"], "new_evidence":new["evidence"],
                    "reason":new["reason"], "proposal_file":d["proposal_file"], "proposal_case":d["proposal_case"],
                    "change_type":"compound_atom_split" if len(replacements)==2 else "measurement_erratum",
                })
            consumed.add(k)
        rows.extend(replacements)
    for c in coverage:
        c["claim_count"] = sum(r["section"] == c["section"] and r["source_start"] == c["source_start"] and r["source_end"] == c["source_end"] for r in rows)
    verify_rows(info, rows, inputs, coverage)
    # Every unreviewed row must be structurally identical, including fact key and evidence.
    by_id = {r["claim_id"]:r for r in rows}
    require(all(by_id[r["claim_id"]] == r for r in old_rows if (directory,r["claim_id"]) not in changes), "Unreviewed atom changed")
    return {"parent":directory, "destination":destination, "info":info, "old_rows":old_rows,
            "rows":rows, "coverage":coverage, "inputs":inputs, "mappings":mappings, "consumed":consumed}


def write_audit(plan, decision_hash):
    output = Path(plan["destination"])
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output/"adjudicated_claims.csv", plan["rows"])
    (output/"claims.csv").write_bytes((output/"adjudicated_claims.csv").read_bytes())
    csv_write(output/"text_coverage.csv", plan["coverage"])
    csv_write(output/"errata_atom_mapping.csv", plan["mappings"])
    info = copy.deepcopy(plan["info"])
    csv_write(output/"grounding_summary.csv", [{"ticker":info["ticker"],"run_id":info["run_id"],"section":section,
        **metrics(plan["rows"] if section=="ALL" else [r for r in plan["rows"] if r["section"]==section])} for section in [*FIELDS,"ALL"]])
    spec = {"protocol_version":"machine-adjudication-v1+accepted-post-release-errata-v1",
        "parent_audit":plan["parent"],"root_decision_sha256":decision_hash,
        "run_id":info["run_id"],"evidence_run_id":plan["rows"][0]["evidence_run_id"],"sections":{}}
    for section in FIELDS:
        units=[]
        for c in plan["coverage"]:
            if c["section"]!=section: continue
            matched=[r for r in plan["rows"] if r["section"]==section and r["source_start"]==c["source_start"] and r["source_end"]==c["source_end"]]
            atoms=[{**{f:r[f] for f in ("claim_id","claim","claim_type","classification","numeric_status","fact_key","reason")},"evidence":json.loads(r["evidence"])} for r in matched]
            units.append({"unit_id":int(c["unit_id"]),"atoms":atoms,"nonfactual_reason":c["nonfactual_reason"]})
        spec["sections"][section]={"source_sha256":info["source_hashes"][section],"evidence_sha256":info["evidence_hashes"][section],"units":units}
    write_json(output/"reviewed_specification.json",spec)
    (output/"parent_manifest.json").write_bytes((Path(plan["parent"])/"manifest.json").read_bytes())
    (output/"adjudication_instructions.md").write_bytes((Path(plan["parent"])/"adjudication_instructions.md").read_bytes())
    info.update(auditor_version="machine-adjudication-v1+post-release-errata-v1",
        completed_at_utc=utc_now(),parent_audit=plan["parent"],
        parent_claim_csv_sha256=sha256((Path(plan["parent"])/"claims.csv").read_bytes()),
        claim_csv_sha256=sha256((output/"claims.csv").read_bytes()),
        text_coverage_sha256=sha256((output/"text_coverage.csv").read_bytes()),
        reviewed_specification_sha256=sha256((output/"reviewed_specification.json").read_bytes()),
        root_decision_sha256=decision_hash,measurement_only_correction=True,source_text_modified=False,
        new_paid_api_calls=0,human_review_performed=False,
        errata_review_scope="Only accepted31 original atoms; other canonical rows unchanged. Inherited exhaustive eight-section audit, not a fresh human/full re-review.",
        fact_key_update_policy="Changed atoms use a conservative lowercased lexical claim key; unchanged keys are preserved. Lexical retention changes are measurement artifacts, not semantic loss estimates.",
        limitations=info.get("limitations","")+" Accepted post-release scope/benchmark/quantifier errata alter labels and one compound atom. This is measurement correction, not a verifier improvement.")
    write_json(output/"manifest.json",info)


def write_annotations(pairs, output):
    allrows, results, facts = [], [], []
    for pair in pairs:
        bi,b=load_audit(pair["before"]); ai,a=load_audit(pair["after"])
        section_paras={s:{p["paragraph_id"]:p for p in paragraphs((Path(pair["final_run"])/"output"/ai["ticker"]/"analysis"/(s+".txt")).read_text(encoding="utf-8"))} for s in FIELDS}
        rows=[]
        for row in a:
            paragraph=section_paras[row["section"]][int(row["paragraph_id"])]
            rows.append(dict(row,warning_applies=bool(re.match(r"^\s*\[UNVERIFIED\b",paragraph["text"])),warning_basis="Whole paragraph prefix; independent underlying assertion classification unchanged"))
        for section in [*FIELDS,"ALL"]:
            results.append({"ticker":ai["ticker"],"section":section,**annotation_metrics(rows if section=="ALL" else [r for r in rows if r["section"]==section])})
        bykey={}
        for row in rows: bykey.setdefault(fact_identity(row),[]).append(row)
        for row in b:
            matched=bykey.get(fact_identity(row),[])
            facts.append({"ticker":bi["ticker"],"section":row["section"],"before_claim_id":row["claim_id"],"before_classification":row["classification"],"claim":row["claim"],"after_matching_claim_ids":json.dumps([x["claim_id"] for x in matched]),"lexically_retained":bool(matched),"remaining_same_label_instances":sum(x["classification"]==row["classification"] for x in matched),"interpretation":"Exact independent fact key only; nonmatch may be paraphrase, change, or removal, not confirmed correction."})
        allrows+=rows
    for section in [*FIELDS,"ALL"]:
        results.append({"ticker":"ALL","section":section,**annotation_metrics(allrows if section=="ALL" else [r for r in allrows if r["section"]==section])})
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/"annotation_scorecard.csv",results)
    csv_write(output/"annotated_claim_inventory.csv",allrows)
    csv_write(output/"lexical_retention_and_error_recurrence.csv",facts)
    csv_write(output/"unflagged_errors.csv",[r for r in allrows if not r["warning_applies"] and r["classification"]!="Supported"])
    write_json(output/"manifest.json",{"pairs":pairs,"human_review_performed":False,"measurement_only_correction":True,"method_source":"scripts/analyze_verifier_annotations.py summarize/key; whole-paragraph warning rule unchanged","limitations":"Reclassified labels alter measured recall/precision without any change to report warnings. Lexical identity and evidence span retention are proxies."})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix",default="corrected",help="New output label; existing outputs are never overwritten")
    p.add_argument("--validate-only",action="store_true")
    args=p.parse_args()
    require(re.fullmatch(r"[a-z][a-z0-9_]{0,40}",args.suffix),"Invalid output suffix")
    os.chdir(ROOT)
    for name,digest in PINNED.items(): require(sha256((REVIEW/name).read_bytes())==digest,"Proposal differs from accepted version: "+name)
    require(sha256(PAIR_FILE.read_bytes())==PAIR_HASH,"Parent pairs changed")
    pairs=read_json(PAIR_FILE)
    changes=build_changes(read_json(REVIEW/"comparison_proposals.json"),read_json(REVIEW/"cagr_proposals.json"),pairs)
    new_pairs=[]; plans=[]
    for pair in pairs:
        ticker=Path(pair["final_run"]).name.split("_")[0]
        new=dict(pair)
        for stage in ("before","after"):
            dest=f"audits/grounding/{ticker}_{stage}_reasoning_v8_2_{args.suffix}"
            plans.append(apply_audit(pair[stage],dest,changes))
            new[stage]=dest
        new_pairs.append(new)
    consumed=set().union(*(x["consumed"] for x in plans))
    require(consumed==set(changes),"Unapplied accepted atom decisions")
    require(sum(len(x["rows"])-len(x["old_rows"]) for x in plans)==1,"Unexpected atom-count change")
    pair_output=Path(f"config/grounding_reasoning_v8_2_{args.suffix}_pairs.json")
    comparison_output=Path(f"audits/grounding/comparison_reasoning_v8_2_{args.suffix}")
    annotation_output=Path(f"audits/grounding/annotation_reasoning_v8_2_{args.suffix}")
    before_summary=Path(f"audits/grounding/summary_before_reasoning_v8_2_{args.suffix}")
    after_summary=Path(f"audits/grounding/summary_after_reasoning_v8_2_{args.suffix}")
    result_dir=REVIEW/args.suffix
    destinations=[Path(x["destination"]) for x in plans]+[pair_output,comparison_output,annotation_output,before_summary,after_summary,result_dir]
    if not args.validate_only:
        require(all(not x.exists() for x in destinations),"Refusing to overwrite a destination; select an unused --suffix")
    validated={"accepted_original_atoms":len(changes),"mapped_new_atoms":sum(len(x["mappings"]) for x in plans),"unchanged_source_texts":48,"new_paid_api_calls":0,"validation":"source/unit coverage, original labels, quotes, evidence, proposal hashes, unchanged unreviewed rows verified","totals":[{"ticker":x["info"]["ticker"],"run_id":x["info"]["run_id"],**metrics(x["rows"])} for x in plans]}
    if args.validate_only:
        print(json.dumps(validated,indent=2));return
    # Snapshot every input audit file and every original section/prompt before writing.
    protected={}
    for plan in plans:
        for path in Path(plan["parent"]).rglob("*"):
            if path.is_file(): protected[str(path)]=sha256(path.read_bytes())
        for section,item in plan["inputs"].items():
            for path in (Path("runs")/plan["info"]["run_id"]/"output"/plan["info"]["ticker"]/"analysis"/(section+".txt"),Path("runs")/plan["rows"][0]["evidence_run_id"]/"agents"/section/"prompt.txt"):
                protected[str(path)]=sha256(path.read_bytes())
    result_dir.mkdir(parents=True,exist_ok=False)
    decision=root_decision(args.suffix)
    write_json(result_dir/"root_machine_review_decision.json",decision)
    decision_hash=sha256((result_dir/"root_machine_review_decision.json").read_bytes())
    for plan in plans: write_audit(plan,decision_hash)
    write_json(pair_output,new_pairs)
    csv_write(result_dir/"old_new_atom_mapping.csv",[r for x in plans for r in x["mappings"]])
    write_json(result_dir/"protected_input_hashes.json",protected)
    compare(SimpleNamespace(pairs=pair_output,output=comparison_output,allow_partial=False))
    annotations=write_annotations(new_pairs,annotation_output)
    summarize_audits(SimpleNamespace(audits=[Path(x["before"]) for x in new_pairs],output=before_summary))
    summarize_audits(SimpleNamespace(audits=[Path(x["after"]) for x in new_pairs],output=after_summary))
    comparison_manifest=read_json(comparison_output/"manifest.json")
    comparison_manifest.update(measurement_only_correction=True,root_decision_sha256=decision_hash,original_comparison="audits/grounding/comparison_reasoning_v8_2",report_text_changed=False,new_paid_api_calls=0)
    write_json(comparison_output/"manifest.json",comparison_manifest)
    require(all(sha256(Path(path).read_bytes())==digest for path,digest in protected.items()),"Protected original changed during materialization")
    scorecard=[x for x in read_csv(comparison_output/"before_after_scorecard.csv") if x["section"]=="ALL"]
    result={**validated,"protected_files_verified_unchanged":len(protected),"root_decision_sha256":decision_hash,"pair_config":str(pair_output),"measurement_only_correction":True,"human_review_performed":False,"scorecard":scorecard,"warning_scorecard":[x for x in annotations if x["section"]=="ALL"],"reproduction_command":f"python scripts/materialize_post_release_errata.py --suffix {args.suffix}_reproduction","script_sha256":sha256(Path(__file__).read_bytes()),"created_at_utc":utc_now()}
    write_json(result_dir/"results.json",result)
    report=["# Accepted post-release measurement errata", "", "These are new machine-adjudication versions of unchanged V8.2 reports. They correct measurement, not the verifier. No human annotation or new paid model call occurred.", "", "The31 accepted original atoms map to32 atoms because AMD's supported PE decline is retained separately from its unsupported elevated-level judgment. All other atoms, source texts and original audits remain unchanged.", "", "|Company|Before S/U/C/total|After S/U/C/total|", "|---|---|---|"]
    for row in scorecard:
        report.append("|"+row["ticker"]+"|"+"/".join(row["before_"+k] for k in ("supported","unsupported","contradicted","total_claims"))+"|"+"/".join(row["after_"+k] for k in ("supported","unsupported","contradicted","total_claims"))+"|")
    total=next(x for x in scorecard if x["ticker"]=="ALL")
    warning=next(x for x in annotations if x["ticker"]=="ALL" and x["section"]=="ALL")
    report += ["",f"Corrected warning recall: {warning['warning_recall']:.8%}; precision: {warning['warning_precision']:.8%}; unflagged unsupported/contradicted: {warning['unflagged_unsupported_or_contradicted']}, including {warning['unflagged_contradicted']} contradictions.","",f"Supported lexical-key retention: {float(total['independent_supported_fact_retention']):.8%}; evidence-span retention: {float(total['supported_evidence_span_retention']):.8%}. Changed wording/atomization affects these proxies; they do not establish semantic factual recall.","","Original source errors remain: seven explicit CAGR-scope contradictions and the AMD numerical-range error after treatment. Unsupported labels do not imply truth or successful repair.","","Reproduce from the submission root in the recorded dedicated environment:","","```powershell",f"python scripts/materialize_post_release_errata.py --validate-only",f"python scripts/materialize_post_release_errata.py --suffix {args.suffix}_reproduction","```","","The script pins proposal/config hashes, verifies every original source unit and evidence quote, refuses existing output destinations, and compares protected input hashes after writing. `old_new_atom_mapping.csv` preserves every accepted change and its reason. `results.json` contains exact counts and metrics; the new paired config points to versioned audit directories.",""]
    (result_dir/"RESULTS.md").write_text("\n".join(report),encoding="utf-8",newline="\n")
    print(json.dumps({"results":str(result_dir/"results.json"),"scorecard":scorecard,"warning":warning},indent=2))


if __name__ == "__main__":
    main()
