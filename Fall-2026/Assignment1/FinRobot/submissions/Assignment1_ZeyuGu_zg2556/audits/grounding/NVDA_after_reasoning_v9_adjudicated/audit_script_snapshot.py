"""Offline independent delta adjudication with exact all-section audit inheritance.

No verifier module, checked.json, proposed verdict, or saved verifier calculation
is read. New CAGR propositions are evaluated solely from final report text and
the original writer prompt. Unchanged units retain prior machine adjudications;
this is not a new human review of those units. See generated method.md.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import time
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
from types import SimpleNamespace

from audit_io import sha256, utc_now, write_json
from grounding_audit import FIELDS, csv_write, load_audit, load_inputs, read_json, source_units
from materialize_adjudication import materialize

VERSION = "independent-cagr-delta-adjudication-v1"
METHOD = """# Independent V9 delta adjudication

This is machine-assisted review, not human validation. All eight final sections
are covered. Prior corrected judgments transfer only for byte-identical Unicode
source-unit text, in the same section and with the identical original evidence
prompt SHA-256. Sequence alignment preserves occurrence/order and permits no
normalization, warning removal, or paraphrase matching. Inherited labels remain
subject to the limitations and errors of the prior full machine adjudication.

Every changed unit is explicitly decomposed into (1) the supplied CAGR cell,
(2) actual-endpoint CAGR, (3) full-table endpoint CAGR, or (4) forecast status.
Each receives its own classification, numeric status, original prompt citation
and reason. Two independent clauses are not accepted as a single grounded fact.
The independent parser reads the original financial table; the independent
calculator evaluates 100*(exp((ln(end)-ln(start))/elapsed_years)-1) using Decimal
precision 60 and rounds to the precision stated in the report, HALF_UP. It does
not import the production power-based calculator or use its saved outputs.
Positive finite revenue endpoints and a positive year difference are required.
A/E roles must agree with prompt columns; an E endpoint is not realized history.
The report's literal display-cell statement is separate from computed CAGR.
No statistical period is inferred merely from a table-scope description.

Unknown changed prose aborts this narrow offline audit and requires additional
explicit review. A recognized conflicting scalar is Contradicted; absent data
is Unsupported. None of the verifier's own verdicts, citations, or calculated
results is supplied to the evaluator. An unchanged Unsupported annotation stays
Unsupported. The evidence scope is the original writer prompt, not external
truth or later data. The implementation author saw the production rule during
review; this is independent code and evidence checking, not a blind held-out
evaluation. The twelve development examples informed both implementation and
review. Repetition adds claim instances and must not be confused with new facts.

New calculations and forecast clarifications use explicit semantic fact keys;
unchanged lexical keys are inherited. Retention is still a proxy and cannot
measure exact semantic recall across paraphrases. No direct paid API is called;
task-agent authoring costs/tokens remain uninstrumented, not assumed zero.
"""


def prompt_revenue(prompt):
    """Find a unique titled financial Revenue row by labeled dictionary columns."""
    lines = prompt.splitlines()
    found = []
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        columns = tuple(cell.strip() for cell in line.split("|")[1:-1])
        if "metrics" not in columns or "CAGR" not in columns:
            continue
        if len(columns) != len(set(columns)):
            raise ValueError("Ambiguous duplicated financial column names")
        title = next((lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), "")
        if "Financial Metrics and Forecasts" not in title:
            raise ValueError("Financial table lacks original forecast context")
        for j in range(i + 1, len(lines)):
            if not lines[j].startswith("|"):
                break
            cells = [cell.strip() for cell in lines[j].split("|")[1:-1]]
            if len(cells) != len(columns):
                raise ValueError("Unequal financial row/header widths")
            row = dict(zip(columns, cells))
            if row.get("metrics") != "Revenue":
                continue
            years = [key for key in columns if re.fullmatch(r"\d{4}[AE]", key)]
            if len(years) < 2 or [int(y[:4]) for y in years] != sorted(set(int(y[:4]) for y in years)):
                raise ValueError("Ambiguous/nonchronological financial years")
            values = {year: Decimal(row[year].replace(",", "")) for year in years}
            if not all(v.is_finite() for v in values.values()):
                raise ValueError("Nonfinite prompt Revenue value")
            actual = [y for y in years if y.endswith("A")]
            future = [y for y in years if y.endswith("E")]
            if not actual or not future or int(actual[-1][:4]) >= int(future[0][:4]):
                raise ValueError("Ambiguous actual/forecast ordering")
            found.append({"years": years, "values": values, "actual": actual,
                          "displayed": row["CAGR"], "header_line": i + 1,
                          "row_line": j + 1, "title_line": i, "title": title})
    if len(found) != 1:
        raise ValueError("Expected exactly one original Revenue financial table")
    return found[0]


def independently_compute(values, start, end):
    if start not in values or end not in values:
        return None
    n = int(end[:4]) - int(start[:4])
    a, b = values[start], values[end]
    if n <= 0 or a <= 0 or b <= 0 or not (a.is_finite() and b.is_finite()):
        return None
    with localcontext() as ctx:
        ctx.prec = 60
        result = (((b.ln() - a.ln()) / Decimal(n)).exp() - Decimal(1)) * Decimal(100)
    return {"start": start, "end": end, "start_value": str(a), "end_value": str(b),
            "elapsed_years": n, "percent": str(result), "precision": 60,
            "formula": "100 * (exp((ln(end_value) - ln(start_value)) / elapsed_years) - 1)",
            "start_role": "actual" if start.endswith("A") else "forecast",
            "end_role": "actual" if end.endswith("A") else "forecast"}


def scalar_matches(stated, actual):
    stated = Decimal(stated)
    quantum = Decimal(1).scaleb(stated.as_tuple().exponent)
    return Decimal(actual).quantize(quantum, rounding=ROUND_HALF_UP) == stated


def atom(claim, classification, numeric_status, references, reason, key):
    return {"claim": claim, "claim_type": "quantitative", "classification": classification,
            "numeric_status": numeric_status, "evidence": references, "reason": reason, "fact_key": key}


def assess_changed(unit, evidence, ticker):
    """Explicitly recognize the two reviewed sentence forms; no generic inference."""
    table = prompt_revenue(evidence)
    refs = [{"line_start": table[k], "line_end": table[k]} for k in ("header_line", "row_line")]
    a = re.fullmatch(r"The supplied Revenue CAGR cell is (?P<cell>[-+]?\d+(?:\.\d+)?)%; the (?P<start>\d{4}[AE]) to (?P<end>\d{4}[AE]) actual endpoints imply (?P<rate>[-+]?\d+(?:\.\d+)?)% CAGR\.\s*", unit["text"])
    b = re.fullmatch(r"Across the table's (?P<start>\d{4}[AE]) to (?P<end>\d{4}[AE]) endpoints, the implied Revenue CAGR is (?P<rate>[-+]?\d+(?:\.\d+)?)%; (?P<forecast>\d{4}[AE]) is a model forecast\.\s*", unit["text"])
    if not (a or b):
        raise ValueError(f"Unreviewed changed source unit {unit['unit_id']}: {unit['text']!r}")
    m = a or b
    start, end, rate = m["start"], m["end"], m["rate"]
    calculation = independently_compute(table["values"], start, end)
    arithmetic = calculation is not None and scalar_matches(rate, calculation["percent"])
    scope = (start.endswith("A") and end.endswith("A")) if a else (start == table["years"][0] and end == table["years"][-1])
    if calculation is None:
        classification, numeric = "Unsupported", "unverifiable"
        why = "The original prompt does not provide positive finite endpoints and positive elapsed years for this claim."
    else:
        classification, numeric = ("Supported", "correct") if arithmetic and scope else ("Contradicted", "incorrect")
        why = (f"Independent Decimal ln/exp calculation: Revenue({end})={calculation['end_value']}, "
               f"Revenue({start})={calculation['start_value']}; elapsed fiscal-year intervals={calculation['elapsed_years']}; "
               f"100*(exp((ln(end)-ln(start))/n)-1)={calculation['percent']}%. "
               f"At the sentence's displayed precision the arithmetic agrees={arithmetic}; asserted endpoint role/scope agrees={scope}. "
               "This is prompt-conditional arithmetic, not external verification of the financial inputs.")
    computed = atom(f"{ticker} Revenue CAGR from {start} to {end} is {rate}%, " +
                    ("using actual endpoints." if a else "using the full table's endpoints, including the forecast endpoint."),
                    classification, numeric, refs if calculation else [], why,
                    f"{ticker.lower()}|revenue_cagr|{start.lower()}:{end.lower()}|{rate}|percent")
    atoms = []
    if a:
        correct = table["displayed"] == m["cell"] + "%"
        atoms.append(atom(f"{ticker}'s supplied Revenue row displays a CAGR cell of {m['cell']}%.",
                          "Supported" if correct else "Contradicted", "correct" if correct else "incorrect", refs,
                          f"Literal original Revenue CAGR cell is {table['displayed']}. This statement concerns the supplied cell, not an inferred statistical period.",
                          f"{ticker.lower()}|revenue_cagr_display_cell|generic|{m['cell']}|percent"))
        atoms.append(computed)
    else:
        atoms.append(computed)
        label = m["forecast"]
        known = label in table["values"]
        forecast = known and label.endswith("E")
        atoms.append(atom(f"{ticker}'s {label} Revenue endpoint is a model forecast.",
                          "Supported" if forecast else "Contradicted" if known else "Unsupported",
                          "correct" if forecast else "incorrect" if known else "unverifiable",
                          refs if known else [],
                          f"The original table is explicitly headed Financial Metrics and Forecasts; {label} " +
                          ("has the E estimate label, not an A actual label. This supports model/table forecast status only, not management guidance or realization." if forecast else "does not establish the stated forecast role."),
                          f"{ticker.lower()}|revenue_period_role|{label.lower()}|forecast|categorical"))
    return atoms, {"source_unit_id": unit["unit_id"], "source_text": unit["text"], "calculation": calculation,
                   "header_line": table["header_line"], "revenue_line": table["row_line"], "atoms": atoms}


def build(run, prior, output, evidence_run=None):
    started = time.perf_counter()
    prior_info, prior_rows = load_audit(prior)
    evidence_run = evidence_run or Path(prior_info["evidence_run"])
    old_manifest, old = load_inputs(Path(prior_info["source_run"]), evidence_run)
    manifest, new = load_inputs(run, evidence_run)
    spec_path = prior / "reviewed_specification.json"
    if sha256(spec_path.read_bytes()) != prior_info["reviewed_specification_sha256"]:
        raise ValueError("Prior reviewed specification hash mismatch")
    old_spec = read_json(spec_path)
    spec = {"protocol_version": VERSION, "run_id": manifest["run_id"],
            "evidence_run_id": next(iter(new.values()))["metadata"]["evidence_run_id"],
            "raw_judge_audit": None, "sections": {}}
    mapping, calculations, removed, counts = [], [], [], []
    for section in FIELDS:
        old_item, item = old[section], new[section]
        if old_item["metadata"]["evidence_prompt_sha256"] != item["metadata"]["evidence_prompt_sha256"]:
            raise ValueError("Evidence changed; exact-unit inheritance is invalid")
        if prior_info["source_hashes"][section] != old_item["metadata"]["source_sha256"]:
            raise ValueError("Prior audited source hash mismatch")
        old_units, units = source_units(old_item["text"]), source_units(item["text"])
        review = old_spec["sections"][section]
        if review["source_sha256"] != old_item["metadata"]["source_sha256"] or review["evidence_sha256"] != item["metadata"]["evidence_prompt_sha256"]:
            raise ValueError("Prior spec evidence/source binding mismatch")
        old_reviews = {r["unit_id"]: r for r in review["units"]}
        if set(old_reviews) != set(range(len(old_units))):
            raise ValueError("Prior source coverage incomplete")
        matcher = difflib.SequenceMatcher(a=[u["text"] for u in old_units], b=[u["text"] for u in units], autojunk=False)
        reviewed, inherited, changed = [], 0, 0
        for operation, i, j, k, l in matcher.get_opcodes():
            if operation == "equal":
                for oi, ni in zip(range(i, j), range(k, l)):
                    if old_units[oi]["text"] != units[ni]["text"]:
                        raise AssertionError("Exact alignment failed")
                    row = copy.deepcopy(old_reviews[oi]); row["unit_id"] = ni
                    reviewed.append(row); inherited += 1
                    mapping.append({"ticker": manifest["ticker"], "section": section,
                                    "source_unit_id": ni, "prior_unit_id": oi, "basis": "exact_text_same_section_same_evidence",
                                    "source_unit_sha256": sha256(units[ni]["text"]), "inherited_claim_count": len(row["atoms"]),
                                    "prior_audit": str(prior), "evidence_sha256": item["metadata"]["evidence_prompt_sha256"]})
            else:
                if operation != "replace" or j - i != 1 or l - k != 2:
                    raise ValueError(f"Unexpected edit shape requiring explicit review: {section}/{operation}/{i}:{j}/{k}:{l}")
                removed.append({"section": section, "prior_unit": old_units[i], "prior_atoms": old_reviews[i]["atoms"]})
                for ni in range(k, l):
                    atoms, record = assess_changed(units[ni], item["evidence"], manifest["ticker"])
                    record.update(section=section, evidence_prompt_sha256=item["metadata"]["evidence_prompt_sha256"])
                    calculations.append(record); changed += 1
                    reviewed.append({"unit_id": ni, "atoms": atoms, "nonfactual_reason": ""})
                    mapping.append({"ticker": manifest["ticker"], "section": section, "source_unit_id": ni,
                                    "prior_unit_id": i, "basis": "independent_changed_unit_review",
                                    "source_unit_sha256": sha256(units[ni]["text"]), "inherited_claim_count": 0,
                                    "prior_audit": str(prior), "evidence_sha256": item["metadata"]["evidence_prompt_sha256"]})
        spec["sections"][section] = {"source_sha256": item["metadata"]["source_sha256"],
                                    "evidence_sha256": item["metadata"]["evidence_prompt_sha256"], "units": reviewed}
        counts.append({"section": section, "source_units": len(units), "exact_inherited_units": inherited,
                       "independently_reviewed_changed_units": changed, "claims": sum(len(u["atoms"]) for u in reviewed)})
    if output.exists():
        raise FileExistsError("Preserve prior audit; choose a new output directory")
    spec_file = output.parent / (output.name + "_review.json")
    if spec_file.exists():
        raise FileExistsError("Preserve prior review specification")
    write_json(spec_file, spec)
    materialize(SimpleNamespace(run=run, evidence_run=evidence_run, spec=spec_file, output=output))
    csv_write(output / "unit_provenance.csv", mapping)
    csv_write(output / "delta_coverage.csv", counts)
    write_json(output / "independent_changed_unit_review.json", calculations)
    write_json(output / "superseded_source_units.json", removed)
    (output / "method.md").write_text(METHOD, encoding="utf-8")
    info = read_json(output / "manifest.json")
    info.update(delta_audit_version=VERSION, inherited_audit=str(prior),
                inherited_claims_sha256=prior_info["claim_csv_sha256"],
                inherited_specification_sha256=prior_info["reviewed_specification_sha256"],
                exact_inherited_unit_count=sum(r["exact_inherited_units"] for r in counts),
                independently_reviewed_changed_unit_count=len(calculations),
                independently_reviewed_changed_claim_count=sum(len(c["atoms"]) for c in calculations),
                removed_old_source_unit_count=len(removed), removed_old_claim_count=sum(len(c["prior_atoms"]) for c in removed),
                independent_method_sha256=sha256(METHOD), audit_script_sha256=sha256(Path(__file__).read_bytes()),
                offline_execution_runtime_seconds=time.perf_counter()-started,
                new_direct_paid_api_calls=0, human_review_performed=False,
                review_scope="All eight sections: exact unchanged units inherit corrected prior full audit; every changed unit independently decomposed and recalculated from original writer prompt.")
    info["limitations"] += " Inherited judgments are not newly re-adjudicated. Development-cohort-informed delta calculation checks, not blind held-out validation. See method.md and unit_provenance.csv."
    for name in ("unit_provenance.csv", "delta_coverage.csv", "independent_changed_unit_review.json", "superseded_source_units.json"):
        info[name.replace(".", "_") + "_sha256"] = sha256((output / name).read_bytes())
    write_json(output / "manifest.json", info)
    return {"ticker": manifest["ticker"], "units": sum(c["source_units"] for c in counts), "counts": counts,
            "changed_units": len(calculations), "changed_claims": sum(len(c["atoms"]) for c in calculations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--evidence-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.prior_audit, args.output, args.evidence_run), indent=2))


if __name__ == "__main__":
    main()
