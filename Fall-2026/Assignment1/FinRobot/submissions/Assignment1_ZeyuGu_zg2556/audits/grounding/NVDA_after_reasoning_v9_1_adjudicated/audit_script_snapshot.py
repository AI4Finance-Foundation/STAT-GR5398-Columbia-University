"""Independent exact-text V9 inheritance and explicit EV/EBITDA range audit.

Uses only original writer prompts and the final source sentences. No production
guard, verifier checked.json, computed guard records or proposed labels is read.
"""
from __future__ import annotations
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from audit_reasoning_v9 import atom, build, scalar_matches

VERSION = "independent-ev-ebitda-range-delta-adjudication-v1"
METHOD = """# Independent V9.1 EV/EBITDA delta adjudication

Machine-assisted, not human validation. Every final source unit in all eight
sections is covered. V9 judgments transfer only for exactly identical unit text
within the same section and original-evidence SHA-256, preserving occurrence and
order. No warning stripping, whitespace normalization, or paraphrase matching.
Inherited labels, including V9 CAGR arithmetic, are not independently rejudged
again; their complete prior evidence and machine limitations remain traceable.

The only new prose has two sentences. This evaluator reads the final text and
original writer prompt, never the production guard module/results/labels. The
original subject-specific financial-header A columns define the selected actual
years. The separately labeled Peer EV/EBITDA Comparison table identifies subject
values by column name. We require every selected actual year to have a finite
value, inspect all such rows, and use Python Decimal comparisons for min/max.
We independently check the explicit first/last years, extremal values, extremal
years, and alignment of the stated scope with the subject's last A column. The
two extremal propositions and two scope propositions are separately labeled.

This is a statement about supplied table rows, not all-time trading extrema,
intra-year highs/lows, external market truth, price-adjustment conventions, or
the fundamental validity of the pipeline multiples. Missing or ambiguous rows
do not establish an extreme. Foreign-company observations and future rows cannot
fill gaps. Ties may support any tied year; chronology comes from numeric years,
not row order. Negative finite multiples remain comparable; nonfinite values
inside the claimed scope cause failure and require manual machine reassessment.
Unknown changed wording aborts rather than borrowing a guard verdict. A warning
on unchanged prose never makes its underlying proposition supported.

There was no direct paid API call. Task-agent authoring cost is uninstrumented,
not zero. This is a cohort-informed correction and review of one development
example, not a blinded or held-out validation. Added explanatory scope claims
and repeated claims change raw denominators, not necessarily unique information.
Fact keys remain retention proxies, not measured semantic recall. All original
V9 source/audits and V9.1 independently reviewed records are retained.
"""


def original_range_data(prompt, ticker):
    lines = prompt.splitlines()
    identity = f"({ticker}):"
    if not lines or not lines[0].endswith(identity):
        raise ValueError("Prompt identity mismatch")
    financial = []
    for i, line in enumerate(lines):
        columns = [c.strip() for c in line.split("|")[1:-1]]
        if "metrics" in columns and "CAGR" in columns:
            actual = [int(c[:-1]) for c in columns if re.fullmatch(r"\d{4}A", c)]
            if len(actual) < 2 or actual != sorted(set(actual)):
                raise ValueError("Ambiguous actual fiscal-year labels")
            financial.append((i + 1, actual))
    if len(financial) != 1:
        raise ValueError("Expected one subject financial header")
    header_line, actual = financial[0]
    headings = [i for i, line in enumerate(lines) if line.strip() == "## Peer EV/EBITDA Comparison"]
    if len(headings) != 1:
        raise ValueError("Expected unique EV/EBITDA table")
    title = headings[0]
    columns = [c.strip() for c in lines[title + 1].split("|")[1:-1]]
    if columns.count(ticker) != 1 or columns.count("year") != 1:
        raise ValueError("Ambiguous subject/year columns")
    yi, vi = columns.index("year"), columns.index(ticker)
    observed = {}
    for i in range(title + 3, len(lines)):
        if not lines[i].startswith("|"):
            break
        row = [c.strip() for c in lines[i].split("|")[1:-1]]
        if len(row) != len(columns) or not re.fullmatch(r"\d{4}", row[yi]):
            raise ValueError("Malformed original multiple table")
        year = int(row[yi])
        if year in observed:
            raise ValueError("Duplicate multiple year")
        try:
            value = Decimal(row[vi])
        except InvalidOperation:
            raise ValueError("Malformed multiple")
        observed[year] = {"year": year, "value": value, "line": i + 1}
    if any(y not in observed or not observed[y]["value"].is_finite() for y in actual):
        raise ValueError("A selected actual year has missing/nonfinite multiple")
    selected = [observed[y] for y in actual]
    low = min(r["value"] for r in selected)
    high = max(r["value"] for r in selected)
    refs = [{"line_start": header_line, "line_end": header_line},
            {"line_start": title + 1, "line_end": title + 2}]
    refs.extend({"line_start": r["line"], "line_end": r["line"]} for r in selected)
    return {"actual_years": actual, "selected_rows": selected, "minimum": low,
            "maximum": high, "minimum_years": [r["year"] for r in selected if r["value"] == low],
            "maximum_years": [r["year"] for r in selected if r["value"] == high], "references": refs}


def assess_range(unit, evidence, ticker):
    table = original_range_data(evidence, ticker)
    refs = table["references"]
    first = re.fullmatch(r"Across the supplied EV/EBITDA rows aligned with (?P<ticker>[A-Z0-9.-]+)'s (?P<start>\d{4})A through (?P<end>\d{4})A actual columns, the minimum is (?P<minimum>[-+]?\d+(?:\.\d+)?)x \((?P<minimum_year>\d{4})A\) and the maximum is (?P<maximum>[-+]?\d+(?:\.\d+)?)x \((?P<maximum_year>\d{4})A\)\.\s*", unit["text"])
    second = re.fullmatch(r"This scope uses the supplied annual rows through (?P<ticker>[A-Z0-9.-]+)'s last actual column\.\s*", unit["text"])
    if not (first or second) or (first or second)["ticker"] != ticker:
        raise ValueError("Unreviewed changed EV/EBITDA prose or wrong subject")
    if first:
        start, end = int(first["start"]), int(first["end"])
        scope_ok = start == table["actual_years"][0] and end == table["actual_years"][-1]
        interval = [r for r in table["selected_rows"] if start <= r["year"] <= end]
        if not interval:
            raise ValueError("No available actual rows in asserted interval")
        scope_reason = (f"The subject financial header lists actual columns {table['actual_years']}; its last actual is {table['actual_years'][-1]}A. "
                        "The original Peer EV/EBITDA table provides the same-year subject observations. "
                        "Alignment is a table convention, not an external market-timing claim.")
        atoms = [atom(f"The supplied {ticker} EV/EBITDA range uses rows aligned with its {start}A through {end}A actual columns.",
                      "Supported" if scope_ok else "Contradicted", "correct" if scope_ok else "incorrect", refs,
                      scope_reason, f"{ticker.lower()}|ev_ebitda_scope|{start}a:{end}a|subject_actual_columns|annual")]
        observations = "; ".join(f"{r['year']}={r['value']}x" for r in interval)
        for kind, choose in (("minimum", min), ("maximum", max)):
            expected = choose(r["value"] for r in interval)
            years = [r["year"] for r in interval if r["value"] == expected]
            value, year = first[kind], int(first[kind + "_year"])
            correct = scope_ok and scalar_matches(value, expected) and year in years
            atoms.append(atom(f"Across the supplied {ticker} {start}A-{end}A EV/EBITDA rows, the {kind} is {value}x in {year}A.",
                              "Supported" if correct else "Contradicted", "correct" if correct else "incorrect", refs,
                              f"Independent Decimal comparisons over all selected original rows: {observations}. "
                              f"The {kind} is {expected}x, attained in {years}. The final prose's value/year and explicit actual-year scope agree={correct}. "
                              "No intra-year/all-time price range is asserted or inferred.",
                              f"{ticker.lower()}|ev_ebitda_{kind}|{start}a:{end}a|{value}@{year}a|multiple"))
    else:
        atoms = [atom(f"The stated {ticker} EV/EBITDA range scope uses supplied annual rows through its last actual column.",
                      "Supported", "not_numeric", refs,
                      f"The subject actual labels end at {table['actual_years'][-1]}A and every selected EV/EBITDA row is aligned with those actual labels. "
                      "The immediately preceding reviewed range explicitly ends at this same last actual column; future/other-company rows are excluded.",
                      f"{ticker.lower()}|ev_ebitda_range_scope|last_subject_actual|supplied_annual_rows|categorical")]
        atoms[0]["claim_type"] = "factual"
    record = {"source_unit_id": unit["unit_id"], "source_text": unit["text"],
              "original_actual_years": table["actual_years"],
              "independent_observations": [{**r, "value": str(r["value"])} for r in table["selected_rows"]],
              "independent_minimum": str(table["minimum"]), "independent_maximum": str(table["maximum"]),
              "minimum_years": table["minimum_years"], "maximum_years": table["maximum_years"], "atoms": atoms}
    return atoms, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prior-audit", type=Path, required=True)
    p.add_argument("--evidence-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.run, a.prior_audit, a.output, a.evidence_run,
                           reviewer=assess_range, method=METHOD, version=VERSION, audit_script=Path(__file__)), indent=2))


if __name__ == "__main__":
    main()
