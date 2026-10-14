"""Narrow deterministic clarification of a displayed historical Revenue CAGR.

This fragment consumes only the original writer prompt and released sentence.
It never consumes evaluation labels, company-specific rates, or external data.
The vocabulary gate deliberately declines compound or comparative prose.
"""
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext


def _cagr_prompt_table(evidence):
    lines = evidence.splitlines()
    identity = re.fullmatch(r"Financial data for (.+) \(([A-Z0-9.-]+)\):", lines[0]) if lines else None
    if not identity:
        return None
    for position, line in enumerate(lines):
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if "metrics" not in cells or "CAGR" not in cells:
            continue
        years = [(i, c) for i, c in enumerate(cells) if re.fullmatch(r"(?:19|20)\d{2}[AE]", c)]
        if len(years) < 2 or len({c for _, c in years}) != len(years):
            return None
        if [int(c[:4]) for _, c in years] != sorted({int(c[:4]) for _, c in years}):
            return None
        for row_number in range(position + 1, len(lines)):
            if not lines[row_number].lstrip().startswith("|"):
                break
            row = [x.strip() for x in lines[row_number].strip().strip("|").split("|")]
            if len(row) != len(cells) or row[cells.index("metrics")] != "Revenue":
                continue
            try:
                values = {c: Decimal(row[i].replace(",", "")) for i, c in years}
                displayed = row[cells.index("CAGR")]
                if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", displayed):
                    return None
                if not all(x.is_finite() for x in values.values()):
                    return None
            except InvalidOperation:
                return None
            actual = [c for _, c in years if c.endswith("A")]
            forecast = [c for _, c in years if c.endswith("E")]
            if len(actual) < 2 or not forecast or max(int(c[:4]) for c in actual) >= min(int(c[:4]) for c in forecast):
                return None
            return {"company": identity[1], "ticker": identity[2], "values": values,
                "actual": actual, "forecast": forecast, "years": [c for _, c in years],
                "displayed": displayed, "header_line": position + 1, "row_line": row_number + 1,
                "header_quote": lines[position], "row_quote": lines[row_number]}
    return None


def _endpoint_cagr(values, start, end):
    intervals = int(end[:4]) - int(start[:4])
    if intervals <= 0 or values[start] <= 0 or values[end] <= 0:
        return None
    with localcontext() as context:
        context.prec = 36
        rate = ((values[end] / values[start]) ** (Decimal(1) / Decimal(intervals)) - 1) * 100
    return {"start": start, "end": end, "start_value": str(values[start]),
        "end_value": str(values[end]), "year_intervals": intervals,
        "percent": str(rate), "display_percent": str(rate.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)),
        "formula": "100 * ((end_value / start_value) ** (1 / year_intervals) - 1)",
        "contains_forecast_endpoint": end.endswith("E")}


def clarify_revenue_cagr(sentence, evidence):
    """Clarify a short, potentially ambiguous CAGR/table-horizon sentence.

    A table's extent is not automatically the claimed statistical interval. This
    rule therefore supplies both identifiable calculations, without labeling an
    ambiguous original sentence Contradicted. It acts only if the displayed cell
    rounds to the actual-endpoint calculation and the sentence mentions a future
    column. Missing/negative endpoints, other entities and extra premises abstain.
    """
    table = _cagr_prompt_table(evidence)
    if not table:
        return None
    text = re.sub(r"^\[[^\]]+\]\s*", "", sentence).strip()
    if not re.search(r"\brevenue\b", text, re.I) or not re.search(r"\bCAGR\b", text, re.I):
        return None
    percentages = re.findall(r"[-+]?\d+(?:\.\d+)?%", text)
    years = re.findall(r"(?<!\w)(?:19|20)\d{2}[AE](?!\w)", text)
    if len(percentages) != 1 or not years or any(y not in table["values"] for y in years):
        return None
    if not any(y.endswith("E") for y in years):
        return None
    subject_aliases = [table["company"], table["ticker"], table["company"].split()[0]]
    scrubbed = text
    for alias in sorted(set(subject_aliases), key=len, reverse=True):
        scrubbed = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?:['’]s)?(?!\w)", "", scrubbed, flags=re.I)
    scrubbed = re.sub(r"(?:19|20)\d{2}[AE]|[-+]?\d+(?:\.\d+)?%", "", scrubbed)
    allowed = set("revenue cagr is has a of in the table row spanning from to through for period shown across at projected shows lists compound annual growth rate between and".split())
    # Additional predicates, other companies, causes or comparisons are not erased.
    if re.search(r"\d", scrubbed) or set(re.findall(r"[A-Za-z]+", scrubbed.lower())) - allowed:
        return None
    actual = _endpoint_cagr(table["values"], table["actual"][0], table["actual"][-1])
    complete = _endpoint_cagr(table["values"], table["years"][0], table["years"][-1])
    if actual is None or complete is None:
        return None
    displayed = Decimal(table["displayed"][:-1])
    claim_value = Decimal(percentages[0][:-1])
    precision = Decimal(10) ** displayed.as_tuple().exponent
    if claim_value != displayed or Decimal(actual["percent"]).quantize(precision, rounding=ROUND_HALF_UP) != displayed:
        return None
    # Retain every explicit original endpoint, even if a future author changes
    # the sentence to an interior interval: do not silently replace its horizon.
    if any(y not in {complete["start"], complete["end"]} for y in years):
        return None
    revised = (
        f"The supplied Revenue CAGR cell is {table['displayed']}; the "
        f"{actual['start']} to {actual['end']} actual endpoints imply "
        f"{actual['display_percent']}% CAGR. "
        f"Across the table's {complete['start']} to {complete['end']} endpoints, "
        f"the implied Revenue CAGR is {complete['display_percent']}%; "
        f"{complete['end']} is a model forecast."
    )
    return {"rule": "displayed_revenue_cagr_and_table_extent_clarification",
        "original_released_text": sentence, "revised_released_text": revised,
        "original_semantic_verdict": "Not inferred by this rule; table extent may be ambiguous.",
        "company": table["company"], "ticker": table["ticker"],
        "displayed_cell": table["displayed"], "historical_calculation": actual,
        "table_endpoint_calculation": complete,
        "evidence": [
            {"line_number": table["header_line"], "quote": table["header_quote"]},
            {"line_number": table["row_line"], "quote": table["row_quote"]}],
        "limitations": "A narrow deterministic clarification, not general semantic validation. The development cohort informed this rule. No held-out claim."
    }


def apply_cagr_horizon_guard(result, evidence):
    replacements = {}
    for row in result["claims"]:
        if not row["released_text"]:
            continue
        checked = clarify_revenue_cagr(row["released_text"], evidence)
        if checked is None:
            continue
        prior = row["released_text"]
        row["pre_semantic_guard_release"] = {
            key: row[key] for key in ("released_text", "retained_as_verified", "verification_status", "release_text_source")
        }
        row["semantic_horizon_check"] = checked
        row["released_text"] = checked["revised_released_text"]
        row["retained_as_verified"] = True
        row["verification_status"] = "deterministic_horizon_clarified"
        row["release_text_source"] = "original_prompt_endpoints_deterministic_clarification"
        replacements[prior] = row["released_text"]
    if replacements:
        final = []
        for unit in result["revised_text"].split("\n\n"):
            changed = replacements.get(unit, unit)
            if changed not in final:
                final.append(changed)
        result["revised_text"] = "\n\n".join(final)
    result["verifier_version"] = "9.0-explicit-cagr-endpoint-clarification"
    result["semantic_horizon_check_count"] = sum("semantic_horizon_check" in row for row in result["claims"])
    result["verified_claim_count"] = sum(row["retained_as_verified"] for row in result["claims"])
    result["unverified_annotation_count"] = sum(row["verification_status"] in {"unverified_annotation", "unverified_deterministic_failure"} for row in result["claims"])
    result["supported_valid_retention_count"] = sum(row["classification"] == "Supported" and row["retained_as_verified"] for row in result["claims"])
    result["limitations"] += " V9 additionally clarifies a narrowly recognized displayed Revenue CAGR using explicit original-prompt endpoints; historical and forecast intervals are separated. Original machine classifications and failed binding diagnostics remain unchanged for traceability."
    return result
