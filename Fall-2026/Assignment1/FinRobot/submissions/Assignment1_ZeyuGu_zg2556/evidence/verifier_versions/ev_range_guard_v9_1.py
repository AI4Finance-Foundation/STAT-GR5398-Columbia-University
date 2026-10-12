"""Clarify an isolated EV/EBITDA range using original supplied actual rows.

No evaluator labels, external data, ticker list or company-specific constants
are inputs to the trigger. Unresolved identity, dates, values or extra predicates
cause abstention. Synthetic fixtures in tests are not experimental evidence.
"""
import copy
import hashlib
import re
from decimal import Decimal, InvalidOperation


def _ev_range_cells(line):
    if not line.strip().startswith("|") or not line.strip().endswith("|"):
        return None
    return [value.strip() for value in line.strip()[1:-1].split("|")]


def _ev_range_table(evidence):
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    lines = evidence.splitlines()
    identity = re.fullmatch(r"Financial data for (.+) \(([A-Z0-9.-]+)\):", lines[0])
    if not identity:
        return None
    company, ticker = identity.groups()
    financial_headers = []
    for index, line in enumerate(lines):
        cells = _ev_range_cells(line)
        if cells and "metrics" in cells:
            financial_headers.append((index, cells))
    if len(financial_headers) != 1:
        return None
    financial_index, financial_cells = financial_headers[0]
    if any(c not in {"", "metrics", "CAGR"} and not re.fullmatch(r"(?:19|20)\d{2}[AE]", c) for c in financial_cells):
        return None
    columns = [c for c in financial_cells if re.fullmatch(r"(?:19|20)\d{2}[AE]", c)]
    actual = [c for c in columns if c.endswith("A")]
    forecast = [c for c in columns if c.endswith("E")]
    years = [int(c[:4]) for c in columns]
    actual_years = [int(c[:4]) for c in actual]
    if (len(actual) < 2 or years != sorted(set(years)) or
            actual_years != list(range(actual_years[0], actual_years[-1] + 1)) or
            (forecast and actual_years[-1] >= min(int(c[:4]) for c in forecast))):
        return None
    headings = [i for i,line in enumerate(lines) if line.strip() == "## Peer EV/EBITDA Comparison"]
    if len(headings) != 1:
        return None
    heading = headings[0]
    header_index = heading + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if header_index >= len(lines):
        return None
    headers = _ev_range_cells(lines[header_index])
    if (not headers or headers.count("year") != 1 or headers.count(ticker) != 1 or
            len(headers) != len(set(headers))):
        return None
    year_column, value_column = headers.index("year"), headers.index(ticker)
    observations, row_years = [], []
    for index in range(header_index + 1, len(lines)):
        cells = _ev_range_cells(lines[index])
        if cells is None:
            break
        if len(cells) != len(headers):
            return None
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        match = re.fullmatch(r"((?:19|20)\d{2})([AE]?)", cells[year_column])
        if not match:
            # Do not turn calendar dates, quarter labels or fiscal-end dates
            # into a fiscal-year match by taking their first four digits.
            return None
        year, status = int(match[1]), match[2]
        row_years.append(year)
        if year not in actual_years:
            continue
        if status == "E":
            return None
        if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?:[eE][-+]?\d+)?", cells[value_column]):
            return None
        try:
            value = Decimal(cells[value_column].replace(",", ""))
        except (InvalidOperation, ValueError):
            return None
        if not value.is_finite() or value <= 0:
            return None
        observations.append({"year":year, "actual_label":f"{year}A", "value":str(value),
            "source_value":cells[value_column], "source_row_label":cells[year_column],
            "line_number":index + 1, "quote":lines[index]})
    if row_years != sorted(set(row_years)):
        return None
    if [row["year"] for row in observations] != actual_years:
        return None
    return {"company":company, "ticker":ticker, "actual_columns":actual,
        "first_actual":actual[0], "last_actual":actual[-1], "observations":observations,
        "identity_line":1, "financial_header_line":financial_index + 1,
        "metric_heading_line":heading + 1, "peer_header_line":header_index + 1,
        "lines":lines}


def clarify_ev_ebitda_range(sentence, evidence):
    """Return a narrower, explicitly scoped table range, or abstain.

    Only the complete standalone range sentence matches. Its vague recent-years
    window is not reverse-engineered. The replacement expressly describes all
    supplied annual rows aligned to the subject's actual columns through its
    last actual column. Original prose remains in the returned provenance.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    table = _ev_range_table(evidence)
    if table is None:
        return None
    plain = re.sub(r"^\[UNVERIFIED - (?:evidence not supplied|deterministic evidence/number check failed)\]\s*", "", sentence).strip()
    aliases = "|".join(re.escape(s) for s in sorted({table["ticker"],table["company"]},key=len,reverse=True))
    # A closed, deliberately small numeric-bound grammar. It recognizes the
    # isolated proposition without interpreting fuzzy words as exact thresholds.
    bound = r"(?:(?:the|about|around|approximately|near)\s+)?(?:\d+(?:\.\d+)?(?:x)?|(?:low|mid|high)[- ]\d+s)"
    pattern = (rf"(?P<subject>{aliases})(?:['’]s)?\s+EV/EBITDA(?:\s+(?:ratio|multiple|multiples))?\s+"
        rf"(?:has|have)\s+ranged\s+from\s+(?P<lower>{bound})\s+to\s+(?P<upper>{bound})\s+"
        r"(?:in|over)\s+recent\s+years\.")
    matched = re.fullmatch(pattern,plain,re.I)
    if not matched:
        return None
    values = [Decimal(row["value"]) for row in table["observations"]]
    low, high = min(values), max(values)
    minima = [row["actual_label"] for row in table["observations"] if Decimal(row["value"]) == low]
    maxima = [row["actual_label"] for row in table["observations"] if Decimal(row["value"]) == high]
    revised = (
        f"Across the supplied EV/EBITDA rows aligned with {table['ticker']}'s "
        f"{table['first_actual']} through {table['last_actual']} actual columns, "
        f"the minimum is {low}x ({', '.join(minima)}) and the maximum is "
        f"{high}x ({', '.join(maxima)}). "
        f"This scope uses the supplied annual rows through {table['ticker']}'s last actual column."
    )
    evidence_numbers = [table[k] for k in ("identity_line","financial_header_line","metric_heading_line","peer_header_line")]
    evidence_numbers += [row["line_number"] for row in table["observations"]]
    return {
        "rule":"isolated_ev_ebitda_range_supplied_actual_rows",
        "rule_version":"9.1",
        "original_released_text":sentence,
        "original_range_expression":{"lower":matched["lower"],"upper":matched["upper"],"window":"recent years"},
        "revised_released_text":revised,
        "original_semantic_verdict":"Not inferred; the original recent-years interval is unspecified.",
        "ticker":table["ticker"],"company":table["company"],
        "evidence_prompt_sha256":hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        "selection_rule":"Use every peer-table subject value whose year matches an explicitly A-labeled subject column, with contiguous actual years and complete positive finite values; no forecast column enters the calculation.",
        "first_actual":table["first_actual"],"last_actual":table["last_actual"],
        "observations":table["observations"],
        "calculation":{"formula":"minimum = min(subject EV/EBITDA values over selected actual rows); maximum = max(subject EV/EBITDA values over the same rows)",
            "minimum":str(low),"minimum_years":minima,"maximum":str(high),"maximum_years":maxima,
            "selected_row_count":len(values),"interpolation":False,"forecast_values_used":False},
        "evidence":[{"line_number":n,"quote":table["lines"][n-1]} for n in sorted(set(evidence_numbers))],
        "limitations":"Only a supplied-table range clarification. Fiscal-year labels do not prove identical calendar period ends across companies. This rule does not infer the original recent-years window, assess a valuation premium or validate the provider's EV/EBITDA calculation. Development examples informed the grammar; no held-out claim.",
    }


def apply_ev_ebitda_range_guard(result, evidence):
    """Keep the paid review intact and change only exact complete releases."""
    result = copy.deepcopy(result)
    released_units = result["revised_text"].split("\n\n")
    replacements = {}
    for row in result["claims"]:
        prior = row.get("released_text", "")
        if not prior or prior not in released_units:
            continue
        checked = clarify_ev_ebitda_range(prior,evidence)
        if checked is None:
            continue
        row["pre_ev_range_guard_release"] = {k:copy.deepcopy(row.get(k)) for k in
            ("released_text","retained_as_verified","verification_status","release_text_source")}
        row["ev_ebitda_range_check"] = checked
        row["released_text"] = checked["revised_released_text"]
        row["retained_as_verified"] = True
        row["verification_status"] = "deterministic_actual_range_clarified"
        row["release_text_source"] = "original_prompt_actual_rows_deterministic_range"
        replacements[prior] = row["released_text"]
    if replacements:
        # Keep pre-existing deduplication policy. Different paid claim records
        # remain separate and traceable even when their exact releases coincide.
        final = []
        for prior in released_units:
            revised = replacements.get(prior,prior)
            if revised not in final:
                final.append(revised)
        result["revised_text"] = "\n\n".join(final)
    result["verifier_version"] = "9.1-explicit-actual-ev-ebitda-range"
    result["ev_ebitda_range_check_count"] = sum("ev_ebitda_range_check" in row for row in result["claims"])
    result["verified_claim_count"] = sum(row["retained_as_verified"] for row in result["claims"])
    result["unverified_annotation_count"] = sum(row["verification_status"] in {"unverified_annotation","unverified_deterministic_failure"} for row in result["claims"])
    result["supported_valid_retention_count"] = sum(row["classification"] == "Supported" and row["retained_as_verified"] for row in result["claims"])
    result["limitations"] += " V9.1 replaces only isolated, identified EV/EBITDA range sentences with explicitly scoped supplied-actual-row extrema. It retains prior paid claims, evidence, labels and number-check failures; the development probe is not an independent accuracy score."
    return result
