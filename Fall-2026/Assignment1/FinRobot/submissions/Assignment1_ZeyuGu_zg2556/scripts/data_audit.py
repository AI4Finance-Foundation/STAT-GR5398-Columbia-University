"""Compare actual frozen pipeline metrics with accession-specific SEC references.

Requires real FMP income statement rows to prove date/currency alignment. The
SEC-side values are independently extracted from the original filing/iXBRL.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from audit_io import sha256, write_json


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    result = float(str(value).replace(",", "").replace("%", ""))
    if not math.isfinite(result):
        raise ValueError("Non-finite financial observation")
    return result


def optional_number(row, key):
    value = row.get(key)
    return numeric(value) if value not in (None, "", "None", "nan", "NaN") else None


def locate_analysis(analysis_root, ticker):
    directory = analysis_root/ticker/"analysis"
    return directory if directory.exists() else analysis_root/ticker


def aligned_raw(directory, reference):
    matches = [row for row in csv_rows(directory/"income_statement_raw_data.csv") if row["date"][:10] == reference["period_end"]]
    if len(matches) != 1:
        raise ValueError(f"{reference['ticker']}: cannot uniquely align filing period {reference['period_end']} to FMP")
    row = matches[0]
    if row.get("period") not in (None, "", "FY", "annual"):
        raise ValueError("Annual SEC reference cannot be compared to a quarterly pipeline record")
    return row


def ebitda_bridge(reference, raw, ingredients):
    """Quantitative two-definition reconciliation; never silently invent a component."""
    source = {r["ingredient"]:r for r in ingredients
              if r["ticker"] == reference["ticker"] and r["accn"] == reference["sec_accession"] and r["end"] == reference["period_end"]}
    keys = ["net_income", "tax", "interest", "operating_income", "revenue"]
    da_keys = ["other_da", "acquisition_amortization"] if reference["ticker"] == "AMD" else ["da"]
    if not all(key in source for key in keys + da_keys):
        return {"status":"MISSING_SEC_COMPONENT", "components":[], "definition_reconciled":False}
    sec = {key:numeric(source[key]["val"]) for key in keys + da_keys}
    sec["da"] = sum(numeric(source[key]["val"]) for key in da_keys)
    mapping = {"net_income":"netIncome", "tax":"incomeTaxExpense", "interest":"interestExpense",
               "operating_income":"operatingIncome", "da":"depreciationAndAmortization", "revenue":"revenue"}
    components=[]
    for key,field in mapping.items():
        observed=optional_number(raw,field)
        pointers=[source[k]["json_pointer"] for k in (da_keys if key=="da" else [key])]
        components.append({"component":key,"fmp_field":field,"fmp_value":observed,"sec_value":sec[key],
                           "difference":observed-sec[key] if observed is not None else None,"sec_json_pointers":pointers})
    actual = {x["component"]:x["fmp_value"] for x in components}
    fmp_ebitda=optional_number(raw,"ebitda")
    sec_operating=sec["operating_income"]+sec["da"]
    sec_ni=sec["net_income"]+sec["tax"]+sec["interest"]+sec["da"]
    ni_ready=all(actual[k] is not None for k in ["net_income","tax","interest","da"])
    op_ready=all(actual[k] is not None for k in ["operating_income","da"])
    fmp_ni=sum(actual[k] for k in ["net_income","tax","interest","da"]) if ni_ready else None
    fmp_op=actual["operating_income"]+actual["da"] if op_ready else None
    component_match=all(x["difference"] is not None and abs(x["difference"]) <= .5 for x in components)
    matches_ni=fmp_ebitda is not None and fmp_ni is not None and abs(fmp_ebitda-fmp_ni)<=.5
    matches_op=fmp_ebitda is not None and fmp_op is not None and abs(fmp_ebitda-fmp_op)<=.5
    adjusted=None
    exclusion_records=[]
    matches_adjusted=False
    if all(k in source for k in ["continuing_income","discontinued_income","equity_income","pretax_before_equity"]):
        extras={k:numeric(source[k]["val"]) for k in ["continuing_income","discontinued_income","equity_income","pretax_before_equity"]}
        adjusted=sec_ni-extras["discontinued_income"]-extras["equity_income"]
        fmp_continuing=optional_number(raw,"netIncomeFromContinuingOperations")
        fmp_discontinued=optional_number(raw,"netIncomeFromDiscontinuedOperations")
        fmp_ebit=optional_number(raw,"ebit")
        matches_adjusted=(fmp_ebitda is not None and abs(fmp_ebitda-adjusted)<=.5
            and abs(sec["net_income"]-extras["continuing_income"]-extras["discontinued_income"])<=.5
            and abs(extras["pretax_before_equity"]-(extras["continuing_income"]+sec["tax"]-extras["equity_income"]))<=.5
            and fmp_continuing is not None and abs(fmp_continuing-extras["continuing_income"])<=.5
            and fmp_discontinued is not None and abs(fmp_discontinued-extras["discontinued_income"])<=.5
            and fmp_ebit is not None and abs(fmp_ebit-extras["pretax_before_equity"]-sec["interest"])<=.5)
        exclusion_records=[{"component":k,"sec_value":v,"sec_json_pointer":source[k]["json_pointer"],
                            "filing_ixbrl_matches":source[k].get("filing_ixbrl_matches","")} for k,v in extras.items()]
    status = ("RECONCILED_NET_INCOME_BASED_DEFINITION" if component_match and matches_ni
              else "RECONCILED_OPERATING_INCOME_BASED_DEFINITION" if component_match and matches_op
              else "RECONCILED_CONTINUING_PRE_EQUITY_DEFINITION" if component_match and matches_adjusted
              else "FMP_NET_INCOME_FORMULA_MATCH_BUT_COMPONENT_DIFFERENCES" if matches_ni
              else "FMP_OPERATING_FORMULA_MATCH_BUT_COMPONENT_DIFFERENCES" if matches_op
              else "UNRESOLVED_EBITDA_DEFINITION_OR_COMPONENTS")
    selected=sec_ni if matches_ni else sec_operating if matches_op else adjusted if matches_adjusted else None
    return {"status":status,"definition_reconciled":component_match and (matches_ni or matches_op or matches_adjusted),
            "components":components,"fmp_ebitda":fmp_ebitda,"fmp_net_income_recomputed":fmp_ni,
            "fmp_operating_recomputed":fmp_op,"sec_operating_ebitda":sec_operating,"sec_net_income_based_ebitda":sec_ni,
            "sec_nonoperating_excluding_interest":sec_ni-sec_operating,
            "sec_continuing_pre_equity_ebitda":adjusted,"sec_selected_definition_ebitda":selected,
            "exclusion_source_records":exclusion_records,
            "residual_after_selected_definition":fmp_ebitda-selected if fmp_ebitda is not None and selected is not None else None,
            "observed_fmp_minus_sec_operating":fmp_ebitda-sec_operating if fmp_ebitda is not None else None,
            "unexplained_residual_against_sec_net_income_definition":fmp_ebitda-sec_ni if fmp_ebitda is not None else None,
            "sec_revenue":sec["revenue"],"fmp_revenue":actual["revenue"],
            "formula":"NI-based EBITDA = net income + tax expense (signed) + interest expense + D&A; operating EBITDA = operating income + D&A",
            "interpretation":"Formula identity and matching components explain this particular observation; they do not prove FMP's undocumented definition for every company/year."}


def compare(references, analysis_root, ingredients=()):
    results = []
    for reference in references:
        ticker = reference["ticker"]
        directory = locate_analysis(analysis_root,ticker)
        metrics_file = directory/"financial_metrics_and_forecasts.csv"
        raw_file = directory/"income_statement_raw_data.csv"
        metrics = {row["metrics"]:row for row in csv_rows(metrics_file)}
        raw = aligned_raw(directory,reference)
        year = numeric(raw["year"])
        if not year.is_integer(): raise ValueError("Nonintegral pipeline year")
        column = str(int(year)) + "A"
        observed = metrics[reference["metric"]][column]
        filing = float(reference["filing_reference_value"])
        value = numeric(observed)
        tolerance = .05000001 if reference["unit"] == "percent" else (.00500001 if reference["unit"] == "USD/share" else .5)
        difference = value-filing
        explanation = reference["explanation"].split(" No pipeline observation exists")[0]
        currency = raw.get("reportedCurrency", "")
        bridge = ebitda_bridge(reference,raw,ingredients) if reference["metric"] in {"EBITDA","EBITDA Margin"} else None
        if currency != reference["currency"]:
            verdict = "CURRENCY_MISMATCH"
            explanation += f" FMP reportedCurrency={currency}; no silent FX conversion permitted."
        elif abs(difference) <= tolerance:
            verdict = "MATCH_WITHIN_DISPLAY_PRECISION"
            explanation += f" Absolute display tolerance={tolerance}; source filing date {reference['filing_date']}."
        elif reference["metric"] in {"EBITDA", "EBITDA Margin"}:
            if bridge and bridge["definition_reconciled"]:
                reference_value = bridge["sec_selected_definition_ebitda"]
                if reference["metric"]=="EBITDA Margin": reference_value = 100*reference_value/bridge["sec_revenue"]
                if abs(value-reference_value)<=tolerance:
                    verdict=bridge["status"]
                    explanation += (f" Exact observed component bridge: FMP EBITDA={bridge['fmp_ebitda']}; SEC operating EBITDA={bridge['sec_operating_ebitda']}; "
                        f"SEC NI-based EBITDA={bridge['sec_net_income_based_ebitda']}; non-operating excluding interest={bridge['sec_nonoperating_excluding_interest']}; "
                        f"unexplained NI-based residual={bridge['unexplained_residual_against_sec_net_income_definition']}. "
                        "All six FMP/SEC revenue and earnings components match; margin display rounding is checked separately. The initial operating-reference delta remains in the table.")
                    if bridge["status"]=="RECONCILED_CONTINUING_PRE_EQUITY_DEFINITION":
                        explanation += (f" SEC continuing pre-equity EBITDA={bridge['sec_continuing_pre_equity_ebitda']}: exclude separately disclosed discontinued-operations income and equity-method income. "
                            f"These exclusions are source-tagged in the bridge; residual after exclusions={bridge['residual_after_selected_definition']}. FMP continuing/discontinued income and EBIT also match this bridge.")
                else:
                    verdict="PIPELINE_VALUE_DIFFERS_FROM_RECONCILED_RAW_EBITDA"
                    explanation += " Raw FMP EBITDA is reconcilable but pipeline amount/margin is not; inspect transformation."
            else:
                verdict = "DEFINITION_DIFFERENCE_REQUIRES_RECONCILIATION"
                explanation += " SEC reference is operating income plus D&A; missing or differing components prevent an exact definition reconciliation."
        else:
            verdict = "MISMATCH_REQUIRES_REVIEW"
            explanation += " Difference is not explained by display rounding; inspect restatement, fiscal period and FMP field semantics."
        results.append({**reference, "pipeline_value":value, "pipeline_display":observed,
            "pipeline_column":column, "pipeline_period_end":raw["date"][:10],
            "pipeline_fiscal_year":raw.get("calendarYear", raw.get("fiscalYear", "")),
            "pipeline_filing_date":raw.get("fillingDate", raw.get("filingDate", "")),
            "pipeline_accepted_date":raw.get("acceptedDate", ""),
            "pipeline_sec_link":raw.get("finalLink",raw.get("link","")),
            "pipeline_source_file":str(metrics_file), "pipeline_source_sha256":sha256(metrics_file.read_bytes()),
            "raw_source_sha256":sha256(raw_file.read_bytes()), "difference":difference,
            "difference_pct":difference/abs(filing)*100 if filing else "N/A",
            "ebitda_bridge_status":bridge["status"] if bridge else "not_applicable",
            "ebitda_bridge":json.dumps(bridge,separators=(",",":")) if bridge else "",
            "verdict":verdict, "explanation":explanation})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec-references", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--sec-ingredients",type=Path,help="Defaults to sec_metric_ingredients.csv beside references.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ingredient_path=args.sec_ingredients or args.sec_references.parent/"sec_metric_ingredients.csv"
    ingredients=csv_rows(ingredient_path)
    rows = compare(csv_rows(args.sec_references), args.analysis_root, ingredients)
    company_metrics={ticker:{r["metric"] for r in rows if r["ticker"]==ticker} for ticker in {r["ticker"] for r in rows}}
    if len(rows) < 20 or sum(len(v)>=10 for v in company_metrics.values())<2:
        raise ValueError("Insufficient real observations for the required audit")
    if len({(r["ticker"],r["period_end"],r["metric"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate data-audit metric/period observations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"data_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows":len(rows), "companies":sorted({row["ticker"] for row in rows}),
               "verdict_counts":{verdict:sum(r["verdict"] == verdict for r in rows) for verdict in sorted({r["verdict"] for r in rows})},
               "requires_explanation": [f"{r['ticker']}/{r['metric']}" for r in rows if r["verdict"] != "MATCH_WITHIN_DISPLAY_PRECISION"],
               "unresolved_rows":[f"{r['ticker']}/{r['metric']}" for r in rows if r["verdict"] not in {"MATCH_WITHIN_DISPLAY_PRECISION","RECONCILED_NET_INCOME_BASED_DEFINITION","RECONCILED_OPERATING_INCOME_BASED_DEFINITION","RECONCILED_CONTINUING_PRE_EQUITY_DEFINITION"}],
               "sec_reference_sha256":sha256(args.sec_references.read_bytes()),"sec_ingredients_sha256":sha256(ingredient_path.read_bytes())}
    component_rows=[]
    for row in rows:
        if row["metric"]=="EBITDA" and row["ebitda_bridge"]:
            bridge=json.loads(row["ebitda_bridge"])
            for component in bridge["components"]:
                component_rows.append({"ticker":row["ticker"],"period_end":row["period_end"],"sec_accession":row["sec_accession"],
                    "component":component["component"],"fmp_field":component["fmp_field"],"fmp_value":component["fmp_value"],
                    "sec_value":component["sec_value"],"difference":component["difference"],"sec_json_pointers":json.dumps(component["sec_json_pointers"]),
                    "bridge_status":bridge["status"],"observed_ebitda_difference":bridge.get("observed_fmp_minus_sec_operating"),
                    "explained_nonoperating_excluding_interest":bridge.get("sec_nonoperating_excluding_interest"),
                    "unexplained_residual_against_simple_ni_definition":bridge.get("unexplained_residual_against_sec_net_income_definition"),
                    "continuing_pre_equity_sec_ebitda":bridge.get("sec_continuing_pre_equity_ebitda"),
                    "exclusion_source_records":json.dumps(bridge.get("exclusion_source_records",[])),
                    "residual_after_selected_definition":bridge.get("residual_after_selected_definition")})
    if component_rows:
        with (args.output_dir/"ebitda_component_reconciliation.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(component_rows[0]));writer.writeheader();writer.writerows(component_rows)
    write_json(args.output_dir/"data_audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
