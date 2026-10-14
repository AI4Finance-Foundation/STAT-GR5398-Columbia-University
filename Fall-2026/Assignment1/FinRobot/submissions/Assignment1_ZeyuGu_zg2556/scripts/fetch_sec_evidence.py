"""Freeze public SEC evidence, then derive references without claiming a pipeline audit.

Python standard library only. Raw HTTP bodies are immutable within a snapshot.
Use --offline to rebuild CSV references exclusively from the frozen responses.
No API credentials, paid data, or FinRobot implementation are used here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import pathlib
import time
import urllib.error
import urllib.request

COMPANIES = {"NVDA": 1045810, "AMD": 2488, "INTC": 50863, "AVGO": 1730168, "QCOM": 804328}
ROOT = pathlib.Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Snapshot:
    def __init__(self, directory, offline, user_agent):
        self.directory = directory
        self.offline = offline
        self.user_agent = user_agent
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, url, relative):
        path = self.directory / relative
        meta_path = pathlib.Path(str(path) + ".meta.json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != meta["sha256"] or meta["url"] != url:
                raise ValueError(f"Snapshot mismatch: {path}")
            return data if meta["status"] == 200 else None
        if self.offline:
            raise FileNotFoundError(f"No frozen response: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        started = now()
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"})
        error = None
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data, status, headers = response.read(), response.status, dict(response.headers)
                final_url = response.url
        except urllib.error.HTTPError as exc:
            data, status, headers, final_url = exc.read(), exc.code, dict(exc.headers), exc.url
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            # Transport failure is logged separately, not saved as a successful empty response.
            with (self.directory / "transport_errors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"url": url, "started_utc": started, "error": str(exc)}) + "\n")
            return None
        path.write_bytes(data)
        meta = {"url": url, "final_url": final_url, "started_utc": started, "retrieved_utc": now(),
                "status": status, "response_headers": headers, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "error": error, "attempts": 1}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{relative}: HTTP {status}, {len(data)} bytes", flush=True)
        time.sleep(0.3)  # Well below SEC's 10 requests/second fair-access ceiling.
        return data if status == 200 else None


def annual_fact(facts, tags, filing, end=None, unit="USD"):
    """Require the exact accession and a full annual duration; never use later restatements."""
    end = end or filing["reportDate"]
    for tag in tags:
        observations = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, [])
        candidates = [dict(x, tag=tag, unit=unit, json_index=index) for index, x in enumerate(observations)
                      if x.get("accn") == filing["accessionNumber"] and x.get("end") == end
                      and x.get("start") and 350 <= (dt.date.fromisoformat(x["end"]) - dt.date.fromisoformat(x["start"])).days <= 380]
        if candidates:
            values = {x["val"] for x in candidates}
            if len(values) != 1:
                raise ValueError(f"Conflicting SEC facts: {tag}, {filing['accessionNumber']}, {end}")
            return sorted(candidates, key=lambda x: (x.get("frame", ""), x["start"]))[-1]
    return None


class InlineFacts(HTMLParser):
    """Preserve filing-level iXBRL numeric IDs and consolidated contexts for cross-checks."""
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.contexts, self.facts = {}, []
        self.context = self.fact = None
        self.element = ""
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.element = tag
        if tag == "xbrli:context":
            self.context = {"id": attrs.get("id"), "start": "", "end": "", "instant": "", "dimensions": []}
        elif self.context is not None and tag == "xbrldi:explicitmember":
            self.context["dimensions"].append(dict(attrs, member=""))
        elif tag == "ix:nonfraction":
            self.fact = dict(attrs, raw_text="", html_line=self.getpos()[0], html_column=self.getpos()[1])

    def handle_endtag(self, tag):
        if tag == "xbrli:context" and self.context is not None:
            self.contexts[self.context["id"]] = self.context
            self.context = None
        elif tag == "ix:nonfraction" and self.fact is not None:
            self.facts.append(self.fact)
            self.fact = None
        self.element = ""

    def handle_data(self, data):
        if self.fact is not None:
            self.fact["raw_text"] += data
        if self.context is not None:
            field = {"xbrli:startdate": "start", "xbrli:enddate": "end", "xbrli:instant": "instant"}.get(self.element)
            if field:
                self.context[field] += data.strip()
            elif self.element == "xbrldi:explicitmember" and self.context["dimensions"]:
                self.context["dimensions"][-1]["member"] += data.strip()

    def matches(self, observation):
        matches = []
        for fact in self.facts:
            context = self.contexts.get(fact.get("contextref"), {})
            if fact.get("name") != "us-gaap:" + observation["tag"] or context.get("dimensions"):
                continue
            if context.get("start") != observation["start"] or context.get("end") != observation["end"]:
                continue
            text = fact["raw_text"].strip().replace(",", "").replace("\u00a0", "")
            try:
                value = 0 if text in ["-", "—", "–"] else float(text)
                value *= 10 ** int(fact.get("scale", "0"))
                if fact.get("sign") == "-":
                    value *= -1
            except ValueError:
                continue
            if abs(value - observation["val"]) <= max(1e-7, abs(value) * 1e-12):
                matches.append({**fact, "numeric_value": value, "context": context})
        return matches


def references(ticker, facts, filing, snapshot):
    tags = {
        "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "cost": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
        "sga": ["SellingGeneralAndAdministrativeExpense"],
        "operating_income": ["OperatingIncomeLoss"],
        "da": ["DepreciationDepletionAndAmortization", "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment", "DepreciationAmortizationAndAccretionNet"],
        "other_da": ["OtherDepreciationAndAmortization"],
        "acquisition_amortization": ["AdjustmentForAmortization"],
        "eps": ["EarningsPerShareBasic"],
        "net_income": ["NetIncomeLoss"],
        "continuing_income": ["IncomeLossFromContinuingOperations"],
        "discontinued_income": ["IncomeLossFromDiscontinuedOperationsNetOfTaxAttributableToReportingEntity"],
        "equity_income": ["IncomeLossFromEquityMethodInvestments"],
        "pretax_before_equity": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
        "tax": ["IncomeTaxExpenseBenefit"],
        "interest": ["InterestExpenseNonoperating", "InterestExpense"],
        "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    }
    chosen = {key: annual_fact(facts, alternatives, filing, unit="USD/shares" if key == "eps" else "USD")
              for key, alternatives in tags.items()}
    rev_observations = facts.get("facts", {}).get("us-gaap", {}).get(chosen["revenue"]["tag"], {}).get("units", {}).get("USD", []) if chosen["revenue"] else []
    prior_ends = sorted({x["end"] for x in rev_observations if x.get("accn") == filing["accessionNumber"] and x["end"] < filing["reportDate"]}, reverse=True)
    chosen["prior_revenue"] = annual_fact(facts, tags["revenue"], filing, end=prior_ends[0]) if prior_ends else None
    filing_path = snapshot.directory / ticker / filing["accessionNumber"] / filing["primaryDocument"]
    inline = InlineFacts(filing_path.read_text(encoding="utf-8")) if filing_path.exists() else None
    raw_rows = []
    for key, obs in chosen.items():
        if obs:
            raw_rows.append({"ticker": ticker, "ingredient": key, **obs,
                             "companyfacts_file": f"{ticker}/companyfacts.json",
                             "json_pointer": f"/facts/us-gaap/{obs['tag']}/units/{obs['unit'].replace('/', '~1')}/{obs['json_index']}",
                             "filing_ixbrl_matches": json.dumps(inline.matches(obs) if inline else [], separators=(",", ":")),
                             "source_url": filing["source_url"]})
    # Normalize optional companyfacts keys to a stable field set.
    raw_fields = ["ticker", "ingredient", "tag", "unit", "val", "start", "end", "accn", "fy", "fp", "form", "filed", "frame", "companyfacts_file", "json_pointer", "filing_ixbrl_matches", "source_url"]
    raw_rows = [{k: x.get(k, "") for k in raw_fields} for x in raw_rows]
    # AMD separately presents acquisition amortization and other D&A in cash flows.
    # Do not treat the smaller line as total D&A or double count the aggregate.
    da_keys = ["other_da", "acquisition_amortization"] if ticker == "AMD" else ["da"]
    definitions = [
        ("Revenue", ["revenue"], lambda a: a[0], "revenue", "USD", "GAAP consolidated revenue."),
        ("Cost of Operations", ["cost"], lambda a: a[0], "cost of revenue", "USD", "Upstream labels costOfRevenue as Cost of Operations; it is not total operating expense."),
        ("SG&A", ["sga"], lambda a: a[0], "selling, general and administrative expense", "USD", "GAAP SG&A; do not combine research and development."),
        ("Contribution Profit", ["revenue", "cost"], lambda a: a[0]-a[1], "revenue - cost of revenue", "USD", "Upstream Contribution Profit corresponds to gross profit, not managerial variable-cost contribution."),
        ("Contribution Margin", ["revenue", "cost"], lambda a: (a[0]-a[1])/a[0]*100, "100 * (revenue - cost of revenue) / revenue", "percent", "Derived gross margin. Upstream rounds historical margins to 0.1 percentage point."),
        ("EBITDA", ["operating_income"] + da_keys, lambda a: sum(a), "GAAP operating income + " + " + ".join(da_keys), "USD", "REFERENCE DEFINITION ONLY: operating EBITDA proxy, not an SEC GAAP line item or assumed FMP definition. AMD includes both other D&A and separately stated acquisition-related amortization. Reconcile non-operating components explicitly before verdict."),
        ("EBITDA Margin", ["operating_income"] + da_keys + ["revenue"], lambda a: sum(a[:-1])/a[-1]*100, "100 * (operating income + " + " + ".join(da_keys) + ") / revenue", "percent", "Same explicit operating EBITDA proxy; FMP EBITDA definition not yet verified."),
        ("SG&A Margin", ["sga", "revenue"], lambda a: a[0]/a[1]*100, "100 * SG&A / revenue", "percent", "Derived from GAAP SG&A and revenue; compare 0.1 percentage point pipeline rounding."),
        ("Revenue Growth", ["revenue", "prior_revenue"], lambda a: (a[0]/a[1]-1)*100, "100 * (current annual revenue / prior annual revenue - 1)", "percent", "Comparative prior-year revenue comes from the same accession. Fiscal durations can be 52 or 53 weeks."),
        ("EPS", ["eps"], lambda a: a[0], "GAAP basic earnings per share", "USD/share", "Upstream prefers eps over epsdiluted. Confirm FMP eps is basic; never mix basic and diluted or split-adjusted periods."),
    ]
    rows = []
    for metric, ingredients, calculation, formula, unit, note in definitions:
        present = all(chosen.get(k) for k in ingredients)
        evidence = [chosen[k] for k in ingredients if chosen.get(k)]
        rows.append({"ticker": ticker, "metric": metric,
                     "fiscal_year": chosen["revenue"].get("fy", "") if chosen["revenue"] else "",
                     "period_start": chosen["revenue"].get("start", "") if chosen["revenue"] else "",
                     "period_end": filing["reportDate"], "form": filing["form"], "filing_date": filing["filingDate"],
                     "pipeline_value": "", "filing_reference_value": calculation([chosen[k]["val"] for k in ingredients]) if present else "",
                     "currency": "USD", "unit": unit, "difference": "", "verdict": "PENDING_PIPELINE_COMPARISON" if present else "MISSING_SEC_COMPONENT",
                     "sec_accession": filing["accessionNumber"], "filing_source_url": filing["source_url"],
                     "xbrl_context": json.dumps([{**{k: x.get(k) for k in ["tag", "unit", "start", "end", "frame", "val"]}, "filing_matches": inline.matches(x) if inline else []} for x in evidence], separators=(",", ":")),
                     "formula": formula, "evidence_snapshot": str(snapshot.directory.relative_to(ROOT)).replace("\\", "/"),
                     "explanation": note + " No pipeline observation exists in this reference table; this row is not a completed audit."})
    return rows, raw_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-10-11", help="Exclude filings after this date (YYYY-MM-DD).")
    parser.add_argument("--snapshot-dir", type=pathlib.Path, default=ROOT / "evidence/sec/snapshot_2026-10-11")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--user-agent", default="AcademicResearch/1.0 (ZeyuGu-zg2556)")
    args = parser.parse_args()
    dt.date.fromisoformat(args.as_of)
    snapshot = Snapshot(args.snapshot_dir.resolve(), args.offline, args.user_agent)
    filings, classifications, all_references, all_ingredients = [], [], [], []
    for ticker, cik in COMPANIES.items():
        submission_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        submission_bytes = snapshot.get(submission_url, f"{ticker}/submissions.json")
        facts_bytes = snapshot.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", f"{ticker}/companyfacts.json")
        if not submission_bytes:
            continue
        submission = json.loads(submission_bytes)
        classifications.append({"ticker": ticker, "cik": cik, "sec_name": submission.get("name"), "sic": submission.get("sic"),
                                "sic_description": submission.get("sicDescription"), "fiscal_year_end_code": submission.get("fiscalYearEnd"),
                                "source_url": submission_url, "companyfacts_available": bool(facts_bytes)})
        recent = submission["filings"]["recent"]
        records = [dict(zip(recent, values)) for values in zip(*recent.values())]
        available = [x for x in records if x["filingDate"] <= args.as_of and x["form"] in ["10-K", "10-K/A", "10-Q"]]
        annuals = [x for x in available if x["form"] == "10-K"]
        if not annuals:
            raise ValueError(f"No annual filing in recent submissions for {ticker}; add historical submission retrieval.")
        latest_annual = max(annuals, key=lambda x: x["filingDate"])
        quarterlies = [x for x in available if x["form"] == "10-Q"]
        selected = [latest_annual]
        selected += [x for x in available if x["form"] == "10-K/A" and x["reportDate"] == latest_annual["reportDate"]]
        if quarterlies:
            selected.append(max(quarterlies, key=lambda x: x["filingDate"]))
        for filing in selected:
            accession = filing["accessionNumber"]
            source_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filing['primaryDocument']}"
            filing["source_url"] = source_url
            raw = snapshot.get(source_url, f"{ticker}/{accession}/{filing['primaryDocument']}")
            filings.append({"ticker": ticker, "cik": cik, "form": filing["form"], "accession": accession,
                            "filing_date": filing["filingDate"], "period_end": filing["reportDate"],
                            "source_url": source_url, "primary_document_saved": bool(raw),
                            "local_file": f"{ticker}/{accession}/{filing['primaryDocument']}"})
        if ticker in ["NVDA", "AMD"] and facts_bytes:
            rows, ingredients = references(ticker, json.loads(facts_bytes), latest_annual, snapshot)
            all_references.extend(rows)
            all_ingredients.extend(ingredients)
    write_csv(snapshot.directory / "classification.csv", classifications)
    write_csv(snapshot.directory / "filings.csv", filings)
    write_csv(snapshot.directory / "sec_metric_references.csv", all_references)
    write_csv(snapshot.directory / "sec_metric_ingredients.csv", all_ingredients)
    bridges = []
    for ticker in ["NVDA", "AMD"]:
        ingredients = {x["ingredient"]: x for x in all_ingredients if x["ticker"] == ticker}
        da_keys = ["other_da", "acquisition_amortization"] if ticker == "AMD" else ["da"]
        required = ["net_income", "tax", "interest", "operating_income"] + da_keys
        if not all(k in ingredients for k in required):
            continue
        values = {k: ingredients[k]["val"] for k in required}
        da = sum(values[k] for k in da_keys)
        operating = values["operating_income"] + da
        net_income_based = values["net_income"] + values["tax"] + values["interest"] + da
        bridges.append({"ticker": ticker, "period_end": ingredients["net_income"]["end"],
                        "sec_accession": ingredients["net_income"]["accn"], "currency_unit": "USD",
                        "net_income": values["net_income"], "income_tax_expense_benefit": values["tax"],
                        "interest_expense": values["interest"], "operating_income": values["operating_income"],
                        "depreciation_and_amortization": da, "operating_ebitda_reference": operating,
                        "net_income_based_ebitda_reference": net_income_based,
                        "difference_nonoperating_excluding_interest": net_income_based - operating,
                        "definition_a": "operating income + D&A",
                        "definition_b": "net income + income tax expense (benefit) + interest expense + D&A",
                        "source_url": ingredients["net_income"]["source_url"],
                        "status": "ALTERNATIVE_SEC_DERIVATIONS_NOT_FMP_RECONCILIATION"})
    write_csv(snapshot.directory / "sec_ebitda_definition_bridge.csv", bridges)
    metadata = [json.loads(p.read_text(encoding="utf-8")) | {"metadata_path": str(p.relative_to(snapshot.directory)).replace("\\", "/")} for p in snapshot.directory.rglob("*.meta.json")]
    manifest = {"schema_version": 1, "as_of": args.as_of, "derived_utc": now(), "offline_derivation": args.offline,
                "upstream_sha": "2717499b8e30f242640af08c4ad9afd1113c2d45", "purpose": "Independent SEC reference evidence; not FMP or completed pipeline audit",
                "responses": metadata, "completed_pipeline_comparisons": 0, "reference_rows": len(all_references),
                "reference_definition_caution": "Operating EBITDA proxy requires reconciliation with FMP definition and each company's D&A scope."}
    (snapshot.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"responses": len(metadata), "reference_rows": len(all_references), "completed_pipeline_comparisons": 0}))


if __name__ == "__main__":
    main()
