import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("output/sec_company_facts")
OUT = Path("experiments/part2_financial_audit")
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 2025
CUTOFF = date(2026, 10, 10)

DIRECT = {
    "Revenue": None,
    "Net Income": None,
    "Operating Income": None,
    "Operating Cash Flow": None,
    "Capital Expenditures": None,
    "Cost of Revenue": None,
    "Income Tax Expense": None,
    "Interest Expense": None,
    "Depreciation and Amortization": None,
    "Depreciation": "Depreciation",
    "Intangible Amortization": "AmortizationOfIntangibleAssets",
}

FALLBACK_CONCEPTS = {
    "QCOM": {
        "Capital Expenditures": "PaymentsToAcquireProductiveAssets"
    }
}


def retrieve_fact(raw, concept, accession, year):
    entry = raw.get("facts", {}).get("us-gaap", {}).get(concept)
    if not entry:
        return None

    candidates = []
    for x in entry.get("units", {}).get("USD", []):
        if x.get("accn") != accession:
            continue
        if x.get("form") not in ("10-K", "10-K/A"):
            continue
        if x.get("fp") != "FY":
            continue
        if not x.get("start") or not x.get("end"):
            continue

        try:
            start = date.fromisoformat(x["start"])
            end = date.fromisoformat(x["end"])
            filed = date.fromisoformat(x["filed"])
        except (KeyError, ValueError, TypeError):
            continue

        if end.year != year:
            continue
        if not 330 <= (end - start).days <= 380:
            continue
        if filed > CUTOFF:
            continue

        candidates.append(x)

    if not candidates:
        return None

    # Prefer the fact covering the latest fiscal-end date.
    candidates.sort(
        key=lambda x: (x["end"], x["start"]),
        reverse=True,
    )

    return candidates[0]


def equal_usd(a, b):
    return bool(
        np.isfinite(a)
        and np.isfinite(b)
        and abs(float(a) - float(b)) <= 1.0
    )


records = []

for ticker in ("AVGO", "QCOM"):
    folder = BASE / ticker
    raw = json.loads(
        (folder / "companyfacts_raw.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (folder / "data_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    df = pd.read_csv(folder / "financials_normalized_usd.csv")
    row = df.loc[df["fiscal_year"] == YEAR].iloc[0]

    source_map = {}
    for item in provenance["metric_audit"]:
        if (
            item["fiscal_year"] == YEAR
            and item["status"] == "found"
        ):
            source_map[item["metric"]] = item

    accession = source_map["Revenue"]["accession"]
    cik = str(provenance["cik"]).lstrip("0")

    filing_index = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession.replace('-', '')}/"
    )

    verified = {}

    def append_record(
        metric, kind, local, sec, concept, formula=""
    ):
        local = float(local) if pd.notna(local) else np.nan
        sec = float(sec) if sec is not None and pd.notna(sec) else np.nan

        ok = equal_usd(local, sec)
        status = "MATCH" if ok else (
            "MISSING" if pd.isna(sec) or pd.isna(local)
            else "MISMATCH"
        )

        records.append({
            "ticker": ticker,
            "fiscal_year": YEAR,
            "metric": metric,
            "type": kind,
            "local_value_usd": local,
            "sec_recomputed_usd": sec,
            "absolute_error_usd": (
                abs(local - sec)
                if pd.notna(local) and pd.notna(sec)
                else np.nan
            ),
            "relative_error_pct": (
                abs(local - sec) / abs(sec) * 100
                if pd.notna(local)
                and pd.notna(sec)
                and sec != 0 else np.nan
            ),
            "status": status,
            "xbrl_concept_or_formula": concept,
            "calculation": formula,
            "accession": accession,
            "source_url": filing_index,
            "review_note": (
                "Internal reconciliation against locally saved "
                "SEC XBRL facts; manual 10-K review pending."
            ),
        })

    for metric in DIRECT:
        if metric not in row.index:
            continue

        local = row[metric]
        if pd.isna(local):
            continue

        if DIRECT[metric] is not None:
            concept = DIRECT[metric]
        elif metric in source_map:
            concept = source_map[metric]["xbrl_concept"]
        else:
            concept = FALLBACK_CONCEPTS.get(
                ticker, {}
            ).get(metric)

        if not concept:
            continue

        sec_fact = retrieve_fact(
            raw, concept, accession, YEAR
        )

        if sec_fact is None:
            append_record(
                metric, "Direct", local, None, concept
            )
            continue

        sec_val = sec_fact["val"]
        verified[metric] = float(sec_val)

        append_record(
            metric, "Direct", local,
            sec_val, concept
        )

    # Derive FCF independently from the extracted SEC facts.
    ocf = verified.get("Operating Cash Flow")
    capex = verified.get("Capital Expenditures")

    if ocf is not None and capex is not None:
        append_record(
            "FCF Proxy",
            "Derived",
            row["FCF Proxy"],
            ocf - capex,
            "Operating Cash Flow - Capital Expenditures",
            "FCF proxy = OCF - CapEx"
        )

    # Derive EBITDA proxy independently from SEC components.
    operating = verified.get("Operating Income")

    if ticker == "AVGO":
        depreciation = verified.get("Depreciation")
        amortization = verified.get("Intangible Amortization")

        if all(
            x is not None
            for x in (operating, depreciation, amortization)
        ):
            append_record(
                "EBITDA Proxy",
                "Derived",
                row["EBITDA Proxy"],
                operating + depreciation + amortization,
                "OperatingIncomeLoss + Depreciation + "
                "AmortizationOfIntangibleAssets",
                "Operating Income + Depreciation + "
                "Intangible Amortization"
            )

    else:
        da = verified.get("Depreciation and Amortization")
        if operating is not None and da is not None:
            append_record(
                "EBITDA Proxy",
                "Derived",
                row["EBITDA Proxy"],
                operating + da,
                "OperatingIncomeLoss + "
                "DepreciationDepletionAndAmortization",
                "Operating Income + D&A"
            )

    subset = [r for r in records if r["ticker"] == ticker]
    matched = sum(r["status"] == "MATCH" for r in subset)

    print(f"\n========== {ticker} 2025 AUDIT ==========")
    for r in subset:
        print(
            f"{r['status']:9s} "
            f"{r['type']:7s} "
            f"{r['metric']:30s} "
            f"${r['local_value_usd'] / 1e9:,.3f}B"
        )

    print(f"Matched: {matched}/{len(subset)}")
    print("10-K accession:", accession)

    if len(subset) < 10:
        print("WARNING: Fewer than 10 checkable metrics")


audit = pd.DataFrame(records)

csv_path = OUT / "financial_audit_2025.csv"
json_path = OUT / "financial_audit_2025.json"

audit.to_csv(csv_path, index=False)
json_path.write_text(
    audit.to_json(orient="records", indent=2),
    encoding="utf-8"
)

print("\n========== TOTAL AUDIT ==========")
print("Rows:", len(audit))
print("MATCH:", (audit["status"] == "MATCH").sum())
print("MISMATCH:", (audit["status"] == "MISMATCH").sum())
print("MISSING:", (audit["status"] == "MISSING").sum())
print("Saved CSV:", csv_path)
print("Saved JSON:", json_path)
print("\nPART 2A AUDIT COMPLETE")
