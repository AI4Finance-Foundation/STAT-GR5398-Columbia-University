import argparse
import json
import re
from pathlib import Path

import pandas as pd


def verify(text, csv_path):
    df = pd.read_csv(csv_path)
    findings = []

    year_cols = [
        str(c) for c in df.columns
        if re.fullmatch(r"20\d{2}[AE]", str(c))
    ]

    metric_col = next(
        (c for c in df.columns
         if c.lower() in ("metrics", "metric")),
        None
    )

    def metric_value(name, year):
        if metric_col is None or year not in df.columns:
            return None

        names = df[metric_col].astype(str).str.strip().str.lower()
        selected = df[names == name.lower()]
        if selected.empty:
            return None

        value = pd.to_numeric(
            selected.iloc[0][year], errors="coerce"
        )
        return float(value) if pd.notna(value) else None

    # Rule 1: actual-year labels incorrectly described as estimates.
    for match in re.finditer(
        r"\b(estimated|forecasted|projected)\s+"
        r"(?:at\s+|to\s+|approximately\s+|around\s+)?"
        r"(?:\$[\d,.]+\s*(?:billion|million|B|M)?\s+"
        r"(?:in|for)\s+)?"
        r"(20\d{2}A)\b",
        text, flags=re.IGNORECASE
    ):
        findings.append({
            "rule": "ACTUAL_FORECAST_MISMATCH",
            "status": "flagged",
            "evidence": match.group(0),
            "reason": "An Actual year is described as an estimate.",
        })

    # Additional pattern: "an estimated $144.55 billion in 2026A".
    for match in re.finditer(
        r"\b(?:an?\s+)?(?:estimated|forecasted|projected)\s+"
        r"\$[\d,.]+\s*(?:billion|million|B|M)\s+"
        r"(?:in|for|by)\s+(20\d{2}A)\b",
        text, flags=re.IGNORECASE
    ):
        if not any(f["evidence"] == match.group(0) for f in findings):
            findings.append({
                "rule": "ACTUAL_FORECAST_MISMATCH",
                "status": "flagged",
                "evidence": match.group(0),
                "reason": "An Actual value is described as estimated.",
            })

    # Rule 2: focus-company EBITDA statements with explicit dollar/year pair.
    # Limit scope to paragraphs/sentences referring to NVIDIA's EBITDA;
    # other-company claims require separate peer-data validation.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    amount_year = re.compile(
        r"\$([\d,.]+)\s*(billion|million|B|M)\s+"
        r"(?:in|by|for)\s+(20\d{2}[AE])\b",
        re.IGNORECASE,
    )

    for sentence in sentences:
        if not re.search(r"\bEBITDA\b", sentence, re.IGNORECASE):
            continue

        pairs = list(amount_year.finditer(sentence))
        if not pairs:
            continue

        # Automated validation is restricted to short, unambiguous claims.
        # Sentences referencing peers may contain multiple companies/metrics.
        if re.search(r"\b(?:AMD|Intel|INTC|Broadcom|Qualcomm)\b", sentence):
            continue

        for match in pairs:
            amount = float(match.group(1).replace(",", ""))
            unit = match.group(2).lower()
            year = match.group(3)

            if unit in ("million", "m"):
                amount /= 1000.0

            # The FinRobot CSV reports financial amounts in $ billions.
            revenue = metric_value("Revenue", year)
            ebitda = metric_value("EBITDA", year)

            if revenue is not None and amount > revenue * 1.02:
                findings.append({
                    "rule": "AMOUNT_EXCEEDS_REVENUE",
                    "status": "flagged",
                    "year": year,
                    "claimed_billions": amount,
                    "revenue_billions": revenue,
                    "evidence": match.group(0),
                    "reason": "Claimed amount exceeds annual revenue; "
                              "check metric attribution.",
                })

            if ebitda is not None and abs(amount - ebitda) > max(
                1.0, 0.05 * abs(ebitda)
            ):
                findings.append({
                    "rule": "EBITDA_VALUE_MISMATCH",
                    "status": "needs_review",
                    "year": year,
                    "claimed_billions": amount,
                    "csv_ebitda_billions": ebitda,
                    "evidence": match.group(0),
                    "reason": "Dollar amount in an EBITDA sentence differs "
                              "from the CSV; verify its metric context.",
                })

    return {
        "years_available": year_cols,
        "metric_column": metric_col,
        "findings": findings,
        "finding_count": len(findings),
        "method": "Deterministic rule-based screening; manual review required",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = verify(
        Path(args.text).read_text(encoding="utf-8"),
        args.csv,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Financial Grounding Verifier")
    print("Metric column:", result["metric_column"])
    print("Years:", ", ".join(result["years_available"]))
    print("Findings:", result["finding_count"])

    for i, finding in enumerate(result["findings"], 1):
        print(f"\n[{i}] {finding['rule']}")
        print("Evidence:", finding["evidence"])
        print("Reason:", finding["reason"])

    print("\nAudit saved to:", out)


if __name__ == "__main__":
    main()
