import argparse
import json
import re
from pathlib import Path

import pandas as pd


AMOUNT_YEAR = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*"
    r"(billion|million|B|M)\s+"
    r"(?:in|by|for|during)\s+"
    r"(20\d{2}[AE])\b",
    re.IGNORECASE,
)

ESTIMATED_ACTUAL = re.compile(
    r"\b(?:an?\s+)?(?:estimated|forecasted|projected)"
    r"\s+(?:at\s+|to\s+|approximately\s+|around\s+)?"
    r"\$?\s*[\d,]+(?:\.\d+)?\s*"
    r"(?:billion|million|B|M)?\s+"
    r"(?:in|by|for)\s+(20\d{2}A)\b",
    re.IGNORECASE,
)



METRIC_LABEL = re.compile(
    r"\b(?:"
    r"cost of operations|cost of revenue|"
    r"operating cash flow|contribution profit|"
    r"gross profit|net income|"
    r"revenue|sales|EBITDA|"
    r"EPS|SG&A"
    r")\b",
    re.IGNORECASE,
)


def attributed_metric(prefix):
    """Find nearest explicit metric before a dollar amount."""
    matches = list(METRIC_LABEL.finditer(prefix[-180:]))
    if not matches:
        return None

    metric = matches[-1].group(0).lower()
    return "Revenue" if metric == "sales" else metric


HISTORICAL_TREND_FORECAST = re.compile(
    r"\bprojected\b[^.!?\\n]{0,200}?"
    r"\bfrom\s+\$[\d,.]+\s*(?:B|M|billion|million)"
    r"\s+in\s+20\d{2}A"
    r"[^.!?\\n]{0,100}?"
    r"\bto\s+\$[\d,.]+\s*(?:B|M|billion|million)"
    r"\s+in\s+20\d{2}A\b",
    re.IGNORECASE,
)


def verify(text, csv_path):
    df = pd.read_csv(csv_path)

    if "metrics" not in df.columns:
        raise ValueError("CSV must contain a metrics column")

    year_cols = [
        str(c) for c in df.columns
        if re.fullmatch(r"20\d{2}[AE]", str(c))
    ]

    def get_billion(metric, year):
        if year not in df.columns:
            return None

        rows = df[
            df["metrics"].astype(str).str.strip().str.lower()
            == metric.lower()
        ]
        if rows.empty:
            return None

        value = pd.to_numeric(
            rows.iloc[0][year], errors="coerce"
        )

        if pd.isna(value):
            return None

        # CSV uses absolute USD, not USD billions.
        return float(value) / 1e9

    findings = []
    seen = set()

    def add(rule, evidence, status, reason, **extra):
        key = (rule, evidence.strip().lower())
        if key in seen:
            return
        seen.add(key)
        findings.append({
            "rule": rule,
            "status": status,
            "evidence": evidence.strip(),
            "reason": reason,
            **extra,
        })

    # 1. Detect "estimated $144.55 billion in 2026A"
    # without recording the same phrase twice.
    for m in ESTIMATED_ACTUAL.finditer(text):
        add(
            "ACTUAL_FORECAST_MISMATCH",
            m.group(0),
            "contradicted",
            "Actual-year figure is described as an estimate.",
            year=m.group(1),
        )

    # Actual-to-actual change described as a projection.
    for m in HISTORICAL_TREND_FORECAST.finditer(text):
        add(
            "HISTORICAL_TREND_CALLED_FORECAST",
            m.group(0),
            "contradicted",
            "Both endpoints are labeled Actual (A), "
            "but the change is described as projected.",
        )

    # 2. Split paragraphs into approximate sentences.
    # Preserve decimal points in financial amounts.
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text.replace("\n", " ")
    )

    for sentence in sentences:
        matches = list(AMOUNT_YEAR.finditer(sentence))
        if not matches:
            continue

        # An EBITDA claim needs an explicit EBITDA context.
        # This is a screening heuristic rather than full semantic parsing.
        if not re.search(r"\bEBITDA\b", sentence, re.IGNORECASE):
            continue

        for m in matches:
            claimed = float(m.group(1).replace(",", ""))
            unit = m.group(2).lower()
            year = m.group(3)

            if unit in ("million", "m"):
                claimed /= 1000.0

            actual = get_billion("EBITDA", year)
            revenue = get_billion("Revenue", year)

            if actual is None:
                continue

            # Determine the text immediately before this claim.
            prefix = sentence[:m.start()]
            local = prefix[-140:]

            metric = attributed_metric(prefix)

            # Do not compare Revenue, Cost of Operations,
            # EPS, etc. against EBITDA.
            if metric != "ebitda":
                continue

            # Avoid comparing clearly attributed peer amounts
            # against NVDA's financial data.
            peer_attribution = bool(re.search(
                r"(?:AMD|Intel|INTC|Broadcom|Qualcomm)"
                r"(?:'s)?\s+(?:EBITDA|earnings)"
                r"[^.;]{0,90}$",
                local,
                re.IGNORECASE,
            ))

            if peer_attribution:
                continue

            # If the sentence includes a peer comparison but the
            # claim is ambiguous, flag attribution for review.
            # The larger mismatch is still retained as a finding.
            possible_peer_context = bool(re.search(
                r"\b(?:AMD|Intel|INTC)\b",
                sentence,
                re.IGNORECASE,
            ))

            # Dollar values are rounded in generated prose.
            tolerance = max(0.15, 0.02 * abs(actual))

            if abs(claimed - actual) > tolerance:
                add(
                    "EBITDA_VALUE_MISMATCH",
                    m.group(0),
                    "needs_review" if possible_peer_context
                    else "contradicted",
                    "Claimed EBITDA differs from NVDA CSV. "
                    "Check company and metric attribution.",
                    year=year,
                    claimed_billions=claimed,
                    csv_ebitda_billions=round(actual, 4),
                    difference_billions=round(
                        claimed - actual, 4
                    ),
                )

            if (
                revenue is not None
                and claimed > revenue * 1.02
            ):
                add(
                    "AMOUNT_EXCEEDS_REVENUE",
                    m.group(0),
                    "needs_review" if possible_peer_context
                    else "contradicted",
                    "Claimed EBITDA exceeds corresponding "
                    "annual revenue; verify attribution.",
                    year=year,
                    claimed_billions=claimed,
                    revenue_billions=round(revenue, 4),
                )

    return {
        "verifier_version": "0.3",
        "units": "CSV in USD; claims converted to USD billions",
        "years_available": year_cols,
        "findings": findings,
        "finding_count": len(findings),
        "limitations": (
            "Heuristic numeric screening. Company and metric "
            "attribution may require manual review."
        ),
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

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("Financial Grounding Verifier v0.3")
    print("Units:", result["units"])
    print("Findings:", result["finding_count"])

    for i, item in enumerate(result["findings"], 1):
        print(f"\n[{i}] {item['rule']}")
        print("Status:", item["status"])
        print("Evidence:", item["evidence"])
        print("Reason:", item["reason"])

        if "csv_ebitda_billions" in item:
            print(
                "CSV EBITDA:",
                item["csv_ebitda_billions"],
                "billion USD",
            )

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
