from pathlib import Path
import re
import pandas as pd

ROOT = Path("output")
OUT = Path("experiments/part2_grounding_audit")
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["NVDA", "AMD", "AVGO"]

SECTIONS = [
    "tagline",
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
    "major_takeaways",
]

rows = []


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()

    # Keep decimal values like 159.85 together.
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            text,
        )
        if len(sentence.strip()) >= 35
    ]


for ticker in TICKERS:
    print(f"\n========== {ticker} ==========")

    candidates = {}

    for section in SECTIONS:
        path = ROOT / ticker / "analysis" / f"{section}.txt"

        if not path.exists():
            raise FileNotFoundError(path)

        sentences = split_sentences(
            path.read_text(encoding="utf-8")
        )

        # Prefer claims with measurable financial evidence.
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (
                bool(re.search(
                    r"\b20\d{2}[AE]?\b"
                    r"|\$[\d,.]+"
                    r"|\d+(?:\.\d+)?\s*%",
                    item[1]
                )),
                bool(re.search(
                    r"revenue|EBITDA|margin|"
                    r"cash flow|growth|valuation",
                    item[1],
                    flags=re.IGNORECASE,
                )),
                -item[0],
            ),
            reverse=True,
        )

        candidates[section] = [
            sentence for _, sentence in ranked
        ]

        if not ranked:
            raise RuntimeError(
                f"No eligible claim: {ticker}/{section}"
            )

    selected = []
    used = set()

    # Select one claim from each of seven sections.
    for section in SECTIONS:
        claim = next(
            (s for s in candidates[section] if s not in used),
            None,
        )

        if claim is None:
            raise RuntimeError(
                f"No unique claim for {ticker}/{section}"
            )

        selected.append((section, claim))
        used.add(claim)

    # Select an additional valuation claim.
    extra = next(
        (
            s for s in candidates["valuation_overview"]
            if s not in used
        ),
        None,
    )

    if extra is None:
        raise RuntimeError(
            f"No additional valuation claim for {ticker}"
        )

    selected.append(("valuation_overview", extra))

    for i, (section, claim) in enumerate(selected, 1):
        claim_id = f"{ticker}_{i:02d}"

        rows.append({
            "claim_id": claim_id,
            "ticker": ticker,
            "section": section,
            "claim_text": claim,
            "evidence_source": "",
            "evidence_value_or_quote": "",
            "label": "",
            "reason": "",
            "reviewer": "manual",
        })

        print(f"\n{claim_id} [{section}]")
        print(claim)


audit = pd.DataFrame(rows)

if len(audit) != 24:
    raise RuntimeError(
        f"Expected 24 claims, got {len(audit)}"
    )

output = OUT / "grounding_audit_template.csv"
audit.to_csv(output, index=False)

print("\n========== EXTRACTION SUMMARY ==========")
print(audit.groupby("ticker").size().to_string())
print("Total claims:", len(audit))
print("Saved:", output)

print("\nPART 2C CANDIDATE EXTRACTION COMPLETE")
