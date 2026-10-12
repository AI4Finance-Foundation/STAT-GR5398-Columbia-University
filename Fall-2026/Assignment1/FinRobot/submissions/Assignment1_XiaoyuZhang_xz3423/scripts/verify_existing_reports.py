import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(
    0, str(ROOT / "finrobot_equity/core/src")
)

from financial_grounding_verifier import verify

OUT = ROOT / "experiments/part3_verifier/real_reports"
OUT.mkdir(parents=True, exist_ok=True)

sections = [
    "tagline",
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
    "major_takeaways",
]

summary = []

for ticker in ["NVDA", "AMD", "AVGO"]:
    folder = ROOT / "output" / ticker / "analysis"
    csv = folder / "financial_metrics_and_forecasts.csv"

    if not csv.exists():
        print("MISSING CSV:", ticker)
        continue

    for section in sections:
        text_path = folder / f"{section}.txt"
        if not text_path.exists():
            print("MISSING TEXT:", ticker, section)
            continue

        text = text_path.read_text(encoding="utf-8")

        try:
            report = verify(text, str(csv))
            findings = report.get("findings", [])

            output_path = OUT / f"{ticker}_{section}.json"
            output_path.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )

            summary.append({
                "ticker": ticker,
                "section": section,
                "findings": len(findings),
                "status": "CHECKED",
            })
        except Exception as exc:
            summary.append({
                "ticker": ticker,
                "section": section,
                "findings": None,
                "status": f"ERROR: {exc}",
            })

import pandas as pd

df = pd.DataFrame(summary)
df.to_csv(
    OUT / "real_report_verification_summary.csv",
    index=False,
)

print("\n========== REAL REPORT VERIFICATION ==========")
print(df.to_string(index=False))
print("\nTotal findings:", df["findings"].fillna(0).sum())
print("Sections checked:", (df["status"] == "CHECKED").sum())
print("Errors:", (df["status"] != "CHECKED").sum())
print("\nREAL REPORT VERIFICATION COMPLETE")
