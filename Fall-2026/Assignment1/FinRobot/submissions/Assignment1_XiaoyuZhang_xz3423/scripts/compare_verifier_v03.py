import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "finrobot_equity/core/src"
OUT = ROOT / "experiments/part3_verifier"
OUT.mkdir(parents=True, exist_ok=True)

CSV = (
    ROOT / "output/NVDA/analysis/"
    "financial_metrics_and_forecasts.csv"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


v02 = load(
    SRC / "financial_grounding_verifier.py", "v02"
)
v03 = load(
    SRC / "financial_grounding_verifier.v03.py", "v03"
)

cases = []

# Real report sections: no additional LLM calls.
for section in [
    "tagline",
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
    "major_takeaways",
]:
    path = ROOT / f"output/NVDA/analysis/{section}.txt"
    if not path.exists():
        continue

    cases.append((
        section,
        path.read_text(encoding="utf-8")
    ))

# Three focused regression examples.
cases += [
    (
        "correct_revenue",
        "NVIDIA revenue was $215.9 billion in 2026A. "
        "EBITDA margin also expanded."
    ),
    (
        "wrong_ebitda",
        "NVIDIA EBITDA is projected at "
        "$382 billion in 2027E."
    ),
    (
        "correct_ebitda",
        "NVIDIA EBITDA was $144.6 billion in 2026A."
    ),
]

records = []

for name, text in cases:
    r02 = v02(text, str(CSV))
    r03 = v03(text, str(CSV))

    records.append({
        "case": name,
        "v02_findings": len(r02["findings"]),
        "v03_findings": len(r03["findings"]),
        "v03_rules": "; ".join(
            x["rule"] for x in r03["findings"]
        ),
    })

    if name in [
        "tagline", "risks",
        "correct_revenue", "wrong_ebitda"
    ]:
        print(f"\n========== {name} ==========")
        print("v0.2:", [
            (x["rule"], x["evidence"])
            for x in r02["findings"]
        ])
        print("v0.3:", [
            (x["rule"], x["evidence"])
            for x in r03["findings"]
        ])

df = pd.DataFrame(records)
df.to_csv(
    OUT / "verifier_v02_v03_real_report_comparison.csv",
    index=False,
)

print("\n========== COMPARISON ==========")
print(df.to_string(index=False))

print("\nV0.2 total findings:", df["v02_findings"].sum())
print("V0.3 total findings:", df["v03_findings"].sum())

print("\nVERIFIER V0.3 COMPARISON COMPLETE")
