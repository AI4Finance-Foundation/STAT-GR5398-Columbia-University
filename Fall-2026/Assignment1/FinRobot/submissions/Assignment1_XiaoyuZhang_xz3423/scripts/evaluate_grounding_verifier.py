import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "experiments/part3_verifier"
OUT.mkdir(parents=True, exist_ok=True)

CSV = ROOT / (
    "output/NVDA/analysis/"
    "financial_metrics_and_forecasts.csv"
)

SOURCE = ROOT / (
    "finrobot_equity/core/src/"
    "financial_grounding_verifier.py"
)

BASELINE = SOURCE.with_suffix(".v01.py")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not BASELINE.exists():
    raise FileNotFoundError(
        f"Baseline missing: {BASELINE}. "
        "Do not reconstruct v0.1 from memory."
    )

if not CSV.exists():
    raise FileNotFoundError(CSV)


def load_verifier(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not callable(getattr(module, "verify", None)):
        raise RuntimeError(
            f"{path} has no verify(text, csv_path) function"
        )
    return module.verify


v01 = load_verifier(BASELINE, "verifier_v01")
v02 = load_verifier(SOURCE, "verifier_v02")

# Gold label:
# 1 = Contains at least one known grounding error
# 0 = No error under the metric/year/unit rules tested.
#
# All financial values refer to the frozen NVDA CSV.
#
# The constructed cases test detection and false alarms.
# They are NOT claimed to be original model outputs.

cases = [
    {
        "id": "S01",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA was $144.6 billion "
            "in 2026A."
        ),
        "reason": "Correct rounded historical EBITDA.",
    },
    {
        "id": "S02",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA is $183.7 billion "
            "in 2029E."
        ),
        "reason": "Correct rounded forecast EBITDA.",
    },
    {
        "id": "S03",
        "kind": "constructed",
        "gold": 1,
        "text": (
            "NVIDIA EBITDA is projected to reach "
            "$382 billion by 2027E, compared "
            "with AMD and Intel."
        ),
        "reason": (
            "NVDA 2027E model EBITDA is "
            "approximately $159.85B, not $382B."
        ),
    },
    {
        "id": "S04",
        "kind": "constructed",
        "gold": 1,
        "text": (
            "NVIDIA EBITDA was an estimated "
            "$144.55 billion in 2026A."
        ),
        "reason": (
            "2026A is historical Actual, "
            "not an estimate."
        ),
    },
    {
        "id": "S05",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA was $11.35 billion "
            "in 2022A."
        ),
        "reason": "Correct rounded historical EBITDA.",
    },
    {
        "id": "S06",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA is projected at "
            "$159.85 billion for 2027E."
        ),
        "reason": "Correct forecast EBITDA.",
    },
    {
        "id": "S07",
        "kind": "constructed",
        "gold": 1,
        "text": (
            "NVIDIA EBITDA will reach "
            "$200 billion in 2027E."
        ),
        "reason": "Wrong EBITDA amount.",
    },
    {
        "id": "S08",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA revenue was about "
            "$215.9 billion in 2026A."
        ),
        "reason": "Correct rounded revenue.",
    },
    {
        "id": "S09",
        "kind": "constructed",
        "gold": 1,
        "text": (
            "NVIDIA EBITDA reached "
            "$400 billion in 2027E."
        ),
        "reason": (
            "Wrong EBITDA amount, "
            "above modeled annual revenue."
        ),
    },
    {
        "id": "S10",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA was "
            "$144.55 billion in 2026A."
        ),
        "reason": "Correct historical EBITDA.",
    },
    {
        "id": "S11",
        "kind": "constructed",
        "gold": 1,
        "text": (
            "NVIDIA EBITDA was projected "
            "at $144.55 billion in 2026A."
        ),
        "reason": "Actual year described as projected.",
    },
    {
        "id": "S12",
        "kind": "constructed",
        "gold": 0,
        "text": (
            "NVIDIA EBITDA margin was "
            "66.9% in 2026A."
        ),
        "reason": "Correct rounded EBITDA margin.",
    },
]

# Two previously generated real AI sections.
# Their errors have already been manually identified.
real_cases = [
    (
        "R01",
        "valuation_overview_after.txt",
        "Contains the incorrect NVDA 2027E $382B claim.",
    ),
    (
        "R02",
        "competitor_analysis_after.txt",
        "Describes 2026A EBITDA as estimated.",
    ),
]

real_folder = ROOT / "experiments/nvda_grounding_test"

for case_id, filename, reason in real_cases:
    path = real_folder / filename
    if path.exists():
        cases.append({
            "id": case_id,
            "kind": "real_agent_output",
            "gold": 1,
            "text": path.read_text(encoding="utf-8"),
            "reason": reason,
        })
    else:
        print("WARNING: Real passage not found:", path)

results = []

for case in cases:
    row = {
        "case_id": case["id"],
        "case_type": case["kind"],
        "gold_error": case["gold"],
        "gold_reason": case["reason"],
        "text": case["text"],
    }

    for version, verifier in [
        ("v01", v01),
        ("v02", v02),
    ]:
        try:
            report = verifier(
                case["text"], str(CSV)
            )
            findings = report.get("findings", [])

            # Passage-level binary decision:
            # One or more findings = detected error.
            # Multiple findings for the same claim
            # do not count as multiple true positives.
            row[f"{version}_detected"] = (
                1 if findings else 0
            )
            row[f"{version}_finding_count"] = len(findings)
            row[f"{version}_rules"] = "; ".join(
                sorted({
                    f.get("rule", "UNKNOWN")
                    for f in findings
                })
            )
            row[f"{version}_error"] = ""
        except Exception as e:
            row[f"{version}_detected"] = None
            row[f"{version}_finding_count"] = None
            row[f"{version}_rules"] = ""
            row[f"{version}_error"] = repr(e)

    results.append(row)

detail = pd.DataFrame(results)
detail_path = OUT / "verifier_case_results.csv"
detail.to_csv(detail_path, index=False)

if (
    detail[["v01_detected", "v02_detected"]]
    .isna()
    .any()
    .any()
):
    print(detail[
        detail["v01_error"].ne("")
        | detail["v02_error"].ne("")
    ][[
        "case_id", "v01_error", "v02_error"
    ]].to_string(index=False))
    raise RuntimeError(
        "Some cases raised errors. "
        "Results saved for inspection; "
        "no scores reported."
    )


def evaluate(df, col):
    gold = df["gold_error"].astype(int)
    pred = df[col].astype(int)

    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


summaries = []

for subset_name, subset in [
    ("constructed", detail[
        detail["case_type"] == "constructed"
    ]),
    ("all_available", detail),
]:
    for version in ["v01", "v02"]:
        metrics = evaluate(
            subset, f"{version}_detected"
        )
        summaries.append({
            "subset": subset_name,
            "version": version,
            "n_cases": len(subset),
            **metrics,
        })

summary = pd.DataFrame(summaries)
summary_path = OUT / "verifier_metrics.csv"
summary.to_csv(summary_path, index=False)

metadata = {
    "evaluation_unit": "one test passage",
    "positive_label": (
        "At least one manually identified "
        "financial grounding error"
    ),
    "negative_label": (
        "No known error under tested rules"
    ),
    "limitations": [
        "Small manually curated benchmark.",
        "Constructed examples are not natural model outputs.",
        "Two original agent sections are included when found.",
        "This does not measure full-report accuracy.",
        "Verifier alerts requiring review count as detections.",
        "Source data and gold labels should be independently reviewed.",
    ],
}

(OUT / "evaluation_method.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\n========== VERIFIER CASE RESULTS ==========")
print(detail[[
    "case_id", "case_type", "gold_error",
    "v01_detected", "v02_detected",
    "v01_finding_count", "v02_finding_count",
]].to_string(index=False))

print("\n========== PRECISION / RECALL / F1 ==========")
print(summary.round(3).to_string(index=False))

print("\nSaved:")
print(detail_path)
print(summary_path)
print(OUT / "evaluation_method.json")

print("\nPART 3 EVALUATION COMPLETE")
