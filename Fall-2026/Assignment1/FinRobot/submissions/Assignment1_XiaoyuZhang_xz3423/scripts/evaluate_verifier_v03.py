import importlib.util
from pathlib import Path
import pandas as pd

root = Path.cwd()
folder = root / "experiments/part3_verifier"
src = root / "finrobot_equity/core/src"

cases = pd.read_csv(folder / "verifier_case_results.csv")
csv_path = root / "output/NVDA/analysis/financial_metrics_and_forecasts.csv"

required = ["case_id", "text", "gold_error", "v01_detected", "v02_detected"]
missing = [c for c in required if c not in cases.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

path = src / "financial_grounding_verifier.v03.py"
spec = importlib.util.spec_from_file_location("verifier_v03", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

predictions = []
rule_names = []

for _, row in cases.iterrows():
    result = module.verify(str(row["text"]), str(csv_path))
    findings = result.get("findings", [])
    predictions.append(int(len(findings) > 0))
    rule_names.append("; ".join(sorted({
        item["rule"] for item in findings
    })))

cases["v03_detected"] = predictions
cases["v03_rules"] = rule_names

def score(version):
    gold = cases["gold_error"].astype(int)
    pred = cases[f"{version}_detected"].astype(int)

    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())

    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0

    return {
        "version": version,
        "n": len(cases),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }

summary = pd.DataFrame([
    score("v01"), score("v02"), score("v03")
])

cases.to_csv(
    folder / "verifier_v01_v02_v03_cases.csv",
    index=False
)
summary.to_csv(
    folder / "verifier_v01_v02_v03_metrics.csv",
    index=False
)

print("========== THREE VERSION COMPARISON ==========")
print(summary.to_string(index=False))

print("\n========== V0.3 MISCLASSIFICATIONS ==========")
errors = cases[
    cases["gold_error"].astype(int) != cases["v03_detected"]
]
if errors.empty:
    print("No errors on this curated benchmark")
else:
    print(errors[
        ["case_id", "gold_error", "v03_detected", "v03_rules"]
    ].to_string(index=False))

print("\nVERIFIER THREE-VERSION EVALUATION COMPLETE")
