from pathlib import Path
import subprocess
import importlib.util
import json
import csv

root = Path.cwd()
src = root / "finrobot_equity/core/src"
out = root / "experiments/part3_verifier"
out.mkdir(parents=True, exist_ok=True)

pipeline = src / "generate_financial_analysis.py"
offline = root / "scripts/verify_existing_reports.py"
v03 = src / "financial_grounding_verifier.v03.py"

assert v03.exists(), "Missing v0.3 verifier"
assert pipeline.exists(), "Missing report generation pipeline"

# Preserve prior working versions and update only the verifier import.
original = pipeline.read_text(encoding="utf-8")
old_import = (
    "from financial_grounding_verifier "
    "import verify as _verify_grounding"
)
new_import = (
    "import importlib.util as _grounding_importlib\n"
    "                    _verifier_source = "
    "_GroundingPath(__file__).resolve().parent / "
    "'financial_grounding_verifier.v03.py'\n"
    "                    _spec = _grounding_importlib.spec_from_file_location("
    "'financial_grounding_verifier_v03', _verifier_source)\n"
    "                    _module = _grounding_importlib.module_from_spec(_spec)\n"
    "                    _spec.loader.exec_module(_module)\n"
    "                    _verify_grounding = _module.verify"
)

# Detect indentation from original import line to make replacement safe.
if old_import in original:
    lines = original.splitlines(keepends=True)
    updated_lines = []
    for line in lines:
        if old_import in line:
            indent = line[:len(line)-len(line.lstrip())]
            replacement_lines = new_import.splitlines()
            updated_lines.extend(
                indent + item.lstrip() + "\n"
                for item in replacement_lines
            )
        else:
            updated_lines.append(line)
    updated = "".join(updated_lines)
    compile(updated, str(pipeline), "exec")
    pipeline.write_text(updated, encoding="utf-8")
elif "financial_grounding_verifier.v03.py" in original:
    print("Pipeline already references v0.3")
else:
    raise RuntimeError(
        "Could not find expected v0.2 integration import. "
        "No pipeline changes made."
    )

# Run an independent offline v0.3 evaluation over existing text.
spec = importlib.util.spec_from_file_location("verifier_v03", v03)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

rows = []
for ticker in ["NVDA", "AMD", "AVGO"]:
    folder = root / "output" / ticker / "analysis"
    financial_csv = folder / "financial_metrics_and_forecasts.csv"
    for section in [
        "tagline", "company_overview", "investment_overview",
        "valuation_overview", "risks",
        "competitor_analysis", "major_takeaways",
    ]:
        text_path = folder / f"{section}.txt"
        if not financial_csv.exists() or not text_path.exists():
            raise FileNotFoundError(f"{ticker}/{section}: missing input")

        result = module.verify(
            text_path.read_text(encoding="utf-8"),
            str(financial_csv)
        )
        report_path = out / "real_reports" / f"{ticker}_{section}_v03.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        rows.append({
            "ticker": ticker,
            "section": section,
            "finding_count": result["finding_count"],
            "version": result["verifier_version"],
        })

summary_path = out / "real_report_v03_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Generate patch with tracked changes plus separately included new files.
patches = []
changed = subprocess.run(
    ["git", "diff", "--",
     "finrobot_equity/core/src/generate_financial_analysis.py",
     "finrobot_equity/core/src/create_equity_report.py",
     "finrobot_equity/core/src/modules/market_data_api.py",
     "finrobot_equity/core/src/modules/text_generator_agents.py"],
    capture_output=True, text=True, check=True
)
patches.append(changed.stdout)

for relative in [
    "finrobot_equity/core/src/financial_grounding_verifier.py",
    "finrobot_equity/core/src/financial_grounding_verifier.v03.py",
]:
    path = root / relative
    if not path.exists():
        continue

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        capture_output=True
    ).returncode == 0

    if tracked:
        result = subprocess.run(
            ["git", "diff", "--", relative],
            capture_output=True, text=True, check=True
        )
        patches.append(result.stdout)
    else:
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", "/dev/null", relative],
            capture_output=True, text=True
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        patches.append(result.stdout)

patch_path = out / "finrobot_improvement_final.patch"
patch_path.write_text(
    "\n".join(x for x in patches if x.strip()),
    encoding="utf-8"
)

print("\n========== PART 3 FINALIZATION ==========")
print("Pipeline v0.3: configured")
print("Actual report sections checked:", len(rows))
print("v0.3 total findings:", sum(x["finding_count"] for x in rows))
print("Patch:", patch_path)
print("Patch size:", patch_path.stat().st_size)
print("Diff sections:", patch_path.read_text().count("diff --git"))
print("Summary:", summary_path)
print("PART 3 FINALIZATION COMPLETE")
