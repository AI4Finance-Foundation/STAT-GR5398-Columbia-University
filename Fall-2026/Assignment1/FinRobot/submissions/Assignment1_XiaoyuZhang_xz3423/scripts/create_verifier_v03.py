from pathlib import Path

root = Path("finrobot_equity/core/src")
old = root / "financial_grounding_verifier.py"
new = root / "financial_grounding_verifier.v03.py"

source = old.read_text(encoding="utf-8")

# Prevent accidentally constructing v0.3 from a newer version.
if '"verifier_version": "0.2"' not in source:
    raise RuntimeError("Source is not v0.2. No changes made.")

# 1. Identify the metric immediately preceding a dollar claim.
# This intentionally limits the verifier to explicitly attributed
# EBITDA amounts. Other metrics are left for separate rules.
helper = '''
METRIC_LABEL = re.compile(
    r"\\b(?:"
    r"cost of operations|cost of revenue|"
    r"operating cash flow|contribution profit|"
    r"gross profit|net income|"
    r"revenue|sales|EBITDA|"
    r"EPS|SG&A"
    r")\\b",
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
    r"\\bprojected\\b[^.!?\\\\n]{0,200}?"
    r"\\bfrom\\s+\\$[\\d,.]+\\s*(?:B|M|billion|million)"
    r"\\s+in\\s+20\\d{2}A"
    r"[^.!?\\\\n]{0,100}?"
    r"\\bto\\s+\\$[\\d,.]+\\s*(?:B|M|billion|million)"
    r"\\s+in\\s+20\\d{2}A\\b",
    re.IGNORECASE,
)

'''

anchor = "\ndef verify(text, csv_path):"
if source.count(anchor) != 1:
    raise RuntimeError("Cannot locate verify() safely.")

source = source.replace(anchor, "\n" + helper + anchor, 1)

# 2. Add a specific historical-trend wording rule.
anchor = "    # 2. Split paragraphs into approximate sentences."
addition = '''    # Actual-to-actual change described as a projection.
    for m in HISTORICAL_TREND_FORECAST.finditer(text):
        add(
            "HISTORICAL_TREND_CALLED_FORECAST",
            m.group(0),
            "contradicted",
            "Both endpoints are labeled Actual (A), "
            "but the change is described as projected.",
        )

'''

if source.count(anchor) != 1:
    raise RuntimeError("Cannot locate sentence-splitting anchor.")

source = source.replace(anchor, addition + anchor, 1)

# 3. Require a local EBITDA attribution for each dollar value.
anchor = "            # Avoid comparing clearly attributed peer amounts"
guard = '''            metric = attributed_metric(prefix)

            # Do not compare Revenue, Cost of Operations,
            # EPS, etc. against EBITDA.
            if metric != "ebitda":
                continue

'''

if source.count(anchor) != 1:
    raise RuntimeError("Cannot locate metric attribution anchor.")

source = source.replace(anchor, guard + anchor, 1)

source = source.replace(
    '"verifier_version": "0.2"',
    '"verifier_version": "0.3"',
    1,
)

source = source.replace(
    "Financial Grounding Verifier v0.2",
    "Financial Grounding Verifier v0.3",
    1,
)

compile(source, str(new), "exec")
new.write_text(source, encoding="utf-8")

print("PASS - v0.2 preserved")
print("PASS - v0.3 created")
print("PASS - Metric attribution added")
print("PASS - Historical trend wording rule added")
print("PASS - Python syntax validated")
print("Created:", new)
print("VERIFIER V0.3 CREATED")
