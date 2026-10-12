from pathlib import Path
import shutil

root = Path.cwd()

report = root / "research_report.md"
readme = root / "README.md"
blog = root / "medium_article.md"

for path in (report, readme, blog):
    shutil.copy2(path, path.with_suffix(path.suffix + ".backup"))

part2b = """
## Updated Part 2B — Native Valuation Engine Reconciliation

In addition to the illustrative AVGO/QCOM DCF scenarios described above,
I independently reproduced NVIDIA's valuation using the original
`ValuationEngine.calculate_ev_ebitda_valuation()` implementation.

The engine selected forecast-year 2029E EBITDA of $183.7151 billion.
Using an explicitly specified 12.0x EV/EBITDA multiple, illustrative
shares outstanding of 24.5 billion, and the engine's assumption that
net debt equals 10% of enterprise value, the native engine returned
an implied value of **$80.9846 per share**.

The independent arithmetic was:

- Enterprise value = $183.7151B x 12 = approximately $2,204.5815B.
- Imputed net debt = 10% of enterprise value.
- Equity value = 90% of enterprise value.
- Implied price = equity value / 24.5B shares = $80.9846.

The reconciliation difference was **$0.00**.

I also ran the original `SensitivityAnalyzer` using revenue
perturbations of +/-5% and EBITDA margin changes of +/-2 percentage
points. Its operating scenarios were translated into per-share
values with the same EV/EBITDA pricing rule. Across 25 joint
scenarios, the implied prices ranged from **$74.8419 to $87.3477**,
or approximately -7.59% to +7.86% relative to the engine baseline.

These values are illustrative model outputs, **not verified price
targets**. Share count and net debt are modeling assumptions.
The engine's constant valuation multiple, simplified balance-sheet
adjustment, and default forecast-year selection remain important
limitations. The native sensitivity module generates operating
scenarios, not share-price estimates by itself; price translation
was an additional deterministic calculation.

Reproduction materials:
`experiments/part2_valuation/native_engine_reconciliation/`
and `scripts/reconcile_native_valuation.py`.

## Updated Part 2C — Full-Candidate Grounding Audit Status

An expanded extraction identified **206 candidate statements** across
NVDA, AMD, and AVGO, covering seven substantive generated sections
per company. News placeholders lacking substantive news claims
were treated as not applicable rather than automatically Supported.

A subsequent evidence-based adjudication assigned final labels to
**19 statements**: 13 Supported, 1 Unsupported, and 5 Contradicted.
The remaining **187 statements remain Pending Review**.

The earlier 24-claim preliminary sample and its 70.8% supported
share are retained as a separate exploratory exercise.
**Neither 13/19 nor 17/24 is a valid grounding rate for all
206 candidate statements.** A complete section-level grounding
rate requires adjudicating the remaining claims.

The audit highlights three mechanisms of unreliable text:
misattribution of financial figures, confusion between historical
Actual and forecast Estimate years, and qualitative causal claims
not established by the agent's evidence tables.

## Updated Part 3 — Verifier v0.3 and Real-Report Evaluation

The final improvement is an opt-in, post-generation, rule-based
Financial Grounding Verifier integrated into the report generation
pipeline. It generates JSON findings while retaining the original
LLM output. It does not automatically rewrite the report text.

Version 0.3 adds local financial-metric attribution to avoid
comparing explicitly labeled Revenue or Cost of Operations amounts
against EBITDA. It also introduces a targeted historical-versus-
forecast wording check.

On the same curated 14-passage benchmark:

| Version | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.1 | 6 | 5 | 1 | 2 | 0.5455 | 0.8571 | 0.6667 |
| v0.2 | 7 | 0 | 0 | 7 | 1.0000 | 1.0000 | 1.0000 |
| v0.3 | 7 | 0 | 0 | 7 | 1.0000 | 1.0000 | 1.0000 |

On seven original NVDA report sections, v0.2 raised three
EBITDA-mismatch alerts that manual review identified as
metric-attribution false positives: one Revenue amount and two
Cost of Operations amounts. Version 0.3 eliminated all three
without losing detection of the constructed $382B 2027E
EBITDA error.

A separate offline run completed successfully across **21 original
sections** from NVDA, AMD, and AVGO. Version 0.3 returned
zero findings, which means **no errors were flagged by the current
rules**, not that every statement was supported or correct.

This is a measured improvement in screening specificity, not
a demonstrated improvement in whole-report Grounding Rate.
Automated flagging does not change original text, and a complete
before/after claim-grounding comparison remains unavailable.

Reproduction materials:
`experiments/part3_verifier/`,
`finrobot_equity/core/src/financial_grounding_verifier.v03.py`,
and `experiments/part3_verifier/finrobot_improvement_final.patch`.

## Setup, Runtime and Cost Disclosure

The workflow used OpenAI-backed text generation and a combination
of FMP and cached SEC data. The exact model identifier, token usage,
per-company elapsed runtime, API expenditure, and total spend
have not been independently reconstructed from complete execution
logs. They are therefore **not numerically reported** rather than
being assigned invented values. Reproduction requires recording
these quantities during future runs.
"""

readme_update = """
## Final experimental update — October 12, 2026

- **Native valuation reconciliation:** NVDA 2029E EBITDA
  $183.7151B; 12.0x EV/EBITDA; illustrative price $80.9846;
  hand-calculation difference $0.00.
- **Native sensitivity analysis:** 25 scenarios spanning revenue
  +/-5% and margin +/-2 percentage points; implied prices
  $74.8419-$87.3477.
- **Expanded grounding audit:** 206 candidate claims; 19 finally
  adjudicated, 187 pending. No complete full-sample Grounding
  Rate is claimed.
- **Verifier v0.3:** curated 14-passage F1 = 1.000; three
  confirmed v0.2 false alerts eliminated in NVDA;
  21 real sections screened with zero v0.3 alerts.
- **Interpretation:** zero alerts is not proof of complete
  grounding. Flagging mode does not automatically improve
  the original report's grounding rate.

### Reproduce the new results

Run from the original FinRobot source checkout with its
environment and inputs available:

```bash
python scripts/reconcile_native_valuation.py
python scripts/evaluate_verifier_v03.py
python scripts/compare_verifier_v03.py
```

See `experiments/part2_valuation/native_engine_reconciliation/`
and `experiments/part3_verifier/` for machine-readable results.

### Public article

`medium_article.md` is a draft for publication. A published
Medium URL has not been provided or verified.
"""

blog_update = """
## Update: Checking the original valuation engine

I later reproduced FinRobot's own EV/EBITDA formula for
NVIDIA using 2029E EBITDA of $183.7151 billion, an illustrative
12x multiple, an assumed 10% enterprise-value adjustment for
net debt, and 24.5 billion assumed shares. Both the native
engine and an independent hand calculation returned $80.9846
per share. Across 25 revenue/margin sensitivity scenarios,
the illustrative values ranged from $74.84 to $87.35.

The verifier also underwent another iteration. Version 0.3
resolved three false EBITDA alerts found in a real NVIDIA
report, while preserving detection of the known $382B
EBITDA discrepancy in a targeted test. These improvements
show why independently reviewing a detector's own errors
is as important as building the detector.

The expanded extraction found 206 candidate statements,
but only 19 have received subsequent final adjudication.
I do not present an overall grounding rate for the 206.
"""

marker = "## Updated Part 2B — Native Valuation Engine Reconciliation"

text = report.read_text(encoding="utf-8")
if marker not in text:
    location = "## 9. Synthesis and Limitations"
    if location in text:
        text = text.replace(location, part2b + "\n" + location, 1)
    else:
        text += "\n" + part2b
    report.write_text(text, encoding="utf-8")
    print("UPDATED research_report.md")
else:
    print("SKIPPED research_report.md: already updated")

for path, section, signature in [
    (readme, readme_update, "## Final experimental update"),
    (blog, blog_update, "## Update: Checking the original valuation engine"),
]:
    text = path.read_text(encoding="utf-8")
    if signature not in text:
        path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
        print("UPDATED", path.name)
    else:
        print("SKIPPED", path.name, ": already updated")

print("FINAL DOCUMENT SOURCES UPDATED")
