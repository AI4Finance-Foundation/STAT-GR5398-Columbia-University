# GR5398 Assignment 1 — FinRobot Equity Research Agent

**Student:** Xiaoyu Zhang  
**Term:** Fall 2026  
**Universe:** NVDA, AMD, INTC, AVGO, QCOM

## Deliverables

- `research_report.pdf` — integrated English research paper.
- `research_report.md` — editable source for the paper.
- `medium_article.md` — public-facing, Medium-style write-up.
- `reports/` — the five generated HTML equity reports.
- `experiments/financials/` — company financial and projection tables.
- `experiments/part2A/` — FY2025 SEC Company Facts reconciliation.
- `experiments/part2B/` — AVGO/QCOM DCF-proxy scenarios and sensitivity tables.
- `experiments/part2C/` — the 24-claim preliminary grounding review.
- `experiments/part2D/` — NVDA repeated-generation outputs and manifest.
- `experiments/part3/` — financial verifier evaluation data, metrics, and methodology.
- `scripts/` — supporting scripts and verifier v0.1 / v0.2.

## Key experimental findings

| Experiment | Result | Important limitation |
|---|---|---|
| Five-company workflow | Five HTML reports, 25/25 internal report checks | HTML completeness does not guarantee financial accuracy |
| SEC financial reconciliation | 22/22 matches (AVGO and QCOM) | Shared underlying XBRL source, not an independent 10-K audit |
| AVGO DCF proxy | $371.348B base; $277.80B–$575.56B grid | Not an equity value or per-share target |
| QCOM DCF proxy | $176.885B base; $132.33B–$274.16B grid | Not an equity value or per-share target |
| Preliminary grounding audit | 17 supported / 4 unsupported / 3 contradicted | 24 manually reviewed compound claims; not independently adjudicated |
| Three-run NVDA stability | Three recurring risk themes; recurring A/E wording issue | No three comparable independent price targets |
| Verifier (14 curated passages) | F1: v0.1 0.667 → v0.2 1.000 | Small targeted benchmark; no generalization claim |

## Data provenance

Financial Modeling Prep (FMP) data were used where available. Broadcom and Qualcomm FY2025 financial inputs were drawn from locally cached SEC EDGAR Company Facts. Original SEC accessions: AVGO `0001730168-25-000121`, QCOM `0000804328-25-000085`. Derived FCF and EBITDA are explicitly **proxies**, not asserted to be issuer-reported non-GAAP figures. Forecast assumptions are illustrative.

## Reproduction and dependencies

The repository preserves analysis outputs and the relevant scripts. It is a **lightweight submission**, not a completely self-contained reimplementation: the scripts were originally executed from the larger FinRobot source tree, and some require previously downloaded SEC JSON, FMP inputs, or the existing module layout.

Example commands from the original FinRobot root with its virtual environment activated:

```bash
python scripts/audit_sec_financials.py
python scripts/independent_dcf.py
python scripts/prepare_grounding_audit.py
python scripts/evaluate_grounding_verifier.py
```

The stability script uses the original FinRobot text-generation module and an OpenAI API key; it is not rerun by default. See individual scripts and their generated output files for experiment details.

## Responsible use

The output is an educational evaluation of LLM-assisted equity research, not investment advice. The tested verifier is rule-based and does not independently validate every statement, filing definition, or market assumption. No private API keys, credentials, or virtual environment files are included in this submission.


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

`medium_article.md` is a draft for publication. Published Medium article: https://medium.com/@xz3423/can-an-ai-equity-research-agent-be-trusted-1f2dd9a9fbae
