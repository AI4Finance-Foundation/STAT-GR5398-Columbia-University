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
| Full candidate grounding audit | 113 Supported / 66 Unsupported / 27 Contradicted (206 total; 54.85%) | Preliminary sentence-level labels; seven substantive sections per company; news-summary placeholders treated as N/A |
| Three-run NVDA stability | Three recurring risk themes; recurring A/E wording issue | No three comparable independent price targets |
| Verifier (14 curated passages) | F1: v0.1 0.667 → v0.2 1.000 → v0.3 1.000 | Curated benchmark only; zero v0.3 alerts on 21 original sections does not imply whole-report correctness |

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


## Final Grading-Relevant Updates

### Final grounding audit status

All 206 candidate statements extracted from NVDA, AMD, and AVGO were reviewed at the sentence level.

| Company | Reviewed candidates | Supported | Unsupported | Contradicted | Grounding Rate |
|---|---:|---:|---:|---:|---:|
| AMD | 72 | 33 | 30 | 9 | 45.83% |
| AVGO | 67 | 49 | 11 | 7 | 73.13% |
| NVDA | 67 | 31 | 25 | 11 | 46.27% |
| **Overall** | **206** | **113** | **66** | **27** | **54.85%** |

Grounding Rate is defined here as:

**Supported candidate statements / Total reviewed candidate statements**

This is a preliminary sentence-level audit rather than an independently adjudicated atomic-claim gold standard.  
Compound statements were not always decomposed into minimal factual units, so the metric should be interpreted as a practical reliability indicator, not a definitive truth score.

### Section-level observation

The audit showed meaningful variation by section.  
`major_takeaways` was the strongest section overall, while `risks` was the weakest and contained the highest concentration of unsupported or weakly evidenced narrative claims.

The final section-level grounding table is included in:

- `experiments/part2_grounding_audit/final_section_grounding_rates.csv`

The full reviewed worklist is included in:

- `experiments/part2_grounding_audit/grounding_review_worklist.csv`

### Verifier update

The financial grounding verifier was improved through three versions:

- **v0.1**: F1 = 0.667
- **v0.2**: F1 = 1.000
- **v0.3**: F1 = 1.000

The curated benchmark contains 14 passages and is targeted rather than production-scale.  
Version 0.3 also removed three false EBITDA alerts from earlier NVDA report screening while preserving detection of the known wrong-EBITDA test case.

However, **zero alerts on 21 original report sections should not be interpreted as proof that the reports are fully grounded**.  
The verifier is a lightweight numerical screening tool, not a full semantic fact-checking system.

### Architecture note

This submission is based on a customized FinRobot equity-research workflow using section-specific prompt generation, structured financial tables, and a lightweight post-generation verifier.

It is **not claimed** that the submitted implementation exactly reproduces the full upstream multi-agent architecture described in the course prompt.  
The contribution here is an evaluated and improved workflow built on top of the FinRobot codebase and related modules.

### Reproducibility and limits

The repository includes:

- five generated HTML reports,
- SEC-based financial reconciliation outputs,
- independent DCF sensitivity experiments,
- native valuation engine reconciliation,
- the full 206-claim grounding audit results,
- stability-testing outputs,
- verifier evaluation artifacts,
- the final research report and Medium article draft.

Not all runtime metadata were captured consistently.  
Per-company wall-clock runtime, exact API token usage, and total API spending should therefore be treated as **not fully recorded**, rather than as zero.

### Published Medium article

Published Medium URL:

- https://medium.com/@xz3423/can-an-ai-equity-research-agent-be-trusted-1f2dd9a9fbae

