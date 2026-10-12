# Evaluating and Improving the Reliability of a FinRobot Equity Research Agent

**STAT GR5398 · Assignment 1 · Fall 2026**  
**Author:** Xiaoyu Zhang · **UNI:** xz3423  
**Coverage universe:** NVIDIA (NVDA), Advanced Micro Devices (AMD), Intel (INTC), Broadcom (AVGO), and Qualcomm (QCOM)  
**Sector:** Semiconductors

## 1. Project Overview

This project investigates whether LLM-assisted equity research is numerically reproducible, grounded in the evidence provided to the model, and stable across repeated generations. A customized FinRobot workflow produced five company research reports. The evaluation combines financial-record reconciliation, independent valuation checks, sentence-level grounding review, repeated-generation analysis, and a post-generation financial grounding verifier.

The central distinction is between **deterministic calculations** (which should be reproducible from the supplied data and assumptions) and **LLM-generated interpretations** (which must be checked against the evidence the model actually received). The findings assess research reliability, not investment performance.

## 2. Repository Contents

| Path | Contents |
| --- | --- |
| `research_report.pdf`, `research_report.md` | Full research paper and editable Markdown source |
| `reports/{TICKER}.html` | Five generated equity research reports |
| `output/{TICKER}/analysis/` | Financial tables, peer comparisons, generated agent narratives, and other available intermediate outputs |
| `experiments/part2_financial_audit/` | FY2025 SEC Company Facts reconciliation results |
| `experiments/part2_valuation/` | DCF proxy scenarios, sensitivity grids, and native-engine reconciliation |
| `experiments/part2_grounding_audit/` | Expanded 206-candidate grounding audit, evidence notes, and section-level results |
| `experiments/part2_stability/` | Three NVDA generations, saved responses, and qualitative stability analysis |
| `experiments/part3_verifier/` | Verifier evaluation outputs, real-report screening, and code patches |
| `finrobot_equity/core/src/` | Submitted modified generation and verifier code |
| `scripts/` | Audit, valuation, stability, and evaluation scripts |
| `medium_article.md` | Manuscript associated with the published Medium article |
| `submission_manifest.json` | Submission metadata |

Some additional folders (`experiments/part2A/`, `part2B/`, `part2C/`, `part2D/`, and `part3/`) preserve earlier experiment outputs. The expanded audit and v0.3 results in the paths above supersede the preliminary 24-claim review and earlier verifier summaries.

## 3. Experimental Results

### 3.1 Multi-Company Research Reports

The submitted coverage universe contains five semiconductor companies: NVDA, AMD, INTC, AVGO, and QCOM. Each has a generated HTML report and saved intermediate financial/analysis artifacts. Twenty-five internal report-generation checks passed (five per company). These checks establish artifact completeness and internal formatting/data checks, **not** the economic accuracy of every statement.

Financial Modeling Prep (FMP) was used where available. Because the available FMP Free access did not provide the required AVGO and QCOM statements, those two firms were processed using locally saved SEC EDGAR Company Facts. Forecast figures are scenario outputs, not management guidance or externally validated analyst forecasts.

### 3.2 Financial Data Audit

Eleven FY2025 values each for AVGO and QCOM were checked against selected SEC XBRL Company Facts, including directly mapped items and independently recomputed proxy metrics.

| Company | Directly mapped | Derived | Matched | Mismatched |
| --- | ---: | ---: | ---: | ---: |
| AVGO | 9 | 2 | 11 | 0 |
| QCOM | 9 | 2 | 11 | 0 |
| **Total** | **18** | **4** | **22** | **0** |

This **22/22** result shows consistency with the selected saved XBRL facts, not an independent audit of every original filing line. Free cash flow and EBITDA proxy definitions require care, especially when comparing across companies. See `experiments/part2_financial_audit/` and the paper for definitions and source limitations.

### 3.3 Valuation Reproduction and Sensitivity

Two illustrative five-year discounted cash-flow *proxy* scenarios start from FY2025 operating cash flow less capital expenditure, using a base 11% discount rate and 3% terminal growth:

| Company | Base present-value proxy | Sensitivity range |
| --- | ---: | ---: |
| AVGO | $371.348B | $277.80B–$575.56B |
| QCOM | $176.885B | $132.33B–$274.16B |

These values are **not** validated equity valuations or per-share targets: the exercise does not complete all balance-sheet adjustments and uses illustrative assumptions.

A separate independent reconciliation reproduces the native FinRobot EV/EBITDA valuation for NVDA: 2029E EBITDA of approximately $183.7151B, a specified 12.0× multiple, illustrative 24.5B shares, and the engine's 10% enterprise-value net-debt adjustment imply **$80.9846 per share**, with a **$0.00** arithmetic reconciliation difference. Across 25 revenue/margin scenarios, the implied values range from **$74.8419 to $87.3477**. These are illustrative outputs rather than investment price targets.

See `experiments/part2_valuation/native_engine_reconciliation/` and `scripts/reconcile_native_valuation.py`.

### 3.4 Evidence Grounding Audit

The expanded assessment assigned preliminary whole-sentence labels to **206 candidate statements** from seven substantive generated sections for each of NVDA, AMD, and AVGO. A statement is classified as **Supported** if its material content is backed by the saved model evidence, **Unsupported** if that evidence is insufficient, and **Contradicted** if the supplied evidence disagrees.

| Company | Candidates | Supported | Unsupported | Contradicted | Grounding rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMD | 72 | 33 | 30 | 9 | 45.83% |
| AVGO | 67 | 49 | 11 | 7 | 73.13% |
| NVDA | 67 | 31 | 25 | 11 | 46.27% |
| **Total** | **206** | **113** | **66** | **27** | **54.85%** |

**Grounding rate = Supported candidate statements / All reviewed candidate statements.** By section, `major_takeaways` had the highest rate (21/22, 95.45%), while `risks` had the lowest (3/17, 17.65%). The three `news_summary` outputs were no-news placeholders, so they are recorded as **N/A**, not as fully grounded eighth-section audits.

These are **preliminary sentence-level judgments**, not a fully atomized or independently adjudicated gold-standard annotation set. Compound sentences may contain a mixture of supported numbers and unsupported interpretations. The evidence boundary is the financial and peer tables actually provided to the customized text-generation workflow; it is not unrestricted outside knowledge.

Machine-readable results: `experiments/part2_grounding_audit/grounding_review_worklist.csv` and `final_section_grounding_rates.csv`.

### 3.5 Repeated-Generation Stability

Three NVDA runs reused a frozen financial snapshot and prompts, using **GPT-4.1-mini** with **temperature 0.7**, to generate investment-overview and risk text. Revenue-growth volatility, valuation-multiple risk, and SG&A-assumption uncertainty appeared in all three runs. Historical FY2026 actuals were also mischaracterized using estimate/forecast language in all three.

The outputs did **not** contain three independently computed ratings or price targets, so rating-agreement and price-target-variance measures are **unavailable**, not zero. One run mentioned an external consensus target already present in the prompt. See `experiments/part2_stability/part2d_stability_analysis.md`.

## 4. Financial Grounding Verifier Improvement

The selected improvement is a **post-generation, rule-based financial grounding verifier** that checks certain amounts, units, accounting periods, and metric attributions against known financial inputs. It is opt-in, emits findings, and does **not** automatically rewrite narrative text.

On a curated **14-passage** binary detection benchmark (12 constructed examples and two original agent passages):

| Version | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.1 | 6 | 5 | 1 | 2 | 0.545 | 0.857 | 0.667 |
| v0.2 | 7 | 0 | 0 | 7 | 1.000 | 1.000 | 1.000 |
| v0.3 | 7 | 0 | 0 | 7 | 1.000 | 1.000 | 1.000 |

Version 0.3 also eliminated three identified metric-attribution false alarms from previous NVDA original-report screening. In a separate offline check of **21 original sections** across NVDA, AMD, and AVGO, v0.3 raised **zero alerts**. This does **not** establish that those sections were fully grounded; the manual grounding audit shows substantial unsupported content that the current rules do not detect. The benchmark is small and targeted, and no actual **before/after whole-report grounding-rate improvement** has been measured, since the verifier does not rewrite the reports.

The submitted workflow uses customized section-specific OpenAI Chat Completions prompts. It should not be confused with an unchanged upstream eight-agent `openai-agents`/Pydantic architecture or described as an independent ninth LLM agent. The improvement patch and evaluation files are under `experiments/part3_verifier/`.

## 5. Reproducibility

This is a **lightweight research submission**, not a fully standalone clone of FinRobot. Reproducing all results requires the original FinRobot checkout, relevant locally cached data, installed dependencies, and—for API-backed generation—private credentials. **Do not commit API keys or `config.ini`.**

From the original FinRobot project root, the relevant experiment entry points include:

```bash
python scripts/audit_sec_financials.py
python scripts/independent_dcf.py
python scripts/reconcile_native_valuation.py
python scripts/prepare_grounding_audit.py
python scripts/run_stability_test.py
python scripts/evaluate_grounding_verifier.py
python scripts/evaluate_verifier_v03.py
python scripts/compare_verifier_v03.py
```

These commands describe the original project layout and are **not guaranteed to run directly from this submission directory** without the original dependencies and data. The saved artifacts allow inspection even when external APIs are unavailable.

**Model and usage records:** The three-run NVDA stability test documents GPT-4.1-mini, temperature 0.7. The customized generator specifies temperature 0.7 and an output-token limit of 1000 per section; the exact model selected for each of the original five-company generation runs is not fully recoverable from submitted logs. Per-company wall-clock runtimes, complete token usage, exact API costs, and total spend were **not consistently recorded** and are not represented as zero.

## 6. Report and Published Article

- **Full research report:** [`research_report.pdf`](research_report.pdf) (editable version: [`research_report.md`](research_report.md))
- **Five company HTML reports:** [`reports/`](reports/)
- **Medium article:** [Can an AI Equity Research Agent Be Trusted?](https://medium.com/@xz3423/can-an-ai-equity-research-agent-be-trusted-1f2dd9a9fbae)
- **Article manuscript:** [`medium_article.md`](medium_article.md)
