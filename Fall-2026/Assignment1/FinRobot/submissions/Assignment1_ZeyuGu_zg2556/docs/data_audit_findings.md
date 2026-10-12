# Measured SEC reconciliation

The real frozen NVDA and AMD FMP analyses have now been compared with the accession-specific SEC references: 20 distinct metric/period observations, ten per company. Sixteen match within display precision. Four EBITDA/EBITDA-margin observations differ from an operating-income-plus-D&A reference but are fully reconciled to explicit alternative definitions. There are no unexplained differences among these selected twenty observations. This is limited evidence about the selected historical rows, not a claim that every API field, forecast, company or narrative is correct.

Reproduce from the submission directory:

```powershell
python scripts/data_audit.py --sec-references evidence/sec/snapshot_2026-10-11/sec_metric_references.csv --analysis-root snapshots --output-dir audits/data
```

`audits/data/data_audit.csv` contains pipeline values, filing reference values, raw/display deltas, fiscal periods, publication dates, currencies, formulas, source links, input hashes and exact XBRL contexts. `ebitda_component_reconciliation.csv` makes the component arithmetic explicit. `data_audit_summary.json` records verdict totals and source hashes. The earlier SEC-only reference CSV remains reference-only; it has not been relabeled as a completed audit.

| Annual observation ($million) | NVDA FY2026 | AMD FY2025 |
|---|---:|---:|
| GAAP operating income | 130,387 | 3,694 |
| D&A used in reconciliation | 2,843 | 3,004 |
| Operating EBITDA reference | 133,230 | 6,698 |
| Net income | 120,067 | 4,335 |
| Signed income-tax expense/(benefit) | 21,383 | -103 |
| Interest expense | 259 | 131 |
| Simple NI + tax + interest + D&A | 144,552 | 7,367 |
| Discontinued-operations income excluded by observed AMD formula | Not needed | 66 |
| Equity-method income excluded by observed AMD formula | Not needed | 26 |
| Actual FMP EBITDA | 144,552 | 7,275 |
| FMP minus operating reference | 11,322 | 577 |
| Residual after identified definition | 0 | 0 |

For NVIDIA, the measured FMP value equals net income plus tax, interest expense and D&A. The $11.322 billion difference from an operating definition reflects non-operating earnings other than interest expense. All six independently compared FMP/SEC components match. Source: [FY2026 consolidated statements and cash flows](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm).

For AMD, applying the same simple net-income formula initially leaves a $92 million gap. The original SEC statement separately discloses $66 million income from discontinued operations and $26 million equity income. Removing both gives $7.275 billion, exactly the FMP observation. An equivalent calculation is $4.140 billion income from continuing operations before income taxes and equity income + $0.131 billion interest expense + $3.004 billion D&A. The resulting EBIT of $4.271 billion also equals FMP's EBIT field. AMD D&A includes $0.750 billion other D&A plus $2.254 billion acquisition amortization. Source: [FY2025 consolidated statements of operations and cash flows](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm), with accession `0000002488-26-000018`; the precise contexts and supplementary equity/discontinued tags are preserved in the bridge.

This is an empirical formula reconciliation for these observations, not proof of one universal FMP EBITDA policy. A provider's apparently standardized EBITDA field can still contain differing non-operating exclusions across issuers. This matters to peer multiples and forecasting: an exact agreement with a reproducible definition does not make that definition economically comparable across firms. Historical EBITDA must not be silently replaced by gross profit less SG&A in projected periods.

The AMD cost-of-operations observation also merits a label clarification. FMP cost of revenue of $17.487 billion corresponds to SEC cost of sales of $16.456 billion plus $1.031 billion acquisition-related amortization classified in cost of sales. It matches the aggregate XBRL cost tag and the gross-profit subtraction, despite not matching the smaller printed cost-of-sales line alone. EPS comparison uses total basic EPS, not continuing-operations-only or diluted EPS. Percentage tolerance is 0.05 percentage point because the pipeline displays one decimal; USD/share tolerance is 0.005, and USD totals use half a dollar. No FX conversion or market-price input was needed for these ten selected metrics.
