> Current release note: Current execution update: all five formal companies now have actual baseline/final reports. AVGO/QCOM use public SEC/Yahoo adapters and cached FMP peers after entitlement denial; WRDS observations are excluded. Actual peers and common-window returns are in `coverage_universe.csv`; any planned or pending source-collection language below records earlier selection work.

# Coverage universe and SEC evidence

Author: Zeyu Gu (zg2556). Evidence cutoff: 2026-10-11. This document describes source collection and cohort selection, not completed FinRobot experiments.

## Cohort definition

The five-company cohort is the semiconductor products industry: companies designing or manufacturing processors, accelerators, networking or wireless integrated circuits as a substantial operating business. This is a business-based definition, not a claim that all companies have the same SEC SIC code. Frozen SEC submissions classify NVDA, AMD, INTC and AVGO as SIC 3674, Semiconductors & Related Devices; QCOM is SIC 3663, Radio & TV Broadcasting & Communications Equipment. Qualcomm remains an economically defensible semiconductor peer because its QCT chip platform business produced $38.367 billion of $44.284 billion consolidated FY2025 revenue (86.64%); QTL licensing materially changes its consolidated margins. The numerator is segment revenue and is not a pure chip-unit revenue measure. See [Qualcomm FY2025 10-K, QCT segment discussion and Note 8](https://www.sec.gov/Archives/edgar/data/804328/000080432825000085/qcom-20250928.htm).

| Company | Business basis and peer rationale | Comparability limitation |
|---|---|---|
| NVDA | Accelerated computing, networking and graphics. AMD and INTC supply competing compute products; AVGO supplies AI accelerators and networking. | Different accelerator, CPU and system mixes; software ecosystem economics affect consolidated multiples. |
| AMD | CPUs, GPUs, adaptive and embedded semiconductor products. NVDA and INTC are the closest covered compute comparators. | Acquisition amortization and embedded exposure affect GAAP earnings and EBITDA reconciliation. |
| INTC | CPU and related semiconductor products, plus manufacturing. AMD and NVDA compare product competition. | Foundry losses, capital intensity and internal segment eliminations make consolidated multiples unlike fabless peers. |
| AVGO | Semiconductor networking, custom accelerators and wireless products. NVDA is an AI/networking comparator; QCOM is a wireless chip comparator. | Infrastructure software contributes 42% of FY2025 revenue; consolidated valuation is not a pure semiconductor comparable. |
| QCOM | Wireless, handset, automotive and IoT integrated circuits. AVGO overlaps in connectivity; AMD overlaps in client/edge compute. | Licensing economics, handset concentration and ARM/x86 product differences restrict direct comparability. |

These relationships are analyst choices based on operating overlap, not SEC endorsement of a valuation peer group. The [NVIDIA FY2026 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm), [AMD FY2025 10-K](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm), [Intel FY2025 10-K](https://www.sec.gov/Archives/edgar/data/50863/000005086326000011/intc-20251227.htm), and [Broadcom FY2025 10-K](https://www.sec.gov/Archives/edgar/data/1730168/000173016825000121/avgo-20251102.htm) provide the business and segment descriptions. Broadcom's segment discussion reports semiconductor revenue of $36.858 billion and total revenue of $63.887 billion (57.69%, rounded to 58% in the filing). Intel's Note 3 explains foundry intersegment revenue and eliminations; adding Intel Products and Intel Foundry revenue would double-count internal activity.

Selection spans compute, wireless, networking and manufacturing business conditions. Actual FMP stable-endpoint probes now return HTTP 200 for NVDA, AMD and INTC, but HTTP 402 for AVGO and QCOM; tested alternatives MU, TXN, ADI, AMAT, LRCX, KLAC and ON are also denied. Thus only three of the five planned same-industry companies have real frozen pipeline runs. Each has the other two as available peers. `coverage_universe.csv` separates intended research peers from the actually run peer map. AAPL and GOOGL have real onboarding runs but do not satisfy the missing semiconductor coverage.

A common 2025-10-10 to 2026-10-09 raw close-price window is now frozen: NVDA +25.18%, AMD +182.97% and INTC +187.87%. This spans differing magnitudes but all three rose; it does not establish balanced winners/losers coverage. These are close-to-close price returns, not dividend-reinvested total returns. The provider's split-adjustment convention has not been independently verified. Exact dates, endpoint, prices, formula and caveat are in the coverage CSV and `snapshots/TICKER/analysis/price_performance.json`. FMP news endpoints return HTTP 402; each available company instead uses one identified official issuer earnings release inside the frozen 90-day window, a limited alternative rather than a comprehensive news feed.

## Frozen sources and periods

`evidence/sec/snapshot_2026-10-11/` preserves 21 successful HTTP responses: five SEC submissions JSON files, five companyfacts JSON files, the latest available annual and quarterly filings for each company, and AMD's annual amendment. Each response has a `.meta.json` sidecar with source and final URL, retrieval UTC, HTTP status, byte count and SHA-256. `manifest.json` indexes them. `filings.csv` separates the reporting period from publication date. Raw API snapshots are current at retrieval; historical values used below are filtered to the specified accession, not taken from later amended observations.

| Company | Annual period end | Annual filing date | Latest quarterly period end | Quarterly filing date |
|---|---|---|---|---|
| NVDA | 2026-01-25 (FY2026) | 2026-02-25 | 2026-07-26 | 2026-08-26 |
| AMD | 2025-12-27 (FY2025) | 2026-02-04 | 2026-06-27 | 2026-08-05 |
| INTC | 2025-12-27 (FY2025) | 2026-01-23 | 2026-06-27 | 2026-07-24 |
| AVGO | 2025-11-02 (FY2025) | 2025-12-18 | 2026-08-02 | 2026-09-10 |
| QCOM | 2025-09-28 (FY2025) | 2025-11-05 | 2026-06-28 | 2026-07-29 |

Annual FY labels are issuer fiscal years. SEC companyfacts `frame` labels are calendar-aligned aggregation aids, not substitutes for fiscal-year names: NVIDIA's FY2026 annual facts carry `CY2025`. The upstream API processor derives its `year` from the financial statement's end-date calendar year; preserve actual start/end dates when matching columns. Neither a filing's publication year nor a current quarter can be used to relabel a prior annual observation.

AMD's [2026-02-04 10-K/A, explanatory note](https://www.sec.gov/Archives/edgar/data/2488/000000248826000021/amd-20251227.htm) corrects transposed percentage changes in the MD&A discussion of client shipments and average selling price. Its correction is 15% shipment growth and 31% ASP growth. It does not replace the consolidated financial statements used here. Both original and amendment are frozen; textual claims using that passage must use the amended disclosure.

## Independent reference values, not completed audits

`sec_metric_references.csv` contains 20 populated filing-side references (10 each for NVDA and AMD), corresponding to actual metrics in upstream `financial_metrics_and_forecasts.csv`: revenue, cost of operations, SG&A, contribution profit, contribution margin, EBITDA, EBITDA margin, SG&A margin, revenue growth and EPS. `pipeline_value` and `difference` are deliberately blank; every verdict is `PENDING_PIPELINE_COMPARISON`. They are a reference resource, not the completed audit. The separate `audits/data/data_audit.csv` now contains the genuine 20-row pipeline comparison: 16 matches within display precision and four fully explained EBITDA-definition differences. See `data_audit_findings.md`. Keeping reference values separate prevents their originally blank pipeline fields from being mistaken for experimental results.

`sec_metric_ingredients.csv` retains the source observations (including supplementary continuing-operations and equity-income components), exact JSON pointers into the raw companyfacts responses, and matching numeric facts in the saved original filing. Each iXBRL match records its fact ID, context reference, tag, scale, sign, start/end dates, dimensions and numeric value. Every ingredient has at least one matching dimension-free original-filing fact. SEC values are in USD, not millions; the filing tables display millions. Percentage references are in percentage units (71.068 means 71.068%), while upstream historical margins are strings rounded to 0.1 percentage point.

The contribution metric is economically gross profit: upstream computes revenue less cost of revenue, not variable-cost contribution. Basic EPS is selected because upstream prioritizes FMP `eps` over `epsdiluted`; the FMP mapping must be checked before a verdict. Historical financial facts are GAAP, not management's adjusted results.

EBITDA is not an SEC GAAP line item. The reference explicitly uses operating income plus cash-flow D&A. NVDA: $130.387 billion + $2.843 billion = $133.230 billion. AMD: $3.694 billion + $0.750 billion other D&A + $2.254 billion acquisition-related amortization = $6.698 billion. AMD separately presents those two add-backs; using $0.750 billion alone understates D&A. These are transparent operating-EBITDA proxies, not an assertion that FMP uses the same definition. Net-income-based EBITDA can differ because of non-operating income and expense; reconciliation must inspect the actual FMP response before calling any gap an error. R&D and stock compensation have not been added back arbitrarily.

Revenue growth uses prior revenue as presented in the same annual accession; do not silently splice in a later restatement. PE and EV/EBITDA are not among these ten reference metrics because SEC statements alone do not provide the required contemporaneous market price and enterprise value. Their audits require separately frozen prices, dates, shares, debt and cash. The frozen SEC collection is a source evidence resource only; it is not inserted retroactively into any agent prompt and cannot make an unsupported baseline claim supported.

## Reproduction

From the submission directory, with Python 3.10 or newer (standard library only):

```powershell
python scripts/fetch_sec_evidence.py --offline
```

This verifies every existing raw-response hash and rebuilds filing-side tables without network traffic. A fresh online capture must use a new snapshot directory so it cannot overwrite the original input state:

```powershell
python scripts/fetch_sec_evidence.py --as-of 2026-10-11 --snapshot-dir evidence/sec/new_snapshot
```

The script records a transport error instead of substituting data, preserves non-200 HTTP response bodies, checks the filing-date cutoff, and restricts fact selection to the exact annual accession and an annual-duration window. It does not infer missing facts or retrieve paid data. When using a new date, review the annual filing selection, amendments and issuer-specific D&A mapping before interpreting output. No secrets or payment authorization are needed for these public SEC requests.
