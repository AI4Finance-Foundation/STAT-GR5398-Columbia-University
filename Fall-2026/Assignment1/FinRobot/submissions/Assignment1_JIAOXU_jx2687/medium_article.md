# Auditing FinRobot: When a Polished Equity Report Outruns Its Evidence

*A five-company experiment in financial verification, forecast design, and the limits of LLM-generated research.*

An equity research report can be fluent, detailed, and professionally formatted while making claims its inputs cannot support. In this project, FinRobot successfully generated reports, but that success did not establish that the reports were ready for investment decisions.

The experiment produced three findings. Financial arithmetic was easier to verify than narrative explanations. Fixing forecast dates improved consistency without improving historical forecast accuracy. Repeated outputs could preserve the same risk categories while changing the substance beneath them.

These findings come from actual executions and saved evidence, rather than example outputs. They also have limits: the claim audit is provisional and partly automated, and the valuation assumptions are deliberately exposed rather than presented as justified investment targets.

## A working pipeline with a narrow evidence base

This project follows the [Columbia STAT GR5398 Fall 2026 FinRobot assignment](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/master/Fall-2026/Assignment1/FinRobot/README.md), using the course-specified `finrobot_equity` module. The tested FinRobot source was pinned to commit `2717499b8e30f242640af08c4ad9afd1113c2d45`.

The formal coverage universe consists of NVIDIA, AMD, Intel, Apple, and Microsoft, all classified as Technology in the retrieved FMP profiles. Each company has at least two peer inputs. Alphabet was also run to complete the course starter set, but is outside the formal Technology universe. Broad sector membership does not make these businesses interchangeable: semiconductor, hardware, and software companies have different economics, which limits peer valuation comparisons.

Financial Modeling Prep supplied annual financial data and peer metrics. GPT-4.1 generated eight specialist sections per company. Financial and market information reflects the saved snapshots available for this project on October 11, 2026; annual financial periods vary by company and should not be interpreted as current trailing-twelve-month figures.

Two compatibility repairs were necessary. New-account FMP access required stable endpoints instead of the legacy endpoints used by the code. The CLI was also connected to the existing typed specialist agents, with their actual prompts, outputs, and token usage saved. These changes are recorded separately from the forecast improvement. Provider failures were changed to raise errors instead of producing canned replacement text.

The evidence supplied to the agents remained narrow: financial tables, peer EBITDA and EV/EBITDA comparisons, and news when available. In this run, the news endpoint returned HTTP 402. Some agent instructions requested information beyond these inputs, without an implemented web-search tool to retrieve it.

## Matching a number does not settle its definition

The financial audit compared ten metrics for NVIDIA and ten for AMD with filing-based references. Sixteen comparisons matched within the stated rounding tolerance. Four differences involved EBITDA or EBITDA margin.

AMD illustrates why a discrepancy needs interpretation. Its original 2025 filing reports $750 million of other depreciation and amortization and $2.254 billion of acquisition-related amortization. Together, they give $3.004 billion. Adding this to operating income produces an operating-EBITDA proxy of $6.698 billion; the FMP-derived pipeline reports EBITDA of $7.275 billion, a $577 million difference. The source is the [AMD 2025 Form 10-K](https://ir.amd.com/financial-information/sec-filings/content/0000002488-26-000018/amd-20251227.htm).

This difference was retained as an unresolved definition issue, rather than automatically labelled a data error. The operating-income-based proxy does not reproduce the provider's measure. A difference in adjustment basis is a possible explanation, but full reconciliation is still needed before treating them as equivalent. NVIDIA checks likewise used its [original annual report](https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-annual-report-web-Hyperlinks.pdf).

The practical lesson is that traceability requires more than finding the same-looking label in two tables. It requires checking the fiscal period, units, definition, and formula.

## Correct forecast dates did not produce better forecasts

The baseline forecast output used fixed labels such as 2025E, 2026E, and 2027E even when actual data had advanced into those years. The patch made forecast labels follow the latest actual period and replaced fixed revenue growth assumptions with a historical two-year compound annual growth rate, capped between −50% and +50%, while holding margins flat.

Across the five-company FMP universe, the share of forecast-year labels strictly later than the latest actual period-end year rose from **8 of 15, or 53.3%, to 15 of 15, or 100%**. This is a chronological validity check; it does not prove economic accuracy or eliminate fiscal-calendar labelling limitations.

Predictive performance was tested separately. A supplemental SEC-data backtest covered NVIDIA, AMD, Intel, Broadcom, and Qualcomm, producing 68 one-year rolling-origin forecasts. Training facts were restricted to filings available at each origin, and the next annual revenue observation was the target. This is a different sample from the formal FMP universe.

The historical-CAGR forecast was compared with a simple 5% revenue-growth baseline. Mean absolute percentage error rose from **13.74% to 15.86%**, a deterioration of about 2.12 percentage points. This does not compare every dimension of the original multi-year forecast: it tests a specific one-year revenue rule.

The patch made the model easier to inspect and corrected its chronology. It did not demonstrate improved aggregate predictive accuracy. A transparent assumption still needs an empirical test.

## The prose often went beyond what the prompt established

The grounding audit extracted 576 factual and quantitative claims across all eight sections for NVIDIA, AMD, and Intel. Claims were classified against the evidence actually supplied to each agent:

- **Supported:** derivable from the prompt's data.
- **Unsupported:** not established by that prompt.
- **Contradicted:** inconsistent with the prompt's evidence.

After five documented review corrections, the provisional labels were **177 Supported, 397 Unsupported, and 2 Contradicted**. Unsupported does not mean necessarily false: a statement may be true outside the experiment while lacking evidence in the agent's input.

The section rates below pool claims across the three companies. Each rate is supported claims divided by all extracted claims in that section, rather than an average of company percentages.

| Section | Extracted claims | Supported | Grounding rate |
| --- | ---: | ---: | ---: |
| Company overview | 105 | 14 | 13.3% |
| Competitor analysis | 73 | 8 | 11.0% |
| Investment overview | 81 | 44 | 54.3% |
| Major takeaways | 63 | 33 | 52.4% |
| News summary | 62 | 18 | 29.0% |
| Risks | 104 | 22 | 21.2% |
| Tagline | 24 | 5 | 20.8% |
| Valuation overview | 64 | 33 | 51.6% |

One concrete failure appeared in NVIDIA's news summary: it described headline prominence and analyst consensus despite the absence of retrieved news. Another described historical 2025A revenue as forward guidance. A third combined correct revenue figures with a customer-adoption explanation that the financial table did not establish.

The evaluator also made mistakes. It initially classified some compound claims as Supported because their numerical component was correct, overlooking an unsupported causal explanation. Those classifications were corrected and recorded. Citation-format mismatches were flagged separately rather than automatically treated as evidence that a claim was unsupported.

These rates are therefore audit findings, not certified accuracy scores. The same model family performed extraction and classification; complete claim coverage and entailment were not independently adjudicated. The results identify where review is needed, but cannot replace that review.

The structural problem is visible in the agent design. Instructions request detailed competition, regulation, and supply-chain analysis while the prompt mainly supplies financial tables. More sections create more opportunities to write; they do not automatically create more evidence.

## Stable headings did not establish stable judgments

NVIDIA was generated three times using identical saved user prompts and unchanged system instructions. The extracted first three risk categories were Market Risk, Competitive Risk, and Operational Risk in every run.

That agreement is weaker evidence than it first appears. The original risk prompt already specifies those categories and their order. The supporting details and wording changed between runs. Category agreement largely reflects the requested structure, not independently demonstrated consistency of the underlying research.

The agent prose did not explicitly emit an investment rating or numeric target price, so their variances could not be calculated. Missing values were kept missing, rather than assigned zero variance. Market-fetched or default fields in the rendered report header were treated separately from agent decisions.

A better stability test would require explicit, evidence-linked decision outputs and compare the underlying risk assertions, not just their headings.

## Reproducing a DCF did not validate the investment thesis

Independent calculations reproduced the upstream DCF arithmetic for all five companies. A full sensitivity grid also recalculated 45 scenarios, combining WACC values of 8%, 10%, and 12% with terminal growth of 1.5%, 2.5%, and 3.5%.

The upstream base case assumed ten projection years, 10% growth for the first five years, 5% for the next five, 10% WACC, and 2.5% terminal growth. It used cash flow from operations less capital expenditure, annual weighted-average diluted shares, and synthetic net debt equal to 10% of enterprise value.

Those choices are material weaknesses. Cash flow from operations less capex is not automatically unlevered free cash flow to the firm. Synthetic net debt is not observed balance-sheet net debt. A sensitivity grid shows how assumptions affect an answer; it does not justify the assumptions themselves.

Intel's negative base cash flow exposed the model boundary: the raw DCF was negative, and the helper floored the price at zero. That is a failure of the simple extrapolation to produce a useful valuation, not evidence that Intel's economic value is zero.

## What I would trust, and what I would change

The most useful outputs are the preserved data, reproducible calculations, explicit assumptions, and audit trails. Even those require attention to definitions. The generated prose is a research draft whose claims need review; it is not ready to be handed to a portfolio manager as a verified recommendation.

A stronger pipeline would require evidence references for factual statements, distinguish historical observations from forecasts, leave unsupported context explicitly unresolved, and abstain when required inputs are unavailable. Its verifier would split numerical and causal assertions before checking them. Forecast improvements would be judged on held-out performance as well as consistency.

There are also open questions this experiment does not answer: whether narrower industry peers improve valuation usefulness; whether independently labelled claims change the grounding rates; and whether alternative forecast rules outperform the fixed-growth baseline across a broader sample.

Initial two-step runtimes were approximately 35.5–42.0 seconds per company, excluding installation and subsequent evaluation. Recorded responses for reports, repetitions, and evaluation totalled about **$1.07** at the cited [GPT-4.1 rates](https://developers.openai.com/api/docs/models/gpt-4.1). This subtotal excludes some diagnostic calls and a truncated evaluator response, so it is not a reconciliation of the account's full bill.

The experiment package preserves raw data, filing comparisons, prompts, outputs, patches, claim labels, stability results, and an executed notebook. It makes the main results checkable without relying on the presentation of the reports.

FinRobot was effective at assembling a readable research draft. The audit showed where that draft outran its evidence. The next improvement should make those boundaries explicit before a reader mistakes fluent analysis for verified research.

---

*Prepared for Columbia STAT GR5398 Fall 2026 Assignment 1. AI assistance was used in implementation, execution, evaluation scaffolding, and drafting. Results are based on saved experiment artifacts; automated evaluation limitations are stated above. This manuscript has not been published. The author should review it and ensure any required course disclosure accompanies publication.*
