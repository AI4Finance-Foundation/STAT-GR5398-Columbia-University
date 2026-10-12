# Can an AI Equity Research Agent Be Trusted? What I Learned from Testing FinRobot on Five Semiconductor Stocks

*By Xiaoyu Zhang | GR5398, Fall 2026*

Large language models are good at writing reports that *sound* like research. But a convincing narrative is not the same as a trustworthy financial conclusion. I wanted to understand what happens when an AI equity research agent is asked to analyze several related companies—and, more importantly, what kind of verification catches the mistakes a polished report can hide.

For this project, I adapted a FinRobot equity research workflow to cover NVIDIA, AMD, Intel, Broadcom, and Qualcomm. The system generated company-level financial tables, forecasts, peer comparisons, and five HTML reports. I then tested the outputs along four dimensions: financial data consistency, valuation sensitivity, evidence grounding, and run-to-run stability. Finally, I improved a lightweight numerical grounding verifier and compared its performance against the previous version.

## Five reports are the start, not the finish

A five-company workflow has an immediate appeal: many of the required steps—fetching financial statements, normalizing figures, calculating ratios, comparing peers, and generating prose—can be automated. But the most revealing result came from a failure in the peer comparison. An earlier version of the NVIDIA table contained an approximately **$382 billion 2027E EBITDA** figure, whereas the primary projection implied about **$159.85 billion**. A fluent paragraph could easily reproduce the wrong number unless the system checks the source table it is using.

That case changed my definition of success. Completing the report is one milestone; proving that its numbers are consistent is another.

## First, trace the numbers to evidence

I used locally saved SEC Company Facts to reconcile selected FY2025 financial records for Broadcom and Qualcomm. The result was **22 matches out of 22 tested records** across direct and derived measures. For example, calculated free-cash-flow proxies—operating cash flow less capital expenditures—were **$26.914 billion** for Broadcom and **$12.820 billion** for Qualcomm.

The caveat matters: both the source and normalized data came from the same SEC XBRL records. This was an *internal reconciliation*, not an independent audit of every line in each 10-K. Correct arithmetic can still accompany an incorrect accounting definition if the taxonomy mapping is wrong.

## A valuation model needs a range, not a magic number

I built illustrative five-year discounted-cash-flow scenarios using those FCF proxies. With assumed growth of 5%, 6%, 4%, 4%, and 4%, an 11% discount rate, and 3% terminal growth, Broadcom's discounted cash-flow proxy was **$371.348 billion** and Qualcomm's was **$176.885 billion**.

These are **not share-price targets**. I did not independently estimate the companies' capital costs, translate the proxy into verified free cash flow to the firm or equity, or adjust for net debt and diluted shares. Instead, I used the model to explore sensitivity. Across 9%–13% discount rates and 2%–4% terminal growth, the Broadcom result ranged from **$277.80 billion to $575.56 billion**. The wide range is a reminder that transparent assumptions are often more useful than a deceptively precise point estimate.

## What happens when you fact-check the prose?

I extracted 24 statements from reports on NVIDIA, AMD, and Broadcom and performed a preliminary evidence review. **17 were marked Supported, 4 Unsupported, and 3 Contradicted**—an initial supported share of **70.8%**. Some mistakes were surprisingly subtle. Historical numbers were described as forecasts. Forecast growth language did not always match the underlying assumptions. Other statements paired correct figures with claims that lacked sufficient evidence.

I treat that 70.8% as a *provisional sample result*. Some paragraphs contain several claims, and the review was not a fully independent, atomized annotation exercise. Nevertheless, it showed why checking whether a paragraph cites a plausible number is not enough.

## Three runs, one repeating problem

To test stability, I generated NVIDIA investment-overview and risk sections three times from the same frozen input data. Revenue-growth volatility, high valuation multiples, and SG&A uncertainty appeared in all three runs. That indicates meaningful stability in the broad risk themes.

But all three runs also showed confusion between FY2026 historical actuals and estimated or projected figures. Only the third run mentioned an external **$339.17 Buy-consensus target**, which had already been provided to the agent; none supplied a comparable independent target price. I therefore did **not** compute a target-price standard deviation. The data were missing, not perfectly stable.

## A small verifier improvement made a measurable difference

The clearest engineering contribution was a targeted Financial Grounding Verifier. In its original version, correct dollar figures sometimes triggered alerts because of unit-handling errors, while the earlier $382 billion EBITDA discrepancy slipped through. I improved amount normalization and selected discrepancy checks, then evaluated both versions on the same 14-passage benchmark.

On that curated dataset, **F1 rose from 0.667 to 1.000**, with false positives falling from five to zero and false negatives from one to zero. That is promising—but not proof of universal reliability. Twelve cases were constructed to test known failure modes, and the two real Agent passages were already detected by both versions. Larger blind tests remain necessary.

## The lesson: let models explain, but let evidence check

My takeaway is not that AI equity research should be abandoned. It is that good automation needs a division of labor. Language models are useful for organizing analysis and explaining trade-offs. Deterministic software is better suited to verifying units, periods, arithmetic, and source relationships. Human review remains necessary for accounting semantics, assumptions, and business judgments.

A robust financial agent should always distinguish **historical actuals from estimates**, **external consensus from independent valuation**, and **verified statements from plausible but unsupported narratives**. The objective is not just faster report generation; it is a research workflow in which every important number can be traced, challenged, and corrected.

*This article summarizes a coursework experiment using a fixed data snapshot. It is not investment advice.*


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
