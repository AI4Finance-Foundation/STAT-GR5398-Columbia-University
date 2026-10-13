# Making an AI Equity Research Agent More Reliable: Auditing and Rebuilding FinRobot's Valuation Pipeline

AI-generated equity research looks convincing long before it is trustworthy. My Assignment 1 experiment with FinRobot made that gap very concrete.

I ran the equity-research pipeline across five Technology-sector companies: NVIDIA, AMD, Intel, Apple, and Microsoft. The system collected financial data, generated forecasts, ran eight specialist LLM sections, and rendered professional-looking research reports. The easy part was producing the reports. The harder question was whether a portfolio manager should rely on them.

## What I audited

I separated the system into three layers: source data, deterministic calculations, and LLM interpretation.

First, I checked filing-linked financial values. For Apple and Microsoft I selected more than ten line items per company and reconciled them with the companies' SEC 10-Ks. Revenue, cost of revenue/sales, SG&A, gross profit, basic EPS, revenue growth, and gross-margin calculations reconciled cleanly. One naming problem surfaced immediately: the pipeline calls `Revenue - Cost of Operations` “Contribution Profit,” but the number is the filing's gross profit/gross margin. The arithmetic is right; the label is misleading.

Second, I audited forecasts and valuation. The forecast layer applies the same 5%, 6%, and 4% revenue-growth path across companies. More importantly, the printed assumption says EBITDA margin improves by 1 percentage point annually, but the generated first-year margin changes were +3.6 points for NVIDIA, +18.0 for AMD, +0.4 for Intel, +7.1 for Apple, and -3.5 for Microsoft. That is a material implementation problem because EV/EBITDA valuation uses forecast EBITDA.

Third, I audited narrative grounding. I extracted factual and quantitative claims from all eight agent sections for NVIDIA, Apple, and Microsoft and classified each as Supported, Unsupported, or Contradicted relative to the exact evidence available to the agents: financial tables, peer tables, and recent news. The overall grounding rate was about 54%. `major_takeaways` performed best because it mostly restated supplied numbers. `company_overview` performed worst because it introduced business-profile facts, products, partnerships, and causal explanations that were not present in the common evidence prompt.

That pattern is structural, not random. If an agent is asked to write a rich company profile or risk discussion while its prompt contains only financial tables and recent news, it has to fill the gap from model prior knowledge.

## The valuation problem

The upstream valuation engine had several shortcuts that made its output difficult to defend.

The EV/EBITDA method could fall back to a 12x multiple and assumed net debt equal to 10% of enterprise value. The DCF could approximate free cash flow as 60% of EBITDA, assume 10% growth for years 1-5 and 5% for years 6-10, and again set net debt to 10% of enterprise value. Peer comparison expected a data shape that the report pipeline did not actually provide, so it could return zero. The synthesis then weighted surviving methods with arbitrary “confidence” values.

I refactored the valuation path to use actual FMP cash flow, actual debt minus cash, historical EV/EBITDA multiples, first-forecast-year EBITDA, a common peer fiscal year, explicit WACC and terminal growth, and auditable equal method weights. The DCF now uses the mean historical free-cash-flow margin from the latest three actual years to translate forecast revenue into near-term FCF, then applies explicit post-forecast assumptions.

For NVIDIA, reconstructing the upstream logic on the same saved inputs produces a synthesized value of about $82.49 per share. The modified pipeline produces $184.52, composed of $287.61 from EV/EBITDA, $59.62 from DCF, and $206.32 from peer comparison. The point is not that $184.52 is “correct.” The improvement is that every major input can now be traced and debated.

Intel provides another useful case. Its latest actual free cash flow was negative, so the modified DCF refuses to generate a positive-looking value and marks the method unavailable. That is a better failure mode than silently manufacturing FCF from EBITDA.

## Stability was not just an LLM problem

I also repeated NVIDIA with the same commands. The top-three risk narratives were identical in all three saved runs. But the live FMP market-data calls were rate-limited in runs 2 and 3. The pipeline then emitted `N/A` for the rating and zero for target price and valuation methods.

That result changed my view of “agent stability.” Reproducibility depends not only on temperature and prompts, but also on data versioning, caching, API error handling, and whether a live failure can masquerade as a real zero.

## Would I hand the report to a portfolio manager?

Not yet.

The filing-derived tables are close to decision-grade when the metric labels are corrected and sources are explicit. The deterministic valuation layer is substantially more defensible after the refactor, but it still depends on simplistic forecast assumptions and heterogeneous peer sets. The LLM-generated prose is useful as a draft, not as evidence. A grounding rate near 54% means too much factual language is not directly supported by the common prompt.

The next version should snapshot the entire evidence base, cache market data, reject API failures rather than converting them to zeros, replace hard-coded forecasts with a backtested forecasting method, and add a verifier that checks every generated factual claim against source tables before publication.

The main lesson is simple: a polished financial report is not the same thing as a reproducible research process. In finance, the value of an AI agent comes from traceability and controlled failure modes, not from how professional the PDF looks.
