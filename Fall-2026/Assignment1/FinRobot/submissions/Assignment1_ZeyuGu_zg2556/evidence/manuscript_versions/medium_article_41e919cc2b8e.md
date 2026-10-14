# A polished AI equity report can still be wrong: auditing FinRobot from source to claims

*Zeyu Gu (zg2556), STAT GR5398, Fall 2026*

**Working draft, not published.** The data and valuation findings below are measured. Full-section before/after grounding, stability extraction, total cost and complete five-company industry coverage are still being finalized. No measured improvement is claimed before those audits finish.

A research report can have attractive charts, a confident thesis and a precise target price while combining observed data, modeling assumptions and an LLM's interpretation. The presentation does not establish that those pieces agree.

This FinRobot project tests that chain of evidence, then investigates one focused change: an additional verifier after the eight writing agents. The findings already show why a reproducible calculation can remain economically questionable, why a verifier itself needs testing, and why incomplete coverage must remain visible.

## Start with the system that actually runs

The course starter contained instructions and a two-step wrapper, not the full equity implementation. The upstream code was pinned to commit `2717499b8e30f242640af08c4ad9afd1113c2d45`. Its active command-line path differs from its typed eight-agent manager. This experiment uses the actual typed agents with their original instructions and shared evidence prompt; that compatibility departure is recorded in every manifest.

The baseline writer is `gpt-4.1-mini`, temperature 0.7, with a 4,000-token output ceiling per section. The eight sections cover the thesis, company overview, investment update, valuation, risks, competitors, takeaways and news. Each call retains its complete prompt, instructions, schema, wire request and response, token usage, runtime and run ID.

A two-step pipeline can fetch new prices or regenerate prose while rendering. Auditing one text and delivering another would invalidate the audit. The audited renderer therefore uses frozen inputs, disables live fetching and regeneration, and verifies all eight rendered text hashes. It labels FMP's consensus target separately from the valuation engine and the LLM's interpretation.

The onboarding companies - NVDA, AMD, INTC, AAPL and GOOGL - all ran. They are not one coherent industry group. The formal plan selects NVDA, AMD, INTC, AVGO and QCOM in semiconductors, with at least two peers per company. At this draft's cutoff, authentic frozen financial runs exist for the first three; AVGO and QCOM data requests returned entitlement failures. Apple and Alphabet remain onboarding examples, not substitutes for missing semiconductor coverage.

## Twenty filing checks found a definition problem

The data audit checks ten metric/period observations each for NVIDIA and AMD against accession-specific SEC filing references. Sixteen of twenty agree within declared display precision. Four EBITDA and EBITDA-margin observations differ from an operating-income-plus-D&A reference, but all four reconcile to explicit alternative definitions. This is evidence about those selected rows, not validation of the entire pipeline.

For NVIDIA FY2026, operating income of $130.387 billion plus D&A of $2.843 billion gives operating EBITDA of $133.230 billion. FMP reports $144.552 billion. The $11.322 billion difference is not rounding: the FMP number equals net income plus tax, interest and D&A, retaining non-operating earnings other than interest expense. [NVIDIA FY2026 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm).

For AMD, the simple net-income formula gives $7.367 billion, versus FMP's $7.275 billion. Removing $66 million of discontinued-operations income and $26 million of equity income closes the $92 million gap. Acquisition amortization also explains the apparent mismatch between an aggregate cost-of-revenue field and the smaller printed cost-of-sales line. [AMD FY2025 10-K](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm).

The conclusion is not that the provider's EBITDA was simply wrong. A standardized field name does not guarantee a comparable economic definition. Agreement with a filing under one formula is necessary evidence, but it does not establish that a peer multiple is useful. The saved audit preserves accession, fiscal period, filing date, unit, formula, values, differences and XBRL context for every observation.

## A forecast can change the meaning of its earnings measure

Default revenue growth is 5%, 6% and 4% over the next three years. Contribution margin rises by one percentage point annually, while SG&A/revenue falls by half a percentage point. These are constants, not estimates inferred from the filings.

The code subsequently calculates forecast EBITDA as contribution profit minus SG&A. This excludes research and development and changes the historical earnings definition. Adjusting an earlier EBITDA-margin setting does not fix the later overwrite.

AMD exposes the consequence: its reconciled historical FMP EBITDA margin is about 21.0%, but the first forecast margin is 39.0%, followed by 40.5% and 42.0%. A jump from 21% to 39% cannot be explained by the documented one-percentage-point improvement. Much of it is a definition discontinuity. NVIDIA's SG&A/revenue also falls from about 2.1% to 0.6% without evidence supporting that precise path.

Fiscal timing matters too. NVIDIA's latest annual actual is FY2026; AMD's is FY2025. Their next forecast labels must start in different years. Fixed 2025E/2026E/2027E labels can overlap actuals. A compatibility fix derives the next three fiscal years from the last actual while preserving baseline formulas for auditability.

A future forecasting project should align earnings definitions and reconcile published quarters before selecting company-specific scenarios. That is not a second formal improvement here. The controlled intervention remains the verifier, whose before/after comparison uses unchanged writer inputs.

## Exact valuation reproduction does not establish fair value

Independent standard-library arithmetic reproduces the saved engine outputs without calling the same engine again. NVIDIA's EV/EBITDA target is $333.71 and its DCF target is $72.52 under their respective assumptions. The saved target, enterprise-value, equity-value and assumed-net-debt reconciliations have zero numerical differences.

Those matches validate implementation agreement, not investment merit. The EV/EBITDA case uses FY2029E EBITDA, a target multiple near 48.885, 24.221 billion shares and estimated net debt equal to 10% of enterprise value. The debt amount is a fallback formula, not observed balance-sheet net debt. Raising enterprise value therefore raises the assumed debt automatically.

FMP's consensus target of $339.17 is a separate observation. Its proximity to the engine's $333.71 does not make the estimates equivalent or validate either method. Their distinct origins must remain visible.

## Sensitivity must reach the price

The native analyzer changes revenue and margins. An EBITDA scenario is not yet a target-price scenario. Each of its 25 combinations was propagated through the same EV/EBITDA calculation, retaining the baseline multiple, shares and debt rule.

NVIDIA's target range is $308.40-$359.93 around a $333.71 base. AMD's is $341.85-$415.62 around $377.84. These are deterministic scenario ranges, not confidence intervals or probabilities.

Revenue shocks are relative changes to the forecast revenue level; margin shocks are absolute percentage-point changes. Under this baseline method, local price elasticity to revenue equals one. Adding one percentage point to margin changes NVIDIA's target by approximately 1.36% and AMD's by 2.38%, before clipping. The difference follows from their starting margins.

![NVIDIA target-price sensitivity from frozen inputs](../audits/target_price_sensitivity/NVDA/target_price_sensitivity.png)

This grid still holds the multiple, shares, debt rule and model structure fixed. It does not measure uncertainty about whether forecast EBITDA is comparable with historical EBITDA. A smooth heatmap cannot resolve that structural question.

## The ninth agent also needs an audit

The verifier receives the original writer prompt and generated section. It enumerates source units and atomic claims, classifies them as Supported, Unsupported or Contradicted, identifies evidence lines and binds numeric tokens to source values. Deterministic checks reject missing coverage, invalid citations, missing years, duplicate bindings and failed conversions.

The first real pilot failed. Three mini-model attempts at NVIDIA's tagline produced structured output, but none passed validation. The model altered table quote spacing, tried to equate a threshold such as 70% with a different observed percentage, and omitted year bindings. These three attempts remain preserved and cost $0.0081568. Structured output was not verified output.

The revised schema asks for evidence line IDs, and code resolves the original lines exactly. Repair diagnostics now identify failing claims and numbers. Unsupported assertions are visibly annotated “UNVERIFIED - evidence not supplied”; they remain Unsupported in the independent after audit. Disclosure does not earn grounding credit.

A second mini-model pilot still failed numeric-binding requirements. The formal verifier and independent judge were then moved to `gpt-4.1`, within the same $20 total authorization. That decision followed inspection of NVIDIA pilot failures; it is not held-out model selection. Baseline writing and the three independent stability generations remain on the original mini model and parameters.

**Pending experiment results:** complete before/after claim counts, grounding and contradiction rates, numeric-error rates, verified retention, additional tokens, runtime and cost will be inserted only after full-section independent evaluation completes. This draft claims no measured grounding improvement. The independent judge does not receive verifier labels as an answer key.

## What a portfolio manager can use

The useful foundation is frozen evidence, explicit fiscal periods, reproducible calculations, component reconciliations and a visible separation between market data, valuation and narrative. These can reduce the time an analyst spends finding and checking a number.

Unreviewed prose cannot yet replace investment judgment. Accurate historical rows can coexist with unsupported competitive claims, misleading forecast language or an inconsistent earnings definition. Reproducible prices can still depend on hidden fallbacks. A verifier can fail. Five polished onboarding reports do not satisfy a missing five-company industry experiment.

The standard is evidence: judge a claim against what the model actually saw; count a repeat only when it is a new generation; preserve failures in the cost ledger; and leave incomplete requirements visibly incomplete. Final measurements will establish whether the verifier improves the report. They cannot retroactively ground the baseline.

## Reproducibility and publication status

The submission includes the pinned upstream SHA and patches, dedicated environment, saved snapshots and prompts, seven baseline runs, independent valuation arithmetic, target-price sensitivity data and figures, and claim-level evaluations as they finish. Offline replay differs from new online acquisition. A saved output replay is not an independent stability run.

This article has not been published. Its genuine Medium and submission links will be added after results are complete and publication is authorized. Local evidence: `audits/data/data_audit.csv`, `snapshots/NVDA/valuation/valuation_reconciliation.csv`, `audits/target_price_sensitivity`, `docs/verifier_version_history.md`, and versioned run manifests.
