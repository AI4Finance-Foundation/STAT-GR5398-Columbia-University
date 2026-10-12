# I ran FinRobot, then audited what its reports actually knew

*Frozen evidence, five semiconductor companies, and what an independently audited verifier can — and cannot — fix*

The most useful number in my FinRobot experiment was not a stock-price target. The original V6 evaluation counted 291 supported factual assertions among 1,478 claims. A later audit of the evaluator corrected that baseline to 286/1,479. The latest verifier treatment yields 570/1985 supported claims, with 1415 still unsupported; the versioned results below distinguish changed judgments from changed text.

That finding came after the system had already done the visible work. It had loaded financial statements, projected revenue and earnings, produced charts, and written eight sections of equity research. The HTML reports looked like research. The question was whether their claims were supported by the information the writing agents had actually received.

For this assignment I ran the real `finrobot_equity` implementation at commit `2717499b8e30f242640af08c4ad9afd1113c2d45`. I froze the inputs, traced the outputs into HTML, checked financial values against SEC filings, independently rebuilt valuation arithmetic, repeated one company's generation three times, and added one improvement: a ninth verifier stage. The result is useful evidence about research automation, including a verifier that helps reviewers find problems but fails to make the prose reliably correct.

## First, make the experiment real

The course repository supplied instructions and wrappers. The upstream project contained the financial and agent implementations, including execution paths that were not interchangeable. I used its actual eight typed specialist agents and saved each complete evidence prompt, instruction, structured response, wire response, model setting, usage record and runtime. The baseline writer was GPT-4.1-mini at temperature 0.7. Every fresh run has a distinct identifier and real provider completion IDs.

The formal cohort is NVIDIA, AMD, Intel, Broadcom and Qualcomm, each with two actual peers. Apple and Alphabet also ran as required onboarding examples, but they do not count as semiconductor coverage. This matters: five successful API calls do not automatically make a coherent industry sample. Qualcomm’s licensing business, Broadcom’s infrastructure software, Intel’s foundry and the fabless companies’ economics also make consolidated comparisons imperfect.

Data access was an experimental constraint. FMP allowed the five onboarding symbols but denied Broadcom, Qualcomm and news endpoints. I did not buy a service or fill gaps with fabricated responses. Broadcom and Qualcomm instead use saved public SEC financial statements, exact filing contexts and Yahoo daily-price responses, with frozen real FMP peers. Missing subject multiples remain missing. Their valuation engine uses an explicitly labeled 12x fallback. News is limited to one issuer release or a literal SEC quarterly table, not a comprehensive media feed.

WRDS was investigated, but its observations were excluded from the model and the submitted package after reviewing the service’s AI restrictions. Public materials provided a workable route. Different data definitions and narrow news remain limitations, even when the system executes successfully.

The common raw-close return window runs from October 10, 2025 to October 9, 2026: Broadcom +11.37%, Qualcomm +14.27%, NVIDIA +25.18%, AMD +182.97% and Intel +187.87%. Those are differing magnitudes, not a balanced winner-and-loser sample. They exclude dividends; provider split conventions were not independently verified. This is a frozen contemporary exercise, not a historical backtest.

## Reproducing a number is different from trusting its meaning

I checked ten pipeline metrics each for NVIDIA and AMD against their original annual SEC filings. Sixteen values matched within display precision. Four EBITDA and EBITDA-margin differences could be reconciled through explicit components. They were not unexplained data errors.

For NVIDIA FY2026, GAAP operating income plus D&A is $133.230 billion, while the FMP EBITDA field is $144.552 billion. The latter equals net income plus tax, interest expense and D&A. AMD needs a different bridge: its simple net-income formula is $92 million above the provider’s $7.275 billion EBITDA; excluding $66 million discontinued-operations income and $26 million equity income explains the difference. The original [NVIDIA](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm) and [AMD](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm) filings make those bridges inspectable.

The forecast then changes the definition again. The default code eventually computes projected EBITDA as gross profit less SG&A, excluding R&D. AMD’s historical EBITDA margin is about 21%, but its first forecast margin becomes 39%. A reader might attribute the jump to operating leverage. The code provides a much simpler explanation: a material expense is missing from the forecast measure.

The default revenue growth path is 5%, 6%, 4%, with gross margin rising one percentage point and SG&A/revenue falling half a point annually. Those are universal assumptions, not company guidance. I then executed a separate offline parameter exercise: NVIDIA growth of 30%, 20%, 10%, and AMD growth of 15%, 10%, 8%, with both gross and SG&A ratios held at the pipeline’s rounded latest-actual levels. The rates describe deceleration from the companies’ observed growth; they are assumptions to interrogate, not fitted forecasts.

- **Company / EBITDA year: NVDA 2029E**; Default model target, USD: 333.71; Adjusted model target, USD: 464.43; Change: +39.17%.

- **Company / EBITDA year: AMD 2028E**; Default model target, USD: 377.84; Adjusted model target, USD: 398.17; Change: +5.38%.

The result illustrates why changing a growth rate is not enough. NVIDIA’s terminal revenue rises 48.25%, but lower margin assumptions reduce the target increase to 39.17%. AMD’s revenue rises 18.03%, while removing assumed margin expansion leaves a target increase of only 5.38%. The same shares, historical multiple and 10%-of-EV debt rule are used in both cases; the debt dollar amount is not held fixed. These conditional prices retain the original method’s economic weaknesses.

More seriously, AMD’s adjusted EBITDA margin is still 37.5%, compared with about 21% in the historical provider data. Holding margins flat did not restore the missing expense definition. I saved all four actual processor runs and independently checked the arithmetic. Both default replays match the original tables; networking is blocked and this adds no API cost. The new scenario outputs are separate from every writer prompt, verifier response and paired audit: the original verifier comparison is unchanged.

## Exact valuation arithmetic can expose weak economics

My independent notebook imports a standard-library calculation, not the upstream valuation engine. Across seven companies, 84 EV/EBITDA and DCF components reconcile. Nine notebooks, including two derived news versions, also execute after relocation without the upstream checkout or a network connection.

The result confirms what the program computes. It does not validate the assumptions. Its main multiple method applies an average historical multiple to third-year forecast EBITDA without discounting. Net debt is estimated as 10% of the resulting enterprise value. The DCF uses generic growth and discount rates; Intel’s negative starting FCF eventually produces a clipped zero target.

For NVIDIA, the multiple method gives $333.71 and DCF gives $72.52. When I propagate relative revenue shocks of +/-5% and margin shocks of +/-2 percentage points through the same multiple method, the target range is $308.40-$359.93. The structural difference between valuation methods is much larger than that local scenario range. A narrow sensitivity plot can look reassuring while leaving the largest uncertainties untouched.

The native sensitivity analyzer reports revenue and EBITDA changes. I therefore saved it separately and explicitly propagated each of its 25 scenarios into per-share valuation. A margin change from 42% to 44% is two percentage points; it is not a 2% relative increase. These grids are deterministic scenarios, not calibrated confidence intervals.

## Original V6 evaluation: audit what was in the prompt

My grounding rule is deliberately strict. If an assertion is true in the world but absent from the generating prompt, it is Unsupported. Later browsing cannot improve its baseline label. Contradicted means the prompt itself supplies a conflict. Supported means the prompt entails the claim or permits an explicit calculation.

The complete audit covers all eight sections for NVIDIA, AMD and Intel. Each nonblank source unit has explicit atomic premises or a nonfactual rationale, with offsets, hashes, exact evidence lines and reasons. This is Codex-assisted machine adjudication, not human validation. Early API-judge failures are preserved. Textual coverage does not prove perfect semantic decomposition, and the same model family can share biases.

The baseline counts are:

- **Company: NVIDIA**; Supported / total: 81 / 486; Unsupported: 388; Contradicted: 17.

- **Company: AMD**; Supported / total: 96 / 553; Unsupported: 447; Contradicted: 10.

- **Company: Intel**; Supported / total: 114 / 439; Unsupported: 312; Contradicted: 13.

- **Company: Total**; Supported / total: 291 / 1,478; Unsupported: 1,147; Contradicted: 40.

Competitor analysis has only 17 supported assertions out of 295. Risks have 20 out of 250. Their prompts request moats, competitive positioning, management execution and company-specific risks from financial tables and a single earnings excerpt. Many resulting claims are plausible. Plausibility does not make the supplied evidence sufficient.

Some errors are direct numerical-semantic contradictions. NVIDIA’s prose says contribution margins were consistently above 70%, while its own historical table includes 64.9% and 56.9%. A historical 68.3% revenue CAGR becomes a projection through 2029, even though the forecast path is 5%, 6%, 4%. Matching the digits alone is not enough; the period and actual-versus-forecast meaning are essential.

## Original V6 result: warnings improved more than correction

The verifier combines an evidence-bounded review with deterministic checks. It resolves citations to actual prompt lines, checks numeric bindings and source coverage, preserves original/revised mappings, and makes the renderer use the treated text. Unsupported claims retain a visible warning. A claim recognized as Contradicted must be corrected from evidence or withheld.

Real failures drove several revisions of this same improvement: exact quote copying failed, numeric bindings used the wrong values, source words were omitted, and a review cited only a company-name line to support a business description. The final V6 rejects structural-only citations. Earlier attempts, source versions and costs remain available. Development used the evaluated cohort, so this is not a held-out benchmark.

Under the original V6 evaluation, independent after auditing found 48 contradictions among 1,590 assertions. Support rises from 19.69% to 22.52%, but more repeated claims change the denominator. Verifiable numeric error falls from 25/171 to 29/217: a lower rate alongside a higher count. It would be misleading to present that as a clean accuracy improvement.

The useful gain is visible issue coverage. There were no baseline warnings. After treatment, warnings cover 1,158 of 1,232 unsupported or contradicted assertions: 93.99% recall, with 78.78% precision. The system retains 96.13% of independently supported lexical fact keys and 100% of supporting evidence spans. Those are imperfect retention proxies, not proof of semantic recall. It did not improve its score by deleting all information; it often preserves too much repeated unverified text.

The remaining failures matter. Seventy-four problematic assertions are unflagged, including eight contradictions. One says NVIDIA’s comparable multiple is above AMD even though the supplied figures are 33.7774x and 47.8456x. The verifier can misclassify a contradiction and then faithfully enforce the wrong label. A warning layer is useful for triage, but it is not a substitute for an analyst.

## Repetition reveals another source of uncertainty

I generated NVIDIA three times with identical frozen evidence, model and parameters. The 24 agent calls have distinct completion IDs; they are not cached copies.

No run states an explicit writer per-share target. One has no rating, one conditionally recommends holding or increasing exposure, and one explicitly says overweight. Numerical rating and target variances are therefore N/A. The cover’s FMP target stays at $339.17 with zero variance, which says nothing about the language model.

The first three distinct risk categories have pairwise Jaccard overlaps of 0.20, 0.20 and 0.50. The ordering is first mention, not economic importance. Only one pair shares enough categories for Kendall correlation, and that pair shares just two. Three draws are too few for a strong stability claim, but they are enough to show that fixed inputs do not imply a fixed investment narrative.

## What I would give a portfolio manager

I would provide the filing-linked historical table, the reconciled definitions, independent calculations, explicit scenario equations and claim-level error maps. These support screening and quality control. I would also show the source gaps and every fallback beside the output that uses them.

I would not hand over an unattended recommendation, position size or fair-value number. The forecasting definition, debt rule, multiple selection, source coverage and narrative entailment remain weak. The verifier improves disclosure but leaves both errors and substantial review burden.

The direct API ledger totals $10.045071 for 347 settled calls, including failed attempts and repairs, within the authorized $20. It records 1,303,571 input and 822,575 output tokens and 5,023.18 seconds of model-call time. These are conservative list-price estimates, not invoices; Codex-assisted authoring and adjudication costs are unavailable rather than assumed zero. Baseline eight-agent generation takes roughly 25–31 seconds per company in this environment, with verifier time and local processing recorded separately.

The experiment’s reproducibility package fixes the upstream commit, separates compatibility and verifier patches, preserves raw responses and prompts, and can replay all seven data pipelines with network sockets disabled. Clean patch application passes 108 tests and six subtests. Those checks establish executable provenance. They do not establish financial truth.

That distinction is the lesson I would keep: research automation needs an evidence contract at every boundary. A data field needs a definition, a forecast needs an assumption, a target needs a valuation method, a sentence needs supplied evidence, and a verifier needs its own evaluation. When those contracts fail, the report should make the failure visible.

## What happened when I strengthened the reviewer

I retained the V6 experiment and ran a separate follow-up with stricter instructions on fiscal periods, company comparisons and growth horizons, plus exact-string deduplication. The full same-input, eight-section study on NVIDIA, AMD and Intel reduced unflagged problematic assertions from 74 to 31. Warning recall rose from 93.99% to 97.70%, with precision rising from 78.78% to 82.42%. That is a measurable improvement in triage. Yet eight contradictions still passed without warnings. The newer output had 325 supported claims among 1,673 assertions (19.43%), 41 contradictions and 23 errors among 180 verifiable numeric claims. Supported lexical-key retention fell to 76.06%; paraphrases affect that proxy, and evidence-span retention was 93.02%. More warnings did not establish a trustworthy editor.

The judge itself supplied another failure case. A model evaluating warned prose skipped 18 NVIDIA units as nonfactual. After I hid only the warning prefixes, it stopped skipping those units but still combined separate facts and missed explicit prompt evidence. Both failed proposals are preserved. A Codex task agent explicitly adjudicated every changed unit against the original evidence; no human validation was performed. The same development cohort influenced the instructions, so these findings are not a held-out test.

A bounded reasoning-model pilot also produced a negative result. One frozen NVIDIA section on GPT-5.4 used all 26,000 allotted output tokens as reasoning, returned no visible review and cost $0.400930. I did not expand that high-effort setting or count it as a successful verifier run; a separate low-effort treatment followed. The raw response is saved. Together these experiments argue for measuring information retention, missed contradictions and evaluator failures alongside warning rates. The complete package retains V6 as the historical original experimental release and reports the V7.1 candidate separately; neither is an unattended investment product.

## I also had to audit the auditor

A full three-company reasoning-model treatment made 24 new reviews over the same frozen writer evidence. The first evaluator result looked stronger than the original reviewer, but a second task-agent review found an additional problem: some labels had silently added a time horizon or comparator that the sentence never specified. A table spanning actuals and forecasts does not by itself say which endpoints produced a CAGR. Ambiguity deserves an Unsupported label; it is not evidence of a false calculation.

I preserved the original labels and recorded a symmetric before/after correction. The corrected reasoning output has 530 supported claims among 1,958, with eight contradictions. This is a measurement correction, not a software improvement. It also withdraws several incorrectly Supported labels, and retains the supported half of a compound assertion while marking its unsupported judgment separately.

Then the software changed. A narrow rule now calculates CAGR from the provided endpoints and spells out both intervals. NVIDIA’s supplied historical 68.3% growth and the 37.49% implied across 2022A-2029E answer different questions; the final year in the latter is a model forecast. A second narrow check names the actual years behind an EV/EBITDA minimum and maximum. Both retain original text, evidence locations, formulas and reasons; neither uses later web facts or the audit labels to choose a number.

- **Version, under corrected evaluation: Matched baseline**; Supported / claims: 286 / 1479; Grounding: 19.34%; Contradicted: 31; Numeric errors / verifiable: 25 / 172.

- **Version, under corrected evaluation: Reasoning V8.2, unchanged prose**; Supported / claims: 530 / 1958; Grounding: 27.07%; Contradicted: 8; Numeric errors / verifiable: 8 / 430.

- **Version, under corrected evaluation: V9: explicit CAGR horizons**; Supported / claims: 566 / 1982; Grounding: 28.56%; Contradicted: 1; Numeric errors / verifiable: 1 / 459.

- **Version, under corrected evaluation: V9.1: explicit actual multiple range**; Supported / claims: 570 / 1985; Grounding: 28.72%; Contradicted: 0; Numeric errors / verifiable: 0 / 461.

The final 1415 Unsupported claims are a substantial remaining limitation. The reported zero numeric errors applies only to the 461 verifiable numeric claims; another 51 numeric claims remain unverifiable. This is not a claim that every numerical assertion is correct. Warning recall is 100.00% and precision 89.90%, with 0 problematic claims still unflagged. Supported lexical retention is 9.68%; evidence-span retention is 81.82%. Those proxies differ, so I cannot claim complete information preservation. The narrow checks were developed on this cohort; a low observed error count is not an out-of-sample guarantee.

The new text was audited across every section, with independently implemented endpoint arithmetic and exact source hashes. That review was machine-assisted and no human claim validation occurred. The real paid reasoning treatment cost $3.0613925; these deterministic derivatives add no API tokens or charges. The complete saved direct-API ledger remains $10.045071, including failed attempts and pilots, within the approved $20 cap. Task-agent authoring/evaluation cost is unavailable, not zero.

I also tested a fresh directory containing the actual submission dependencies, prevented the worker and renderer from reading the original directory, disabled network calls, and forced a future execution clock. Reports must reproduce their saved data and date rather than quietly refetch a current price or drop a valuation when an old absolute path disappears. That is a practical reproducibility test a polished report alone cannot pass.

The useful result is an inspectable analyst aid: filing-linked metrics, reproducible arithmetic, scenario sensitivities and traceable warnings. The remaining unsupported business narratives, causal claims and uncertain forecasts still require review. The project is a measured research experiment, not an autonomous portfolio recommendation.

