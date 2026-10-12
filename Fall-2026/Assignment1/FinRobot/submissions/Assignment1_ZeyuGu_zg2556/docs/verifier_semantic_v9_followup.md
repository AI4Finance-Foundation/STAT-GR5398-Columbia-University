# Evaluator correction and deterministic horizon/range follow-up

## 1. Same-input reasoning treatment and evaluator corrections

The follow-up made 24 fresh reviews, eight each for NVIDIA, AMD and Intel, with `gpt-5.4-2026-03-05`, low reasoning effort, a 20,000 completion-token ceiling and omitted temperature. It held writer drafts, original prompts and the semantic-review instructions fixed. These calls cost $3.0613925, used 287,867 tokens and took 2,062.10 seconds of model-call time. V8.2 revalidated captured responses offline; this is not another independent generation. The failed high-effort pilot cost $0.400930 and the successful low-effort pilot $0.138385; both remain in the cumulative ledger.

A second audit found that the evaluator sometimes added a time window, comparator or universal quantifier absent from the source sentence. For example, a sentence describing a table that extends into forecasts does not necessarily claim that a displayed CAGR was calculated across that full table. Such under-specified assertions are Unsupported, not automatically Contradicted or Supported. The correction reviewed both before and after, including withdrawing several unsupported Supported labels. An AMD baseline assertion was split into a supported observed decline and an unsupported assertion that the level remained elevated. The saved mapping changes 31 original atoms into 32; 120 original source/audit files retain their hashes. No report text, evidence prompt or model response changes in this measurement correction.

The corrected baseline is 286 Supported, 1,162 Unsupported and 31 Contradicted among 1,479 claims. Corrected V8.2 is 530 Supported, 1,420 Unsupported and 8 Contradicted among 1,958. Its warning recall is 1,416/1,428 (99.16%); seven contradictions and five unsupported assertions have no warning. The previously reported 16 contradictions and 12 unflagged contradictions are preserved historical evaluator results, superseded for this comparison. Changing labels is not a verifier performance gain.

## 2. Narrow deterministic semantic checks within Option A

The verifier now has two narrow deterministic release checks. V9 reads the subject's supplied Revenue row and fiscal A/E header, verifies that the displayed CAGR rounds to the actual-endpoint calculation, and explicitly names both actual and full-table horizons. The calculation is `(ending / starting) ** (1 / year_difference) - 1`; year differences, rather than column counts, determine elapsed intervals. Forecast endpoints remain labeled as model forecasts. For NVIDIA, the historical cell is 68.3% while the 2022A-2029E endpoint calculation is 37.49%; AMD's extended result is 13.59% and Intel's is -3.59%. The twelve matching original sentences are preserved alongside each clarification. Ambiguous originals are not retrospectively declared false.

V9.1 also handles one narrowly matched EV/EBITDA range sentence. It replaces an unspecified recent-years window with an explicitly dated window from the supplied actual rows and calculates its minimum and maximum. The rule reads no audit label, hard-coded ticker result or later information. Missing, inconsistent or ambiguous inputs cause abstention. Original model classifications, numeric bindings, release status and revised prose remain separately saved; these checks do not alter paid instructions or captured responses.

| Version, under corrected evaluation | Supported / claims | Grounding | Contradicted | Numeric errors / verifiable |
| --- | --- | --- | --- | --- |
| Matched baseline | 286 / 1479 | 19.34% | 31 | 25 / 172 |
| Reasoning V8.2, unchanged prose | 530 / 1958 | 27.07% | 8 | 8 / 430 |
| V9: explicit CAGR horizons | 566 / 1982 | 28.56% | 1 | 1 / 459 |
| V9.1: explicit actual multiple range | 570 / 1985 | 28.72% | 0 | 0 / 461 |

The final version retains 1985 substantive atomic assertions, 134.21% of the baseline count, including 1415 Unsupported and 0 Contradicted. Numeric claims include 51 that remain unverifiable. A low observed contradiction count is not proof of factual correctness: unknown claims remain unknown, the evaluated cohort informed development, and there is no held-out test. No general semantic-verification guarantee is claimed.

Independent warning recall is 100.00%, precision 89.90%; 0 unsupported/contradicted assertions remain unflagged, including 0 contradictions. Supported lexical-key retention is 9.68%, while source-evidence-span retention is 81.82%. These disagree because rephrasing, expanded endpoint statements and decomposition affect the proxies. Neither is unique-fact recall; claim-count growth does not prove information preservation. Warned claims remain in all substantive denominators.

All eight final sections of each company are audited against their original writer prompts. An identical source unit can inherit its corrected prior label only when the section, text and evidence hashes match. Changed units receive independent decomposition and calculations; the CAGR evaluator uses a separate Decimal logarithm/exponential implementation, not the verifier function. This is task-agent-assisted evaluation, not human validation; its exact task-agent model usage is unavailable in the direct-API ledger. The audit remains fallible.

V9 and V9.1 reuse the same 24 actual paid responses with model-call methods and network connections disabled. V9.1 adds zero API tokens and $0 new model cost; verification plus rendering takes 7.05 seconds across the three saved derivations. The $3.0613925 original-review cost is allocated to this treatment for comparison, not billed again. Separate upstream compatibility and Option A patches apply to the pinned commit; the final Option A patch SHA is `c7d6fab68e32d49f2237cfb201ca8fb25422f823a7b6f09ac4d428e0d046d889`.

The explicit audit correction, V9 and V9.1 scorecards are separate. Original V6/V7.1 and V8.2 prose and labels remain available as experiment history. The final three-company reports are linked from the README; five-company coverage and the original stability experiment are unchanged. Additional safeguards improve traceability within Option A, not a second research option.

## Recompute from saved evidence

The README gives clean-patch offline replay commands. Use `config/grounding_reasoning_v9_1_pairs.json` with `grounding_audit.py compare` and `analyze_verifier_annotations.py` for exact metrics. Historical annotation correction is in `evidence/post_release_review/corrected/`; V9/V9.1 independent change reviews and source hashes are in their selected audit directories. Never promote a saved-response replay to an independent generation. Medium and the course PR remain deferred.
