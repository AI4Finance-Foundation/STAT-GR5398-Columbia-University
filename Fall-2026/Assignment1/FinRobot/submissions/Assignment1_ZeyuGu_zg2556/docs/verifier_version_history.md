# Verifier version history and failure preservation

The baseline writer prompts and eight original agent instructions were not altered. V1, V2 and V3 are versions of the same course Option A ninth-stage verifier, not additional research interventions.

## Observed V1 failure

`runs/NVDA_final_1` preserves all three actual paid tagline attempts: the original and two bounded validation repairs. None passed deterministic validation, and no final verified report was released from that run. Total recorded cost was $0.008157, with 10,212 input tokens, 2,545 output tokens and 16.140 seconds of model-call runtime. The first two attempts are under `attempt_history/verifier/tagline`; the third is the canonical `verifier/tagline` directory. All full request/response, error and cost evidence remains intact.

The model reconstructed table quotes with changed spacing and abbreviated content, so they were not exact substrings. It bound a threshold such as 70% to a different observed value (73.5%), repeated the same number binding, omitted year bindings, and supplied an empty unsupported atomic claim. The original validator rejected the output, but its aggregate failure message did not expose all corrective details.

## V2 changes

- The model returns only evidence line IDs. Code resolves the complete original prompt line verbatim, preserving pipe-table spacing and source identity. Invalid or blank line IDs still fail.
- Instructions explicitly require one binding per distinct numeric token, including years; a forecast year must cite the table header and its metric must cite the matching data row. Threshold language is rephrased to a directly evidenced value rather than forcing an unequal binding.
- Any deterministic claim error now rejects the whole candidate with claim IDs, failing numeric tokens and reasons. This feeds the existing maximum of two validation repairs. Supported claims cannot be silently discarded after a failed binding.
- Unsupported atomic claims remain visible with an explicit `[UNVERIFIED - evidence not supplied]` annotation. They stay classified Unsupported, with `retained_as_verified=false` and `verification_status=unverified_annotation`. This preserves information for review and must not increase grounding scores. The independent after audit judges the underlying assertion, not the annotation.
- Nonfactual units are preserved as commentary with no verification claimed. Truly contradicted claims require an evidence-based correction or removal from the released factual narrative. Source units, original quotes, claims and reasons remain recorded.

Source snapshots and SHA-256 values are in `evidence/verifier_versions`. The actual V1 failure is an offline regression case in `tests/test_evidence_verifier.py`; synthetic controls test missing units, invalid citations, wrong numbers, hidden claim deletion, unsupported retention, and forecast/year bindings. At implementation handoff, 45 combined safety/resume/verifier tests passed. These are software checks, not human validation of financial claims.

## Limits and interpretation

A full evidence line proves citation identity, not semantic entailment or correct alignment to a company/year column. The model can still miss an atomic subclaim, choose the wrong table column, misclassify a causal statement, or label a factual unit nonfactual. Lexical source coverage is necessary but not sufficient for semantic completeness. Independent full-section audits and before/after retention measurements are therefore required. An unsupported annotation is useful disclosure, not a verified fact, and may leave the measured grounding rate unchanged or worse.

A new V2 run ID is required. Resuming the failed V1 run with changed instructions/schema correctly fails the immutable-contract check. Costs of all failed attempts and repairs belong in total experiment costs; successful-section response files alone undercount them.

## V3: deterministic failures handled per claim

Real V2 attempts continued to fail number-binding checks even after trying the larger configured model. The actual request/response/error records remain in `runs/NVDA_final_v2` and `runs/NVDA_final_gpt41`; they are not relabeled as successful V3 generations. V3 changes the release policy so that one malformed numeric binding no longer blocks all independently valid claims in a section.

- Missing/duplicated source units, uncovered source words, absent original quotes and attempts to discard supported claims still reject the whole review. These are structural or information-retention failures.
- A valid Supported claim remains in the released text. A valid evidence-based correction of a Contradicted claim may also be released, with the original and correction both retained in its mapping.
- If a non-Contradicted claim fails deterministic citation or numeric checks, the report retains the exact **original quote**, prefixed `[UNVERIFIED - deterministic evidence/number check failed]`. A failed proposed rewrite never replaces that original. Its `retained_as_verified` is false; exact diagnostics and numeric results remain in the claim record and section failure log. The original model classification is preserved for audit, but does not override the failed verification status.
- A Contradicted claim whose proposed correction fails validation is withheld from the report, while its original quote, attempted correction, evidence and error record remain in the audit. A section with no releasable content contains an explicit `[FACTUAL CONTENT WITHHELD - N/A]` notice rather than fabricated filler. This notice is not a factual claim or an increase in grounding.
- Unsupported claims retain the existing explicit unverified annotation. Every warning is disclosure, not evidence; independent after auditing must judge the underlying proposition and must not award Supported status merely because an annotation describes the draft.
- A narrow deterministic repair can add an omitted year identity binding: the draft must explicitly refer to a year/fiscal period, and that exact numeric token must appear on an already cited line with year context or a table header containing actual/forecast year labels. The selected line and exact quote are recorded with `binding_source=deterministic_exact_cited_year_identity`. This never replaces an existing wrong binding, invents a financial value, performs an inequality transformation or establishes company/period semantics.

V3 output records verified, annotated, withheld and deterministic-failure counts plus claim retention. Keeping an unverified claim is deliberately distinct from verifying it. Because a quote can contain more than one atomic proposition, retaining exact original quotes after failed checks may repeat facts and may leave grounding unchanged or worse. The independent full-section claim audit and its retention/coverage metrics remain the experiment's outcome measures.

At V3 handoff, **50 offline verifier and resume tests passed**. Additional offline regression applied the V3 local validator to preserved real V2 responses without any model call: the mini-model tagline retained three failed claims as warnings and verified none; the larger-model tagline retained one checked claim, three warnings, and withheld one claim. These are implementation regressions documented in `evidence/verifier_versions/v3_offline_regression.json`, not V3 experimental results or human validation. New `*_final_v3` run IDs and new recorded model responses are required for the formal before/after experiment. V1/V2 source archives, raw attempts and their costs remain unchanged.


## V4-V6 final experiment

V4 permits only a finite list of complete uncovered connector/heading spans, retaining exact offsets. Omitted substantive facts still fail. See `verifier_v4_validation_change.md`. Real V4 failures showed that evidence-supported rephrasing with action=correct was being rejected solely because the model labeled the original Supported. V5 allows that action when all source/citation/numeric checks pass; unvalidated proposed rewrites still cannot replace original assertions. Saved reviews are reused only after validating source/evidence hashes, original prompt prefix, provider response and budget settlement; reuse is explicitly recorded, not a new generation.

V6 adds a substantive-line guard: a company identity line, blank line, heading, separator or table header alone cannot establish a business/financial proposition. Supported or corrected claims with only such references fail closed into an unverified original quote (or withholding for rejected contradictions). This narrow guard fixes an observed citation failure; a substantive line can still be semantically irrelevant or point at a wrong company/year.

All seven `*_final_v6` outputs are real verifier treatments. Five inherit prior paid reviews and rerun deterministic validation; AVGO/QCOM have new GPT-4.1 reviews. Exact version hashes are in the source archive manifest. The three-company complete independent audit finds 358/1590 supported and48 contradicted assertions. Warning recall is93.99%, precision78.78%;74 problematic claims are not warned. No improvement is inferred from a software test alone. Baseline and treated narrative errors remain preserved as experimental evidence.

Final patches apply on a clean pinned checkout;101 tests and6 subtests passed, then seven saved data pipelines and14 saved-text renders ran with networking disabled. The final renderer adds the same research-status notice to baseline and final and displays table NaN as N/A; none of the audited section text changes.

## Completed V7.1 follow-up

See `verifier_v7_followup.md` for seven rendered candidates, all-section paired NVDA/AMD/INTC audits, separate applicable development patches, improved warnings and residual semantic failures. Exact source SHA and reuse lineage remain in each run. The main patches remain V6. A subsequent GPT-5.4 single-section pilot failed at its completion-token limit and was not expanded. All charges are in the common ledger; final summaries include failures. Publication is explicitly deferred by the user.
