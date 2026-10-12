# Full reasoning-verifier follow-up

This separate Option A treatment uses `gpt-5.4-2026-03-05`, low reasoning effort, a20,000 completion-token ceiling and omitted temperature. Writer drafts, original evidence prompts and V7 semantic instructions are frozen. Three companies each received eight new reviews; neither the successful low-effort pilot nor an earlier cached review substitutes for a new treatment call. No market data were refetched. Raw paid V8.1 outputs remain immutable; V8.2 derives a separate final text offline using those captured responses and a read-only ledger. Derivation is not independent generation.

V8.2 fixes transport representations, not financial truth: an A/E year wrapper must occur exactly in both the revised sentence and cited original header before a numeric binding alias can be accepted. Currency aliases require identical explicit wrappers. It invents no source line, entity, number or revised prose. Raw bindings and normalization maps remain saved. The clean Option A patch SHA is `f1366e30d9922767f70a58860ce48dc64740e6f46b404ea045a3b93d0cd8073a` on upstream `2717499b8e30f242640af08c4ad9afd1113c2d45`; the clean applied checkout passes the recorded regression suite. Baseline compatibility and research patches stay separate.

| Company | Baseline supported / total | Treatment supported / total | Contradictions | Verifiable numeric errors (counts) |
| --- | --- | --- | --- | --- |
| NVDA | 82 / 486 | 176 / 670 | 17 → 6 | 11 → 4 |
| AMD | 96 / 553 | 142 / 645 | 10 → 7 | 5 → 7 |
| INTC | 114 / 439 | 216 / 643 | 13 → 3 | 9 → 2 |
| ALL | 292 / 1478 | 534 / 1958 | 40 → 16 | 25 → 13 |

Overall grounding changes from 19.76% to 27.27%; contradiction rate changes from 2.71% to 0.82%. Treatment numeric errors are 13 among 435 verifiable numeric claims (2.99%); 51 further numeric claims are unverifiable. The alternate denominator including unverifiable claims yields 2.67%. Company/section denominators and all atomic claims are saved, not sampled.

Warning recall is 99.16%, precision 89.65%, with 12 unflagged unsupported/contradicted assertions including 12 contradictions. Warned assertions remain in the substantive audit denominator. The visible claim count is 132.48% of baseline; supported lexical-key retention is 9.47%, and supporting evidence-span retention is 81.82%. The verifier's own reported visible retention is 99.26%; it is not independent accuracy evidence. There are 214 duplicate lexical claim instances. Rephrasing, added endpoint statements and changed decomposition alter denominators. These proxies are not unique semantic-fact recall or proof that no information was lost; consult original/revised claim mappings.

The three-company paid source reviews cost $3.061393, use 287,867 tokens and take 2062.10 seconds of model-call time. The offline derivative adds $0.000000 in new model usage. These two figures are allocation and incremental usage, not amounts to add twice. The global shared ledger also includes both pilots, preflights, abandoned repairs, earlier candidates and machine-judge failures. Task-agent adjudication usage is unavailable in that direct-API ledger, not zero.

Every source unit was checked against the original writer prompt. Exact already adjudicated text/evidence may carry its prior label; changed text gets a fresh explicit task-agent decomposition. The breakdown is {"AMD": {"carried_units": 42, "explicitly_reviewed_changed_units": 417}, "INTC": {"carried_units": 31, "explicitly_reviewed_changed_units": 458}, "NVDA": {"carried_units": 59, "explicitly_reviewed_changed_units": 460}}. Verifier classifications are never evaluator truth. This remains machine-assisted adjudication without human review; implicit premises or classifications can still be wrong. Development and model selection used this cohort, and there is no held-out generalization or isolated model-effect claim.

One baseline annotation erratum is explicit: the supplied NVIDIA news already reports Data Center sales, which supports the narrow claim that it serves that market. The earlier baseline auditor omitted that evidence. This study corrects that single Unsupported label to Supported, without changing generated text or adding later evidence. NVIDIA's baseline is consequently82/486, rather than81/486 in the preserved V6/V7.1 studies. The correction reduces the apparent improvement; the original audit and exact before/after erratum remain saved. Do not compare totals across these studies without this one-claim adjustment. A separate AMD after-audit erratum removes an auditor-invented comparator: the phrase about NVIDIA's higher multiples does not explicitly say higher than AMD now. Its ambiguous comparator is Unsupported, not a positively disproven current NVIDIA-over-AMD comparison. Both annotation versions are preserved; full-study pairs select the corrected version. Neither correction changes report text or original evidence.

The high-effort pilot returned no visible text after26,000 reasoning/output tokens and cost$0.400930. A separate low-effort pilot completed4,372 input and8,497 output tokens at$0.138385; offline wrapper normalization accepted11 revisions but one citation still failed. Both outcomes and raw responses are preserved. The full treatment's successful runs do not erase those failures.

Residual cases matter. NVIDIA still passes the generic68.3% CAGR as applying through2029E, while original endpoints imply about37.49% for2022A–2029E. It also warns on the true claim that2023A–2025A CAGR exceeds68%; the direct result is119.95%. Earlier baselines and V6/V7.1 measurements remain intact. This candidate cannot be represented as error-free or ready for unattended portfolio decisions. Source-linked tables and independently reproduced valuation calculations can support analyst review; unsupported business narratives, causal interpretations and unflagged horizon errors still need intervention.

Recompute all saved metrics from the submission directory with the documented dedicated Python environment, using new output directories:

```powershell
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_reasoning_v8_2_pairs.json --output reproduction/reasoning_comparison_new
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_reasoning_v8_2_pairs.json --output reproduction/reasoning_annotations_new
& $RunPython scripts/summarize_runs.py
```

Each `runs/TICKER_final_reasoning_v8_2/offline_derivation.json` names its paid source. Applying the separate V8.2 patches to a clean pinned checkout and running `scripts/revalidate_reasoning_offline.py --upstream $CleanV82 --tickers NVDA AMD INTC` reuses actual responses without paid calls. Original data recapture is a different experiment and must use new snapshot/run IDs. Medium and course PR publication remain explicitly deferred; no URL is invented.
