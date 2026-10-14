# Development follow-up: stronger semantic instructions did not eliminate false assertions

This is an adaptive development experiment on the same frozen writer inputs. V6 remains the canonical local release with the completed follow-up reported separately. V7/V7.1 outputs, successful and failed responses, patches and audits remain separate and are not silently substituted for V6.

V7 retains the ninth-stage architecture and adds explicit instructions about actual/forecast periods, entity/metric alignment, comparison horizons, CAGR endpoints, universal trend qualifiers and flattened filing/news columns. It also emits an exactly repeated released string only once per section, preserving all source mappings. Both changes belong to Option A; there is no ablation isolating their effects. The evaluator feedback used in prompt design makes this a development-cohort result, not a held-out efficacy estimate.

All seven companies had genuine new GPT-4.1 verifier draws. INTC stopped after bounded structural repairs because an uncovered “In summary” connector was treated as missing factual content. V7.1 permits only three additional complete discourse spans: “in summary”, “in conclusion”, “overall”. Negation, causal terms and substantive omissions remain rejected. It revalidates real V7 responses with immutable request/wire/cost provenance, and generates only missing sections. Revalidation is not independent generation. Separate patches apply cleanly to the pinned upstream under `evidence/verifier_versions/patches_v7` and `patches_v7_1`. The latter passed 103 tests and six subtests.

Every V7.1 report was rendered with exact text and local assets, with source hashes asserted by `experiment.py`; timing is in `evidence/render_v7_study.json`. These candidate reports have not replaced the canonical V6 browser-QA inventory.

## Complete independent comparison

The original baseline evidence is unchanged for NVDA, AMD and INTC. Across all eight sections, 360 exactly unchanged sentence units reuse earlier independent labels, with text/evidence identity checked. Every one of the 173 changed units was explicitly decomposed and adjudicated by the Codex task agent against the original writer prompt. No human verification occurred. The exact specifications, source offsets, evidence citations, decision rationale and carry-forward maps are retained under each `*_after_v7_1_working`/`*_adjudicated` directory.

| Ticker | Units | Claims | S | U | C |
|---|---:|---:|---:|---:|---:|
| NVDA | 191 | 572 | 91 | 465 | 16 |
| AMD | 175 | 602 | 99 | 492 | 11 |
| INTC | 167 | 499 | 135 | 350 | 14 |

The common baseline has 291/1478 supported claims (19.69%), 40 contradicted claims (2.71%), and 25/171 errors among verifiable numeric claims (14.62%). V7.1 has 325/1673 supported (19.43%), 41 contradicted (2.45%), and 23/180 verifiable numeric errors (12.78%). Including unverifiable numeric claims in the denominator gives 23/222 (10.36%); that broader denominator can dilute apparent error rates.

Exact supported lexical-key retention is 76.06%; supported evidence-span retention is 93.02%. These are conservative lexical/location proxies, not semantic coverage guarantees: new paraphrases and finer decomposition affect keys. Claim count is 113.19% of baseline and verifier-reported visible retention is 98.92%. Neither a larger claim count nor repeated evidence establishes better information retention. Inspect original-span mappings and both retention definitions.

Warning recall rises from V6's 93.99% to 97.70%, and warning precision from 78.78% to 82.42%. Unflagged unsupported/contradicted claims fall from 74 to 31; unflagged contradictions remain 8. Raw denominators differ because of changed phrasing, decomposition and duplicate removal. This is a measured improvement in warning coverage, not proof that the narrative is factually safe. All 41 contradictions, including warned ones, remain errors in the underlying assertions.

The selected three-company V7.1 responses allocate 186,938 tokens, $0.807346 and 276.42 seconds to the treatment. V7.1 itself adds $0.059582; inherited calls were paid in V7. Failed repairs, the other companies, pilot judges and authoring are outside this selected-treatment subtotal and belong in the global accounting. Codex task-agent adjudication usage is unavailable, not free.

## The evaluator also failed

The unblinded GPT-4.1 proposal incorrectly declared 18 changed NVDA units nonfactual because they carried UNVERIFIED annotations. A second proposal run hid only the warning prefix in judge input, preserving the complete original text, offsets and reversible mapping outside that input. It no longer skipped those units, but still combined independent premises and sometimes missed explicit supplied facts—for example, the $89bn NVIDIA Data Center amount on original evidence line45. Neither proposal set is canonical. `evidence/v7_evaluator_failure_summary.json` records coverage failures, and the task-agent specifications replace the inadequate proposals. This demonstrates why automated label counts and schema coverage alone are insufficient.

Known residual issues include period labels, CAGR horizons and peer-comparison semantics. Formal submission and zero-factual-error certification remain unresolved. Medium and the course PR are explicitly deferred by the user; no publication is authorized now.

## Recompute

Run `scripts/grounding_audit.py compare --pairs config/grounding_v7_1_pairs.json --output NEW_DIRECTORY` and `scripts/analyze_verifier_annotations.py --pairs config/grounding_v7_1_pairs.json --output ANOTHER_NEW_DIRECTORY`. These operate only on saved data. The exact paid plans are `config/verifier_v7_plan.json` and `config/verifier_v7_1_plan.json`; they are historical experiments, not a recommendation to repeat paid failures. Freeze a new destination and check the cumulative ledger before any fresh generation.

## Closed reasoning pilot

The one-section GPT-5.4 pilot failed with an empty visible response after its 26,000 completion-token limit. Saved provider usage supports $0.400930 and278.07seconds, now reconciled in the shared ledger. There was no usable review and no rollout. See `evidence/reasoning_pilot/NVDA_company_overview/reconciliation.json`. The consolidated research PDF reports this negative result and both full V6/V7.1 studies. Main V6 release selection is retained; neither treatment satisfies strict factual-correctness certification.
