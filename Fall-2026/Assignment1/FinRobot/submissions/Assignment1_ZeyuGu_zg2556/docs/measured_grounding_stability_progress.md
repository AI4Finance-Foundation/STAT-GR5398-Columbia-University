> Current release note: Historical progress note retained. Current complete results supersede partial counts below:1478 baseline/1590 after atomic claims across NVDA AMD INTC and all8 sections; canonical paths in config/grounding_v6_pairs.json; stability in audits/stability/NVDA_adjudicated; no human validation.

# Measured grounding and stability progress

This note records partial real results, not completion of the assignment. It supplements the earlier research PDF draft and must be incorporated when the report is finalized.

## Explicit baseline claim adjudication

All eight baseline sections were decomposed through a second, explicit machine review, using only each writer's original saved evidence prompt. No human review occurred. Raw API judge proposals and failed attempts remain separate. Source-unit coverage alone is not proof that a reviewer recognized every implicit premise; the full judgments, source quotes, evidence lines and reasons are inspectable.

| Ticker | Reviewed source units | Atomic claims | Supported | Unsupported | Contradicted | Grounding rate | Incorrect numerical claims / all numerical claims |
|---|---:|---:|---:|---:|---:|---:|---:|
| NVDA | 148 | 486 | 81 | 388 | 17 | 16.67% | 11 / 65 |
| INTC | 146 | 439 | 114 | 312 | 13 | 25.97% | 9 / 68 |

Sources: `audits/grounding/NVDA_before_adjudicated/` and `INTC_before_adjudicated/`. Numeric denominators include unverifiable numerical claims; each CSV also supplies the error rate conditional on verifiable numbers. These are two-company baseline results only. No three-company cohort result or verifier improvement is claimed. The evaluation protocol was refined after raw-judge pilot failures and is not a held-out preregistration.

## Three actual independent writer runs

`audits/stability/NVDA_adjudicated/generation_independence.csv` verifies 24 distinct provider responses across three actual eight-agent runs. Their frozen inputs and model settings match. Measurement and failed extraction calls do not count as additional writer replicates.

| Run | Explicit recommendation | Explicit writer per-share target |
|---|---|---|
| NVDA baseline 1 | Not stated | Not stated |
| NVDA baseline 2 | Conditional hold or increase, depending on portfolio strategy | Not stated |
| NVDA baseline 3 | Continued overweight | Not stated |

Only one run supplies a unique standard rating. Rating variance is therefore N/A; a one-observation zero is not presented as stability. Writer target variance is N/A. The separate FMP consensus target is $339.17 in all three frozen inputs, with sample variance zero; this measures input reuse, not LLM behavior.

The complete risk-unit inventory contains every risk sentence and its assigned exposures, with explicit reasons where no additional specific exposure is identified. Three distinct categories are selected by first substantive mention, using a fixed taxonomy order for categories in the same sentence. This convention is not an importance ranking. Revenue/segment concentration is kept separate from customer concentration.

The selected lists are demand cyclicality / macro conditions / competition for run 1; demand cyclicality / revenue concentration / export and geopolitical exposure for run 2; and demand cyclicality / revenue concentration / supply capacity for run 3. Pairwise Jaccard overlaps are 0.20, 0.20 and 0.50. Kendall agreement is N/A for pairs sharing one category, and 1.0 for runs 2 and 3, which share two categories in the same order. Small samples, subjective categories, and arbitrary within-sentence ties substantially limit these numbers; detailed risk language can differ even when categories match.

This explicit review replaces an incomplete measurement attempt in which an API extractor returned only three exposures despite instructions to inventory all risks, and rejected the conditional recommendation as a nonstandard label. All earlier responses, errors and charges remain preserved. Codex authoring/adjudication tokens and cost are not instrumented by the direct API ledger and are marked unavailable rather than zero.

Offline recomputation of this stability review, using a fresh output directory:

```powershell
python scripts/stability_adjudication.py --runs runs/NVDA_baseline_1 runs/NVDA_baseline_2 runs/NVDA_baseline_3 --output audits/stability/NVDA_adjudicated_replay
```

This command recomputes measurements from the three saved generations. It never creates a new independent generation or invokes a model.
