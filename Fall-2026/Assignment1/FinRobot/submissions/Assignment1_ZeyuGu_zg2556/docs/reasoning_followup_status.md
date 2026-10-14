<!-- COMPLETED-REASONING-CHECKPOINT -->
# Completed follow-up: historical checkpoint below

The full three-company experiment and independent machine audit are now complete. The following original status is retained for chronology and is superseded by [the complete study](verifier_reasoning_v8_followup.md), the updated research report, and evidence/reasoning_study_summary.json. All 24 paid source calls settled; offline derivations are not new draws. Publication remains deferred.

---

# Reasoning-verifier follow-up: local work in progress

As of 2026-10-12 05:22 UTC, the three-company treatment is still generating. NVIDIA is complete and independently machine-adjudicated across all eight sections. AMD is generating; Intel has not started. Do not present the NVIDIA-only result as the required complete three-company comparison. No publication is authorized.

The current research PDF, article and main V6 release predate this follow-up. Preserve them as a checkpoint; their old total-cost figure and assertion that no further pilot is running are superseded by this status and the shared ledger. Final consolidation is pending the complete experiment. The shared settled total at this checkpoint is $8.4013865, with one $0.409975 open maximum reservation; the user-approved cumulative cap is $20. The new serial plan has a tighter $14 absolute cumulative guard, not a new $14 allowance.

The real treatment uses pinned `gpt-5.4-2026-03-05`, low reasoning effort, 20,000 completion tokens, no explicit temperature, identical frozen writer prompts/drafts and V7 semantic instructions. Each company's eight reviews are generated independently of the one-section pilot. Paid runs use the immutable clean V8.1 source. V8.2 then revalidates those captured responses offline: it accepts a bare claim-year binding only if the same complete A/E year wrapper occurs in both revised sentence and cited original header. No model response, value, reference or evidence is invented. V8.2 passes 132 offline tests plus six subtests on its clean pinned checkout. Source and clean patches are under `evidence/verifier_versions/`.

NVIDIA's offline derivative changes two sections; the paid source responses and all prior versions remain intact. `ReadOnlyLedger` rejects reserve/settle operations and new model calls are disabled. This derived output is not a new independent generation. The paid NVIDIA run used 101,015 total tokens, cost $1.093175 and took 717.8393 seconds of model time. The V8.2 derivation itself incurred zero new model calls.

| NVIDIA only | Baseline | Reasoning V8.2 |
| --- | --- | --- |
| Supported / total atomic claims | 81 / 486 | 176 / 670 |
| Grounding rate | 16.67% | 26.27% |
| Contradicted claims | 17 | 6 |
| Verifiable numeric errors | 11 / 58 | 4 / 144 |
| Unflagged contradictions | 17 | 4 |

Warning recall is 99.19% and precision 88.93%. Supported lexical-key retention is only 16.46%, while evidence-span retention is 92.31%; extensive rephrasing and finer decomposition make the lexical measure especially fragile. The visible atomic-claim count is 137.86% of baseline, with 81 duplicate lexical instances. These are not unique semantic-fact recall estimates or proof that all original information survived. All 519 source units were covered: 59 exact independent-label transfers and 460 explicitly reviewed changed units. No verifier classification was used as ground truth; there was no human validation.

Four unflagged contradictions still apply the generic 68.3% CAGR cell to a horizon ending in 2029E. Original endpoints give about 37.49% from2022A to2029E. The model also warns on the true assertion that2023A–2025A CAGR exceeds68%; direct calculation gives119.95%. This is not an error-free final candidate. The full source-bound audit, error rows and warning metrics are saved under `audits/grounding/*reasoning_v8_2*`.

Development used the measured cohort, and model choice, semantic instructions and numeric validation were adapted after observed failures. This is a measured engineering follow-up with no held-out generalization claim. Existing V6 and V7.1 comparisons remain available, including negative results. No additional research option is introduced.

Resume locally from this submission directory. First inspect `evidence/progress/verifier_reasoning_v8_1_plan.jsonl`; do not start a second paid driver while one is live. Completed company's review can be derived without paid calls:

```powershell
$env:PYTHONUTF8='1'
& $RunPython scripts/revalidate_reasoning_offline.py --upstream $CleanV82 --tickers NVDA
& $RunPython scripts/prepare_reasoning_adjudication.py AMD --version reasoning_v8_2
```

The preparation command creates an incomplete specification; every changed unit still needs explicit independent review. Completed NVIDIA labels are reproducible with `scripts/build_nvda_reasoning_review.py` and `scripts/materialize_adjudication.py`. Use fresh output directories for comparison recomputation:

```powershell
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_reasoning_v8_2_nvda_partial_pairs.json --allow-partial --output reproduction/nvda_reasoning_comparison_new
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_reasoning_v8_2_nvda_partial_pairs.json --output reproduction/nvda_reasoning_annotations_new
```

The full three-company pair file intentionally fails until all required audit manifests exist. The final PDF, article, acceptance table, secret scan, artifact hashes and local course diff must be refreshed after completion. Medium publication and course PR remain deferred at the user's explicit direction.

## Annotation consistency erratum (2026-10-12)

AMD is now fully generated, derived and independently adjudicated:459source units and645claims (142Supported,495Unsupported,8Contradicted). Intel is still generating. The full-study configuration uses one explicitly corrected NVIDIA baseline annotation: original supplied Data Center sales support the narrow service-to-market assertion. Baseline Supported rises81→82 without any generated-text or evidence change. Original baseline and the earlier NVIDIA-only partial scorecard above remain preserved. See `audits/grounding/NVDA_before_reasoning_erratum_working/annotation_erratum.json`. This correction narrows apparent improvement; no later evidence was supplied.
