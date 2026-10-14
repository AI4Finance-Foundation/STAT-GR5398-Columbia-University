"""Measured development addendum; never replace canonical experimental labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def current_accounting(text):
    cost=json.loads((ROOT/'audits/resources/cost_summary.json').read_text())
    replacements={'3.59449':f"{cost['cost_usd']:.6f}",'209 settled calls':f"{cost['settled_call_count']} settled calls",
        '728,661':f"{cost['input_tokens']:,}",'360,915':f"{cost['output_tokens']:,}",
        '1,662.67':f"{cost['model_runtime_seconds']:,.2f}",'101 tests':'108 tests'}
    for a,b in replacements.items():text=text.replace(a,b)
    return text

def _v7_report_addendum():
    return '''### 9.1 Development follow-up: improved warnings, unresolved semantics

The completed V7.1 follow-up is reported separately from the canonical V6 release. V7 added semantic review instructions about periods, entities, comparisons, CAGR and universal trend claims, plus exact released-string deduplication. V7.1 repaired a bounded validation failure by permitting only three neutral connectors. Both changes remain within Option A. All seven companies received new V7 reviews; V7.1 reused proven saved responses and generated two missing Intel sections. Reuse is not independent generation. Development patches apply to separate clean pinned checkouts; they do not overwrite the main V6 patches.

The complete NVDA/AMD/INTC follow-up covers all eight sections with unchanged baseline evidence: 360 exactly unchanged sentence units carry independent labels with text/evidence identity, and all 173 changed units were explicitly decomposed and judged against the original writer prompts. These are machine-assisted labels, without human review. The task agent used the evaluated cohort to guide development; there is no held-out efficacy claim and no ablation separating prompt and deduplication effects.

| Measure | Baseline | V6 | V7.1 development |
| --- | --- | --- | --- |
| Supported / total | 291 / 1478 | 358 / 1590 | 325 / 1673 |
| Grounding rate | 19.69% | 22.52% | 19.43% |
| Contradicted / total | 40 / 1478 | 48 / 1590 | 41 / 1673 |
| Contradiction rate | 2.71% | 3.02% | 2.45% |
| Verifiable numeric errors | 25 / 171 | 29 / 217 | 23 / 180 |
| Warning recall | 0% | 93.99% | 97.70% |
| Warning precision | N/A | 78.78% | 82.42% |
| Unflagged U/C | 1187 | 74 | 31 |
| Unflagged contradictions | 40 | 8 | 8 |

V7.1 supported lexical-key retention is 76.06%, supporting evidence-span retention is 93.02%, and claim count is 113.19% of baseline. Verifiable numeric error rate is 12.78%; including unverifiable numeric claims gives 23/222 (10.36%). Wording, decomposition and duplicate removal change denominators; these are not unique semantic-fact recall estimates. Selected three-company review allocation is 186,938 tokens, $0.807346 and 276.42 seconds, including $0.059582 of new V7.1 calls. Global accounting also includes failed repairs and evaluation attempts. Full company/section tables, claim maps and retained failures are under `comparison_v7_1`, `annotation_v7_1` and `docs/verifier_v7_followup.md`.

The evaluator also failed. An unblinded GPT-4.1 proposal skipped 18 warned NVIDIA units as nonfactual. Hiding only the warning prefix eliminated those omissions but still under-decomposed claims and missed explicit evidence. Both proposal versions remain saved and neither supplies the canonical labels. Explicit task-agent judgments replaced them. Schema compliance and automated coverage alone do not establish valid financial labels.

A subsequent one-section NVIDIA pilot used pinned `gpt-5.4-2026-03-05`, high reasoning effort and a 26,000 completion-token limit, with identical V7.1 instructions and original frozen inputs. It returned HTTP 200 with `finish_reason=length`, 4,372 input tokens, 26,000 reasoning/output tokens and empty visible content. The SDK then raised MaxTurnsExceeded. No verification or efficacy result exists. The 278.07-second attempt costs $0.400930 at the recorded $2.50/$15 per million input/output prices; its raw wire response and reconciliation are retained. The pilot was not expanded. The provider documents these settings and prices in the [official model reference](https://developers.openai.com/api/docs/models/gpt-5.4).

V6 remains the main preserved experimental release, with the full V7.1 result alongside it. V7.1 improves warning coverage but lowers supported lexical retention and does not reduce the eight unflagged contradictions. Neither candidate is certified for unattended portfolio decisions. This selection is transparent and does not hide the newer negative result or treat V6 as the best-performing model.

'''

def _v7_docs_addendum(kind):
    if kind=='README.md':
        return '''
## Completed development follow-up

The main report links above remain the preserved V6 experimental release. [Full V7.1 study](docs/verifier_v7_followup.md) and `runs/{TICKER}_final_v7_1/` are separate measured candidates. Warning recall improved to 97.70% and unflagged U/C fell to 31, but eight unflagged contradictions remain and supported lexical retention is lower. The research PDF and article report both versions. A GPT-5.4 one-section pilot exhausted its token limit with no visible output; its $0.400930 cost and failed response are included. No further paid pilot or publication is running.

Additional exact offline recomputation commands from this submission directory (use new output folders):

```powershell
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_v7_1_pairs.json --output reproduction/v7_comparison_new
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_v7_1_pairs.json --output reproduction/v7_annotations_new
& $RunPython scripts/summarize_runs.py
```

Development model/prompt/plan versions, clean patch evidence and the complete eight-section evaluations are linked in the study. V7.1 source SHA is `9e4b0da4dc5d259c4ffea48f1c1b4371df5862cdb24f9110643163f0b521b40a`. Main V6 reproduction commands and patches are unchanged. These candidates are experimental artifacts, not an additional research option. Formal publication and the strict zero-factual-error condition remain unmet; user direction is to keep everything local.
'''
    if kind=='blog/medium_article.md':
        return '''
## What happened when I strengthened the reviewer

I retained the V6 experiment and ran a separate follow-up with stricter instructions on fiscal periods, company comparisons and growth horizons, plus exact-string deduplication. The full same-input, eight-section study on NVIDIA, AMD and Intel reduced unflagged problematic assertions from 74 to 31. Warning recall rose from 93.99% to 97.70%, with precision rising from 78.78% to 82.42%. That is a measurable improvement in triage. Yet eight contradictions still passed without warnings. The newer output had 325 supported claims among 1,673 assertions (19.43%), 41 contradictions and 23 errors among 180 verifiable numeric claims. Supported lexical-key retention fell to 76.06%; paraphrases affect that proxy, and evidence-span retention was 93.02%. More warnings did not establish a trustworthy editor.

The judge itself supplied another failure case. A model evaluating warned prose skipped 18 NVIDIA units as nonfactual. After I hid only the warning prefixes, it stopped skipping those units but still combined separate facts and missed explicit prompt evidence. Both failed proposals are preserved. A Codex task agent explicitly adjudicated every changed unit against the original evidence; no human validation was performed. The same development cohort influenced the instructions, so these findings are not a held-out test.

A bounded reasoning-model pilot also produced a negative result. One frozen NVIDIA section on GPT-5.4 used all 26,000 allotted output tokens as reasoning, returned no visible review and cost $0.400930. I did not expand it or count it as a successful verifier run. The raw response is saved. Together these experiments argue for measuring information retention, missed contradictions and evaluator failures alongside warning rates. The complete package retains V6 as the main experimental release and reports the V7.1 candidate separately; neither is an unattended investment product.
'''
    return '\nAdditional development evidence: the full V7.1 follow-up improves warning recall to97.70% but retains eight unflagged contradictions. A failed GPT-5.4 pilot is preserved and billed. Both are disclosed in the research report; neither replaces the main V6 result. Publication remains deferred at the user\'s request.\n'

def reasoning_addendum():
    summary=json.loads((ROOT/'evidence/reasoning_study_summary.json').read_text(encoding='utf-8'))
    if not summary['study_scope_complete']:raise ValueError('Incomplete follow-up cannot be reported as complete')
    return '''### 9.2 Same-input reasoning-model treatment

A separate completed treatment used pinned GPT-5.4-2026-03-05, low reasoning effort, 20,000 completion tokens and no explicit temperature. It generated 24 fresh reviews for NVIDIA, AMD and Intel using unchanged writer drafts, frozen evidence and V7 semantic instructions. The paid runs used V8.1. V8.2 then revalidated the captured responses offline, repairing narrowly defined A/E-year and currency binding representations without adding evidence, values, citations or model text. The clean V8.2 patch passes 132 tests and six subtests. Offline derivation is not a new independent generation.

| Measure | Matched baseline | Reasoning V8.2 |
| --- | --- | --- |
| Supported / total claims | 292 / 1478 | 534 / 1958 |
| Grounding rate | 19.76% | 27.27% |
| Contradicted claims | 40 | 16 |
| Contradiction rate | 2.71% | 0.82% |
| Verifiable numeric errors | 25 / 171 | 13 / 435 |
| Warning recall / precision | 0% / N/A | 99.16% / 89.65% |
| Unflagged contradictions | 40 | 12 |

Company-level after counts are NVIDIA 176 Supported, 488 Unsupported and 6 Contradicted; AMD 142, 496 and 7; Intel 216, 424 and 3. All eight sections of each company were audited. The review covers 1,467 source units: 132 exact independent-label transfers and 1,335 explicitly adjudicated changed units. Labels remain machine-assisted, without human validation. One NVIDIA baseline annotation is corrected because original supplied Data Center sales already support service to that market (82 rather than81 Supported claims). A separate AMD after-audit erratum removes an auditor-invented current comparator. Original annotations are preserved; no source or generated text changes. These small evaluator corrections and their effect on comparisons are explicit in the full study.

The verifiable numeric error rate is 2.99%; including 51 unverifiable numeric claims gives 13/486 (2.67%). Supported lexical-key retention is 9.47%, supporting evidence-span retention is 81.82%, and visible claim count is 132.48% of baseline, including 214 duplicate lexical instances. Extensive rephrasing and decomposition make these retention proxies fragile. The verifier reports 99.26% visible retention, but that is not independent fact recall. More explicit numbers and changed denominators do not by themselves establish more preserved information. This treatment cannot be described as simply retaining all useful facts while improving accuracy.

The 24 paid source reviews used 287,867 tokens, cost $3.061393 and took 2,062.10 seconds. V8.2 added zero model calls. A separate successful low-effort pilot cost $0.138385 (4,372 input/8,497 output tokens,73.76 seconds); it was not reused as a treatment draw. The failed high-effort pilot remains billed at $0.400930. Global accounting includes both, all prior versions and failures. These are development-cohort results with no held-out efficacy claim or ablation separating model, instructions and validator changes.

All 12 unflagged contradictions in this treatment misuse a generic historical CAGR at a horizon extending into forecasts. NVIDIA's2022A–2029E endpoints imply about37.49%, not68.3%; AMD's2021A–2028E imply13.59%, not20.5%; Intel has the analogous horizon error. Warnings also reject supplied facts, including AMD's explicit non-GAAP amounts and NVIDIA's true2023A–2025A CAGR-above68% inequality. The treatment reduces total contradictions but leaves more unflagged contradictions than the eight in V7.1. No candidate is promoted as error-free. V6 remains the preserved main experimental release, with complete V7.1 and reasoning follow-ups alongside it.

Full company/section scorecards, atomic claims, error rows, annotation errata, model versions, clean patches and commands are in `docs/verifier_reasoning_v8_followup.md`, `comparison_reasoning_v8_2`, `annotation_reasoning_v8_2` and `evidence/verifier_versions/patches_v8_2`. The three new HTML reports use exactly the audited V8.2 text; live browser checks found all charts loaded and no section overflow. These presentation checks do not certify semantic truth.

'''

def report_addendum():
    return _v7_report_addendum().replace('The pilot was not expanded.','The high-effort setting was not expanded; the separate low-effort treatment is reported next.')+reasoning_addendum()

def docs_addendum(kind):
    old=_v7_docs_addendum(kind)
    old=old.replace('No further paid pilot or publication is running.','The subsequent completed reasoning treatment is reported below; no paid generation or publication is currently running.')
    old=old.replace('I did not expand it or count it as a successful verifier run.','I did not expand that high-effort setting or count it as a successful verifier run; a separate low-effort treatment followed.')
    if kind=='README.md':
        new='''
## Completed reasoning follow-up

[Full same-input study](docs/verifier_reasoning_v8_followup.md) now covers NVIDIA, AMD and Intel, all eight sections. Grounding is27.27%, contradictions16/1958, verifiable numeric errors13/435 and warning recall99.16%. Twelve contradictions remain unflagged. Supported lexical retention is9.47% and evidence-span retention81.82%; these proxies are affected by rephrasing/decomposition and do not establish full semantic retention. The study records all residual failures and two evaluator errata. No human review or held-out generalization is claimed.

The pinned reasoning verifier is `gpt-5.4-2026-03-05`, low effort,20,000 completion tokens, temperature omitted. All24 paid source calls are under `*_final_reasoning_v8_1`; separate `*_final_reasoning_v8_2` reports derive from those real responses without new calls. Original baseline inputs are unchanged. This treatment costs$3.061393; global ledger accounting above includes both pilots and every other attempt. All reservations are settled.

V8.2 uses separate clean patches under `evidence/verifier_versions/patches_v8_2/`;132 tests plus six subtests pass. The main V6 patches remain separate. Apply these development patches to a fresh checkout at the same pinned SHA, then use the following from this submission directory. `$CleanV82` is that fresh patched checkout; `$RunPython` is the dedicated environment Python defined above.

```powershell
$env:PYTHONUTF8='1'
$CleanV82 = Join-Path $env:TEMP ('finrobot-v82-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$ReplayV82 = Join-Path $env:TEMP ('finrobot-replay-' + [guid]::NewGuid().ToString('N').Substring(0,8))
git clone https://github.com/AI4Finance-Foundation/FinRobot.git $CleanV82
git -C $CleanV82 checkout --detach 2717499b8e30f242640af08c4ad9afd1113c2d45
git -C $CleanV82 apply (Join-Path $PWD 'evidence/verifier_versions/patches_v8_2/compatibility.patch')
git -C $CleanV82 apply (Join-Path $PWD 'evidence/verifier_versions/patches_v8_2/option_a_verifier.patch')
& $RunPython scripts/revalidate_reasoning_offline.py --upstream $CleanV82 --tickers NVDA AMD INTC --output-root $ReplayV82 --disable-network
# Keep the replay path short on Windows to avoid MAX_PATH in chart filenames.
# These new directories preserve all submitted runs; model calls are disabled.
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_reasoning_v8_2_pairs.json --output reproduction/reasoning_comparison_new
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_reasoning_v8_2_pairs.json --output reproduction/reasoning_annotations_new
```

The source paid generation plan is `config/verifier_reasoning_v8_1_plan.json`; it records a$14 absolute cumulative guard within the approved$20 ledger. Do not rerun a finished plan as a stability replicate. A new paid experiment needs new output IDs and an explicit budget policy against the same ledger. Exact snapshots, request schemas, prompts, model parameters and every source call are saved in the respective manifests. Publication remains deferred; the package is local and the strict zero-error narrative condition remains unmet.
'''
    elif kind=='blog/medium_article.md':
        new='''
## A stronger reviewer reduced errors, but still missed the horizon

I then ran a complete, separate low-effort GPT-5.4 treatment on the same NVIDIA, AMD and Intel drafts and original evidence:24 new reviews, all eight sections per company. Small deterministic fixes normalized numeric bindings only when the same fiscal-year or currency wrapper appeared in both sentence and cited source. The paid source responses were preserved; offline revalidation was not counted as another independent run.

The fully reviewed output had534 supported assertions among1,958 (27.27%),16 contradictions and13 errors among435 verifiable numeric claims. The matched baseline had292/1,478 supported assertions and40 contradictions; one original baseline annotation was corrected because it had overlooked Data Center revenue already in the prompt. I also corrected an ambiguous comparator that the later evaluator had overstated. Both audit versions remain available. Those are evaluator corrections, not newly retrieved evidence.

Warning recall reached99.16% and precision89.65%. Yet12 contradictions passed without warning, more than the eight in the previous candidate. They all assigned a historical CAGR to a longer interval containing model forecasts. NVIDIA's endpoints imply about37.49% across2022A–2029E, not68.3%. A number can match a source cell and still describe the wrong horizon. The reviewer also warned on some facts that were plainly supplied.

Information retention remains a real limitation. There were214 duplicate lexical instances, and finer decomposition inflated the number of claims. Supported lexical-key retention was9.47%, while supporting evidence-span retention was81.82%. Those are fragile proxies under rewriting, not proof of semantic recall. The reviewer itself reported99.26% visible retention; I do not treat that as independent evidence. Better error rates alone are insufficient if useful information is lost, duplicated or reframed.

This treatment cost$3.061393 for287,867 tokens and2,062.10 seconds of model time, plus a separate$0.138385 successful low-effort pilot. Both that pilot and the$0.400930 failed high-effort attempt are included in the global total. A Codex task agent explicitly adjudicated 1,335 changed source units and carried 132 exact previously adjudicated units. This was machine adjudication; no human validation was performed. Model selection and development used this cohort, so the result has no held-out generalization claim.

The practical finding is narrower than a trustworthy automatic analyst: verification can improve triage and reduce observed errors, while arithmetic meaning, retained information and the evaluator itself still require scrutiny. The package preserves all three measured versions and their failures. None is certified for unattended investment decisions.
'''
    else:
        new='\nThe full same-input reasoning follow-up adds24 genuine reviews and complete independent machine adjudication of all24 sections. Its27.27% grounding,16 contradictions,12 unflagged contradictions and information-retention limitations are disclosed. Separate V8.2 patches pass132 tests+6subtests. All direct API calls are settled; user direction remains local-only. See docs/verifier_reasoning_v8_followup.md.\n'
    return old+new
