"""Publish a local manuscript revision from measured semantic-guard results.

This never publishes to Medium/GitHub, changes experiment text, or labels claims.
Historical manuscripts and all predecessor measurements remain available.
"""
import csv,json,re
from pathlib import Path
from audit_io import utc_now,write_json,sha256
ROOT=Path(__file__).resolve().parents[1]
VERSION='v9_1'

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8',newline='')))
def overall(p):return next(r for r in rows(p) if r['ticker']=='ALL' and r['section']=='ALL')
def pct(v):return 'N/A' if v=='N/A' else f'{100*float(v):.2f}%'
def n(v):return str(int(float(v)))
def archive(relative):
    p=ROOT/relative;data=p.read_bytes();dest=ROOT/'evidence/document_history/pre_semantic_v9_1'/relative
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    # Rebuilding after a layout fix must start from the immutable predecessor,
    # not reinsert follow-up paragraphs into the already revised manuscript.
    return dest.read_text(encoding='utf-8')
def save(relative,text):
    (ROOT/relative).write_text(text,encoding='utf-8')

def main():
    base=overall('audits/grounding/comparison_reasoning_v8_2_corrected/before_after_scorecard.csv')
    v9=overall('audits/grounding/comparison_reasoning_v9/before_after_scorecard.csv')
    final=overall(f'audits/grounding/comparison_reasoning_{VERSION}/before_after_scorecard.csv')
    warning=overall(f'audits/grounding/annotation_reasoning_{VERSION}/annotation_scorecard.csv')
    pairs=read(f'config/grounding_reasoning_{VERSION}_pairs.json')
    assert len(pairs)==3
    for pair in pairs:
        audit=read(pair['after']+'/manifest.json')
        assert audit['state']=='complete' and len(audit['sections'])==8 and not audit['human_review_performed']
    patch=read(f'evidence/verifier_versions/patches_{VERSION}/manifest.json')
    assert patch['clean_apply']=='passed'
    from validate_delivery import validate_submission_relocation_evidence
    validate_submission_relocation_evidence(VERSION)
    validate_submission_relocation_evidence('v8_2')
    cost=read('audits/resources/cost_summary.json')
    assert not cost['unsettled_call_ids'] and cost['cost_usd']<20
    derivatives=[read(f'runs/{t}_final_reasoning_{VERSION}/offline_derivation.json') for t in ['NVDA','AMD','INTC']]
    assert all(d['new_model_calls']==0 and d['network_disabled_during_verification_and_render'] for d in derivatives)
    runtime=sum(d['runtime_seconds'] for d in derivatives)
    def measurement(r,side):
        return f"{r[side+'_supported']} / {r[side+'_total_claims']} | {pct(r[side+'_grounding_rate'])} | {r[side+'_contradicted']} | {r[side+'_numeric_incorrect']} / {int(r[side+'_numeric_correct'])+int(r[side+'_numeric_incorrect'])}"
    table='\n'.join([
      '| Version, under corrected evaluation | Supported / claims | Grounding | Contradicted | Numeric errors / verifiable |',
      '| --- | --- | --- | --- | --- |',
      '| Matched baseline | '+measurement(base,'before')+' |',
      '| Reasoning V8.2, unchanged prose | '+measurement(base,'after')+' |',
      '| V9: explicit CAGR horizons | '+measurement(v9,'after')+' |',
      '| V9.1: explicit actual multiple range | '+measurement(final,'after')+' |'])
    method='''### 9.2 Same-input reasoning treatment and evaluator corrections

The follow-up made 24 fresh reviews, eight each for NVIDIA, AMD and Intel, with `gpt-5.4-2026-03-05`, low reasoning effort, a 20,000 completion-token ceiling and omitted temperature. It held writer drafts, original prompts and the semantic-review instructions fixed. These calls cost $3.0613925, used 287,867 tokens and took 2,062.10 seconds of model-call time. V8.2 revalidated captured responses offline; this is not another independent generation. The failed high-effort pilot cost $0.400930 and the successful low-effort pilot $0.138385; both remain in the cumulative ledger.

A second audit found that the evaluator sometimes added a time window, comparator or universal quantifier absent from the source sentence. For example, a sentence describing a table that extends into forecasts does not necessarily claim that a displayed CAGR was calculated across that full table. Such under-specified assertions are Unsupported, not automatically Contradicted or Supported. The correction reviewed both before and after, including withdrawing several unsupported Supported labels. An AMD baseline assertion was split into a supported observed decline and an unsupported assertion that the level remained elevated. The saved mapping changes 31 original atoms into 32; 120 original source/audit files retain their hashes. No report text, evidence prompt or model response changes in this measurement correction.

The corrected baseline is 286 Supported, 1,162 Unsupported and 31 Contradicted among 1,479 claims. Corrected V8.2 is 530 Supported, 1,420 Unsupported and 8 Contradicted among 1,958. Its warning recall is 1,416/1,428 (99.16%); seven contradictions and five unsupported assertions have no warning. The previously reported 16 contradictions and 12 unflagged contradictions are preserved historical evaluator results, superseded for this comparison. Changing labels is not a verifier performance gain.
'''
    experiment=f'''### 9.3 Narrow deterministic semantic checks within Option A

The verifier now has two narrow deterministic release checks. V9 reads the subject's supplied Revenue row and fiscal A/E header, verifies that the displayed CAGR rounds to the actual-endpoint calculation, and explicitly names both actual and full-table horizons. The calculation is `(ending / starting) ** (1 / year_difference) - 1`; year differences, rather than column counts, determine elapsed intervals. Forecast endpoints remain labeled as model forecasts. For NVIDIA, the historical cell is 68.3% while the 2022A-2029E endpoint calculation is 37.49%; AMD's extended result is 13.59% and Intel's is -3.59%. The twelve matching original sentences are preserved alongside each clarification. Ambiguous originals are not retrospectively declared false.

V9.1 also handles one narrowly matched EV/EBITDA range sentence. It replaces an unspecified recent-years window with an explicitly dated window from the supplied actual rows and calculates its minimum and maximum. The rule reads no audit label, hard-coded ticker result or later information. Missing, inconsistent or ambiguous inputs cause abstention. Original model classifications, numeric bindings, release status and revised prose remain separately saved; these checks do not alter paid instructions or captured responses.

{table}

The final version retains {final['after_total_claims']} substantive atomic assertions, {pct(final['claim_count_ratio'])} of the baseline count, including {final['after_unsupported']} Unsupported and {final['after_contradicted']} Contradicted. Numeric claims include {final['after_numeric_unverifiable']} that remain unverifiable. A low observed contradiction count is not proof of factual correctness: unknown claims remain unknown, the evaluated cohort informed development, and there is no held-out test. No general semantic-verification guarantee is claimed.

Independent warning recall is {pct(warning['warning_recall'])}, precision {pct(warning['warning_precision'])}; {warning['unflagged_unsupported_or_contradicted']} unsupported/contradicted assertions remain unflagged, including {warning['unflagged_contradicted']} contradictions. Supported lexical-key retention is {pct(final['independent_supported_fact_retention'])}, while source-evidence-span retention is {pct(final['supported_evidence_span_retention'])}. These disagree because rephrasing, expanded endpoint statements and decomposition affect the proxies. Neither is unique-fact recall; claim-count growth does not prove information preservation. Warned claims remain in all substantive denominators.

All eight final sections of each company are audited against their original writer prompts. An identical source unit can inherit its corrected prior label only when the section, text and evidence hashes match. Changed units receive independent decomposition and calculations; the CAGR evaluator uses a separate Decimal logarithm/exponential implementation, not the verifier function. This is task-agent-assisted evaluation, not human validation; its exact task-agent model usage is unavailable in the direct-API ledger. The audit remains fallible.

V9 and V9.1 reuse the same 24 actual paid responses with model-call methods and network connections disabled. V9.1 adds zero API tokens and $0 new model cost; verification plus rendering takes {runtime:.2f} seconds across the three saved derivations. The $3.0613925 original-review cost is allocated to this treatment for comparison, not billed again. Separate upstream compatibility and Option A patches apply to the pinned commit; the final Option A patch SHA is `{patch['patch_sha256']['option_a_verifier.patch']}`.

The explicit audit correction, V9 and V9.1 scorecards are separate. Original V6/V7.1 and V8.2 prose and labels remain available as experiment history. The final three-company reports are linked from the README; five-company coverage and the original stability experiment are unchanged. Additional safeguards improve traceability within Option A, not a second research option.
'''
    study='# Evaluator correction and deterministic horizon/range follow-up\n\n'+method.replace('### 9.2','## 1.')+'\n'+experiment.replace('### 9.3','## 2.')
    study+='\n## Recompute from saved evidence\n\nThe README gives clean-patch offline replay commands. Use `config/grounding_reasoning_v9_1_pairs.json` with `grounding_audit.py compare` and `analyze_verifier_annotations.py` for exact metrics. Historical annotation correction is in `evidence/post_release_review/corrected/`; V9/V9.1 independent change reviews and source hashes are in their selected audit directories. Never promote a saved-response replay to an independent generation. Medium and the course PR remain deferred.\n'
    save('docs/verifier_semantic_v9_followup.md',study)
    report=archive('docs/research_report.md')
    report,replacements=re.subn(r'### 9\.2 Same-input reasoning-model treatment.*?(?=## 10\.)',method+'\n'+experiment+'\n',report,flags=re.S)
    assert replacements==1,'Expected one historical reasoning section'
    report=report.replace('The full three-company claim audit contains 1,478','The original V6 evaluation contains 1,478')
    report=report.replace('The verifier uses `gpt-4.1-2025-04-14`','The original V6 verifier uses `gpt-4.1-2025-04-14`')
    report=report.replace('V6 remains the main preserved experimental release','V6 remains the historical original experimental release')
    report=report.replace('from the canonical V6 release','from the historical V6 release')
    report=report.replace('| Company | Writer tokens in / out |','Original V6 per-company stage accounting (the later reasoning-treatment allocation is stated in Section 9.3):\n\n| Company | Writer tokens in / out |')
    report=report.replace('For the paired comparison, the 24 allocated verifier reviews','For the original V6 paired comparison, the 24 allocated verifier reviews')
    report=report.replace('Both patches apply to a clean copy of the pinned upstream, with 108 tests','The original V6 patches apply to a clean copy of the pinned upstream, with 108 tests')
    report=report.replace('## 7. Complete claim audit of the original prompts','## 7. Original V6 claim audit (historical evaluation)')
    report=report.replace('## 9. Option A: verifier gains and remaining failures','## 9. Option A: original experiment and measured follow-ups')
    lead=f"The latest three-company follow-up is separately measured in Sections 9.2-9.3: after explicit evaluator corrections, grounding changes from {pct(base['before_grounding_rate'])} to {pct(final['after_grounding_rate'])}, with {base['before_contradicted']} versus {final['after_contradicted']} contradicted claims. Unsupported claims and machine-evaluation limitations remain. Earlier V6/V7.1 tables below retain their original evaluation version and must not be mixed with corrected denominators.\n\n"
    report=report.replace('## 2. Implementation, environment and evidence boundary',lead+'## 2. Implementation, environment and evidence boundary')
    report=report.replace('## 12. Submission, sources and repeatability','''The complete submission dependencies were also copied to a new directory and replayed with Python access to the original submission denied, network and paid calls disabled, and an execution clock forced to 2099. Exact HTML, section text, chart assets, market inputs and valuation hashes are checked against saved originals. The original report date is preserved while new execution time is recorded separately. This closes the earlier output-directory-only replay gap; the isolation is a tested Python guard, not an operating-system security sandbox.

## 12. Submission, sources and repeatability''')
    save('docs/research_report.md',report)
    readme=archive('README.md')
    readme=readme.split('## Completed reasoning follow-up')[0]
    readme=readme.replace('Primary results: baseline support','Historical V6 results: baseline support')
    readme=readme.replace('seven final verifier outputs.','seven original V6 verifier outputs; later treatments are versioned separately.')
    readme=readme.replace('The verifier is an experimental review aid: the independent after audit still finds 48','In the original V6 evaluation, the verifier is an experimental review aid: the independent after audit found 48')
    readme=readme.replace('Canonical label locations and exact pairs are','Original V6 label locations and exact pairs are')
    readme=readme.replace('are primary results; older development outputs are explicitly noncanonical.','are preserved original V6 results; the latest evaluated follow-up is linked above.')
    readme=readme.replace('Verifier: `gpt-4.1`','Original V6 verifier: `gpt-4.1`')
    readme=readme.replace('The main report links above remain the preserved V6 experimental release.','The original five-company report table retains the historical V6 release; the latest three-company reports are linked separately above.')
    readme=readme.replace('The subsequent completed reasoning treatment is reported below;','The subsequent reasoning and semantic treatments are linked above;')
    readme=readme.replace('## Completed development follow-up',"The submission's `.gitattributes` uses `* -text` to preserve raw evidence/prompt bytes across Git checkouts and `*.pdf -diff` to keep PDFs binary. `prepare_course_diff.py` disables textconv/external diff, compares every scoped staged blob with saved bytes and verifies actual clean-index application. See `evidence/reproducibility/course_git_bytes_*.json` for the initial newline-conversion failure and byte-preserving recheck, and `course_export_textconv_failure.json` in that directory for the rejected PDF-to-text export. Only an export that passes actual-application byte checks is accepted.\n\n## Completed development follow-up")
    guide=f'''## Latest evaluated Option A follow-up

Read [the semantic follow-up](docs/verifier_semantic_v9_followup.md) alongside the research report. The older V6 links below preserve the five-company original experiment. Latest separately evaluated reports are [NVDA](runs/NVDA_final_reasoning_v9_1/output/NVDA/report/Audited_Equity_Report_NVDA.html), [AMD](runs/AMD_final_reasoning_v9_1/output/AMD/report/Audited_Equity_Report_AMD.html), and [INTC](runs/INTC_final_reasoning_v9_1/output/INTC/report/Audited_Equity_Report_INTC.html). Keep their sibling assets and analysis directories.

{table}

The corrected V8.2 row changes evaluator judgments only; V9/V9.1 change generated text using original evidence. There are still {final['after_unsupported']} Unsupported claims and {final['after_numeric_unverifiable']} unverifiable numeric claims. Final warning recall is {pct(warning['warning_recall'])}, precision {pct(warning['warning_precision'])}; {warning['unflagged_unsupported_or_contradicted']} problematic claims remain unflagged. This is complete machine-assisted scope coverage, not human validation or investment-ready prose. Claims added by explicit endpoint explanations change denominators. Full-assignment status remains incomplete: publication is intentionally deferred and strict error-free narrative certification is unavailable.

The latest paid reviewer is `gpt-5.4-2026-03-05`, low reasoning effort, omitted temperature and a 20,000 completion-token ceiling. V9/V9.1 reuse its captured reviews with no new model calls; the historical V6 reviewer below uses GPT-4.1. Clean-patch verification and the expanded regression log (235 tests and 6 subtests) are saved separately.

'''
    readme=readme.replace('## Read the result',guide+'## Read the original experiment')
    repro='''## Reproduce the latest follow-up without new API calls

Start in this submission with `$RunPython` from the clean-environment setup above. These new short directories preserve original runs. Existing saved responses, evidence and cost records are required; private credentials are not used.

```powershell
$NewId = [guid]::NewGuid().ToString('N').Substring(0,8)
$CleanV91 = Join-Path $env:TEMP "fr91src-$NewId"
$ReplayV91 = Join-Path $env:TEMP "fr91out-$NewId"
git clone --no-checkout https://github.com/AI4Finance-Foundation/FinRobot.git $CleanV91
git -C $CleanV91 checkout --detach 2717499b8e30f242640af08c4ad9afd1113c2d45
git -C $CleanV91 apply (Join-Path $PWD 'evidence/verifier_versions/patches_v9_1/compatibility.patch')
git -C $CleanV91 apply (Join-Path $PWD 'evidence/verifier_versions/patches_v9_1/option_a_verifier.patch')
& $RunPython scripts/revalidate_reasoning_offline.py --upstream $CleanV91 --verifier-version v9_1 --output-root $ReplayV91 --disable-network
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_reasoning_v9_1_pairs.json --output "reproduction/comparison-$NewId"
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_reasoning_v9_1_pairs.json --output "reproduction/annotations-$NewId"
& $RunPython scripts/validate_submission_relocation.py --upstream $CleanV91 --verifier-version v9_1 --evidence-dir "reproduction/relocation-$NewId"
```

The clone step downloads source; the replay itself rejects network and model calls. A new online capture is a different experiment. This replay is not a fourth stability draw. Apply V8.2 or V9 patches and select their version to reproduce those historical treatments; their immutable patch manifests and reports remain under `evidence/verifier_versions/` and `runs/`.

Current academic readiness and remaining requirements are in [submission readiness](docs/submission_readiness.md) and `docs/acceptance.csv`. `evidence/delivery_validation.json` records integrity checks, not a guarantee of semantic truth. Medium URL and course-master PR remain absent by the user's explicit direction.
'''
    save('README.md',readme.rstrip()+'\n\n'+repro)
    blog=archive('blog/medium_article.md').split('## A stronger reviewer reduced errors, but still missed the horizon')[0]
    blog=blog.replace('It was 291: the number of supported factual assertions in a baseline inventory of 1,478 claims.',f"The original V6 evaluation counted 291 supported factual assertions among 1,478 claims. A later audit of the evaluator corrected that baseline to 286/1,479. The latest verifier treatment yields {final['after_supported']}/{final['after_total_claims']} supported claims, with {final['after_unsupported']} still unsupported; the versioned results below distinguish changed judgments from changed text.")
    blog=blog.replace('## Audit what was in the prompt','## Original V6 evaluation: audit what was in the prompt')
    blog=blog.replace('## The ninth agent helped with warnings, not reliable correction','## Original V6 result: warnings improved more than correction')
    blog=blog.replace('Independent after auditing still finds 48','Under the original V6 evaluation, independent after auditing found 48')
    blog=blog.replace('I kept V6 as the main experimental release','I preserved V6 as the historical original experimental release')
    blog=blog.replace('retains V6 as the main experimental release','retains V6 as the historical original experimental release')
    blog+=f'''## I also had to audit the auditor

A full three-company reasoning-model treatment made 24 new reviews over the same frozen writer evidence. The first evaluator result looked stronger than the original reviewer, but a second task-agent review found an additional problem: some labels had silently added a time horizon or comparator that the sentence never specified. A table spanning actuals and forecasts does not by itself say which endpoints produced a CAGR. Ambiguity deserves an Unsupported label; it is not evidence of a false calculation.

I preserved the original labels and recorded a symmetric before/after correction. The corrected reasoning output has 530 supported claims among 1,958, with eight contradictions. This is a measurement correction, not a software improvement. It also withdraws several incorrectly Supported labels, and retains the supported half of a compound assertion while marking its unsupported judgment separately.

Then the software changed. A narrow rule now calculates CAGR from the provided endpoints and spells out both intervals. NVIDIA's supplied historical 68.3% growth and the 37.49% implied across 2022A-2029E answer different questions; the final year in the latter is a model forecast. A second narrow check names the actual years behind an EV/EBITDA minimum and maximum. Both retain original text, evidence locations, formulas and reasons; neither uses later web facts or the audit labels to choose a number.

{table}

The final {final['after_unsupported']} Unsupported claims are a substantial remaining limitation. Warning recall is {pct(warning['warning_recall'])} and precision {pct(warning['warning_precision'])}, with {warning['unflagged_unsupported_or_contradicted']} problematic claims still unflagged. Supported lexical retention is {pct(final['independent_supported_fact_retention'])}; evidence-span retention is {pct(final['supported_evidence_span_retention'])}. Those proxies differ, so I cannot claim complete information preservation. The narrow checks were developed on this cohort; a low observed error count is not an out-of-sample guarantee.

The new text was audited across every section, with independently implemented endpoint arithmetic and exact source hashes. That review was machine-assisted and no human claim validation occurred. The real paid reasoning treatment cost $3.0613925; these deterministic derivatives add no API tokens or charges. The complete saved direct-API ledger remains ${cost['cost_usd']:.6f}, including failed attempts and pilots, within the approved $20 cap. Task-agent authoring/evaluation cost is unavailable, not zero.

I also tested a fresh directory containing the actual submission dependencies, prevented the worker and renderer from reading the original directory, disabled network calls, and forced a future execution clock. Reports must reproduce their saved data and date rather than quietly refetch a current price or drop a valuation when an old absolute path disappears. That is a practical reproducibility test a polished report alone cannot pass.

The useful result is an inspectable analyst aid: filing-linked metrics, reproducible arithmetic, scenario sensitivities and traceable warnings. The remaining unsupported business narratives, causal claims and uncertain forecasts still require review. The project is a measured research experiment, not an autonomous portfolio recommendation. Publication remains deferred; this is the complete local manuscript, with no invented public URL.
'''
    save('blog/medium_article.md',blog)
    archive('docs/submission_readiness.md')
    save('docs/submission_readiness.md',f'''# Local research readiness and formal submission

The local package contains actual five-company semiconductor coverage with two peers each, the requested onboarding runs, twenty SEC-linked data checks, independent valuation notebooks,125 target-price scenarios, and three independent same-input NVDA generations. Every section of three companies has complete before/after atomic-claim inventories. Data, prompts, responses, usage and failures are frozen and traceable.

The original V6/V7.1 studies remain preserved. The current comparison corrects evaluator scope errors symmetrically, then separately measures narrow deterministic CAGR and EV/EBITDA range clarifications within Option A. See [the measured follow-up](verifier_semantic_v9_followup.md). Final grounding is {pct(final['after_grounding_rate'])}; {final['after_unsupported']} claims remain Unsupported, {final['after_contradicted']} Contradicted, and {final['after_numeric_unverifiable']} numeric claims unverifiable. Machine-assisted labels are not human validation or a gold standard. Development used the evaluated cohort and there is no held-out generalization claim.

Reproduction now tests copied submission dependencies with original-root Python reads denied, networking and paid calls disabled, and a future execution clock. Rendered text, assets, market data and valuations must match. `evidence/delivery_validation.json`, the latest relocation manifest, PDF visual QA, secret scan and scoped diff metadata give the actual check outcomes; old output-only replay is historical evidence, not migration proof.

This is a substantial, reviewable academic research submission, provided the current integrity/PDF checks pass. The course asks for an honest account of residual failures. The stricter user criterion requiring every delivered narrative number to be correct cannot be certified: historical experimental outputs intentionally preserve mistakes, and unresolved claims remain labeled. An independent human review of the claim judgments would strengthen the work. These limits preclude a guarantee of a grade or unattended portfolio use.

Medium publication, its real URL and the course master PR remain unperformed because the user explicitly requested local-only materials. Do not publish or ask again without a later user instruction. Thus the full assignment is not declared complete. The README gives exact reproduction commands; no additional credential or paid call is currently needed for local replay. Current direct-API spend is ${cost['cost_usd']:.6f} of the authorized $20.
''')
    archive('docs/course_pr_body.md')
    save('docs/course_pr_body.md',f'''## FinRobot Assignment 1 — Zeyu Gu (zg2556)

Adds the specified FinRobot submission folder with real five-company semiconductor coverage, two peers per company, onboarding reports, frozen API/SEC evidence,20 filing reconciliations, independent valuations,125 target-price scenarios and three independent NVDA generations. All eight sections of three companies receive complete atomic-claim before/after audits.

Option A adds a ninth verifier with numeric/evidence checks, preserved revisions and explicit warnings. Original experiments and failures remain alongside the corrected evaluator and deterministic horizon/range follow-up: final grounding {pct(final['after_grounding_rate'])}, {final['after_contradicted']} contradicted and {final['after_unsupported']} unsupported claims. Evaluation is machine-assisted, not human-validated; retention proxies and cohort-development limitations are disclosed. Separate patches apply to pinned upstream; the course tree contains no upstream clone or credentials.

Validation includes offline source/response replay, actual submission relocation with original-root reads denied, independent metric recomputation, HTML asset/text checks, PDF visual review, and credential scanning. The shared course checkout branch/index are not changed by packaging. Direct API spend is ${cost['cost_usd']:.6f}, below the approved $20 cap.

Publication status: local draft only. The user deferred Medium and the course PR; no public URL is claimed. The README links the research PDF/source, full analytical article, exact commands and acceptance register.
''')
    evidence_inputs=[f'audits/grounding/{d}_reasoning_{VERSION}/{f}' for d,f in [('comparison','before_after_scorecard.csv'),('annotation','annotation_scorecard.csv')]]
    write_json(ROOT/'evidence/semantic_study_summary.json',{'generated_at_utc':utc_now(),'version':VERSION,'study_scope_complete':True,
        'human_review_performed':False,'held_out_evaluation':False,'publication_authorized':False,
        'corrected_v8_2':base,'v9':v9,'scorecard':final,'warnings':warning,'offline_runtime_seconds':runtime,
        'input_hashes':{p:sha256((ROOT/p).read_bytes()) for p in evidence_inputs},
        'study_sha256':sha256((ROOT/'docs/verifier_semantic_v9_followup.md').read_bytes()),
        'old_documents':'evidence/document_history/pre_semantic_v9_1'})
    print('Local manuscripts updated from measured results; PDF/QA/inventory refresh still required.')

if __name__=='__main__':main()
