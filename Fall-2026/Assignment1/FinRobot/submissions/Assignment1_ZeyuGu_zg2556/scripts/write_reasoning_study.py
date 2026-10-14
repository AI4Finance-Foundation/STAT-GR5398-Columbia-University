"""Write measured follow-up prose only after all three full audits are complete.

Consumes the saved offline scorecards. Does not infer a verdict from verifier
labels, make provider calls, promote a release or authorize publication.
"""
import csv,json
from pathlib import Path
from audit_io import write_json,sha256,utc_now
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
def pct(v):return 'N/A' if v=='N/A' else f'{100*float(v):.2f}%'
def num(v):return f'{int(float(v)):,}'

def study():
    comp=ROOT/'audits/grounding/comparison_reasoning_v8_2';ann=ROOT/'audits/grounding/annotation_reasoning_v8_2'
    manifest=read(comp/'manifest.json')
    assert manifest['course_scope_complete'] and set(manifest['companies'])=={'NVDA','AMD','INTC'}
    sc=rows(comp/'before_after_scorecard.csv');ac=rows(ann/'annotation_scorecard.csv')
    total=next(r for r in sc if r['ticker']=='ALL' and r['section']=='ALL')
    warning=next(r for r in ac if r['ticker']=='ALL' and r['section']=='ALL')
    independent={};derivations={}
    for t in ['NVDA','AMD','INTC']:
        work=ROOT/'audits/grounding'/f'{t}_after_reasoning_v8_2_working'
        selected=next(pair for pair in manifest['pairs'] if Path(pair['final_run']).name.startswith(t+'_'))
        reviewed=read(ROOT/selected['after']/'manifest.json')
        assert reviewed['state']=='complete' and len(reviewed['sections'])==8 and not reviewed['human_review_performed']
        independent[t]={'carried_units':len(read(work/'exact_source_carry_forward.json')),
                        'explicitly_reviewed_changed_units':len(read(work/'explicit_changed_unit_review.json'))}
        derivations[t]=read(ROOT/'runs'/f'{t}_final_reasoning_v8_2/offline_derivation.json')
        assert derivations[t]['new_model_calls']==0
    patch=read(ROOT/'evidence/verifier_versions/patches_v8_2/manifest.json')
    assert patch['clean_apply']=='passed' and patch['offline_test_returncode']==0
    return sc,total,warning,independent,derivations,patch

def make_text():
    sc,t,a,review,derivation,patch=study()
    table=[]
    for r in sc:
        if r['section']=='ALL':
            table.append(f"| {r['ticker']} | {r['before_supported']} / {r['before_total_claims']} | {r['after_supported']} / {r['after_total_claims']} | {r['before_contradicted']} → {r['after_contradicted']} | {r['before_numeric_incorrect']} → {r['after_numeric_incorrect']} |")
    texts=f'''# Full reasoning-verifier follow-up

This separate Option A treatment uses `gpt-5.4-2026-03-05`, low reasoning effort, a20,000 completion-token ceiling and omitted temperature. Writer drafts, original evidence prompts and V7 semantic instructions are frozen. Three companies each received eight new reviews; neither the successful low-effort pilot nor an earlier cached review substitutes for a new treatment call. No market data were refetched. Raw paid V8.1 outputs remain immutable; V8.2 derives a separate final text offline using those captured responses and a read-only ledger. Derivation is not independent generation.

V8.2 fixes transport representations, not financial truth: an A/E year wrapper must occur exactly in both the revised sentence and cited original header before a numeric binding alias can be accepted. Currency aliases require identical explicit wrappers. It invents no source line, entity, number or revised prose. Raw bindings and normalization maps remain saved. The clean Option A patch SHA is `{patch['patch_sha256']['option_a_verifier.patch']}` on upstream `{patch['upstream_sha']}`; the clean applied checkout passes the recorded regression suite. Baseline compatibility and research patches stay separate.

| Company | Baseline supported / total | Treatment supported / total | Contradictions | Verifiable numeric errors (counts) |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

Overall grounding changes from {pct(t['before_grounding_rate'])} to {pct(t['after_grounding_rate'])}; contradiction rate changes from {pct(t['before_contradiction_rate'])} to {pct(t['after_contradiction_rate'])}. Treatment numeric errors are {t['after_numeric_incorrect']} among {int(t['after_numeric_correct'])+int(t['after_numeric_incorrect'])} verifiable numeric claims ({pct(t['after_numeric_error_rate_verifiable'])}); {t['after_numeric_unverifiable']} further numeric claims are unverifiable. The alternate denominator including unverifiable claims yields {pct(t['after_numeric_error_rate'])}. Company/section denominators and all atomic claims are saved, not sampled.

Warning recall is {pct(a['warning_recall'])}, precision {pct(a['warning_precision'])}, with {a['unflagged_unsupported_or_contradicted']} unflagged unsupported/contradicted assertions including {a['unflagged_contradicted']} contradictions. Warned assertions remain in the substantive audit denominator. The visible claim count is {pct(t['claim_count_ratio'])} of baseline; supported lexical-key retention is {pct(t['independent_supported_fact_retention'])}, and supporting evidence-span retention is {pct(t['supported_evidence_span_retention'])}. The verifier's own reported visible retention is {pct(t['verifier_reported_claim_retention_all_sections'])}; it is not independent accuracy evidence. There are {a['duplicate_lexical_claim_instances']} duplicate lexical claim instances. Rephrasing, added endpoint statements and changed decomposition alter denominators. These proxies are not unique semantic-fact recall or proof that no information was lost; consult original/revised claim mappings.

The three-company paid source reviews cost ${float(t['verifier_extra_cost_usd_all_sections']):.6f}, use {num(t['verifier_extra_tokens_all_sections'])} tokens and take {float(t['verifier_extra_runtime_seconds_all_sections']):.2f} seconds of model-call time. The offline derivative adds ${float(t['verifier_incremental_cost_usd_this_run']):.6f} in new model usage. These two figures are allocation and incremental usage, not amounts to add twice. The global shared ledger also includes both pilots, preflights, abandoned repairs, earlier candidates and machine-judge failures. Task-agent adjudication usage is unavailable in that direct-API ledger, not zero.

Every source unit was checked against the original writer prompt. Exact already adjudicated text/evidence may carry its prior label; changed text gets a fresh explicit task-agent decomposition. The breakdown is {json.dumps(review,sort_keys=True)}. Verifier classifications are never evaluator truth. This remains machine-assisted adjudication without human review; implicit premises or classifications can still be wrong. Development and model selection used this cohort, and there is no held-out generalization or isolated model-effect claim.

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
'''
    return texts,{'generated_at_utc':utc_now(),'scorecard':t,'warnings':a,'independent_review':review,
                  'study_scope_complete':True,'human_review_performed':False,'publication_authorized':False,
                  'input_hashes':{str(p.relative_to(ROOT)):sha256(p.read_bytes()) for p in [ROOT/'audits/grounding/comparison_reasoning_v8_2/before_after_scorecard.csv',ROOT/'audits/grounding/annotation_reasoning_v8_2/annotation_scorecard.csv']}}

def main():
    text,manifest=make_text();p=ROOT/'docs/verifier_reasoning_v8_followup.md';p.write_text(text,encoding='utf-8')
    manifest['study_sha256']=sha256(p.read_bytes());write_json(ROOT/'evidence/reasoning_study_summary.json',manifest)
    print('Full three-company study written from complete saved scorecards.')
if __name__=='__main__':main()
