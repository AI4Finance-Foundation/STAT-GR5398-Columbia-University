"""Write the fully measured V7/V7.1 development follow-up without promotion."""
import csv,json,shutil
from pathlib import Path
from audit_io import write_json,utc_now
ROOT=Path(__file__).resolve().parents[1]
def rows(path):return list(csv.DictReader((ROOT/path).open(encoding='utf-8',newline='')))
score=next(x for x in rows('audits/grounding/comparison_v7_1/before_after_scorecard.csv') if x['ticker']==x['section']=='ALL')
ann=next(x for x in rows('audits/grounding/annotation_v7_1/annotation_scorecard.csv') if x['ticker']==x['section']=='ALL')
old=next(x for x in rows('audits/grounding/annotation_v6/annotation_scorecard.csv') if x['ticker']==x['section']=='ALL')
def pct(s):return f'{100*float(s):.2f}%'
table='| Ticker | Units | Claims | S | U | C |\n|---|---:|---:|---:|---:|---:|\n'
for t in ['NVDA','AMD','INTC']:
    s=next(x for x in rows(f'audits/grounding/{t}_after_v7_1_adjudicated/grounding_summary.csv') if x['section']=='ALL')
    u=len(rows(f'audits/grounding/{t}_after_v7_1_adjudicated/text_coverage.csv'))
    table+=f"| {t} | {u} | {s['total_claims']} | {s['supported']} | {s['unsupported']} | {s['contradicted']} |\n"
text=f'''# Development follow-up: stronger semantic instructions did not eliminate false assertions

This is an adaptive development experiment on the same frozen writer inputs. V6 remains the canonical local release while further validation is in progress. V7/V7.1 outputs, successful and failed responses, patches and audits remain separate and are not silently substituted for V6.

V7 retains the ninth-stage architecture and adds explicit instructions about actual/forecast periods, entity/metric alignment, comparison horizons, CAGR endpoints, universal trend qualifiers and flattened filing/news columns. It also emits an exactly repeated released string only once per section, preserving all source mappings. Both changes belong to Option A; there is no ablation isolating their effects. The evaluator feedback used in prompt design makes this a development-cohort result, not a held-out efficacy estimate.

All seven companies had genuine new GPT-4.1 verifier draws. INTC stopped after bounded structural repairs because an uncovered “In summary” connector was treated as missing factual content. V7.1 permits only three additional complete discourse spans: “in summary”, “in conclusion”, “overall”. Negation, causal terms and substantive omissions remain rejected. It revalidates real V7 responses with immutable request/wire/cost provenance, and generates only missing sections. Revalidation is not independent generation. Separate patches apply cleanly to the pinned upstream under `evidence/verifier_versions/patches_v7` and `patches_v7_1`. The latter passed 103 tests and six subtests.

Every V7.1 report was rendered with exact text and local assets, with source hashes asserted by `experiment.py`; timing is in `evidence/render_v7_study.json`. These candidate reports have not replaced the canonical V6 browser-QA inventory.

## Complete independent comparison

The original baseline evidence is unchanged for NVDA, AMD and INTC. Across all eight sections, 360 exactly unchanged sentence units reuse earlier independent labels, with text/evidence identity checked. Every one of the 173 changed units was explicitly decomposed and adjudicated by the Codex task agent against the original writer prompt. No human verification occurred. The exact specifications, source offsets, evidence citations, decision rationale and carry-forward maps are retained under each `*_after_v7_1_working`/`*_adjudicated` directory.

{table}
The common baseline has {score['before_supported']}/{score['before_total_claims']} supported claims ({pct(score['before_grounding_rate'])}), {score['before_contradicted']} contradicted claims ({pct(score['before_contradiction_rate'])}), and {score['before_numeric_incorrect']}/{int(score['before_numeric_correct'])+int(score['before_numeric_incorrect'])} errors among verifiable numeric claims ({pct(score['before_numeric_error_rate_verifiable'])}). V7.1 has {score['after_supported']}/{score['after_total_claims']} supported ({pct(score['after_grounding_rate'])}), {score['after_contradicted']} contradicted ({pct(score['after_contradiction_rate'])}), and {score['after_numeric_incorrect']}/{int(score['after_numeric_correct'])+int(score['after_numeric_incorrect'])} verifiable numeric errors ({pct(score['after_numeric_error_rate_verifiable'])}). Including unverifiable numeric claims in the denominator gives {score['after_numeric_incorrect']}/{score['after_numeric_claims']} ({pct(score['after_numeric_error_rate'])}); that broader denominator can dilute apparent error rates.

Exact supported lexical-key retention is {pct(score['independent_supported_fact_retention'])}; supported evidence-span retention is {pct(score['supported_evidence_span_retention'])}. These are conservative lexical/location proxies, not semantic coverage guarantees: new paraphrases and finer decomposition affect keys. Claim count is {pct(score['claim_count_ratio'])} of baseline and verifier-reported visible retention is {pct(score['verifier_reported_claim_retention_all_sections'])}. Neither a larger claim count nor repeated evidence establishes better information retention. Inspect original-span mappings and both retention definitions.

Warning recall rises from V6's {pct(old['warning_recall'])} to {pct(ann['warning_recall'])}, and warning precision from {pct(old['warning_precision'])} to {pct(ann['warning_precision'])}. Unflagged unsupported/contradicted claims fall from {old['unflagged_unsupported_or_contradicted']} to {ann['unflagged_unsupported_or_contradicted']}; unflagged contradictions remain {ann['unflagged_contradicted']}. Raw denominators differ because of changed phrasing, decomposition and duplicate removal. This is a measured improvement in warning coverage, not proof that the narrative is factually safe. All {ann['contradicted']} contradictions, including warned ones, remain errors in the underlying assertions.

The selected three-company V7.1 responses allocate {int(score['verifier_extra_tokens_all_sections']):,} tokens, ${float(score['verifier_extra_cost_usd_all_sections']):.6f} and {float(score['verifier_extra_runtime_seconds_all_sections']):.2f} seconds to the treatment. V7.1 itself adds ${float(score['verifier_incremental_cost_usd_this_run']):.6f}; inherited calls were paid in V7. Failed repairs, the other companies, pilot judges and authoring are outside this selected-treatment subtotal and belong in the global accounting. Codex task-agent adjudication usage is unavailable, not free.

## The evaluator also failed

The unblinded GPT-4.1 proposal incorrectly declared 18 changed NVDA units nonfactual because they carried UNVERIFIED annotations. A second proposal run hid only the warning prefix in judge input, preserving the complete original text, offsets and reversible mapping outside that input. It no longer skipped those units, but still combined independent premises and sometimes missed explicit supplied facts—for example, the $89bn NVIDIA Data Center amount on original evidence line45. Neither proposal set is canonical. `evidence/v7_evaluator_failure_summary.json` records coverage failures, and the task-agent specifications replace the inadequate proposals. This demonstrates why automated label counts and schema coverage alone are insufficient.

Known residual issues include period labels, CAGR horizons and peer-comparison semantics. Formal submission and zero-factual-error certification remain unresolved. Medium and the course PR are explicitly deferred by the user; no publication is authorized now.

## Recompute

Run `scripts/grounding_audit.py compare --pairs config/grounding_v7_1_pairs.json --output NEW_DIRECTORY` and `scripts/analyze_verifier_annotations.py --pairs config/grounding_v7_1_pairs.json --output ANOTHER_NEW_DIRECTORY`. These operate only on saved data. The exact paid plans are `config/verifier_v7_plan.json` and `config/verifier_v7_1_plan.json`; they are historical experiments, not a recommendation to repeat paid failures. Freeze a new destination and check the cumulative ledger before any fresh generation.
'''
(ROOT/'docs/verifier_v7_followup.md').write_text(text,encoding='utf-8')
write_json(ROOT/'evidence/progress/v7_study_complete.json',{'completed_utc':utc_now(),
    'whole_assignment_complete':False,'canonical_release_unchanged':'v6',
    'comparison':'config/grounding_v7_1_pairs.json','fully_audited_development_companies':['NVDA','AMD','INTC'],
    'changed_units_explicitly_reviewed':173,'exact_units_carried':360,'all_eight_sections':True,
    'no_human_validation':True,'publication':'explicitly deferred by user',
    'next':'Evaluate the bounded reasoning-model pilot; then consolidate costs/docs/PDF/QA before any release selection.'})
print('Wrote measured V7.1 follow-up; did not change canonical release.')
