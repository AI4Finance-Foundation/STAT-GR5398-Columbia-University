"""Evidence-backed release checkpoint; preserve verified publication status."""
import csv,json
from pathlib import Path
from audit_io import sha256,utc_now,write_json
ROOT=Path(__file__).resolve().parents[1]
def publication_context():
    publication=json.loads((ROOT/'config/publication_status.json').read_text(encoding='utf-8'))
    medium_url=publication.get('medium_url')
    medium_verified=bool(medium_url and publication.get('medium_status')=='published_by_user_verified_public')
    medium_note=(f'Medium was published by the user and its public URL verified: {medium_url}.'
        if medium_verified else 'Medium publication and its real URL are not yet verified.')
    pr_url=publication.get('course_pull_request_url')
    pr_note=(f'Course PR URL is recorded at {pr_url}; check its current submission status separately.'
        if pr_url else 'Course master PR has not been created; the user plans to submit from macOS.')
    remaining=[] if medium_verified else ['Verify Medium publication and its real URL']
    remaining.append('Verify the recorded course master PR' if pr_url else 'Create the course master PR from the prepared scoped package')
    return publication,medium_verified,medium_note+' '+pr_note,remaining

def main():
    cost=json.loads((ROOT/'audits/resources/cost_summary.json').read_text())
    pdf=json.loads((ROOT/'evidence/submission_pdf_review/selection.json').read_text())
    publication,medium_verified,publication_note,publication_remaining=publication_context()
    updates={}
    def setstatus(ids,evidence,status='verified',next_action='Preserve frozen evidence and rerun checks only when relevant files change.'):
        for key in ids.split():updates[key]=(status,evidence,next_action)
    setstatus('U01 U02a U02b C01 C02','Requirement/source inspection docs, pinned upstream SHA, exact environment and source lineage; typed agents and calculation/LLM boundary documented.')
    setstatus('U02c U20','patches/manifest.json: initial clean apply 101 tests +6 subtests; expanded local_final_tests.log 108 +6. Reasoning V8.2 clean patch passes 132 +6. evidence/reproducibility/clean7_v3 records seven offline pipelines and fourteen real-text report replays.')
    setstatus('U03 C03','Only the named FinRobot submission is modified; upstream cache external, no shared branch switch/reset; checkpoint, immutable evidence and cost/error logs retained.')
    setstatus('U04a U04b U04c',f"Saved real FMP/model preflights, $20 approved shared cap, ${cost['cost_usd']:.6f} settled estimate, {len(cost['unsettled_call_ids'])} unresolved reservations; public inputs after WRDS decision; no purchase. Failed reasoning-pilot usage reconciled from saved HTTP200 provider response.")
    setstatus('U05','Real experiments, English report/blog and Chinese progress; explicit remaining publication and semantic limitations.')
    setstatus('U06a','Five requested onboarding symbols run; nine actual baseline runs including three NVDA independent draws.')
    setstatus('U06b U07 U08a','coverage_universe.csv: five business-defined semiconductor companies with two actual peers each; AVGO/QCOM public SEC/Yahoo adapters disclosed; all required native analyzers and real eight-agent baseline/final stages run.')
    setstatus('U08b U10b U15e','Fourteen main HTML reports plus two NVDA repeat reports; exact saved/rendered section hashes; local assets checked in browser. Optional AVGO/QCOM EPS-PE chart explicitly unavailable because PE absent.')
    setstatus('U09a U09b U09c','Raw response bytes and safe metadata, fiscal/publication provenance, complete prompts/instructions/typed+wire responses and per-call timing/usage/failure IDs; evidence/costs.jsonl and audits/resources.')
    setstatus('U10a U10c U10d U10e','Actual+3 labels; original configuration retained; field_lineage.csv; frozen headline versus engine fields; explicit valuations/fallbacks and distinct native/target-price sensitivities.')
    setstatus('U11a U11b U11c','audits/data/data_audit.csv:20 SEC-context records for NVDA/AMD;16 matches and4 explained EBITDA differences. Ratios show formulas; market prices separately dated, not misrepresented as SEC metrics.')
    setstatus('U12a U12b','Default 5/6/4% revenue and gross/SG&A trajectories quantified; R&D omission diagnosed; evidence-linked analyst scenario proposals separated from unchanged experiment.')
    setstatus('U13a U13b U14a U14b','84 independently reconciled valuation components across seven companies; nine relocated notebooks; five formal tickers each have native analyzer and25 target-price scenarios, assumptions, ranges and elasticities.')
    setstatus('U15a U15b U15c U15d','Canonical config/grounding_v6_pairs.json: all8 sections for3 companies before/after;1478/1590 atomic claims. Full V7.1 and reasoning V8.2 follow-ups also cover all3x8 sections; reasoning after1958 claims from1467 source units. Exact original-prompt evidence and independent machine judgments, no human review. Two annotation errata preserved explicitly.')
    setstatus('U16a U16b U16c','audits/stability/NVDA_adjudicated:24 distinct real calls in3 identical-input runs; all76 risk units, explicit rating/target provenance; Jaccard/rank/missing-variance interpretation plus per-run tokens/time/cost.')
    setstatus('U17 U18','V6 verifier source and actual checked claims retain original/revised/released text, evidence, diagnostics and states; numeric and citation checks; renderer uses checked text. Independent semantic failures remain disclosed.')
    setstatus('U19a U19b U19c','V6, V7.1 and complete reasoning V8.2 matched full-section comparisons: grounding, contradiction, numeric error, retention proxies, warning recall/precision, costs and timing. Reasoning grounding19.76% to27.27%, C40 to16, verifiable numeric errors25/171 to13/435;12 unflagged contradictions remain and retention proxies disagree. No human-validation or held-out claim.')
    setstatus('U21a U21b U22a U22b U22c',f"Complete local artifact set, {pdf['pages']}-page selected research PDF, analytical Medium manuscript, README and inventory; report sources and older editions remain local-only. Includes full V7.1 and three-company reasoning follow-ups plus both pilots; main V6 release remains explicit. PM-use boundaries and failures disclosed. {publication_note}")
    setstatus('U23a','validate_delivery confirms quantitative scope, source-text bindings and exact before/after input equality; authoritative files allow metrics to be recomputed.')
    setstatus('U23b','Layout/asset/text encoding checks pass. Preserved V6 has48 contradicted after assertions,8 unflagged; reasoning V8.2 has16,12 unflagged. Errors are retained as experimental observations with complete independent audit, not certified correct.','partial','Course Part3 explicitly requests residual-error analysis and does not require zero hallucinations. The stricter user request for delivered reports without erroneous numbers cannot be certified for experimental generated prose; keep research and investment-use readiness distinct. See docs/submission_readiness.md.')
    setstatus('U23c','Actual credential-byte scan and forbidden-path checks; no private config/WRDS observations; offline/fresh-online modes explicitly separated.')
    setstatus('U24','README includes exact commands/SHA/model/key-free config/costs/dates; real Medium URL absent.','partial','Keep local per current user direction; only insert a genuine URL after a later authorized publication succeeds.')
    setstatus('U25a','Complete blog/medium_article.md and reviewable PDF, audit scorecards and patch prepared.','partial','Publication is explicitly deferred by the user; keep the complete article local and do not request authorization again unless the user resumes submission.')
    setstatus('U25b','Scoped new course folder and docs/course_pr_body.md prepared; no PR created or claimed.','missing','The user will submit the prepared package from macOS; use an isolated checkout/fork without switching/resetting the shared worktree.')
    setstatus('U26',f'Full-assignment submission status remains false. {publication_note} Residual experimental errors remain disclosed; course research readiness and error-free investment-use readiness are explicitly distinct.')
    semantic_path=ROOT/'evidence/semantic_study_summary.json'
    if semantic_path.exists():
        study=json.loads(semantic_path.read_text(encoding='utf-8'));s=study['scorecard'];a=study['warnings']
        latest=f"Latest V9.1: baseline {s['before_supported']}/{s['before_total_claims']} Supported, {s['before_contradicted']} Contradicted; final {s['after_supported']}/{s['after_total_claims']} Supported, {s['after_unsupported']} Unsupported and {s['after_contradicted']} Contradicted. Independent machine-assisted full3x8-section audit; evaluator errata separate from output changes, originals preserved; no human/held-out claim."
        setstatus('U15a U15b U15c U15d U19a U19b U19c',latest+' Complete scorecards include numeric-error denominators, both retention proxies, warnings, allocated/new costs and runtime.')
        setstatus('U02c U20','Separate canonical and versioned compatibility/Option A patches; latest evidence/verifier_versions/patches_v9_1 clean apply and full tests. Genuine copied-submission replay: original-root reads denied, networking/paid calls disabled, future execution clock; exact HTML,24 sections,9 assets and market/valuation hashes.')
        setstatus('U21a U21b U22a U22b U22c',f"Complete local research artifacts; {pdf['pages']}-page selected PDF; report sources and older editions are local-only by user direction and full analytical article. Original experiment, evaluator corrections and V9/V9.1 measured follow-ups distinguished. Actual PDF QA and final integrity/secret checks recorded. {publication_note}")
        setstatus('U23b',f"Current layout/asset/text identity QA recorded separately. {latest} Original experimental prose retains mistakes; {s['after_numeric_unverifiable']} latest numeric claims unverifiable.",'partial','No error-free narrative certification: retain measured failures and Unknown/Unsupported statuses. See docs/submission_readiness.md.')
    scenario_path=ROOT/'audits/forecast_parameter_scenarios/run_v2/manifest.json'
    if scenario_path.exists():
        scenario=json.loads(scenario_path.read_text(encoding='utf-8'))
        assert scenario['status']=='PASS' and scenario['all_checks_passed'] and scenario['new_api_calls']==0
        setstatus('U12a U12b','Executed frozen-input NVDA/AMD default and analyst-deceleration cases in audits/forecast_parameter_scenarios/run_v2;220 baseline fields match,140 implementation arithmetic checks and670 independently authored numerical/integrity checks pass. Annual growth30/20/10% and15/10/8%,zero margin increments,rounded historical ratios,EPS/EBITDA limitations and same-method target deltas explicit. All9,083 protected original submission files and verifier comparisons remain unchanged. No fitted forecast/backtest claim.')
        setstatus('U25a','Updated full blog manuscript plus blog/medium_metadata.json: title,subtitle,tags and article hash; docs/publication_preparation.md gives local review and publication sequence.','partial','User requested preparation only; do not publish or create an online draft without later direction.')
        setstatus('U25b','Updated docs/course_pr_body.md,exact required folder and local export preparation;latest observed master and actual-application proof are referenced by evidence/course_submission_diff.json. No PR created.','missing','Before the user submits from macOS,verify current master,article URL,secret-safe scope and actual patch application again.')
    # Apply current publication facts after historical/scenario-specific defaults.
    if medium_verified:
        setstatus('U24',f"README includes exact commands/SHA/model/key-free config/costs/dates and the verified Medium URL: {publication['medium_url']}.")
        setstatus('U25a',f"The user published the complete analytical article; its public URL was verified: {publication['medium_url']}. Current manuscript and publication evidence are recorded in blog/medium_metadata.json.",next_action='Keep the verified URL and manuscript bindings; prepare the remaining course master PR. No assistant remote write is authorized.')
    path=ROOT/'docs/acceptance.csv';raw=path.read_bytes();archive=ROOT/'evidence/progress'/('acceptance_before_'+sha256(raw)[:12]+'.csv');archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():archive.write_bytes(raw)
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if row['id'] in updates:row['status'],row['verified_evidence'],row['blocker_or_next_action']=updates[row['id']]
    with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    checkpoint={'as_of_utc':utc_now(),'assignment_complete':False,'live_paid_driver_session_id':None,
      'run_states':{p.parent.name:json.loads(p.read_text())['state'] for p in (ROOT/'runs').glob('*/manifest.json')},
      'settled_api_cost_usd':cost['cost_usd'],'unsettled_reservations':cost['unsettled_call_ids'],'approved_cap_usd':20,
      'formal_real_tickers':['NVDA','AMD','INTC','AVGO','QCOM'],'grounding_complete_paired_tickers':['NVDA','AMD','INTC'],
      'wrds_status':'Private exports only; user selected public SEC/Yahoo materials; no WRDS values in prompts/submission.',
      'parallel_agent_status':'Root and task agents completed preserved work, evaluator errata, narrow guard review and migration validation; no extra credits purchased.',
      'canonical_pairs':'config/grounding_v6_pairs.json','stability_directory':'audits/stability/NVDA_adjudicated',
      'completed_development_pairs':[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'config').glob('grounding_*pairs.json')) if p.name!='grounding_v6_pairs.json'],
      'reasoning_pilot':'High effort failed with reconciled paid usage; low effort succeeded. Separate24 fresh reviews across3 companies completed and independently machine-adjudicated. No active paid driver.',
      'research_pdf':'output/pdf/research_report_submission.pdf','medium_draft':'blog/medium_article.md',
      'publication':publication,'publication_summary':publication_note,
      'remaining':publication_remaining+['Generated experiment narratives retain measured errors; user-requested error-free numbers cannot be certified. Course itself requires honest residual-error reporting, not a perfect verifier.'],
      'safe_resume_commands':['python scripts/summarize_runs.py','python scripts/validate_delivery.py','python scripts/checkpoint_release.py'],
      'resume_warning':'No further paid call is needed for current results. Never count saved-review revalidation or offline replay as independent generation. Do not restart historical failed plans.'}
    write_json(ROOT/'evidence/progress/checkpoint.json',checkpoint)
    print(json.dumps({'assignment_complete':False,'status_counts':{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)},'cost':cost['cost_usd']}))
if __name__=='__main__':main()
