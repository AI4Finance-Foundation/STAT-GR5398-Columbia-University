"""Assemble the editable research manuscript from the completed saved audits.

Keeps the prior manuscript by content hash. No truth labels or model outputs are
created here: the tables and charts are summaries of separately preserved work.
"""
import csv,json,re
from pathlib import Path
from audit_io import sha256,write_json,utc_now
from grounding_audit import verifier_accounting,csv_write
from update_coverage_status import ACTIVE_SNAPSHOTS
from release_followup import report_addendum,current_accounting
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig',newline='')))
def pct(x):return f'{100*float(x):.2f}%'
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in data]])
def archive(path):
    raw=path.read_bytes();dest=ROOT/'evidence/manuscript_versions'/(path.stem+'_'+sha256(raw)[:12]+path.suffix)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_bytes(raw)

def main():
    path=ROOT/'docs/research_report.md';old=path.read_text(encoding='utf-8');archive(path)
    # The selected SEC audit and forecasting discussion have already been reviewed.
    mid=old[old.index('## 4. Data audit:'):re.search(r'^## 7\.',old,re.M).start()]
    mid=mid.split('Additional publicly sourced semiconductor valuation results:')[0]
    mid=mid.replace('sixty reconciled values across five available companies','eighty-four reconciled values across seven distinct companies (including the two onboarding-only names)')
    mid=mid.replace('Seven saved notebooks','Nine saved notebooks').replace('`evidence/notebook_portability.json`','`evidence/notebook_portability_v6.json`')
    mid=mid.replace('CSVs, native analyzer outputs and heatmaps are under `audits/target_price_sensitivity/`.','CSVs, native analyzer outputs and heatmaps are under `audits/target_price_sensitivity/` for NVDA/AMD/INTC and `audits/sensitivity/` for AVGO/QCOM.')
    score=rows('audits/grounding/comparison_v6/before_after_scorecard.csv');companies=[r for r in score if r['section']=='ALL' and r['ticker']!='ALL'];overall=next(r for r in score if r['section']=='ALL' and r['ticker']=='ALL')
    annotation=rows('audits/grounding/annotation_v6/annotation_scorecard.csv');ann=next(r for r in annotation if r['ticker']=='ALL' and r['section']=='ALL')
    scope=rows('coverage_universe.csv');cost=read('audits/resources/cost_summary.json');billing={r['run_or_audit_id']:r for r in rows('audits/resources/cost_runtime_by_run.csv')}
    public=[];sens=[];per_report=[]
    for ticker,snapshot in ACTIVE_SNAPSHOTS.items():
        v=read(f'snapshots/{snapshot}/valuation/engine_valuation.json');v={x['method']:x for x in v['results']}
        if ticker in {'AVGO','QCOM'}:
            public.append([ticker,f"{v['EV/EBITDA']['assumptions']['Current Price']:.2f}",f"{v['EV/EBITDA']['target_price']:.2f}",f"{v['DCF']['target_price']:.2f}",'2028E; 12x fallback'])
        if ticker in {'NVDA','AMD','INTC','AVGO','QCOM'}:
            folder='audits/sensitivity' if ticker in {'AVGO','QCOM'} else 'audits/target_price_sensitivity'
            s=read(f'{folder}/{ticker}/target_price_sensitivity_summary.json')
            sens.append([ticker,f"{s['base_target_price']:.2f}",f"{s['minimum_target_price']:.2f}-{s['maximum_target_price']:.2f}",f"{s['local_price_pct_change_per_plus_1pp_margin']:.2f}%"])
        b=billing[ticker+'_baseline_1'];responses,unknown,records=verifier_accounting(ROOT/'runs'/f'{ticker}_final_v6')
        sm=read(f'snapshots/{snapshot}/manifest.json')
        per_report.append({'ticker':ticker,'snapshot':snapshot,'data_analysis_seconds':sm['runtime_seconds'],
            'writer_input_tokens':int(b['input_tokens']),'writer_output_tokens':int(b['output_tokens']),
            'writer_seconds':float(b['model_runtime_seconds']),'writer_cost_usd':float(b['cost_usd']),
            'verifier_allocated_seconds':sum(r['runtime_seconds'] for r in responses),
            'verifier_allocated_cost_usd':sum(r['cost_usd'] for r in responses),
            'verifier_incremental_v6_cost_usd':sum(r['incremental_cost_usd_this_run'] for r in records),
            'verifier_incremental_v6_tokens':sum(r['incremental_tokens_this_run'] for r in records),
            'verifier_unresolved_attempts':len(unknown)})
    render={r['run_id']:r['runtime_seconds'] for r in read('evidence/release_render_ui7.json')['records']}
    for r in per_report:
        r['baseline_render_seconds']=render[r['ticker']+'_baseline_1'];r['final_render_seconds']=render[r['ticker']+'_final_v6']
        r['accounting_note']='Stage times measured separately, not an observed uninterrupted end-to-end run; inherited verifier calls allocated once, not new V6 calls.'
    csv_write(ROOT/'audits/resources/report_stage_runtime.csv',per_report)
    pre='''# Can FinRobot support portfolio research? An evidence audit of its equity pipeline

**Zeyu Gu | UNI zg2556**  
STAT GR5398, Fall 2026, Assignment 1  
Frozen analytical clock: October 11, 2026, 21:10 UTC  
Upstream: AI4Finance-Foundation/FinRobot, `2717499b8e30f242640af08c4ad9afd1113c2d45`

**Release status:** real experiments and machine-assisted audits are recorded below. The research finds residual factual errors; it does not certify the generated investment reports. The user has explicitly deferred Medium publication and the course master pull request; the materials remain local. No human validation of claims is asserted.

## 1. Research question and principal findings

This experiment separates running an equity-report system from establishing that its output is reliable. Seven distinct companies have real eight-agent baseline and ninth-stage verifier outputs; five constitute the formal semiconductor cohort. NVIDIA has two additional independent baseline generations, making nine actual baseline runs. All consume saved real observations. Frozen-input reruns, saved-response revalidation and actual new generations are labeled separately.

Twenty selected financial observations for NVIDIA and AMD reconcile to accession-specific SEC filings: sixteen match directly and four EBITDA-related differences are explained by explicit definition bridges. Independently implemented EV/EBITDA and DCF arithmetic reproduces 84 engine components across seven companies. Neither result endorses the forecast: the default EBITDA formula omits R&D, historical multiple valuation is undiscounted, and debt is assumed to be 10% of estimated enterprise value.

The full three-company claim audit contains 1,478 baseline atomic assertions across all 24 agent sections. Only 291 (19.69%) are supported by the actual generating prompts; 40 contradict them. The verifier-treated versions contain 1,590 assertions, including 358 supported and 48 contradicted. Its useful measured contribution is transparent warning coverage: 93.99% of final Unsupported/Contradicted assertions receive an unverified warning, with 78.78% precision. Seventy-four problematic assertions remain unflagged, including eight contradictions. This is an improvement in issue disclosure, not a demonstrated reliable correction system.

## 2. Implementation, environment and evidence boundary

The course supplied wrappers and audit headers rather than the full implementation. The actual upstream data processor, eight typed agents, valuation, sensitivity and rendering code were inspected at the fixed commit. The older AutoGen annual-report tutorial was not substituted. The upstream CLI also contains a plain-chat path; `experiment.py` deliberately invokes the actual eight typed agent implementations described by the course.

The dedicated Windows runtime is CPython 3.11.15; the complete recorded dependency set is `requirements-lock.txt` and environment details are in `evidence/preflight_configured/preflight.json`. Writer model: OpenAI `gpt-4.1-mini-2025-04-14`, temperature 0.7, 4,000 maximum output tokens per agent, structured outputs, SDK retries disabled. The verifier uses `gpt-4.1-2025-04-14`, temperature 0, up to 26,000 output tokens and at most two explicit validation repairs. Exact per-attempt settings, prompts, instructions, schemas, wire responses, usage, timings and failures are saved. Private configuration stays outside the course directory.

Compatibility repairs are separate from the single research improvement, Option A. They adapt entitled FMP stable endpoints, roll forecast labels forward from the last actual, expose missing inputs and engine assumptions, and render exact saved text with frozen market and valuation fields. AVGO/QCOM also need a public-source adapter after real FMP 402 denials. These are shared execution/data-availability changes, not competing verifier treatments. Public-source provenance is appended to those two writer prompts and saved; the three-company comparison uses unchanged original FMP-based prompts.

The final measurement uses explicit Codex-assisted claim adjudication under a saved common protocol. The task-agent model ID, token use and runtime are not exported, so they are unavailable, not zero. Earlier GPT-4.1 API judge proposals had extraction/semantic defects and remain as development evidence with costs. The machine adjudicator saw each complete source unit and its original prompt, decomposed factual premises, and recorded reasons; it did not use verifier verdicts as labels. Exact unchanged after sentences inherit their previously reviewed facts; all changed sentences have an explicit separate review. Shared model-family biases and development on the evaluated cohort limit independence. No human review took place.

The render manifest binds every final HTML section to the audited text hash. Later filings or online facts cannot make an original unsupported assertion supported. Offline replay rejects network connections and absent cached requests; it is neither a fresh observation nor another independent generation.

| Field | Source and interpretation |
|---|---|
| Cover price | Frozen FMP observation, or public Yahoo daily close for AVGO/QCOM |
| Cover target | FMP consensus target, not an LLM or engine target; missing for public adapter names |
| Cover grade | FMP analyst grade/action; neither LLM recommendation nor financial-health score |
| Engine target | Explicit EV/EBITDA, peer or DCF method with separate assumptions |
| Narrative | Eight structured writer outputs; ninth-stage revision separately mapped and audited |
| Public-source EBITDA | GAAP operating income + cash-flow D&A; differs from FMP peer definition |

Legacy names also matter: an upstream `fwd_pe` field holds a historical ratio, and a net-debt/equity label can contain debt/equity. Renderer labels disclose these distinctions. `docs/field_lineage.csv` and `docs/data_valuation_source_audit.md` give source locations and formulas.

## 3. Industry universe, peers and actual coverage

The formal universe is semiconductor products, defined by material processor, accelerator, networking or wireless integrated-circuit businesses. SEC SIC is 3674 for NVDA/AMD/INTC/AVGO and 3663 for QCOM. Qualcomm's QCT platform segment is $38.367 billion of $44.284 billion FY2025 revenue (86.64%); Broadcom's semiconductor segment is $36.858 billion of $63.887 billion (57.69%). These support business-based membership while licensing and software constrain comparability. Sources: [Qualcomm FY2025 10-K](https://www.sec.gov/Archives/edgar/data/804328/000080432825000085/qcom-20250928.htm) and [Broadcom FY2025 10-K](https://www.sec.gov/Archives/edgar/data/1730168/000173016825000121/avgo-20251102.htm).

'''
    pre+=table(['Company','Actual peers','Common-window price return','Important comparability limit'],[[r['ticker'],r['actual_peer_tickers'].replace(';',', '),f"{float(r['close_price_return_pct']):+.2f}%",r['comparability_caveat']] for r in scope])
    pre+='''

The common window is October 10, 2025 to October 9, 2026. All five raw-close returns are positive; their magnitudes differ, but this is not balanced winner/loser coverage. Dividends are excluded and provider split conventions are not independently verified. FMP supplies the first three series and public Yahoo chart responses the last two. AAPL and GOOGL are onboarding-only, with cross-industry demonstration peers, and never count toward the formal five.

FMP returned 402 for AVGO/QCOM financials and for news access. Existing WRDS web exports were privately retained, but the user chose public materials after review of WRDS AI restrictions; no WRDS observations enter a model or this submission. Public AVGO/QCOM inputs use exact SEC 10-K comparative FY2023-2025 facts, matching iXBRL contexts, dated cover-page shares, and a saved Yahoo price response. AVGO D&A includes right-of-use amortization. Missing subject multiples remain missing, and the valuation fallback is explicitly 12x with a 3x spread, not a measured multiple. Their FMP peer snapshots are real and frozen but have different EBITDA definitions and fiscal calendars.

All five run sensitivity, catalyst and enhanced-news functions. News substitutes are narrow: one actual issuer release for the original FMP companies, one literal current SEC 10-Q income-statement table for AVGO/QCOM. They are not comprehensive press coverage. The latter table flattening creates quarter-versus-year-to-date ambiguity that the writer sometimes misreads. Native default earnings dates, heuristic 95% event probabilities and assumed 15% scenario standard deviations are explicitly assumptions. The original INTC/AAPL news-filter omission is preserved alongside the repaired `news_v2` analyses. The public data cutoff is an analytical clock, while HTTP fetches occurred later that day; this is a contemporary snapshot exercise, not proof of historical point-in-time availability or a backtest.

'''
    mid+='\nAdditional publicly sourced semiconductor valuation results:\n\n'+table(['Company','Quote USD','EV/EBITDA target','DCF target','EBITDA horizon / multiple'],public)+'\n\nAll five formal target-price sensitivity results (relative revenue +/-5%, margin +/-2 percentage points):\n\n'+table(['Company','Base target USD','Target range USD','Price effect of +1pp margin'],sens)+'\n\nAVGO and QCOM shares are dated August 28 and July 27, 2026, respectively, rather than contemporaneous share counts. Their FCF is GAAP operating cash flow less capex, not demonstrated unlevered FCF. These additional assumptions limit economic interpretation of the reproduced targets.\n\n'
    later='''## 7. Complete claim audit of the original prompts

The saved protocol covers all nonblank source units, with every unit receiving atomic factual premises or an explicit nonfactual reason. Baseline source-unit counts are 148 NVDA, 145 AMD and 146 INTC; after counts are 188, 175 and 158. This is complete textual coverage and explicit machine decomposition, not proof that every implicit semantic premise was recognized. Exact quotes, offsets, run IDs, original prompt hashes, evidence line ranges, classifications, numeric status and rationale are in the claim CSVs. A zero-claim section would be N/A; none occurs here.

Supported means entailed or arithmetically derivable from the original prompt. Unsupported means insufficient evidence, even if a later web search could establish truth. Contradicted requires an explicit conflict with supplied evidence. Numeric errors include wrong period, issuer, units and actual/forecast meaning, not just wrong digits. Qualitative causal and market-position claims require evidence too.

'''
    later+=table(['Company','Baseline S / total','Unsupported','Contradicted','Grounding'],[[r['ticker'],r['before_supported']+' / '+r['before_total_claims'],r['before_unsupported'],r['before_contradicted'],pct(r['before_grounding_rate'])] for r in companies]+[['ALL','291 / 1,478','1,147','40','19.69%']])
    later+='\n\nAcross all three companies, every section is included:\n\n'+table(['Section','Before S / total','Before U / C','After S / total','After U / C'],[[r['section'].replace('_',' '),r['before_supported']+' / '+r['before_total_claims'],r['before_unsupported']+' / '+r['before_contradicted'],r['after_supported']+' / '+r['after_total_claims'],r['after_unsupported']+' / '+r['after_contradicted']] for r in score if r['ticker']=='ALL' and r['section']!='ALL'])
    later+='''

Competitor analysis has the weakest baseline support (17/295, 5.76%), followed by risks (20/250, 8.00%) and company overview (28/295, 9.49%). These agents request moats, management history, customer exposure and causal explanations from prompts primarily containing financial/peer tables and one issuer excerpt. The instructions invite evidence gaps. Investment update and news summary have relatively more supplied earnings facts, but still add consensus comparisons and causal claims not in those excerpts.

One concrete error is NVIDIA's historical contribution margin described as consistently above 70%, although its prompt includes 64.9% and 56.9%. Another repeatedly carries the historical 68.3% revenue CAGR forward through 2029, despite future annual assumptions of 5%, 6%, 4%. AMD's 20.5% actual CAGR likewise appears as if it describes the entire horizon through 2028. Exact table matching without period semantics does not detect these failures.

## 8. Three independent runs and stability

`NVDA_baseline_1`, `_2`, `_3` are three real generations: 24 paid writer calls with identical per-agent request hashes, frozen data, model and parameters, and distinct provider completion/call IDs. Risk extraction explicitly covers all 76 risk source units. It uses the first three distinct risk categories mentioned, with a fixed taxonomy and tie convention, not a claim of economic importance ranking.

| Run | Explicit writer rating | Writer target | First three risk categories |
|---|---|---|---|
| 1 | Not stated | Not stated | Demand cyclicality; macro/currency; competition |
| 2 | Conditional hold or increase | Not stated | Demand cyclicality; revenue concentration; export controls/geopolitics |
| 3 | Overweight | Not stated | Demand cyclicality; revenue concentration; supply capacity |

Writer target variance is N/A because no explicit per-share target is present. Rating variance is N/A because only one draw has a unique standard rating; one observation is not a zero-variance finding. The frozen FMP target is $339.17 in all three runs, with sample variance zero, which provides no evidence about LLM stability. Risk-set Jaccard overlaps are 0.20, 0.20 and 0.50. Kendall agreement over shared categories is N/A, N/A and 1.0; the last pair shares only two categories, so it is weak evidence. All quotes, risk mappings, missing fields and generation independence checks remain under `audits/stability/NVDA_adjudicated/`. Three draws cannot characterize the full output distribution.

## 9. Option A: verifier gains and remaining failures

The ninth stage combines structured evidence review with deterministic citation and numeric-token binding. Code reconstructs verbatim prompt lines, rejects nonexistent or structural-only citations, checks complete source-unit/word coverage, and maps each original span to a retained, corrected, annotated or withheld release. Unsupported facts are visibly annotated, not approved on plausibility. A claim labeled Contradicted by the verifier must be corrected with evidence or withheld; a proposed correction that fails deterministic checks is not released. However, the verifier can itself misclassify a contradiction, which explains independently observed residual conflicts.

V1-V4 real failures exposed quote copying, number/period bindings, missing words and overly strict action validation. V5 allowed evidence-bounded rephrasing and revalidated saved model responses under immutable contracts. V6 rejects citations containing only a company identity, heading or table header. The first five V6 reports revalidate existing paid reviews without new model generation; AVGO/QCOM have new V6 calls. All earlier attempts and charges remain visible. This adaptive development uses the evaluated cohort and is not a preregistered held-out test. Source snapshots and version hashes preserve each iteration.

Before and after use the same original evidence and independent protocol for the complete eight sections of NVDA, AMD and INTC. The final HTML consumes exactly the after-audited text. The quantitative outcome is:

'''
    later+=table(['Company','Grounding before -> after','Contradiction before -> after','Verifiable numeric error before -> after'],[[r['ticker'],pct(r['before_grounding_rate'])+' -> '+pct(r['after_grounding_rate']),pct(r['before_contradiction_rate'])+' -> '+pct(r['after_contradiction_rate']),pct(r['before_numeric_error_rate_verifiable'])+' -> '+pct(r['after_numeric_error_rate_verifiable'])] for r in [*companies,overall]])
    later+='''

Numeric errors are 25/171 verifiable numeric claims before and 29/217 after. Including unverifiable numeric claims in the denominator gives 25/205 (12.20%) and 29/258 (11.24%); both denominators are reported to prevent missing evidence from disguising error. Raw supported share rises 2.83 percentage points, but contradiction count rises from 40 to 48. More repeated assertions change the denominator: final text contains 155 duplicate lexical-key instances. These are not demonstrated unique-fact improvements.

Visible claim count is 107.58% of baseline; independently supported lexical fact-key retention is 96.13%, and supporting evidence-span retention is 100%. The verifier reports 99.80% visible retention but only 10.06% retained as verified. Lexical matching can miss paraphrases and table spans can support multiple facts, so neither proxy proves semantic fact recall. The claim mapping and remaining-error casebook must accompany these rates. The system did not improve grounding by deleting all information; it often preserves too much repeated unverified material.

'''
    later+=table(['Company','Warning recall on U/C','Warning precision','Unflagged U/C','Unflagged C'],[[r['ticker'],pct(r['warning_recall']),pct(r['warning_precision']),r['unflagged_unsupported_or_contradicted'],r['unflagged_contradicted']] for r in annotation if r['section']=='ALL'])
    later+='''

Before treatment there are no warnings, so warning recall is zero and warning precision is N/A. After treatment, 1,158 of 1,232 problematic assertions are covered by warnings; 312 supported assertions are also inside warned paragraphs. Annotation coverage is a measurable review aid, but most text becomes warned and reviewer burden is high. Eight contradictions remain unflagged: examples include NVIDIA's latest comparable multiple being above AMD when the prompt shows 33.7774x versus 47.8456x, and actual 2025 values labeled as projections. Warning status is computed over the whole released paragraph, independently of verifier labels.

The appropriate conclusion is a partial engineering success and a semantic failure: the verifier exposes many evidence gaps while it remains unreliable as a correction gate. Residual errors and failed repairs are preserved, not cleaned out of the scorecard after seeing results. AVGO/QCOM and the two onboarding-only names have verifier outputs, but no full independent claim score is asserted for them. Contradicted content means the generated reports do not satisfy a strict zero-factual-error release standard. The research report describes those errors as experimental observations and does not recommend trading on them.

![Measured grounding and contradiction rates on all 24 sections; repeated claims affect denominators.](../audits/grounding/comparison_v6/before_after_rates.png)

## 10. Cost, runtime and reproducibility

'''
    later+=f"The direct API ledger contains {cost['settled_call_count']} settled calls, {cost['input_tokens']:,} input tokens and {cost['output_tokens']:,} output tokens, totaling **${cost['cost_usd']:.5f}** of the authorized $20, with zero unresolved reservations. Sum of measured model-call time is {cost['model_runtime_seconds']:.2f} seconds. This includes preflight, abandoned verifier/judge attempts, repairs and all nine baseline generations. Prices are the saved $0.40/$1.60 per million input/output for mini and $2/$8 for GPT-4.1; cached tokens are conservatively charged full price. It is a reproducible list-price upper estimate, not an invoice. Codex-assisted authoring/adjudication usage is not available in that ledger and is not asserted to be free. No data subscription upgrade was purchased.\n\n"
    later+=table(['Company','Writer tokens in / out','Writer seconds / USD','Allocated verifier seconds / USD','Baseline / final render seconds'],[[r['ticker'],f"{r['writer_input_tokens']:,} / {r['writer_output_tokens']:,}",f"{r['writer_seconds']:.2f} / {r['writer_cost_usd']:.5f}",f"{r['verifier_allocated_seconds']:.2f} / {r['verifier_allocated_cost_usd']:.5f}",f"{r['baseline_render_seconds']:.2f} / {r['final_render_seconds']:.2f}"] for r in per_report])
    later+='''

These stages were measured separately, so their sum is a staged reproduction estimate rather than an observed uninterrupted end-to-end service latency. Snapshot analysis times are separately recorded in `report_stage_runtime.csv`. NVIDIA replicates 2 and 3 used 20,261 and 19,894 total tokens, $0.01279 and $0.01220, and 27.21 and 24.38 seconds. For the paired comparison, the 24 allocated verifier reviews total 170,352 tokens, $0.763494 and 310.62 seconds. V6's incremental paid cost for these three inherited reviews is zero; this does not erase their original paid cost or the global development overhead. New AVGO/QCOM verifier costs are $0.245748 and $0.296004, including a QCOM repair.

Both patches apply to a clean copy of the pinned upstream, with 101 tests and 6 subtests passing. The clean patched implementation then reproduces all seven data pipelines and 14 saved baseline/final reports with sockets disabled (`reproduction/clean7_v3/` execution summarized in saved evidence). Nine relocated independent notebooks execute without upstream imports, pandas or a network, with all original valuation CSV hashes unchanged. A replay defect in derived snapshots was discovered and fixed by resolving and hash-checking the original sibling HTTP capture; failed runs remain logged. No replay is counted as a new generation.

## 11. What a portfolio manager can use

Accession-linked historical observations, the twenty reconciled data rows, inspectable scenario equations and source/error maps are useful for analyst screening and quality control. A reviewer can reproduce a selected input, locate a discrepant earnings definition and see how a stated assumption moves a target. The warnings offer a measurable triage aid, with substantial false positives and missed errors.

Unattended recommendations, position sizes and fair-value targets remain unreliable. Forecast EBITDA changes definition, R&D is omitted, the peer projection uses incompatible growth mechanics and dates, debt and discounting are crude, subject multiples may be fallbacks, and news is narrow. Machine narratives overreach the supplied evidence. A precise matching calculation and an attractive HTML report do not resolve those weaknesses. Human analysts must validate decisions and material claims; this experiment provides no human certification.

## 12. Submission, sources and repeatability

The course submission is the new `Assignment1_ZeyuGu_zg2556` folder, with no nested upstream clone or credentials. `README.md` gives exact clean-install, offline replay, fresh-generation and audit commands. `coverage_universe.csv`, `snapshots/`, `runs/`, `audits/`, `patches/` and the artifact inventory form the evidence chain. Immutable raw inputs and model responses are preserved, including failures; reproducible summary metrics come from saved CSVs and manifests. New online retrieval requires new timestamps/run IDs and may fail or differ as service permissions and market data change. Public-source acquisition scripts record real response bytes; rerunning the frozen adapter itself is offline.

The complete analytical Medium article is a separate discussion of these experiments rather than pasted stock research. Its public URL and the course master PR will be added only after actual publication/submission. Publication is deferred at the user’s request, so this document does not declare the entire assignment formally submitted. A strict requirement of zero errors in generated narrative also remains unmet, transparently quantified above. Visual/asset QA and credential scanning have separate manifests and do not substitute for semantic correctness.
'''
    later=later.replace('## 10. Cost, runtime and reproducibility',report_addendum()+'## 10. Cost, runtime and reproducibility')
    later=later.replace('and $2/$8 for GPT-4.1;', ', $2/$8 for GPT-4.1 and $2.50/$15 for GPT-5.4;')
    path.write_text(current_accounting(pre+mid+later),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric,title in zip(axes,['grounding_rate','contradiction_rate'],['Supported / all assertions','Contradicted / all assertions']):
        x=list(range(3));ax.bar([i-.18 for i in x],[100*float(r['before_'+metric]) for r in companies],width=.36,label='Baseline',color='#527c91');ax.bar([i+.18 for i in x],[100*float(r['after_'+metric]) for r in companies],width=.36,label='Verifier V6',color='#d79549');ax.set_xticks(x,[r['ticker'] for r in companies]);ax.set_ylabel('Percent');ax.set_title(title);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'audits/grounding/comparison_v6/before_after_rates.png',dpi=180);plt.close(fig)
    print('Updated measured research manuscript and per-report resource accounting from saved records.')
if __name__=='__main__':main()
