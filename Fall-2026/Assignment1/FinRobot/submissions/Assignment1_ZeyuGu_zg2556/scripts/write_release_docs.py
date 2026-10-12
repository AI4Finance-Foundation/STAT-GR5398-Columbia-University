"""Write the submission entry point, article and publication-ready PR text."""
from pathlib import Path
from audit_io import sha256
from release_followup import current_accounting,docs_addendum
ROOT=Path(__file__).resolve().parents[1]

README=r'''# FinRobot: execution, evidence and verifier audit

**Zeyu Gu | zg2556 | STAT GR5398 Fall 2026 Assignment 1**

Five formal semiconductor companies (NVDA, AMD, INTC, AVGO, QCOM), each with two real frozen peers, have baseline and verifier-treated HTML reports. The five requested onboarding names also ran; AAPL/GOOGL remain onboarding-only. NVIDIA has three independent generations with identical inputs. There are nine real baseline runs and seven final verifier outputs.

The deliverable contains 20 SEC financial reconciliations; independent valuation notebooks; native and target-price sensitivities; complete machine-assisted claim inventories across all eight sections of NVDA/AMD/INTC before and after; stability evidence; applicable upstream patches; and an English [research PDF](output/pdf/research_report.pdf), [editable source](docs/research_report.md) and [complete Medium article](blog/medium_article.md).

**Publication status:** Medium URL and course master PR are **not published: user requested local-only materials**. This is not a declaration of formal submission. The verifier is an experimental review aid: the independent after audit still finds 48 contradicted assertions, including eight without an unverified warning. HTML notices and the report disclose these failures. No human claim validation is asserted, and no zero-factual-error certification is offered.

## Read the result

| Formal company | Baseline HTML | Final V6 HTML | Frozen snapshot / peers |
|---|---|---|---|
| NVDA | [Baseline](runs/NVDA_baseline_1/output/NVDA/report/Audited_Equity_Report_NVDA.html) | [Final](runs/NVDA_final_v6/output/NVDA/report/Audited_Equity_Report_NVDA.html) | `NVDA`; AMD, INTC |
| AMD | [Baseline](runs/AMD_baseline_1/output/AMD/report/Audited_Equity_Report_AMD.html) | [Final](runs/AMD_final_v6/output/AMD/report/Audited_Equity_Report_AMD.html) | `AMD`; NVDA, INTC |
| INTC | [Baseline](runs/INTC_baseline_1/output/INTC/report/Audited_Equity_Report_INTC.html) | [Final](runs/INTC_final_v6/output/INTC/report/Audited_Equity_Report_INTC.html) | `INTC_news_v2`; NVDA, AMD |
| AVGO | [Baseline](runs/AVGO_baseline_1/output/AVGO/report/Audited_Equity_Report_AVGO.html) | [Final](runs/AVGO_final_v6/output/AVGO/report/Audited_Equity_Report_AVGO.html) | `AVGO_public_v1`; NVDA, AMD |
| QCOM | [Baseline](runs/QCOM_baseline_1/output/QCOM/report/Audited_Equity_Report_QCOM.html) | [Final](runs/QCOM_final_v6/output/QCOM/report/Audited_Equity_Report_QCOM.html) | `QCOM_public_v1`; AMD, INTC |

Open each HTML with its sibling `assets/` directory intact. `runs/AAPL_baseline_1`, `runs/GOOGL_baseline_1` and their `final_v6` directories contain the onboarding-only reports. Full `output/{TICKER}/analysis` directories are included in every run. NVDA `_baseline_2` and `_baseline_3` are the two additional real stability draws, not copies.

Primary results: baseline support 291/1,478 (19.69%); final support 358/1,590 (22.52%); contradiction 40 -> 48. Warning recall on final unsupported/contradicted assertions is 93.99%, precision 78.78%, supported lexical-fact retention 96.13%. Repeated claims change denominators, so raw support growth is not proof of successful factual correction. All labels are machine-assisted under a disclosed common protocol.

## Version, source boundary and cost

- Upstream: https://github.com/AI4Finance-Foundation/FinRobot, commit **2717499b8e30f242640af08c4ad9afd1113c2d45**. `patches/compatibility.patch` applies first, then `patches/option_a_verifier.patch`. `patches/manifest.json` gives hashes, source paths and clean-apply evidence. No upstream `.git` is submitted.
- Dedicated runtime: Python 3.11.15 on Windows; `requirements-lock.txt`, `evidence/preflight_configured/preflight.json`. Clean patch tests: 101 tests + 6 subtests; genuine offline replay covers seven data pipelines and fourteen saved reports with sockets disabled.
- Writer: `gpt-4.1-mini` resolved to `gpt-4.1-mini-2025-04-14`, temperature 0.7, maximum 4,000 output tokens per agent. Verifier: `gpt-4.1` resolved to `gpt-4.1-2025-04-14`, temperature 0, maximum 26,000, at most two validation repairs. Exact instructions, requests, schemas, wire/structured responses and parameters are saved for every call.
- Formal evaluation: explicit Codex-assisted adjudication, exact model ID/usage unavailable; no human validation. Earlier direct-API judge proposals and failures are development artifacts, not the canonical labels.
- Recorded direct API cost: **$3.59449**, 209 settled calls, 728,661 input / 360,915 output tokens, zero unsettled reservations, within the single **$20** authorized cap. This includes abandoned versions, repairs, preflight and pilot evaluators. It conservatively uses saved list prices, not an invoice. Codex task-agent authoring cost is unknown, not zero. `audits/resources/` separates new calls from allocated reused verifier cost.
- Analytical clock: `2026-10-11T21:10:00+00:00`; individual later same-day fetch times remain explicit. Price window: 2025-10-10 to 2026-10-09. News lookback: 90 days. This is a contemporary frozen snapshot, not a backtest or proof of historical availability at exactly 21:10.
- AVGO/QCOM use real public SEC annual/quarterly filings and Yahoo chart responses after FMP HTTP402; their subject multiple is unavailable and the explicit valuation fallback is 12x +/-3x. Other financial snapshots and all peers use real FMP responses. Issuer releases or literal SEC 10-Q tables substitute for denied FMP news. These are narrow sources, not a comprehensive news service.
- WRDS observations are excluded from models and submission; private export metadata documents the decision. No protected LLM service was available, and the user chose public material. No additional data purchase was made.

## Clean environment

Start PowerShell **inside this submission directory**. Install Git and uv from their official distributions if absent. Choose a new runtime directory outside the course tree; the script refuses to reset an existing clone.

```powershell
.\scripts\bootstrap.ps1 -RuntimeRoot "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction" -PythonVersion 3.11.15
$RunPython = "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction\venv\Scripts\python.exe"
$RunUpstream = "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction\upstream"
```

Bootstrap pins upstream, applies both patches, installs the lock and runs offline tests without paid calls or switching the shared course branch. HTML needs Matplotlib only. Research-PDF generation additionally needs Python `reportlab`, `pypdf`, `Pillow`, Windows Arial fonts and Poppler `pdftoppm`; `requirements-pdf.txt` and `docs/pdf_build.md` give exact setup. A fully configured Codex bundled runtime was used for the delivered PDF.

## Offline replay of every experiment

No credentials are needed. Use a new output path on every invocation. This replays all seven canonical data pipelines, verifies byte-identical analysis, recalculates valuations, and renders copies of every baseline-1/final report from saved text. Network sockets are rejected in child processes.

```powershell
& $RunPython scripts\reproduce_offline.py --upstream $RunUpstream --output reproduction\fresh_offline --render
& $RunPython scripts\check_notebook_portability.py --output reproduction\notebook_portability.json
& $RunPython scripts\fetch_sec_evidence.py --offline
& $RunPython scripts\data_audit.py --sec-references evidence\sec\snapshot_2026-10-11\sec_metric_references.csv --analysis-root snapshots --output-dir reproduction\data_audit
```

The two `news_v2` derived snapshots reference hash-checked sibling HTTP captures. AVGO/QCOM adapters verify real SEC facts against filing iXBRL and use cached FMP peers. Original snapshots are never replaced. The earlier clean7 replay failures and their fix are preserved in reproduction logs summarized under `evidence/reproducibility/`.

Every `snapshots/*/valuation/independent_valuation.ipynb` uses relative paths and standard-library arithmetic, not the engine. To independently recalculate and compare with the engine, or regenerate the target-price sensitivity:

```powershell
& $RunPython scripts\independent_valuation.py --analysis-dir snapshots\NVDA\analysis --valuation-inputs snapshots\NVDA\valuation_inputs.json --upstream-src "$RunUpstream\finrobot_equity\core\src" --output-dir reproduction\NVDA_valuation
& $RunPython scripts\target_price_sensitivity.py --analysis-dir snapshots\NVDA\analysis --valuation-inputs snapshots\NVDA\valuation_inputs.json --upstream-src "$RunUpstream\finrobot_equity\core\src" --output-dir reproduction\NVDA_sensitivity
```

Substitute each snapshot from the table above for the other formal companies. `audits/target_price_sensitivity/{NVDA,AMD,INTC}` and `audits/sensitivity/{AVGO,QCOM}` are canonical outputs. Revenue shocks are relative level changes, margin shocks are percentage-point changes; every scenario is propagated into the same per-share EV/EBITDA method.

Canonical label locations and exact pairs are `config/grounding_v6_pairs.json`. Recompute every score without a model call:

```powershell
& $RunPython scripts\grounding_audit.py compare --pairs config\grounding_v6_pairs.json --output reproduction\comparison
& $RunPython scripts\analyze_verifier_annotations.py --pairs config\grounding_v6_pairs.json --output reproduction\annotations
& $RunPython scripts\stability_adjudication.py --runs runs\NVDA_baseline_1 runs\NVDA_baseline_2 runs\NVDA_baseline_3 --output reproduction\stability
& $RunPython scripts\summarize_runs.py
```

To rebuild a claim CSV from its explicit saved judgments, run `scripts/materialize_adjudication.py --run runs/NVDA_final_v6 --evidence-run runs/NVDA_baseline_1 --spec audits/grounding/NVDA_after_v6_adjudicated/reviewed_specification.json --output reproduction/NVDA_after`. Other exact audit directories are in the pairs file. Re-materialization validates every source unit, citation, hash and label structure; it is not an independent new semantic judgment. Original prompts do not acquire later SEC/web evidence.

## New independent model generations over the same inputs

These incur API usage. Keep the same cumulative ledger and authorized policies, never run concurrent paid drivers and never clear reservations. Configure credentials locally outside the course folder; only a private file path is read below. `config/public.example.ini` is a key-free example. Environment variables `FMP_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` are also supported.

```powershell
$PrivateConfig = Read-Host 'Absolute path of private FinRobot config.ini (outside submission)'
$env:OPENAI_MODEL = 'gpt-4.1-mini'
1..3 | ForEach-Object {
  & $RunPython scripts\experiment.py --upstream $RunUpstream narrate --analysis snapshots\NVDA\analysis --ticker NVDA --company-name 'NVIDIA Corporation' --run "reproduction\NVDA_fresh_$_" --config $PrivateConfig --budget config\budget.approved.json --ledger evidence\costs.jsonl --max-tokens 4000
  if ($LASTEXITCODE -ne 0) { throw 'Generation failed; inspect saved attempt before continuing' }
}
```

The original three draws are `runs/NVDA_baseline_1` through `_3`. A new run ID and fresh successful provider calls define independence; `--resume` continues the same run. For all other actual baseline-1 inputs:

| Ticker | Analysis directory | Exact company name |
|---|---|---|
| AMD | snapshots/AMD/analysis | Advanced Micro Devices, Inc. |
| INTC | snapshots/INTC_news_v2/analysis | Intel Corporation |
| AVGO | snapshots/AVGO_public_v1/analysis | Broadcom Inc. |
| QCOM | snapshots/QCOM_public_v1/analysis | QUALCOMM Incorporated |
| AAPL | snapshots/AAPL_news_v2/analysis | Apple Inc. |
| GOOGL | snapshots/GOOGL/analysis | Alphabet Inc. |

Substitute the ticker, analysis, name and a new `--run` path in the same narrate command. Generate and render a new verifier treatment as follows:

```powershell
$env:OPENAI_MODEL = 'gpt-4.1'
& $RunPython scripts\experiment.py --upstream $RunUpstream verify --run runs\NVDA_baseline_1 --final-run reproduction\NVDA_new_verifier --config $PrivateConfig --budget config\budget.evaluation.json --ledger evidence\costs.jsonl --max-tokens 26000
& $RunPython scripts\experiment.py --upstream $RunUpstream render --run reproduction\NVDA_new_verifier
```

Use the same command for every company. Exact executed plans are retained in `config/*plan.json` with execution journals in `evidence/progress/`. Historical V1-V5 failures are not recommended reruns. The first five final-V6 runs use explicitly recorded `--reuse-review-run` inputs; this revalidates real saved reviews at zero incremental model cost and is **not** an independent new generation. New generations can differ despite frozen evidence; saved responses reproduce the delivered text exactly.

The direct-API `grounding_audit.py run` remains available for fresh judge proposals, but its outputs are not equivalent to the independently reviewed canonical labels. Full source-unit and atomic semantic review is required before treating fresh proposals as measured claims. No automatic command can confer human validation.

## New online acquisition is a different experiment

FMP example, with existing entitlement and a new timestamp/directory:

```powershell
$CaptureTime = [DateTimeOffset]::UtcNow.ToString('o')
& $RunPython scripts\snapshot_pipeline.py --upstream $RunUpstream --config $PrivateConfig --ticker NVDA --company-name 'NVIDIA Corporation' --peers AMD INTC --snapshot reproduction\NVDA_new_data --as-of $CaptureTime --years-limit 5 --news-days-back 90 --news-limit 50 --mode record
```

This command cannot reconstruct past market values. Without a new authorized news source, a denied news endpoint stays missing. Never relabel the October 2026 issuer excerpt as current news. The submitted public SEC/Yahoo acquisition utilities were used to record the dated original responses; `public_semiconductor_pipeline.py --upstream $RunUpstream --ticker AVGO --output reproduction/AVGO_public` reprocesses those frozen public inputs offline, not a new download. A new public capture must update source accession, filing period, market date and adapter cutoff deliberately, with new hashes/run IDs.

## Artifact map and release

`docs/acceptance.csv` maps every user/course criterion to code and evidence. `ARTIFACTS.csv` lists the submission files and hashes. `evidence/progress/checkpoint.json` gives recoverable state. `docs/field_lineage.csv`, forecast recommendations and SEC contexts explain accounting assumptions; source/development/version files preserve failures. `audits/grounding/comparison_v6` and `annotation_v6` are primary results; older development outputs are explicitly noncanonical. PDF and HTML QA do not certify investment claims.

Before publication:

```powershell
& $RunPython scripts\secret_scan.py --config $PrivateConfig
& $RunPython scripts\validate_delivery.py
```

`config.ini`, `.env`, credentials, WRDS observations, local runtime checkouts, caches and duplicate reproduction trees are excluded. Publication-ready PR text is in `docs/course_pr_body.md`. The user has explicitly requested that Medium and course-master PR publication be deferred. Keep these materials local until the user resumes submission; no public URL is invented.
'''

BLOG='''# I ran FinRobot, then audited what its reports actually knew

*Zeyu Gu (zg2556) | STAT GR5398, Fall 2026*  
*Complete article prepared for Medium; not yet published.*

The most useful number in my FinRobot experiment was not a stock-price target. It was 291: the number of supported factual assertions in a baseline inventory of 1,478 claims.

That finding came after the system had already done the visible work. It had loaded financial statements, projected revenue and earnings, produced charts, and written eight sections of equity research. The HTML reports looked like research. The question was whether their claims were supported by the information the writing agents had actually received.

For this assignment I ran the real `finrobot_equity` implementation at commit `2717499b8e30f242640af08c4ad9afd1113c2d45`. I froze the inputs, traced the outputs into HTML, checked financial values against SEC filings, independently rebuilt valuation arithmetic, repeated one company's generation three times, and added one improvement: a ninth verifier stage. The result is useful evidence about research automation, including a verifier that helps reviewers find problems but fails to make the prose reliably correct.

## First, make the experiment real

The course repository supplied instructions and wrappers. The upstream project contained the financial and agent implementations, including execution paths that were not interchangeable. I used its actual eight typed specialist agents and saved each complete evidence prompt, instruction, structured response, wire response, model setting, usage record and runtime. The baseline writer was GPT-4.1-mini at temperature 0.7. Every fresh run has a distinct identifier and real provider completion IDs.

The formal cohort is NVIDIA, AMD, Intel, Broadcom and Qualcomm, each with two actual peers. Apple and Alphabet also ran as required onboarding examples, but they do not count as semiconductor coverage. This matters: five successful API calls do not automatically make a coherent industry sample. Qualcomm's licensing business, Broadcom's infrastructure software, Intel's foundry and the fabless companies' economics also make consolidated comparisons imperfect.

Data access was an experimental constraint. FMP allowed the five onboarding symbols but denied Broadcom, Qualcomm and news endpoints. I did not buy a service or fill gaps with fabricated responses. Broadcom and Qualcomm instead use saved public SEC financial statements, exact filing contexts and Yahoo daily-price responses, with frozen real FMP peers. Missing subject multiples remain missing. Their valuation engine uses an explicitly labeled 12x fallback. News is limited to one issuer release or a literal SEC quarterly table, not a comprehensive media feed.

WRDS was investigated, but its observations were excluded from the model and the submitted package after reviewing the service's AI restrictions. Public materials provided a workable route. Different data definitions and narrow news remain limitations, even when the system executes successfully.

The common raw-close return window runs from October 10, 2025 to October 9, 2026: Broadcom +11.37%, Qualcomm +14.27%, NVIDIA +25.18%, AMD +182.97% and Intel +187.87%. Those are differing magnitudes, not a balanced winner-and-loser sample. They exclude dividends; provider split conventions were not independently verified. This is a frozen contemporary exercise, not a historical backtest.

## Reproducing a number is different from trusting its meaning

I checked ten pipeline metrics each for NVIDIA and AMD against their original annual SEC filings. Sixteen values matched within display precision. Four EBITDA and EBITDA-margin differences could be reconciled through explicit components. They were not unexplained data errors.

For NVIDIA FY2026, GAAP operating income plus D&A is $133.230 billion, while the FMP EBITDA field is $144.552 billion. The latter equals net income plus tax, interest expense and D&A. AMD needs a different bridge: its simple net-income formula is $92 million above the provider's $7.275 billion EBITDA; excluding $66 million discontinued-operations income and $26 million equity income explains the difference. The original [NVIDIA](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm) and [AMD](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm) filings make those bridges inspectable.

The forecast then changes the definition again. The default code eventually computes projected EBITDA as gross profit less SG&A, excluding R&D. AMD's historical EBITDA margin is about 21%, but its first forecast margin becomes 39%. A reader might attribute the jump to operating leverage. The code provides a much simpler explanation: a material expense is missing from the forecast measure.

The default revenue growth path is 5%, 6%, 4%, with gross margin rising one percentage point and SG&A/revenue falling half a point annually. Those are universal assumptions, not company guidance. I documented alternative deceleration scenarios and their rationale, but kept the baseline unchanged for the verifier comparison. Changing both the forecast and the verifier would make the experiment harder to interpret.

## Exact valuation arithmetic can expose weak economics

My independent notebook imports a standard-library calculation, not the upstream valuation engine. Across seven companies, 84 EV/EBITDA and DCF components reconcile. Nine notebooks, including two derived news versions, also execute after relocation without the upstream checkout or a network connection.

The result confirms what the program computes. It does not validate the assumptions. Its main multiple method applies an average historical multiple to third-year forecast EBITDA without discounting. Net debt is estimated as 10% of the resulting enterprise value. The DCF uses generic growth and discount rates; Intel's negative starting FCF eventually produces a clipped zero target.

For NVIDIA, the multiple method gives $333.71 and DCF gives $72.52. When I propagate relative revenue shocks of +/-5% and margin shocks of +/-2 percentage points through the same multiple method, the target range is $308.40-$359.93. The structural difference between valuation methods is much larger than that local scenario range. A narrow sensitivity plot can look reassuring while leaving the largest uncertainties untouched.

The native sensitivity analyzer reports revenue and EBITDA changes. I therefore saved it separately and explicitly propagated each of its 25 scenarios into per-share valuation. A margin change from 42% to 44% is two percentage points; it is not a 2% relative increase. These grids are deterministic scenarios, not calibrated confidence intervals.

## Audit what was in the prompt

My grounding rule is deliberately strict. If an assertion is true in the world but absent from the generating prompt, it is Unsupported. Later browsing cannot improve its baseline label. Contradicted means the prompt itself supplies a conflict. Supported means the prompt entails the claim or permits an explicit calculation.

The complete audit covers all eight sections for NVIDIA, AMD and Intel. Each nonblank source unit has explicit atomic premises or a nonfactual rationale, with offsets, hashes, exact evidence lines and reasons. This is Codex-assisted machine adjudication, not human validation. Early API-judge failures are preserved. Textual coverage does not prove perfect semantic decomposition, and the same model family can share biases.

The baseline counts are:

| Company | Supported / total | Unsupported | Contradicted |
|---|---|---|---|
| NVIDIA | 81 / 486 | 388 | 17 |
| AMD | 96 / 553 | 447 | 10 |
| Intel | 114 / 439 | 312 | 13 |
| Total | 291 / 1,478 | 1,147 | 40 |

Competitor analysis has only 17 supported assertions out of 295. Risks have 20 out of 250. Their prompts request moats, competitive positioning, management execution and company-specific risks from financial tables and a single earnings excerpt. Many resulting claims are plausible. Plausibility does not make the supplied evidence sufficient.

Some errors are direct numerical-semantic contradictions. NVIDIA's prose says contribution margins were consistently above 70%, while its own historical table includes 64.9% and 56.9%. A historical 68.3% revenue CAGR becomes a projection through 2029, even though the forecast path is 5%, 6%, 4%. Matching the digits alone is not enough; the period and actual-versus-forecast meaning are essential.

## The ninth agent helped with warnings, not reliable correction

The verifier combines an evidence-bounded review with deterministic checks. It resolves citations to actual prompt lines, checks numeric bindings and source coverage, preserves original/revised mappings, and makes the renderer use the treated text. Unsupported claims retain a visible warning. A claim recognized as Contradicted must be corrected from evidence or withheld.

Real failures drove several revisions of this same improvement: exact quote copying failed, numeric bindings used the wrong values, source words were omitted, and a review cited only a company-name line to support a business description. The final V6 rejects structural-only citations. Earlier attempts, source versions and costs remain available. Development used the evaluated cohort, so this is not a held-out benchmark.

Independent after auditing still finds 48 contradictions among 1,590 assertions. Support rises from 19.69% to 22.52%, but more repeated claims change the denominator. Verifiable numeric error falls from 25/171 to 29/217: a lower rate alongside a higher count. It would be misleading to present that as a clean accuracy improvement.

The useful gain is visible issue coverage. There were no baseline warnings. After treatment, warnings cover 1,158 of 1,232 unsupported or contradicted assertions: 93.99% recall, with 78.78% precision. The system retains 96.13% of independently supported lexical fact keys and 100% of supporting evidence spans. Those are imperfect retention proxies, not proof of semantic recall. It did not improve its score by deleting all information; it often preserves too much repeated unverified text.

The remaining failures matter. Seventy-four problematic assertions are unflagged, including eight contradictions. One says NVIDIA's comparable multiple is above AMD even though the supplied figures are 33.7774x and 47.8456x. The verifier can misclassify a contradiction and then faithfully enforce the wrong label. A warning layer is useful for triage, but it is not a substitute for an analyst.

## Repetition reveals another source of uncertainty

I generated NVIDIA three times with identical frozen evidence, model and parameters. The 24 agent calls have distinct completion IDs; they are not cached copies.

No run states an explicit writer per-share target. One has no rating, one conditionally recommends holding or increasing exposure, and one explicitly says overweight. Numerical rating and target variances are therefore N/A. The cover's FMP target stays at $339.17 with zero variance, which says nothing about the language model.

The first three distinct risk categories have pairwise Jaccard overlaps of 0.20, 0.20 and 0.50. The ordering is first mention, not economic importance. Only one pair shares enough categories for Kendall correlation, and that pair shares just two. Three draws are too few for a strong stability claim, but they are enough to show that fixed inputs do not imply a fixed investment narrative.

## What I would give a portfolio manager

I would provide the filing-linked historical table, the reconciled definitions, independent calculations, explicit scenario equations and claim-level error maps. These support screening and quality control. I would also show the source gaps and every fallback beside the output that uses them.

I would not hand over an unattended recommendation, position size or fair-value number. The forecasting definition, debt rule, multiple selection, source coverage and narrative entailment remain weak. The verifier improves disclosure but leaves both errors and substantial review burden.

The direct API ledger totals $3.59449 for 209 settled calls, including failed attempts and repairs, within the authorized $20. It records 728,661 input and 360,915 output tokens and 1,662.67 seconds of model-call time. These are conservative list-price estimates, not invoices; Codex-assisted authoring and adjudication costs are unavailable rather than assumed zero. Baseline eight-agent generation takes roughly 25-31 seconds per company in this environment, with verifier time and local processing recorded separately.

The experiment's reproducibility package fixes the upstream commit, separates compatibility and verifier patches, preserves raw responses and prompts, and can replay all seven data pipelines with network sockets disabled. Clean patch application passes 101 tests and six subtests. Those checks establish executable provenance. They do not establish financial truth.

That distinction is the lesson I would keep: research automation needs an evidence contract at every boundary. A data field needs a definition, a forecast needs an assumption, a target needs a valuation method, a sentence needs supplied evidence, and a verifier needs its own evaluation. When those contracts fail, the report should make the failure visible.
'''

PR='''## FinRobot Assignment 1 — Zeyu Gu (zg2556)

Adds the required new `Fall-2026/Assignment1/FinRobot/submissions/Assignment1_ZeyuGu_zg2556/` folder. The work runs pinned FinRobot, audits financial and narrative evidence, and evaluates one Option A verifier improvement.

- Five formal semiconductor companies with at least two peers; five required onboarding names; nine real baseline generations and seven verifier-treated outputs with HTML/assets and full analysis.
- Twenty SEC reconciliations, independent valuation notebooks, target-price sensitivity, three independent NVDA runs, and complete before/after claim inventories for three companies × eight sections.
- Separate applicable compatibility/verifier patches, immutable inputs/prompts/responses, $3.59449 direct API accounting, reproducibility commands, research PDF/editable source and analytical Medium article.

The verifier improves warning coverage (93.99% recall, 78.78% precision) but does not reliably correct claims. Residual contradictions and machine-adjudication limitations remain explicit. No human claim validation is asserted. Private credentials and WRDS observations are excluded; no other track is changed.

Validation: clean patch application; 101 tests plus 6 subtests; seven offline data pipelines and fourteen report replays with network disabled; nine relocated independent notebooks; exact HTML/text hash checks, loaded local assets, PDF page review and credential scan. See saved QA manifests for precise scope.

Medium: pending actual authorized publication; insert the genuine URL before opening this PR.
'''

def main():
    for relative,content in [('README.md',README),('blog/medium_article.md',BLOG),('docs/course_pr_body.md',PR)]:
        content=current_accounting(content)+docs_addendum(relative)
        path=ROOT/relative
        if path.exists():
            raw=path.read_bytes();archive=ROOT/'evidence/manuscript_versions'/(path.stem+'_'+sha256(raw)[:12]+path.suffix);archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():archive.write_bytes(raw)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    print('Wrote reviewable README, complete analytical article and scoped course PR description.')
if __name__=='__main__':main()
