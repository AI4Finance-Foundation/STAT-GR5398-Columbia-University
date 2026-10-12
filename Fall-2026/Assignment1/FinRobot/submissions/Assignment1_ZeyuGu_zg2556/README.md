# FinRobot: execution, evidence and verifier audit

**Zeyu Gu | zg2556 | STAT GR5398 Fall 2026 Assignment 1**

Five formal semiconductor companies (NVDA, AMD, INTC, AVGO, QCOM), each with two real frozen peers, have baseline and verifier-treated HTML reports. The five requested onboarding names also ran; AAPL/GOOGL remain onboarding-only. NVIDIA has three independent generations with identical inputs. There are nine real baseline runs and seven original V6 verifier outputs; later treatments are versioned separately.

The deliverable contains 20 SEC financial reconciliations; independent valuation notebooks; native and target-price sensitivities; complete machine-assisted claim inventories across all eight sections of NVDA/AMD/INTC before and after; stability evidence; applicable upstream patches; and an English [selected research PDF](output/pdf/research_report_submission.pdf) and [complete Medium article](blog/medium_article.md).

**Publication status:** The author published the [Medium article](https://medium.com/@zeyu.gu0302/i-ran-finrobot-then-audited-what-its-reports-actually-knew-fda7a2d6f1e4); signed-out access and the complete body were verified. The course PR against `master` is still pending the author's Mac submission. This is not a declaration of formal submission. In the original V6 evaluation, the verifier is an experimental review aid: the independent after audit found 48 contradicted assertions, including eight without an unverified warning. HTML notices and the report disclose these failures. No human claim validation is asserted, and no zero-factual-error certification is offered.

## Latest evaluated Option A follow-up

Read [the semantic follow-up](docs/verifier_semantic_v9_followup.md) alongside the research report. The older V6 links below preserve the five-company original experiment. Latest separately evaluated reports are [NVDA](runs/NVDA_final_reasoning_v9_1/output/NVDA/report/Audited_Equity_Report_NVDA.html), [AMD](runs/AMD_final_reasoning_v9_1/output/AMD/report/Audited_Equity_Report_AMD.html), and [INTC](runs/INTC_final_reasoning_v9_1/output/INTC/report/Audited_Equity_Report_INTC.html). Keep their sibling assets and analysis directories.

| Version, under corrected evaluation | Supported / claims | Grounding | Contradicted | Numeric errors / verifiable |
| --- | --- | --- | --- | --- |
| Matched baseline | 286 / 1479 | 19.34% | 31 | 25 / 172 |
| Reasoning V8.2, unchanged prose | 530 / 1958 | 27.07% | 8 | 8 / 430 |
| V9: explicit CAGR horizons | 566 / 1982 | 28.56% | 1 | 1 / 459 |
| V9.1: explicit actual multiple range | 570 / 1985 | 28.72% | 0 | 0 / 461 |

The corrected V8.2 row changes evaluator judgments only; V9/V9.1 change generated text using original evidence. There are still 1415 Unsupported claims and 51 unverifiable numeric claims. Final warning recall is 100.00%, precision 89.90%; 0 problematic claims remain unflagged. This is complete machine-assisted scope coverage, not human validation or investment-ready prose. Claims added by explicit endpoint explanations change denominators. Full-assignment status remains incomplete until the course PR exists; strict error-free narrative certification is unavailable.

The latest paid reviewer is `gpt-5.4-2026-03-05`, low reasoning effort, omitted temperature and a 20,000 completion-token ceiling. V9/V9.1 reuse its captured reviews with no new model calls; the historical V6 reviewer below uses GPT-4.1. Clean-patch verification and the expanded regression log (235 tests and 6 subtests) are saved separately.

## Read the original experiment

| Formal company | Baseline HTML | Final V6 HTML | Frozen snapshot / peers |
|---|---|---|---|
| NVDA | [Baseline](runs/NVDA_baseline_1/output/NVDA/report/Audited_Equity_Report_NVDA.html) | [Final](runs/NVDA_final_v6/output/NVDA/report/Audited_Equity_Report_NVDA.html) | `NVDA`; AMD, INTC |
| AMD | [Baseline](runs/AMD_baseline_1/output/AMD/report/Audited_Equity_Report_AMD.html) | [Final](runs/AMD_final_v6/output/AMD/report/Audited_Equity_Report_AMD.html) | `AMD`; NVDA, INTC |
| INTC | [Baseline](runs/INTC_baseline_1/output/INTC/report/Audited_Equity_Report_INTC.html) | [Final](runs/INTC_final_v6/output/INTC/report/Audited_Equity_Report_INTC.html) | `INTC_news_v2`; NVDA, AMD |
| AVGO | [Baseline](runs/AVGO_baseline_1/output/AVGO/report/Audited_Equity_Report_AVGO.html) | [Final](runs/AVGO_final_v6/output/AVGO/report/Audited_Equity_Report_AVGO.html) | `AVGO_public_v1`; NVDA, AMD |
| QCOM | [Baseline](runs/QCOM_baseline_1/output/QCOM/report/Audited_Equity_Report_QCOM.html) | [Final](runs/QCOM_final_v6/output/QCOM/report/Audited_Equity_Report_QCOM.html) | `QCOM_public_v1`; AMD, INTC |

Open each HTML with its sibling `assets/` directory intact. `runs/AAPL_baseline_1`, `runs/GOOGL_baseline_1` and their `final_v6` directories contain the onboarding-only reports. Full `output/{TICKER}/analysis` directories are included in every run. NVDA `_baseline_2` and `_baseline_3` are the two additional real stability draws, not copies.

Historical V6 results: baseline support 291/1,478 (19.69%); final support 358/1,590 (22.52%); contradiction 40 -> 48. Warning recall on final unsupported/contradicted assertions is 93.99%, precision 78.78%, supported lexical-fact retention 96.13%. Repeated claims change denominators, so raw support growth is not proof of successful factual correction. All labels are machine-assisted under a disclosed common protocol.

## Version, source boundary and cost

- Upstream: https://github.com/AI4Finance-Foundation/FinRobot, commit **2717499b8e30f242640af08c4ad9afd1113c2d45**. `patches/compatibility.patch` applies first, then `patches/option_a_verifier.patch`. `patches/manifest.json` gives hashes, source paths and clean-apply evidence. No upstream `.git` is submitted.
- Dedicated runtime: Python 3.11.15 on Windows; `requirements-lock.txt`, `evidence/preflight_configured/preflight.json`. Clean patch tests: 108 tests + 6 subtests; genuine offline replay covers seven data pipelines and fourteen saved reports with sockets disabled.
- Writer: `gpt-4.1-mini` resolved to `gpt-4.1-mini-2025-04-14`, temperature 0.7, maximum 4,000 output tokens per agent. Original V6 verifier: `gpt-4.1` resolved to `gpt-4.1-2025-04-14`, temperature 0, maximum 26,000, at most two validation repairs. Exact instructions, requests, schemas, wire/structured responses and parameters are saved for every call.
- Formal evaluation: explicit Codex-assisted adjudication, exact model ID/usage unavailable; no human validation. Earlier direct-API judge proposals and failures are development artifacts, not the canonical labels.
- Recorded direct API cost: **$10.045071**, 347 settled calls, 1,303,571 input / 822,575 output tokens, zero unsettled reservations, within the single **$20** authorized cap. This includes abandoned versions, repairs, preflight and pilot evaluators. It conservatively uses saved list prices, not an invoice. Codex task-agent authoring cost is unknown, not zero. `audits/resources/` separates new calls from allocated reused verifier cost.
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

Bootstrap pins upstream, applies both patches, installs the lock and runs offline tests without paid calls or switching the shared course branch. HTML needs Matplotlib only. The selected research PDF is the user-provided 16-page file copied unchanged. The user requested that research-report LaTeX/Markdown sources and earlier report editions stay local and be excluded from the course PR. Experiment and metric reproduction remain available from the submitted code, notebooks and frozen evidence. `docs/pdf_build.md` explains selected-PDF validation; it does not promise rebuilding the external report from submitted manuscript sources.

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

Original V6 label locations and exact pairs are `config/grounding_v6_pairs.json`. Recompute every score without a model call:

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

## Executed offline forecast-parameter scenarios

The separate audit in `audits/forecast_parameter_scenarios/run_v2/` executes NVDA growth 30%/20%/10% and AMD 15%/10%/8%, with zero annual margin changes. It retains the original rounded historical ratios, valuation inputs and EBITDA-definition limitations. Each company has an actual default replay and adjusted processor output; all 220 default fields match the saved baseline. This is a conditional parameter exercise, not a second research improvement. It adds $0 API cost and leaves all original verifier inputs, outputs and scorecards unchanged.

| Company / EBITDA year | Default model target, USD | Adjusted model target, USD | Change |
|---|---:|---:|---:|
| NVDA 2029E | 333.71 | 464.43 | +39.17% |
| AMD 2028E | 377.84 | 398.17 | +5.38% |


Exact offline commands, after the clean-environment setup above (choose new output directories):

```powershell
$ScenarioId = [guid]::NewGuid().ToString('N').Substring(0,8)
& $RunPython scripts/forecast_parameter_scenarios.py --upstream $RunUpstream --output "reproduction/forecast-parameters-$ScenarioId"
& $RunPython scripts/independent_forecast_scenario_review.py --submission-root . --output-directory "reproduction/forecast-reference-$ScenarioId"
& $RunPython scripts/compare_forecast_scenario_review.py --submission-root . --scenario-output "reproduction/forecast-parameters-$ScenarioId" --reference-directory "reproduction/forecast-reference-$ScenarioId" --output-directory "reproduction/forecast-check-$ScenarioId"
```

Read `forecast_comparison.csv`, `target_price_comparison.csv`, `definition_caveats.csv`, the per-case configuration/processor CSV/engine JSON, and `manifest.json` under the recorded run. A separate reference is saved in `evidence/forecast_parameter_scenarios_review/independent_reference/`. Integrity review binds the original snapshots, prompts, responses, paired audits and API ledger. Neither replay is an independent LLM generation.

Medium and course PR preparation is in [the local handoff](docs/publication_preparation.md), with title/tags in `blog/medium_metadata.json` and the full description in `docs/course_pr_body.md`. The Medium article is published at the link above; the course PR remains pending. Public readback evidence is in `evidence/medium_publication/verification.json`.

## Artifact map and release

`docs/acceptance.csv` maps every user/course criterion to code and evidence. `ARTIFACTS.csv` lists the submission files and hashes. `evidence/progress/checkpoint.json` gives recoverable state. `docs/field_lineage.csv`, forecast recommendations and SEC contexts explain accounting assumptions; source/development/version files preserve failures. `audits/grounding/comparison_v6` and `annotation_v6` are preserved original V6 results; the latest evaluated follow-up is linked above. PDF and HTML QA do not certify investment claims.

Before exporting the course submission:

```powershell
& $RunPython scripts\secret_scan.py --config $PrivateConfig
& $RunPython scripts\validate_delivery.py
```

`config.ini`, `.env`, credentials, WRDS observations, local runtime checkouts, caches and duplicate reproduction trees are excluded. Publication-ready PR text is in `docs/course_pr_body.md`. Medium is now published by the author. The author will create the course-master PR from their Mac using a fresh scoped export; no GitHub push or PR has been performed by the assistant.

The submission's `.gitattributes` uses `* -text` because raw evidence and prompt hashes depend on exact bytes. Git's platform-dependent newline conversion must not rewrite them. `prepare_course_diff.py` verifies every staged blob against unfiltered saved bytes, applies the actual patch to a clean temporary course index, and checks every applied blob again. The failed conversion check and subsequent byte-preserving check are saved under `evidence/reproducibility/course_git_bytes_*.json`. Shared Git state and other tracks remain untouched.

The exporter also disables external diff/text-conversion drivers (`--no-ext-diff --no-textconv`), and `*.pdf -diff` keeps PDF changes binary. A Windows system PDF driver otherwise turned PDF contents into extracted text inside an apparently applicable patch; the actual-application byte check rejected that export. Its failure is preserved in `evidence/reproducibility/course_export_textconv_failure.json`. Only a later export whose manifest confirms actual clean-index application and exact bytes is the release candidate.

## Completed development follow-up

The original five-company report table retains the historical V6 release; the latest three-company reports are linked separately above. [Full V7.1 study](docs/verifier_v7_followup.md) and `runs/{TICKER}_final_v7_1/` are separate measured candidates. Warning recall improved to 97.70% and unflagged U/C fell to 31, but eight unflagged contradictions remain and supported lexical retention is lower. The research PDF and article report both versions. A GPT-5.4 one-section pilot exhausted its token limit with no visible output; its $0.400930 cost and failed response are included. The subsequent reasoning and semantic treatments are linked above; no paid generation or publication is currently running.

Additional exact offline recomputation commands from this submission directory (use new output folders):

```powershell
& $RunPython scripts/grounding_audit.py compare --pairs config/grounding_v7_1_pairs.json --output reproduction/v7_comparison_new
& $RunPython scripts/analyze_verifier_annotations.py --pairs config/grounding_v7_1_pairs.json --output reproduction/v7_annotations_new
& $RunPython scripts/summarize_runs.py
```

Development model/prompt/plan versions, clean patch evidence and the complete eight-section evaluations are linked in the study. V7.1 source SHA is `9e4b0da4dc5d259c4ffea48f1c1b4371df5862cdb24f9110643163f0b521b40a`. Main V6 reproduction commands and patches are unchanged. These candidates are experimental artifacts, not an additional research option. The course PR and strict zero-factual-error certification remain outstanding. Historical studies retain their original publication-state notes; current publication status is recorded in `config/publication_status.json`.

## Reproduce the latest follow-up without new API calls

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

Current academic readiness and remaining requirements are in [submission readiness](docs/submission_readiness.md) and `docs/acceptance.csv`. `evidence/delivery_validation.json` records integrity checks, not a guarantee of semantic truth. The real Medium URL is recorded above; the course-master PR remains pending.

## Final course submission check

See [the four-step submission check](docs/submission_readiness.md#course-submission-requirements-check). The named folder and local code/notebooks/report/README are ready for submission preparation. The Medium URL requirement is satisfied; the master PR remains outstanding. The selected PDF is [the uploaded 16-page edition](output/pdf/research_report_submission.pdf). Research-report sources and earlier editions are retained locally only, as requested; they are excluded from the PR along with report-document history. No experiment was rerun or modified to select this PDF.
