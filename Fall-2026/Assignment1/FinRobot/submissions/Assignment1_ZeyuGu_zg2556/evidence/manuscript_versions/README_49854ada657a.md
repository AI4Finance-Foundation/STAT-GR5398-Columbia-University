# FinRobot: from generated prose to traceable research

**Zeyu Gu | zg2556 | STAT GR5398 Fall 2026, Assignment 1**

This submission contains real provider calls, frozen financial evidence, original model requests and responses, independent valuation calculations, and an Option A evidence verifier. It is a research-system audit, not a stock recommendation. The acceptance register is `docs/acceptance.csv`; a requirement is complete only when its recorded evidence meets the entire criterion.

## Current delivery status

Five onboarding tickers (NVDA, AMD, INTC, AAPL, GOOGL) have genuine eight-agent runs and HTML reports. NVDA additionally has three independent generations over identical frozen inputs. The formal semiconductor universe is NVDA, AMD, INTC, AVGO and QCOM. **Only the first three have complete FMP snapshots so far.** FMP returned HTTP 402 for AVGO/QCOM and other tested semiconductor candidates. WRDS access was offered and local connection setup is pending. Apple and Alphabet do not fill the missing semiconductor slots.

The data audit contains 20 substantive SEC comparisons, with all four EBITDA-related differences reconciled by explicit components. Independent EV/EBITDA and DCF calculations and 25-case target-price sensitivities are available. Explicit machine grounding adjudication currently covers all eight baseline sections for NVDA (486 atomic claims) and INTC (439); AMD and the complete paired final-text audits remain pending. Three-run stability measurements are saved in `audits/stability/NVDA_adjudicated/`. Failed API extraction and verifier attempts are preserved and charged, not counted as successful experiments. Publication is pending: **no Medium URL or course PR is claimed.** `evidence/progress/checkpoint.json` records the latest recoverable state; the research PDF is still an incomplete draft.

For WRDS setup in the user's own PowerShell terminal, run `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_wrds.ps1` from this submission directory. This uses the Python available in that terminal and creates a separate WRDS-only environment. Enter credentials only at the local hidden prompts. An earlier environment-launcher error occurred before any WRDS login, so it is not evidence of denied WRDS access.

## Version and environment

- Upstream: `https://github.com/AI4Finance-Foundation/FinRobot`, commit `2717499b8e30f242640af08c4ad9afd1113c2d45`.
- Python 3.11.15; installed dependencies: `requirements-lock.txt`. Upstream implementation and the environment live outside this course folder.
- Writer: OpenAI `gpt-4.1-mini`, resolved in actual responses to `gpt-4.1-mini-2025-04-14`; temperature 0.7, maximum output 4,000 tokens, SDK retries disabled. All eight original typed agents and their original instructions are retained.
- Formal verifier and independent evaluator: `gpt-4.1`, temperature 0, maximum output 26,000 tokens. This model choice follows saved procedural failures of the smaller model in the NVDA pilot. It is disclosed and is not a change to the original writer/stability experiments.
- One task-wide approved ceiling: **USD 20**. Both budget policies use the same `evidence/costs.jsonl`; they are not separate allowances. `audits/resources/cost_summary.json` is computed from settled usage. Costs conservatively charge cached input at full list price and are not a provider invoice.
- Financial/news cutoff parameter: `2026-10-11T21:10:00+00:00`; individual HTTP fetch timestamps are separately recorded. News window: 90 days. FMP news access was denied, so one real archived issuer earnings release per company is used and labeled accordingly. This is not comprehensive independent media coverage.

The pinned CLI actually uses a legacy plain chat wrapper; the course describes the typed eight-agent manager. `scripts/experiment.py` deliberately runs the actual typed implementations. This architecture choice, endpoint migrations, rolling forecast labels and exact-text offline rendering are compatibility changes shared by baseline and treatment. The sole research improvement is the ninth verifier stage. See `docs/course_requirements.md`, `docs/field_lineage.csv`, and `docs/verifier_version_history.md`.

## Clean installation

Use PowerShell with Git and `uv` on PATH. Begin in this submission directory. Choose a new runtime directory: the bootstrap refuses to reset an existing checkout.

```powershell
.\scripts\bootstrap.ps1 -RuntimeRoot "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction" -PythonVersion 3.11
$RunPython = "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction\venv\Scripts\python.exe"
$RunUpstream = "$env:LOCALAPPDATA\FinRobot-zg2556-reproduction\upstream"
```

Bootstrap checks out the pinned commit in an isolated clone, applies `patches/compatibility.patch` then `patches/option_a_verifier.patch`, installs the lock, and runs offline regression tests. It makes no paid model request. The course repository branch is never switched or reset. Patch application evidence is in `patches/manifest.json` and `patches/clean_apply_tests.log`.

Keep credentials outside the submission. The scripts accept environment variables `FMP_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, or a private INI with section `[API_KEYS]` and the corresponding lower-case keys. Do not commit that file. The following asks only for its local path:

```powershell
$PrivateConfig = Read-Host 'Absolute path to your private FinRobot config.ini'
& $RunPython scripts\preflight.py --help
```

An API key alone does not imply symbol/news entitlement. Saved preflight responses in `evidence/stable_endpoint_preflight/` show actual access decisions. No subscription was purchased or upgraded. Optional WRDS setup is `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_wrds.ps1`; both username and password are entered only in the local terminal. Its private PostgreSQL login file is outside this submission.

## Offline reproduction: no new data and no new LLM result

Offline reproduction reuses frozen responses. It cannot be counted as another independent generation. The canonical snapshots used for narration are `NVDA`, `AMD`, `INTC_news_v2`, `AAPL_news_v2`, and `GOOGL`. The original INTC/AAPL snapshots remain intact: the `news_v2` variants correct a company-name filtering defect by replaying the same raw HTTP evidence.

```powershell
& $RunPython scripts\fetch_sec_evidence.py --offline
& $RunPython scripts\check_notebook_portability.py
& $RunPython scripts\data_audit.py --help
& $RunPython scripts\update_coverage_status.py
& $RunPython scripts\audit_peer_forecasts.py
& $RunPython scripts\experiment.py --upstream $RunUpstream render --run runs\NVDA_baseline_1
& $RunPython scripts\qa_html.py --report-dir runs\NVDA_baseline_1\output\NVDA\report --output-dir evidence\html_qa\NVDA_reproduction
```

Each `snapshots/*/valuation/independent_valuation.ipynb` runs with relative submission paths. The underlying standard-library calculation does not import the valuation engine. For an explicit independent calculation and separate engine comparison:

```powershell
& $RunPython scripts\independent_valuation.py --analysis-dir snapshots\NVDA\analysis --valuation-inputs snapshots\NVDA\valuation_inputs.json --upstream-src "$RunUpstream\finrobot_equity\core\src" --output-dir reproduction\NVDA_valuation
& $RunPython scripts\target_price_sensitivity.py --help
```

The sensitivity output distinguishes a relative revenue-level shock from an absolute margin change in percentage points. Native EBITDA sensitivity is saved separately from actual per-share target-price propagation. Reconciliation reproduces assumptions; matching the engine does not establish that its assumptions are economically sound.

## Fresh model experiments over frozen inputs

These commands incur API usage. Keep the approved policies and shared ledger; do not delete prior reservations to make budget room. Use a new run ID for every independent generation. `--resume` continues a partially completed run and revalidates completed calls; it does not create a new independent observation.

```powershell
$env:OPENAI_MODEL = 'gpt-4.1-mini'
& $RunPython scripts\experiment.py --upstream $RunUpstream narrate --analysis snapshots\NVDA\analysis --ticker NVDA --company-name 'NVIDIA Corporation' --run reproduction\NVDA_fresh_1 --config $PrivateConfig --budget config\budget.approved.json --ledger evidence\costs.jsonl --max-tokens 4000
& $RunPython scripts\experiment.py --upstream $RunUpstream narrate --analysis snapshots\NVDA\analysis --ticker NVDA --company-name 'NVIDIA Corporation' --run reproduction\NVDA_fresh_2 --config $PrivateConfig --budget config\budget.approved.json --ledger evidence\costs.jsonl --max-tokens 4000
& $RunPython scripts\experiment.py --upstream $RunUpstream narrate --analysis snapshots\NVDA\analysis --ticker NVDA --company-name 'NVIDIA Corporation' --run reproduction\NVDA_fresh_3 --config $PrivateConfig --budget config\budget.approved.json --ledger evidence\costs.jsonl --max-tokens 4000
```

For the other original runs, substitute the following exact inputs. Their saved run IDs are `{TICKER}_baseline_1`:

| Ticker | Snapshot | Company name |
|---|---|---|
| AMD | `snapshots/AMD/analysis` | Advanced Micro Devices, Inc. |
| INTC | `snapshots/INTC_news_v2/analysis` | Intel Corporation |
| AAPL | `snapshots/AAPL_news_v2/analysis` | Apple Inc. |
| GOOGL | `snapshots/GOOGL/analysis` | Alphabet Inc. |

Verifier and independent evaluator use only the generation-time writer evidence. SEC audit files and later web discoveries are not injected into that evidence:

```powershell
$env:OPENAI_MODEL = 'gpt-4.1'
& $RunPython scripts\experiment.py --upstream $RunUpstream verify --run runs\NVDA_baseline_1 --final-run reproduction\NVDA_final --config $PrivateConfig --budget config\budget.evaluation.json --ledger evidence\costs.jsonl --max-tokens 26000
& $RunPython scripts\grounding_audit.py run --run runs\NVDA_baseline_1 --output reproduction\NVDA_before --config $PrivateConfig --budget config\budget.evaluation.json --ledger evidence\costs.jsonl --max-tokens 26000
& $RunPython scripts\grounding_audit.py run --run reproduction\NVDA_final --evidence-run runs\NVDA_baseline_1 --output reproduction\NVDA_after --config $PrivateConfig --budget config\budget.evaluation.json --ledger evidence\costs.jsonl --max-tokens 26000
& $RunPython scripts\experiment.py --upstream $RunUpstream render --run reproduction\NVDA_final
```

Repeat the same before/after commands for AMD and INTC. The independent auditor does not receive verifier verdicts. Every deterministic source unit must appear; each factual premise is requested as an atomic claim. This is machine-assisted semantic labeling, not human verification. Explicit UNVERIFIED annotations retain the underlying Unsupported classification in the independent audit. A heading with no factual claim receives N/A, not a perfect score.

## Fresh online acquisition

An online acquisition creates a new snapshot directory and data timestamp. It is a different experiment and must not overwrite or be silently compared with the frozen baseline. `scripts/snapshot_pipeline.py --help` lists the endpoint-recording and strict no-network replay modes. Raw original body bytes, safe source URLs, retrieval times, status codes and content hashes are retained. FMP secrets are excluded from request URLs and headers are not persisted. Official issuer source bytes and excerpt offsets are under `evidence/issuer_news/`.

Formal coverage may resume when AVGO and QCOM have sufficient authorized data. WRDS-derived fields require an explicit source adapter and fiscal/unit mapping; a WRDS login by itself is not a completed five-company result. Missing market values must remain missing rather than adopting undocumented fallback constants.

## Artifact navigation and release gate

- `coverage_universe.csv`, `onboarding_universe.csv`: formal candidates, actual peer mappings, eligibility and observed price windows.
- `snapshots/`: immutable data, full analysis, native sensitivity/catalyst/enhanced-news outputs and explicit valuation runs.
- `runs/`: original/final eight-section text, exact prompts/instructions/schema/wire responses, usage and report assets.
- `audits/data/`, `audits/target_price_sensitivity/`, `audits/peer_projection_audit.csv`: substantive quantitative audits.
- `audits/grounding/`, `audits/stability/`: full machine-assisted claim and repetition evidence, including incomplete pilot attempts explicitly marked as such.
- `evidence/sec/`: original SEC filing/companyfacts responses, contexts, accessions and exact provenance.
- `docs/`: methods, source traps, forecast proposals and the acceptance register.
- `patches/`: clean compatibility and verifier patches; no nested upstream Git repository.
- `output/pdf/` and `blog/`: research document and analytical Medium draft when generated; publication is separate.

Before release, recompute resource accounting and scan actual local secrets without printing them:

```powershell
& $RunPython scripts\summarize_runs.py
& $RunPython scripts\secret_scan.py --config $PrivateConfig
```

The main report must state which outputs can support a portfolio manager and which still need analyst review. The Medium article analyzes the experiment rather than copying generated stock research. Actual Medium publication and a course PR against `master` require final release authorization after the complete material and scoped diff are reviewable. An unpublished draft is not a publication link, and three semiconductor companies are not five.
