# Requirements and evidence contract

Zeyu Gu · zg2556 · STAT GR5398 Fall 2026 Assignment 1 · FinRobot

This is the requirements register, not a completed research report. `acceptance.csv` lists each user requirement (1–26), the related course section, its proof obligation, and the current evidence status. A planned path is not proof that an artifact exists. `missing` means no qualifying result has been inspected. `partial` means a specific preparatory result exists, while the full requirement remains unmet. Only `verified` means the entire row's acceptance criterion has been inspected and met.

## Authoritative instructions read

- `Assignment1/README.md`: three-track context, separate track submissions, reproducibility, secret hygiene, Medium article, PR against `master`, and October 12, 2026 deadline.
- `Assignment1/FinRobot/README.md`: all sections, including the eight-agent architecture, two-stage baseline, five-company coverage, data/forecast/valuation/grounding/stability audits, one improvement, and research PDF.
- `Assignment1/FinRobot/source_code/README.md` and `run_pipeline.sh`: the course repository supplies a wrapper, not the implementation.
- All four files in `source_code/audit_templates/`: these contain headers only and are not experimental results.
- `Assignment1/FinRobot/submissions/README.md`: use the new `Assignment1_ZeyuGu_zg2556/` directory; a branch does not substitute for the directory.
- User's complete 26-item objective, read from its attachment on October 11, 2026.

No `AGENTS.md` was found in the course repository (including hidden files) or the checked ancestors `C:/`, `C:/Users/`, `C:/Users/zeyug/`, and `C:/Users/zeyug/PycharmProjects/`. No `AGENTS.md` was found in the external upstream checkout. Preserve all other tracks and the shared checkout's branch/state.

The upstream commit used for this source inspection is `2717499b8e30f242640af08c4ad9afd1113c2d45`, from `AI4Finance-Foundation/FinRobot`. All source references below and in `field_lineage.csv` refer to that immutable version, under `finrobot_equity/core/src/` unless otherwise stated. Source inspection is evidence about code behavior, not evidence of successful API execution.

## Course requirements resolved into experiments

| Experiment | Minimum qualifying evidence |
|---|---|
| Onboarding | NVDA, AMD, INTC, AAPL, GOOGL run successfully through the two stages; reuse is allowed only for identical version, input, and configuration. |
| Formal coverage | At least five verified same-industry companies, at least two explained peers per company, real data, complete analysis folders, baseline/final HTML with local assets. Semiconductor candidates are NVDA, AMD, INTC, AVGO, QCOM, pending classification and endpoint verification. |
| Data reconciliation | At least two companies × ten pipeline metrics; SEC accession, financial period, units, values, formulas, deltas, verdicts, and locatable filing/XBRL contexts. Derived or market metrics are not misrepresented as directly disclosed filing values. |
| Forecast audit | The default 5%/6%/4% annual revenue assumptions, +1 percentage point contribution/EBITDA margins, and −0.5 percentage point SG&A margin are printed, interpreted, and compared with evidence-based proposed adjustments. This audit is not a second research improvement. |
| Independent valuation | A separately implemented executable notebook or spreadsheet reproduces a chosen method with all explicit and implicit assumptions and reconciles each component. Importing and calling the same valuation engine is not independent reproduction. |
| Sensitivity | Native analyzer output plus revenue/margin scenarios propagated through the same valuation formula to target price. Distinguish relative revenue change from margin percentage points. |
| Grounding | Exhaustive atomic factual/numeric claims in all eight sections for at least three companies, located in exact final-render text and classified against the generation-time prompt. Later external corroboration never upgrades baseline support. |
| Stability | Three independent LLM generations for one ticker with identical frozen input hashes, model, and parameters, separate run IDs and provider responses; rating, target price, top-three risks, cost, and latency compared. FMP headline fields and LLM conclusions are reported separately. |
| One improvement | Option A ninth verifier stage; original and revised sections, evidence links, claim mapping, numerical checks, repair/removal reasons, and final renderer consumption. Same-input before/after evaluation covers all eight sections for the same three or more companies. |
| Reproducibility and delivery | Pinned upstream and dependency lock, separate compatibility and verifier patches, clean application test, recorded data/prompt/model versions, manifests, costs, errors, exact commands, English PDF/source, original analytical blog, actual Medium URL, and course master PR. |

The older AutoGen annual-report tutorial is optional, ungraded warm-up. Its successful execution would not satisfy any required baseline, audit, or improvement experiment here. Individual company PDFs are optional; the research-report PDF is mandatory. Adanos retail sentiment is optional and must not be presented as present when unavailable.

## Actual upstream differs from the course description

The course describes eight specialist `Agent` objects with Pydantic output schemas, all receiving `EquityResearchAgentManager._prepare_financial_data_prompt()`. Those implementations exist in `modules/equity_agents/`, and their complete instructions and output fields must be recorded if that route is run. However, this pinned commit's production CLI imports `generate_text_section` from `modules/text_generator_agents.py` (Step 1 line 17 and Step 2 line 33). That function directly calls `OpenAI.chat.completions.create`, uses its own shorter `SYSTEM_PROMPTS`, and receives no `response_format` or Pydantic schema. Its default is `gpt-4o-mini`, temperature 0.7, and `max_tokens=1000`; configuration may override the model and endpoint in Step 1.

That active direct-Chat route includes recent news only for `news_summary`, limits news to ten articles, and truncates each news text to 200 characters. The dormant manager route includes all supplied news in every agent's shared prompt. These are materially different evidence bases. An audit must identify the route that actually ran, store its actual instructions and evidence prompt, and avoid attributing unused long specialist instructions to the observed failures. Restoring the course-described typed route, if chosen, must be explicitly documented as a compatibility/architecture choice; it must not be hidden inside the verifier treatment. The manager's `ModelSettings(model=custom_model)` call also requires compatibility verification against the installed SDK.

## Verified source risks to resolve before formal experiments

1. Step 1 builds forecast keys `2025E`, `2026E`, `2027E` regardless of the latest actual financial year. Preserve original output behavior in the audit and use a documented date-alignment repair for valid forward experiments.
2. The course wrapper omits valuation analysis and omits the news-summary file from Step 2. It does enable text regeneration. Correct pipeline flags and section wiring must be recorded explicitly.
3. Step 2 regeneration changes `competitor_analysis` and `major_takeaways` whenever enabled with a key, even if the existing text is valid. The regeneration call does not forward the configured model or base URL. CSV-like content can trigger generic fallback prose. This is not evidence-grounded repair.
4. Even with regeneration disabled, Step 2 rewrites month/year text in the first three lines of company, investment, and valuation overviews to the report date. Real event dates can be corrupted. Audits must use the actual rendered strings; a compatibility patch should avoid silent rewriting.
5. `--skip-auto-fetch` skips comprehensive market metrics but does not itself prevent technical-indicator or other later API reads. Fully offline rerendering requires demonstrated network isolation, not an assumption based on this flag.
6. Page-one price, analyst target, and analyst grade are FMP fields; the engine's target is a separate object; narrative conclusions are LLM text. Upstream preserves these distinctions poorly in presentation. `field_lineage.csv` records their actual sources and assumptions.
7. The valuation engine selects the last A/E column in DataFrame order. The wrapper passes only analysis/current price/shares, allowing 12× ±3× multiple defaults, assumed net debt of 10% of EV, and FCF of 60% of EBITDA. The Step 2 shares fallback is one billion; a present but null shares key can defeat that fallback. None is a verified observation.
8. Step 2 passes peer data as `{'ev_ebitda': DataFrame}`, while the engine expects a mapping of ticker to dictionaries containing scalar `ev_ebitda`. Validate this path and record any compatibility repair.
9. The native revenue sensitivity scales forecast revenue levels by ±5%; it does not recompute annual growth paths. Margin sensitivity adds ±2 percentage points. The combined table reports EBITDA in billions, not per-share fair value. Its nominal confidence interval uses an assumed 15% standard deviation and is not an empirically estimated confidence interval.
10. FMP market fields include mislabeled or assumed values: `fwd_pe` uses historical ratio fields, `net_debt_to_equity` uses debt/equity, free float defaults to 95%, and missing sector defaults to Technology. The current source also manufactures optimistic fallback prose after LLM failure. Missingness and these assumptions must be visible and must not count as successful runs.

## Evaluation conventions

Every final-render section receives a content hash and version. Atomic claims are split by factual proposition, preserving original spans; a sentence with several propositions can produce several rows. Each row records `ticker`, `section`, `run_id`, source span, claim, label, precise evidence location, and rationale. Supported means the actual prompt directly supports or permits a shown deterministic derivation. Unsupported includes plausible prior knowledge, unfurnished citations, or unavailable evidence. Contradicted requires a locatable conflict. Zero-claim sections have grounding rate N/A and remain in coverage reporting.

Before/after metrics include supported/total, contradiction rate, numeric error rate with an explicit numeric-claim denominator, retention and factual coverage, incremental tokens/cost/runtime, and remaining failures. A verifier cannot improve the score by deleting all useful content. Evaluation labels must be independently generated or separately reviewed; machine assistance is disclosed. No human review is claimed without an actual human review record.

Data snapshots keep original payloads, redacted request metadata, timestamps, publication/filing dates, news windows, and input hashes. A fresh online run produces a new snapshot and run ID. Offline replay uses saved data and recorded outputs; it cannot be counted as an independent stochastic generation. Three fresh LLM responses over the same frozen financial snapshot satisfy stability only when model/settings and evidence hashes agree.

## Publication boundary and completion

Prepare the complete reviewable submission and diff before requesting final publication authorization if none exists. Publication and the course master PR are outstanding until real URLs are verified. Never publish credentials, `config.ini`, `.env`, upstream `.git`, or credential-bearing logs/notebook outputs. Dependency installation, schema tests, empty audit tables, and attractive report formatting alone do not satisfy the assignment.

This register is intentionally conservative: source findings and engineering work can be complete while all empirical acceptance rows remain missing. Missing credentials, endpoint entitlement, authorized spending, or publication permission must be stated with exact resume commands and minimum required input in the submission README. They do not justify fabricated results or a claim of full completion.

## Renderer compatibility implementation and control tests

The compatibility changes to `create_equity_report.py` and `modules/html_renderer.py` introduce `--audited-render` and optional `--market-snapshot <JSON>`. Audited rendering requires all eight nonempty supplied text files, forbids regeneration, skips config/credential access and all live market reads, keeps dates and punctuation intact, displays unavailable market fields as N/A, and avoids canned narrative. It emits `Audited_Equity_Report_{TICKER}.html`, a local `assets/` directory, and `rendered_section_texts.json` containing the exact eight strings, their SHA-256 hashes, and the HTML hash. The static audited template includes all eight sections, financial/peer tables, available charts, and supplied analysis JSON. This is a reproducibility repair applied equally to baseline and verifier treatment, not a second research improvement. The original renderer remains available without this flag.

`--skip-auto-fetch` also gates the previously uncontrolled technical-indicator fetch. Audited mode preserves supplied catalyst classifications rather than silently rerunning the renderer's positive-keyword heuristic. Frozen market snapshots must themselves expose assumptions: the renderer does not turn provider defaults into verified observations. For example, legacy 95% free float must be marked assumed or omitted by the data-capture layer.

On October 11, 2026, dedicated Python 3.11 environment control tests verified: CLI import/help; exact preservation of all eight strings including original June 2024 dates, Unicode, HTML-like characters, and comma-heavy text; rejection of a missing news section; matching text/HTML SHA-256 hashes; no config, socket, model, or market calls; N/A absent fields; and explicit unavailable-chart messages. Test fixtures were temporary, clearly labeled control fixtures, and deleted by temporary-directory cleanup. These tests establish renderer invariants only. They are not stock research, API runs, audit observations, or final visual QA, and do not satisfy the empirical minimums.
