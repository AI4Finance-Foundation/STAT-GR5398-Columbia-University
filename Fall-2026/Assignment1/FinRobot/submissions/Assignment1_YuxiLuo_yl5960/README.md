# Assignment 1 — FinRobot Equity Research: Run, Audit, Improve

**Yuxi Luo (yl5960)** · STAT GR5398, Fall 2026 · Medium: [Auditing AI Equity Research with FinRobot](https://medium.com/@yl5960/auditing-ai-equity-research-with-finrobot-f2ed4a9320e2)

I ran `finrobot_equity` on ten companies, audited its numbers, valuation and prose, and improved the part that failed
most often: the text sections presented reported results as forecasts, attached the historical CAGR to forecast
years and cited guidance or consensus that the data does not contain. Labelling the evidence plus a verifier cut
contradicted claims from **16.3% to 1.3%** of all claims (held-out healthcare names: 16.6% to 1.3%) while the number of
supported claims rose from 413 to 686.

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Reports for 10 companies | `reports/*_Equity_Research_Report.html` (self-contained, charts embedded) |
| 2 | Artifacts behind them | `artifacts/<TICKER>/analysis/` (CSV/JSON/TXT) + `step1.log`, `step2.log` per run; also the three stability runs and the Part 3 verifier run |
| 3 | Audit tables | `audit/`: `data_audit.csv` (2.1), `forecast_assumptions.csv`, `valuation_assumptions.csv`, `valuation_bridge.csv`, `valuation_sensitivity.csv` (2.2), `grounding_claims.csv`, `grounding_rates.csv`, `grounding_review.csv`, `stability_grounding.csv` (2.3), each with its notebook |
| 4 | Improvement | `improvement/0001-*.patch`, `improvement/0002-*.patch`; evaluation in `audit/part3_evaluation.ipynb` (+ `part3_sections.csv`, `part3_claims.csv`, `part3_texts.jsonl`) |
| 5 | Research report | `research_report.pdf` |
| 6 | This README | commands, models, config, cost |

Notebooks, in order: `2_1_data_audit`, `2_2_forecast_valuation_audit`, `2_3_grounding_audit`, `2_3b_stability`,
`part3_evaluation`. All are saved executed, with outputs.

## Coverage universe

| Set | Companies (peers) | Why |
|---|---|---|
| Starter | NVDA (AMD, INTC), AMD (NVDA, INTC), INTC (NVDA, AMD), AAPL (MSFT, GOOGL), GOOGL (MSFT, META) | Required starting set |
| Own sector: healthcare | PFE (JNJ, ABBV), ABBV (PFE, JNJ), JNJ (PFE, ABBV), UNH (JNJ, PFE), MRNA (PFE, JNJ) | Different trends: ABBV and JNJ up 80–106% over three years, PFE flat, UNH down 23%, MRNA a turnaround (+717% in a year). The sector also stresses the pipeline: R&D at 15–161% of revenue, a loss-making company (MRNA), and one without an SG&A line (UNH). |

The FMP free plan does not cover LLY, MRK, BMY, AMGN, GILD or any managed-care company, so the healthcare set uses the
five large names it does cover; UNH's peers are therefore JNJ and PFE.

## Setup

| | |
|---|---|
| Code | `AI4Finance-Foundation/FinRobot` at `2717499b` + patch 0001 (required to run at all) + patch 0002 (Part 3) |
| Python | 3.12 (`numpy<2` in the requirements has no wheels for 3.13) |
| Data | Financial Modeling Prep, free plan (250 calls/day; no news endpoint; symbol whitelist) |
| Writer model | `gpt-4.1` (`openai_model` in `config.ini`), temperature 0.7 as hard-coded in `text_generator_agents.py` |
| Step 2 regeneration | `gpt-4o-mini` — the code's default, because `--enable-text-regeneration` does not pass the configured model |
| Audit judge | `gpt-4.1`, temperature 0, structured output (`audit/grounding.py`) |
| Verifier (Part 3) | `gpt-4.1-mini`, temperature 0 (`--enable-verifier`, `--verifier-model`) |

`config.ini` holds the FMP and OpenAI keys and is not included. `scripts/package_submission.py` scans every file in
this folder for both keys before packaging (result: none found).

### Why patch 0001 is needed

FMP serves its legacy `/api/v3` and `/api/v4` endpoints only to accounts that subscribed before 2025-08-31; any new key
gets `403 Legacy Endpoint` on all 13 endpoints the pipeline uses, so Step 1 fetches no data. Patch 0001 moves them to
`/stable/`, maps the renamed fields back so downstream code is unchanged, and stops the API key from being printed in
error messages.

## Reproduce

```bash
git clone https://github.com/AI4Finance-Foundation/FinRobot.git && cd FinRobot
git checkout 2717499b && git am --keep-cr ../improvement/0001-*.patch   # add 0002 for the Part 3 version
# --keep-cr is needed because upstream news_integrator.py has CRLF line endings
python -m venv .venv && .venv/Scripts/pip install -r finrobot_equity/requirements.txt   # (README says requirements-equity.txt; that file moved)
cp finrobot_equity/core/config/config.ini.example finrobot_equity/core/config/config.ini   # add FMP + OpenAI keys
```

`scripts/run_pipeline.py` runs Step 1 and Step 2 for one company with every analysis flag and the data files Step 2
can use (the assignment's Step 2 command omits `--peer-ebitda-csv`, `--sensitivity-analysis-file`,
`--enable-valuation-analysis` and `--enable-enhanced-charts`, without which the report has no valuation section). It
expects the layout `<root>/FinRobot`, `<root>/scripts`, `<root>/.venv` and writes to `FinRobot/output/<TICKER>[/<tag>]`.

```bash
python scripts/run_pipeline.py NVDA  "NVIDIA Corporation"              --peers AMD INTC
python scripts/run_pipeline.py AMD   "Advanced Micro Devices, Inc."    --peers NVDA INTC
python scripts/run_pipeline.py INTC  "Intel Corporation"               --peers NVDA AMD
python scripts/run_pipeline.py AAPL  "Apple Inc."                      --peers MSFT GOOGL
python scripts/run_pipeline.py GOOGL "Alphabet Inc."                   --peers MSFT META
python scripts/run_pipeline.py PFE   "Pfizer Inc."                     --peers JNJ ABBV
python scripts/run_pipeline.py ABBV  "AbbVie Inc."                     --peers PFE JNJ
python scripts/run_pipeline.py JNJ   "Johnson & Johnson"               --peers PFE ABBV
python scripts/run_pipeline.py UNH   "UnitedHealth Group Incorporated" --peers JNJ PFE
python scripts/run_pipeline.py MRNA  "Moderna, Inc."                   --peers PFE JNJ

# stability (2.3): same input three times
for i in 1 2 3; do python scripts/run_pipeline.py NVDA "NVIDIA Corporation" --peers AMD INTC --tag stab$i; done

# Part 3, end to end (with patch 0002 applied)
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker NVDA --company-name "NVIDIA Corporation" \
  --config-file finrobot_equity/core/config/config.ini --peer-tickers AMD INTC \
  --output-dir output/NVDA/part3/analysis --generate-text-sections --enable-sensitivity-analysis --enable-verifier

# Part 3 evaluation (2 x 2, 10 companies x 7 sections, cached) and the notebooks
python audit/part3_experiment.py --workers 8
python audit/notebook_builders/build_part3.py --findings
python audit/notebook_builders/build_stability.py --findings
```

The audit notebooks read the run folders (`FinRobot/output/<TICKER>`), so re-executing them needs that layout; the
copies under `artifacts/` are the same files. Every LLM call made by the audit and by the Part 3 experiment is cached
in `audit/data/` (`grounding_cache.jsonl`, `part3_cache.jsonl`), as are the SEC XBRL facts, FY-end prices, shares
outstanding and analyst consensus, so re-running the notebooks costs nothing and reproduces the saved numbers. The
unit tests for patch 0002 are in `finrobot_equity/core/tests/test_verifier.py` (5 tests, no network).

## Cost and runtime

| Item | Calls | Cost (USD) |
|---|---|---|
| Pipeline runs (14 Step 1 + Step 2) | ~100 gpt-4.1, ~30 gpt-4o-mini | ~0.90 (estimated) |
| Grounding judge (2.3, stability, Part 3) | 330 gpt-4.1 | 7.03 (1.07M input / 0.61M output tokens, recorded) |
| Part 3 writer | 140 gpt-4.1 | ~1.29 (estimated) |
| Part 3 verifier | 140 gpt-4.1-mini | 0.39 (recorded) |
| **Total** | | **~9.6** (FMP: free plan) |

Runtime per report: 42–63 s (Step 1 26–43 s, Step 2 15–29 s). Scoring 438 claims: about 12 minutes sequentially.

## Results at a glance

- **Data (2.1).** 27 line items for NVDA and AAPL checked against 10-K XBRL: 18 matches within rounding, 6 explained by FMP's
  EBITDA definition (pre-tax income + interest + D&A, i.e. NVIDIA's $11.1bn of non-operating income counted in), 1
  split restatement, 2 mislabelled periods (forecast columns hard-coded `2025E–2027E`).
- **Forecast and valuation (2.2).** The default growth (5/6/4%) puts next-year revenue 9–45% below consensus. Forecast
  EBITDA omits R&D. The engine's target reproduces to the cent and reduces to `10.84 x FY+3 EBITDA / shares` for every
  company. Ten of its sixteen assumptions are not shown, and its peer method is silently dropped by an input-shape bug.
  Implied moves range from −98% (MRNA) to +161% (PFE).
- **Grounding (2.3).** 345 Step 1 claims: 57% supported, 26% unsupported, 18% contradicted (≈44% supported after
  correcting for the judge's measured leniency). With the two sections Step 2 rewrites, 438 claims and 79
  contradictions, two thirds of them period/status errors. Judge + rules agree with
  the hand review on 59/65 claims; all 30 rule overrides were right.
- **Stability (2.3).** The page-1 numbers are identical across three runs because no LLM produces them; the top-3
  risks are not stable (overlap 0.5–1.0); section-level grounding varies by up to 100 points.
- **Improvement (3).** See the first paragraph; most of the gain comes from labelling the evidence, and the verifier
  helps only when it is given the report date.
