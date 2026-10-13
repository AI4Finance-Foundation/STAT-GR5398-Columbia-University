# Assignment 1: FinRobot audit and data-driven forecasts

**Xinyu Zhang · UNI xz3490 · STAT GR5398 · Fall 2026**

**Medium blog:** [Checking FinRobot’s Equity Reports](https://medium.com/@xz3490/checking-finrobots-equity-reports-abea81e4ab65) (published 12 October 2026).

## Results

The coverage universe is Technology: NVDA, AMD, INTC, AAPL and CSCO, with two peers per company. The filing audit contains 20 checks across NVIDIA and Apple FY2025: 16 matches within rounding and four explained EBITDA-definition differences. The grounding audit labels 288 statements across all eight saved sections for NVIDIA, AMD and Intel: 88 Supported, 150 Unsupported and 50 Contradicted.

The improvement is **Option B: data-driven forecasts**. Ten retrospective one-year folds reduce revenue MAPE from 14.03% to 10.21% and EBITDA WAPE from 29.49% to 20.14%. Revenue median APE worsens from 6.64% to 7.57%; Intel revenue worsens in both folds. The small current-vintage backtest is not point-in-time or a trading-performance test. No improved LLM prose was generated.

The three same-input NVIDIA runs have mean top-three-risk Jaccard similarity 0.667 and same-rank agreement 0.778. Headline ratings are frozen FMP inputs. All displayed zero target prices reflect missing data after HTTP 402; valid target-price variance is undefined.

## Deliverables

| Material | Location |
| --- | --- |
| Research report | `Research_Report.pdf` |
| Five baseline company reports | `reports/index.html` links to original HTML files and their adjacent charts |
| Original statements, forecasts and section text | `artifacts/baseline/{TICKER}/analysis/` |
| Exact prompts, responses, commands and timestamps | `artifacts/baseline/{TICKER}/audit/` and `render_*/audit/` |
| Four required audit tables | `audits/data_audit.csv`, `forecast_valuation_audit.csv`, `grounding_audit.csv`, `stability_audit.csv` |
| Grounding rates and stability statistics | `audits/grounding_summary.csv`, `grounding_by_section.csv`, `stability_pairs.csv`, `stability_metrics.json` |
| Independent valuation and price sensitivity | `valuation_reproduction.ipynb`, `audits/valuation_reconciliation.csv`, `target_price_sensitivity.csv` |
| Clean improvement diff | `improvement/data_driven_forecasts.patch` |
| Same-company before/after evaluation | `improvement/backtest.csv`, `backtest_summary.csv`, `{TICKER}/` |
| Numerical reproduction and boundary checks | `code/evaluate.py`, `code/test_forecast.py` |
| Live-run recording helpers | `code/finrobot_a1_*.py` |
| Baseline source and infrastructure diff | `artifacts/source_snapshot/`, `code/baseline_setup.patch` |
| Repeat-run evidence | `artifacts/stability/run2/`, `run3/`, `artifacts/stability_audit/` |
| Model cost and runtime | `audits/api_usage.csv`, `runtime_cost.csv`, `cost_summary.json` |
| Blog text | `Medium_Blog.txt` |

## Model, configuration and spend

The audited FinRobot commit is `2717499b8e30f242640af08c4ad9afd1113c2d45`. Source fork: https://github.com/Nancyxy/FinRobot. A separate Stable API compatibility patch changes the baseline data transport; it is not the evaluated improvement.

- Model alias: `gpt-4.1`; every captured response: `gpt-4.1-2025-04-14`.
- Endpoint: `https://api.openai.com/v1/chat/completions`; temperature `0.7`; `max_tokens=1000`.
- Five annual periods, two peers per company, text generation and sensitivity enabled.
- Enhanced-news and catalyst flags were not enabled. News retrieval was attempted but blocked. No Adanos key was configured.
- The active CLI calls plain Chat Completions sequentially through `text_generator_agents.py`; the separate typed SDK manager is not used.
- Rendering disables text regeneration and makes zero model calls. It preserves original analysis hashes but replaces competitor prose through its CSV heuristic; this is documented in the audit.
- Python 3.10.6, OpenAI 1.109.1, openai-agents 0.3.3, requests 2.34.1, pandas 2.3.3, NumPy 1.26.4 on the live-run machine. Offline evaluation used Python 3.12.14, pandas 2.2.3 and NumPy 2.3.5; see `audits/environment.json`.

49 unique successful model calls comprise 35 baseline and 14 repeat calls. Captured usage is 70,714 input tokens, including 7,680 cached tokens, and 16,835 output tokens. At $2/$0.50/$8 per million uncached-input/cached-input/output tokens, estimated recorded spend is **$0.264588**. This is a token estimate, not a billing export; unrecorded setup checks are excluded. Existing FMP Basic access is assumed to add no per-call charge. Pricing source: https://developers.openai.com/api/docs/models/gpt-4.1.

Baseline combined analysis/render times are NVDA 29.63s, AMD 30.97s, INTC 32.77s, AAPL 28.18s and CSCO 26.86s. Time waiting between stages is excluded. Full per-stage times and costs are in `runtime_cost.csv`.

## Reproduce the saved numerical analysis offline

Run from this submission folder. No API keys or network calls are required for these commands.

```bash
python3.12 -m venv .audit-venv
.audit-venv/bin/python -m pip install -r code/requirements-audit.txt
.audit-venv/bin/python code/evaluate.py
.audit-venv/bin/python code/test_forecast.py
```

`evaluate.py` applies the improvement patch in a temporary checkout, recalculates the five forecasts and all 20 metric forecasts in the backtest, reproduces the valuation engine, runs the original sensitivity analyzer, and writes the audit CSVs. It also verifies source excerpts, original analysis hashes, same-input CSVs and identical model request bodies. Grounding labels are manual, reviewable inputs in `code/grounding_labels.json`; the harness aggregates them rather than pretending to discover truth automatically.

The valuation notebook has already been executed without API calls. Its hand arithmetic reads raw archived inputs before calling the engine for comparison. Open it with the notebook working directory set to this submission folder.

## Reproduce live baseline runs

Fresh API data and stochastic model text can differ from the archived results. The exact original argv arrays are in each run/render manifest; `code/exact_commands_used.txt` collects all fourteen original stage commands. The portable commands below reproduce the run design.

From this submission folder, create a separate checkout:

```bash
PACKAGE="$(pwd)"
git clone https://github.com/Nancyxy/FinRobot.git ../FinRobot_reproduction
cd ../FinRobot_reproduction
git checkout 2717499b8e30f242640af08c4ad9afd1113c2d45
REPO="$(pwd)"
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-equity.txt
git apply "$PACKAGE/code/baseline_setup.patch"
cp finrobot_equity/core/config/config.ini.example finrobot_equity/core/config/config.ini
printf '\nfinrobot_equity/core/config/config.ini\n' >> .git/info/exclude
```

Configure local `config.ini` under `[API_KEYS]` with your own FMP and OpenAI keys, `openai_model = gpt-4.1` and `openai_base_url = https://api.openai.com/v1`. Keep the file untracked and ignored. No private configuration is included in this submission.

Run and render NVIDIA, then the AMD/Intel batch, then the Apple/Cisco completion batch:

```bash
.venv/bin/python "$PACKAGE/code/finrobot_a1_run.py" --repo "$REPO" \
  --company-ticker NVDA --company-name "NVIDIA Corporation" --peer-tickers AMD INTC
NVDA_RUN="$(.venv/bin/python -c 'from pathlib import Path; print(sorted(Path("output/a1_runs").glob("NVDA_baseline_*") )[-1].resolve())')"
.venv/bin/python "$PACKAGE/code/finrobot_a1_render.py" --repo "$REPO" --run-dir "$NVDA_RUN" --analyst-name "Xinyu Zhang"
.venv/bin/python "$PACKAGE/code/finrobot_a1_batch.py" --repo "$REPO" --tickers AMD INTC --include-run "$NVDA_RUN"
.venv/bin/python "$PACKAGE/code/finrobot_a1_complete.py" --repo "$REPO"
```

The original successful NVIDIA run was `NVDA_baseline_20261012_093437_937103Z`; AMD `AMD_baseline_20261012_181316_703880Z`; Intel `INTC_baseline_20261012_181349_328717Z`; Apple `AAPL_baseline_20261012_183858_023825Z`; Cisco `CSCO_baseline_20261012_183927_756856Z`. Peers are NVDA→AMD/INTC, AMD→NVDA/INTC, INTC→AMD/NVDA, AAPL→CSCO/NVDA and CSCO→AAPL/NVDA.

An initial QCOM run returned HTTP 402 on all five fundamental endpoints and made no model calls. AVGO was not run. The excluded QCOM artifacts are retained separately and are not counted as successful coverage. Existing free-plan symbol access motivated the broad Technology universe.

## Reproduce the three-run stability design

Use the original NVIDIA archive bundled here to preserve the financial input vintage. The helper replays captured FMP responses but makes two new sets of seven live model calls. It fails on missing replay inputs or prompt drift.

```bash
.venv/bin/python "$PACKAGE/code/finrobot_a1_stability.py" --repo "$REPO" \
  --baseline-run "$PACKAGE/artifacts/baseline/NVDA"
```

The saved original repeat directories are `NVDA_baseline_20261012_192520_441274Z` and `NVDA_baseline_20261012_192549_200109Z`. Both replay 15 analysis and eight render FMP calls, with zero FMP network calls and no replay misses. Exact request bodies and returned model names match the baseline.

## Apply the improvement to the live pipeline

Apply the improvement only after the baseline run; its default mode remains legacy.

```bash
git apply --check "$PACKAGE/improvement/data_driven_forecasts.patch"
git apply "$PACKAGE/improvement/data_driven_forecasts.patch"
.venv/bin/python finrobot_equity/core/src/generate_financial_analysis.py \
  --company-ticker NVDA --company-name "NVIDIA Corporation" --peer-tickers AMD INTC \
  --config-file finrobot_equity/core/config/config.ini --period annual --years-limit 5 \
  --forecast-method historical_median --enable-sensitivity-analysis \
  --output-dir output/NVDA_historical_median/analysis
```

This example produces numerical analysis without generating new prose. The reported five-company before/after experiment is reproduced offline by `evaluate.py`, using the same archived input vintage for both methods. The improvement patch is clean against the pinned upstream processor and CLI; the compatibility patch is separate.

## Audit conventions and limitations

SEC facts come from NVIDIA and Apple FY2025 10-Ks, archived as selected consolidated annual inline-XBRL facts. Operating EBITDA is operating income plus cash-flow D&A; FMP EBITDA uses a different definition. Basic EPS is used. Filing URLs and formulas appear in every data-audit row.

The EV/EBITDA reconciliation uses final estimate-column EBITDA, mean historical multiples, shares inferred from the same quote’s market cap/price, a 10%-of-EV debt proxy and no discounting. The alternate balance-debt price illustrates a convention difference. Neither arithmetic equality nor the local sensitivity range validates present fair value. The price grid uses a revenue-level ±5% shock and a margin ±2 percentage-point shock with fixed multiple/shares.

Grounding is evaluated against each company’s actual captured prompt, not external facts the model might know. Unsupported includes unverified scenarios and causal judgments. Numeric chains and product lists expressing one relationship count as one claim. A single-reviewer annotation and its granularity affect rates; exact excerpts allow independent review. Postprocessed takeaways and unavailable-news placeholders have explicit origins.

The historical-median model uses explicit -25%/+50% bounds, a 5% anchor and 0.5 decay. It preserves actual history and dynamically labels three annual forecast periods. The retrospective test has only ten folds and uses a current data vintage. Peer forecasts, unavailable endpoints, rendering fallbacks and prose grounding are not fixed by this numerical improvement.

## Submission location

Course fork: https://github.com/Nancyxy/STAT-GR5398-Columbia-University, branch `master`.

Place this complete folder at:

```text
Fall-2026/Assignment1/FinRobot/submissions/Assignment1_Xinyu_Zhang_xz3490/
```

The folder name includes the confirmed UNI. The published Medium article is linked above.
