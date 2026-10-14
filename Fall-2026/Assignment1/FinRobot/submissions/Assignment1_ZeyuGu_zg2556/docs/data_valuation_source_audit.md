# Data, forecast, valuation, and sensitivity source audit

Author: Zeyu Gu (zg2556). Source revision: `AI4Finance-Foundation/FinRobot@2717499b8e30f242640af08c4ad9afd1113c2d45`.

This is a code audit, not a completed company experiment. The conclusions below follow from the actual source implementation. No FMP response, SEC reconciliation, observed company valuation, or experimental performance is implied by the mathematical regression tests. The single research improvement remains Option A (the verifier); the changes below are compatibility and evidence disclosure, not a replacement forecasting model or DCF.

## Primary source files

- [financial_data_processor.py at fixed revision](https://github.com/AI4Finance-Foundation/FinRobot/blob/2717499b8e30f242640af08c4ad9afd1113c2d45/finrobot_equity/core/src/modules/financial_data_processor.py)
- [market_data_api.py at fixed revision](https://github.com/AI4Finance-Foundation/FinRobot/blob/2717499b8e30f242640af08c4ad9afd1113c2d45/finrobot_equity/core/src/modules/market_data_api.py)
- [generate_financial_analysis.py at fixed revision](https://github.com/AI4Finance-Foundation/FinRobot/blob/2717499b8e30f242640af08c4ad9afd1113c2d45/finrobot_equity/core/src/generate_financial_analysis.py)
- [valuation_engine.py at fixed revision](https://github.com/AI4Finance-Foundation/FinRobot/blob/2717499b8e30f242640af08c4ad9afd1113c2d45/finrobot_equity/core/src/modules/valuation_engine.py)
- [sensitivity_analyzer.py at fixed revision](https://github.com/AI4Finance-Foundation/FinRobot/blob/2717499b8e30f242640af08c4ad9afd1113c2d45/finrobot_equity/core/src/modules/sensitivity_analyzer.py)

## Financial metrics: observations and derived quantities

The original API client assigns `year = date.dt.year` to statements and ratios. This is the calendar year in which the period ends, not necessarily the issuer's named fiscal year or a calendar-year financial total. The audit must retain FMP `date`, `calendarYear`/fiscal metadata where provided, `period`, `filingDate`, and `acceptedDate`, and align SEC fiscal periods explicitly. A fiscal year ending in January can be named differently from its calendar ending year. Forecast labels inherit this annual-period convention. They must not be represented as calendar-year projections.

The API extraction accepts balance-sheet and cash-flow DataFrames but does not use them in the main metrics table. In particular, net income, balance-sheet net debt, and observed free cash flow are not rows of that table. They cannot be described as audited pipeline metrics unless a separately identified data path adds them.

| Pipeline metric | Data path / equation | Audit caveat |
|---|---|---|
| Revenue | First truthy `revenue`, `totalRevenue`, `netSales`, `totalNetSales` | Units are native response currency, not automatically millions. |
| Cost of Operations | First truthy `costOfRevenue`, `costOfGoodsSold`, `totalCostOfSales` | The label denotes cost of revenue, not all operating costs. |
| SG&A | First truthy `sellingGeneralAndAdministrativeExpenses`, `sellingAndMarketingExpenses`, `sellingGeneralAdministrative` | Selling/marketing alone need not equal total SG&A. |
| Contribution Profit | Revenue − cost of revenue; otherwise `grossProfit` | This is gross profit under the source definition; “contribution” need not be a variable-cost contribution measure. |
| Contribution Margin | Contribution Profit / Revenue | Formatted to one decimal percentage point. |
| EBITDA | Direct FMP `ebitda` for actuals | Must reconcile the provider's definition; SEC does not necessarily disclose EBITDA. |
| EBITDA Margin | EBITDA / Revenue | Formatted to one decimal percentage point. |
| SG&A Margin | SG&A / Revenue | Formatted to one decimal percentage point. |
| Revenue Growth | Revenue_t / Revenue_previous − 1 | First available year is N/A; observation spacing must be checked. |
| EPS | First truthy `eps`, `epsdiluted`, `netIncomePerShare` | Basic/diluted shares, split adjustments, and continuing operations can differ. |
| PE Ratio | Same ending-year row in ratios (`priceEarningsRatio`/`peRatio`), then key metrics | Needs market-price date and earnings definition; SEC alone does not establish a market multiple. |

The original uses Python truthiness (`or`, `if value`) rather than explicit missing-value checks. Valid zeros can disappear, and NaN can bypass fallbacks because it is truthy. Same-year records are joined without matching full fiscal dates. The compatibility change rejects duplicated statement years so quarterly observations cannot silently overwrite one another. The baseline formulas otherwise remain unchanged; the evidence audit must identify these remaining limitations for every affected real metric.

Twenty substantive SEC reconciliation rows should use ten actual pipeline metrics for each of two companies, with exact source periods. Repeated figures from one SEC filing are not independent sources. Derived EBITDA or margin rows must show the formula and constituent filing/XBRL locations. A directly reported item and a calculated item must be distinguished, and an absent GAAP-equivalent measure must remain unavailable rather than be invented.

## Forecast defaults and the compatibility boundary

The original Step 1 selects the latest actual as revenue base, but hard-codes estimates as `2025E`, `2026E`, `2027E`. Once 2025 or 2026 actuals are present, the forecast can appear to predict an already observed period using later actuals. The patch maps the three ordered CLI growth slots to `last_actual + 1`, `+2`, `+3`, preserves original labels in `forecast_label_mapping`, and records `forecast_origin`. It rejects an explicitly older base while later actuals remain in the input; a historical backtest must first truncate data to its origin. This changes date alignment, not growth assumptions.

| Assumption | Original default / actual behavior |
|---|---|
| Revenue growth, horizons 1/2/3 | +5%, +6%, +4%, applied sequentially. Revenue after 3 years = base × 1.05 × 1.06 × 1.04 = base × 1.15752. |
| Contribution Margin | +1 percentage point per forecast year from the *rounded* historical margin. |
| SG&A Margin | −0.5 percentage point per year from the rounded historical margin; can become negative because there is no lower bound. |
| EBITDA Margin input | Initially +1pp/year, then overwritten by `(Contribution Profit − SG&A)/Revenue`. It still affects EPS growth. |
| EBITDA forecast | Revenue × Contribution Margin − SG&A; excludes R&D and other expenses from the projected profit bridge. It is not definitionally continuous with provider-reported historical EBITDA. |
| Resulting forecast EBITDA margin | Base Contribution Margin − base SG&A Margin + 1.5pp × horizon, subject to presentation rounding. The first forecast can jump markedly from historical EBITDA margin. |
| EPS | Prior EPS × (1 + revenue growth) × (1 + EBITDA-margin-improvement input); no net-income or share-count bridge. A percentage-point margin assumption becomes a relative EPS growth factor. |
| PE Ratio | Historical PE × 0.95^horizon; a fixed 5% annual multiple compression assumption. |
| `ebitda_growth_factor = 1.05` | Set in Step 1 but never read by `calculate_growth_and_forecasts`. |
| Boundaries | No caps on margin >100%, negative SG&A, or negative contribution margin. |

For the real forecast audit, preserve a row for each company and each horizon with base period, projected revenue, margin, SG&A, implied EBITDA, and exact CLI assumptions. Explain company-specific plausibility from contemporaneous filings or guidance. Proposed evidence-based alternative assumptions should be reported as analysis, without silently replacing the frozen baseline or introducing a second research improvement. No company-specific adjustment has been asserted in this source audit because no actual financial snapshot is attached here.

## Valuation assumptions and field ownership

The assignment wrapper does not call the valuation engine, and the original `generate_financial_analysis.py` does not import it. An agent's valuation narrative is not evidence that EV/EBITDA, peer comparison, or DCF ran. `independent_valuation.py` explicitly executes the engine as a comparator and separately calculates the equations below without importing it into the calculation functions.

| Field | Owner/source | Consequence |
|---|---|---|
| Report cover price/rating/target | Step 2 market-data inputs/FMP unless explicitly supplied and auto-fetch disabled | These are neither the engine target nor a sampled LLM rating. |
| Model target | `ValuationEngine` using financial table and explicitly passed share count | Save method-level output and assumptions separately from the cover fields. |
| Prose | Eight agents' structured text, potentially regenerated in Step 2 | Audit the text actually rendered and its exact prompt/version. |
| EBITDA used by engine | Last A/E column in the table, normally third forecast horizon | No discounting of terminal forecast EBITDA back to the data date. |
| Shares | Caller-supplied `shares_outstanding`, otherwise literal 1 | Independent harness requires observed positive shares and rejects the fallback. |
| Target EV/EBITDA | Caller override; else arithmetic mean of `enterpriseValueOverEBITDA` | Historical sample, not an independently justified target. |
| Missing historical multiples | Mean 12, standard deviation 3 | Hard-coded assumptions, never observations. |
| Multiple range | Mean ± sample standard deviation, ddof=1 | One observation yields NaN upstream; independent harness stops rather than emitting a spurious report. |
| EV/EBITDA net debt | 10% of modeled EV | Not actual cash/debt; creates circular proportional debt. |
| Peer method | Mean/min/max of supplied peers' `ev_ebitda`; debt = 10% × subject EBITDA × peer mean | No peer normalization, fiscal alignment, or separate observed debt. |
| DCF FCF base | Caller `free_cash_flow`; if falsy, 60% × selected EBITDA | Zero observed FCF is treated as missing; fallback can seed a ten-year DCF from third-horizon forecast EBITDA. |
| DCF growth | 10% for years 1–5, 5% for 6–10 | No company-specific driver model. |
| DCF terminal growth / WACC | 2.5% / 10% | No financing-cost estimation or capital structure. |
| DCF net debt | 10% of DCF enterprise value | Not balance-sheet net debt. |
| DCF low/high range | Only the terminal discount denominator uses WACC ±1pp | Explicit-period PV and terminal-value denominator remain fixed; this is not full WACC sensitivity. |
| Synthesis | Positive targets weighted by 0.7 EV/EBITDA, 0.6 peers, 0.5 DCF | Heuristic “confidence” weights, not calibrated probabilities. Nonpositive method prices are excluded. |
| Price floor | `max(price, 0)` | Negative model equity value becomes zero and nonlinear clipping affects local elasticities. |

Independent EV/EBITDA formulas (native currency and actual shares):

```
EV = selected EBITDA × target multiple
assumed net debt = 0.10 × EV
equity value = EV − assumed net debt
target price = max(equity value / shares, 0)
low price = max((EBITDA × (target multiple − sample std) − base assumed debt) / shares, 0)
high price = max((EBITDA × (target multiple + sample std) − base assumed debt) / shares, 0)
```

The clean patch adds assumption metadata to results without changing these equations. A numerical match only demonstrates faithful reproduction; it does not make the financial assumptions defensible. The DCF recomputation also saves every projected FCF and PV term.

## Native sensitivity versus target-price sensitivity

The native revenue analyzer multiplies each forecast revenue **level** by 0.95 through 1.05. It does not change and compound annual growth assumptions despite its “Growth Delta” label. The native margin analyzer adds −2pp through +2pp to the one-decimal displayed EBITDA margin. Its combined matrix is terminal-year EBITDA rounded to $0.1bn. Its purported 95% confidence interval assumes standard deviation = 15% of the forecast, uses 1.96, and does not estimate historical volatility.

The patch preserves native calculations but explicitly describes the units and uncalibrated scenario bands. `target_price_sensitivity.py` runs all three native analyses and saves their outputs. It then uses the same fixed share count, multiple and baseline net-debt rule to calculate scenario target prices. It saves both:

1. The exact EBITDA/revenue margin grid, preserving the engine's zero-shock target price.
2. The native displayed-margin grid, exposing any price gap caused by margin rounding.

For each scenario `r` (relative revenue change) and `d` (absolute margin fraction), `EBITDA_scenario = Revenue_base × (1+r) × (EBITDA_base/Revenue_base+d)`. The model EV and assumed debt are then recalculated by the exact baseline method. No rounded "$x.xB" string is used as a numeric input. Under positive valuation without clipping, local price elasticity to revenue is 1; percent price change from a +1pp margin move is `1 / base_margin_fraction`. These are algebraic consequences of this simplified valuation, not empirical company estimates. Report the range and omitted uncertainties rather than presenting the grid as a probability interval.

## Offline commands after genuine data capture

Run from the submission root in the dedicated Python environment. `UPSTREAM_SRC` is the patched `finrobot_equity/core/src` directory. `ANALYSIS` points to the immutable run's saved analysis folder. `INPUTS` is that run's `valuation_inputs.json`, with provenance for price, shares and any FCF.

```powershell
& $PY scripts/independent_valuation.py --analysis-dir $ANALYSIS --valuation-inputs $INPUTS --upstream-src $UPSTREAM_SRC --output-dir $VALUATION_OUTPUT
& $PY scripts/target_price_sensitivity.py --analysis-dir $ANALYSIS --valuation-inputs $INPUTS --upstream-src $UPSTREAM_SRC --output-dir $SENSITIVITY_OUTPUT
$env:FINROBOT_UPSTREAM_SRC = $UPSTREAM_SRC
& $PY -m unittest discover -s tests -p test_data_valuation.py -v
```

Required `valuation_inputs.json` keys: `ticker`, `run_id`, `current_price`, `shares_outstanding`. Optional keys: `free_cash_flow`, `free_cash_flow_period`, `ebitda`, `target_multiple`, `dcf_assumptions`, `peer_data` (`ticker -> {ev_ebitda: number}`), and `source_metadata` (`field -> {path, source, date, unit}`). Omit an unavailable FCF instead of inventing it; the engine's 60%-of-EBITDA fallback will be explicitly flagged. Share provenance and date remain mandatory evidence even though the arithmetic interface cannot authenticate a caller's metadata. The input file must contain no API credentials.

Generated deliverables include engine outputs, independent results, source hashes, assumption and reconciliation CSVs, DCF cash flows, rerunnable notebook, native sensitivity tables, scenario-band assumptions, a full-precision target-price CSV, numerical summary, PNG and SVG. These commands are offline and make no market-data requests or paid model calls. They cannot establish that the underlying inputs were genuine; that is the responsibility of the saved API-response manifest and SEC audit.

Regression tests cover dynamic labels, legacy-date compatibility, rejection of future information in an older-base backtest, quarterly collapse, independent EV/EBITDA and DCF agreement against the actual engine, zero-price clipping, margin-rounding reconciliation, and percent-versus-percentage-point shocks. Test fixtures are explicitly synthetic and isolated; no test fixture is a submitted company result.

## Observed stable-API compatibility work (2026-10-11)

The supplied credential was tested without displaying or saving the secret. The account denies the original legacy endpoints. Recorded stable-endpoint probes are in `evidence/stable_endpoint_preflight/`, including raw responses and their fetched-at timestamps. This is real endpoint evidence, separate from the unit-test fixtures.

Stable APIs rename several fields: `priceToEarningsRatio` to the legacy consumer alias `priceEarningsRatio`, `debtToEquityRatio` to `debtEquityRatio`, and `evToEBITDA` to `enterpriseValueOverEBITDA`. Profile aliases preserve `marketCap`, `averageVolume`, `lastDividend`, and `exchange` while adding names expected by the renderer. Raw HTTP bodies retain the original stable schema unchanged. Provider `fiscalYear` remains in raw tables, and `financial_period_provenance.csv` records it alongside the original pipeline's calendar ending year.

The stable quote omits shares; `stable/shares-float` supplies actual `outstandingShares`, `freeFloat`, source, and observation date. The compatibility client uses those observations, eliminates the legacy automatic 95% float/Technology-sector substitutions, and leaves unavailable observations missing. `stable/grades` supplies analyst grades; `stable/ratings-snapshot` is a financial-health letter score and is not substituted for an analyst recommendation. The available `price-target-consensus` returns a consensus target, not the legacy latest single-analyst target. That semantic change is explicit in the market-field source metadata and must remain explicit in the report.

The provider's [official stable API reference](https://site.financialmodelingprep.com/developer/docs) documents these endpoints and the distinction between grades, ratings, and target consensus. Actual entitlement is established by saved requests, not by documentation alone.

The real account returned HTTP 200 for the five assignment warmup tickers' income statements, but HTTP 402 for AVGO, QCOM and the tested alternative semiconductor companies MU, TXN, ADI, AMAT, LRCX, KLAC, and ON. Both stock-specific and general latest-news endpoints returned HTTP 402. These limitations do not establish a five-company same-industry baseline. The supported warmup set must not be relabeled as a semiconductor universe.

`snapshot_pipeline.py` runs the actual upstream data, forecasting, peer, sensitivity, news-classification and catalyst functions. It freezes HTTP requests and the processing clock. It records denied news responses as missing source entitlement; an explicitly supplied `--news-snapshot` can provide separately identified real SEC/issuer-event excerpts. Such excerpts are alternative primary-source evidence, not proof of FMP news access or equivalent press-news coverage. Source URLs and document/excerpt hashes are retained. The native default earnings catalyst at as-of plus 90 days and its 95% heuristic probability are disclosed as assumptions, not observed calendar events.

Snapshot status `frozen` only means that the collected data and deterministic artifacts are immutable. `coverage_status`, `missing_data.json`, and `fallback_register.json` state completeness and limitations. A frozen snapshot alone does not complete the assignment or establish the truth of company claims. Offline replay validates the original hashes and requires a separate output directory; it never falls through to the network when a request is missing, and it checks byte equality of deterministic analysis artifacts.

## Measured parallel forecast inconsistency

The original `project_ebitda_for_peers()` forecasts each column with the **arithmetic mean of historical EBITDA year-over-year growth**, not the main revenue/margin assumptions. It includes the subject company in the peer table. It labels the next row with the latest actual year **across the entire table** plus one, even if a particular company has an older last actual. Only one growth step is applied. Negative or very small historical EBITDA denominators can make arithmetic growth averages unstable.

The real frozen NVDA evidence illustrates both problems. `audits/peer_projection_audit.csv` independently reconstructs 15 projections across the five supported snapshots and records raw API source hashes, historical values, all growth inputs, labels, and comparisons to the corresponding company's main forecast. It does not modify any baseline prompt.

| Company in NVDA peer table | Latest actual ending year | Native label | Mean historical growth | Peer-table EBITDA | Own main forecast for same label | Peer/main difference |
|---|---:|---:|---:|---:|---:|---:|
| NVDA | 2026 | 2027 | 164.27% | $382.001bn | $159.848bn | +138.98% |
| AMD | 2025 | 2027 | 18.23% | $8.601bn | $15.614bn | −44.92% |
| INTC | 2025 | 2027 | 229.89% | $47.352bn | $17.118bn | +176.62% |

For AMD and INTC the peer table advances a label by two ending years while applying one growth step. These should not be presented as comparable contemporaneous forecasts. For NVDA two inconsistent 2027 EBITDA values appear in the same evidence prompt. Grounding classification must identify which table supports a claim; “supported by one prompt cell” does not resolve internal evidence conflicts. A verifier should flag the conflict rather than select whichever value suits the narrative. This is a measured baseline limitation, not a reason to rewrite a prompt after auditing it.

The audit can be repeated offline with `python scripts/audit_peer_forecasts.py --snapshots-root snapshots --output audits/peer_projection_audit.csv`.

## News relevance compatibility derivation

The native `NewsIntegrator` can accept a company name, but its usual caller passes only the ticker. Real Intel and Apple releases were excluded when their title/first 500 characters contained the company name but not INTC/AAPL. The original captures remain frozen at `snapshots/INTC` and `snapshots/AAPL`. Before narrative generation, the wrapper was corrected to pass `company_name`; `snapshots/INTC_news_v2` and `snapshots/AAPL_news_v2` were derived by offline replay of the unchanged raw responses and unchanged issuer releases. Their manifests have new run IDs, original manifest hashes, correction reasons, and artifact comparisons. Each retains one genuine issuer article. This is a compatibility correction to evidence ingestion; it is separate from the Option A before/after verifier experiment.

These derivative snapshots reference the original snapshots for raw HTTP. Reproduce a derivative by using the original as `--snapshot`, a new `--replay-output`, `--derive-analysis`, and an explicit `--derivation-reason`. Do not represent this as an exact replay of the original analysis, whose omission is preserved as evidence.
