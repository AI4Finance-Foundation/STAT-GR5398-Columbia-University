# Forecast audit: implications and analyst alternatives

The parameter alternatives below have now been executed as separate offline audit scenarios on the frozen NVDA/AMD inputs. The sole formal system improvement remains Option A, the verifier; its original inputs and outputs are unchanged. No fitted forecasting model, backtest or forecast-accuracy improvement is claimed. The saved results distinguish conditional parameter effects from the unchanged upstream EBITDA-definition limitation.

## Observations and timing

NVIDIA's FY2026 revenue was $215.938 billion for 2025-01-27 through 2026-01-25, compared with $130.497 billion for the prior fiscal year: growth of 65.47%. AMD's FY2025 revenue was $34.639 billion for 2024-12-29 through 2025-12-27, compared with $25.785 billion: growth of 34.34%. NVIDIA's reported gross margin was 71.068% and SG&A/revenue 2.121%; AMD's corresponding figures were 49.516% and 11.963%. Values and original-filing iXBRL locations are reproducible from `evidence/sec/snapshot_2026-10-11/sec_metric_ingredients.csv` and `sec_metric_references.csv`. [NVIDIA FY2026 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm); [AMD FY2025 10-K](https://www.sec.gov/Archives/edgar/data/2488/000000248826000018/amd-20251227.htm).

The analysis cutoff is 2026-10-11. The next annual NVIDIA period is FY2027; AMD's next is FY2026. These fiscal periods are already partially elapsed at the cutoff and quarterly releases are publicly available. This is a current forecast exercise based on a latest-annual pipeline, not an out-of-sample historical backtest. Annual observations published after a hypothetical older forecast date must never be used to explain that historical prediction. The selected baseline prompt may omit available quarterly evidence; the auditor must still judge the prompt actually supplied.

## What the default constants imply

Default sequential revenue growth of 5%, 6%, 4% multiplies the last actual revenue by 1.15752 over three years. Gross/contribution margin rises by 1 percentage point per year and SG&A margin falls by 0.5 percentage point per year. These are additive changes to margins, not 1% and 0.5% relative changes. Initial margins are rounded to one decimal percentage point by the upstream processor before forecasting.

| Company and year | Default revenue ($bn) | Contribution margin | SG&A margin | Final reported forecast EBITDA margin |
|---|---:|---:|---:|---:|
| NVDA FY2027E | 226.735 | 72.1% | 1.6% | 70.5% |
| NVDA FY2028E | 240.339 | 73.1% | 1.1% | 72.0% |
| NVDA FY2029E | 249.953 | 74.1% | 0.6% | 73.5% |
| AMD FY2026E | 36.371 | 50.5% | 11.5% | 39.0% |
| AMD FY2027E | 38.553 | 51.5% | 11.0% | 40.5% |
| AMD FY2028E | 40.095 | 52.5% | 10.5% | 42.0% |

These are exact formula implications at the SEC reference values, rounded here for display, not independent forecasts. Revenue formula: `R[k] = R[k-1] * (1 + growth[k])`; final margin formula: `round(gross_margin_actual, 1) + k*1.0 - (round(SGA_margin_actual, 1) - k*0.5)` in percentage points.

The forecast code first updates EBITDA margin but later overwrites it with contribution profit minus SG&A, divided by revenue. That operation excludes research and development and changes the historical earnings definition. SEC operating-income-plus-D&A margins are 61.698% for NVDA and 19.337% for AMD; simple net-income-plus-tax-interest-D&A margins are 66.941% and 21.268%. The actual reconciled AMD FMP measure excludes discontinued and equity income and is 21.002%; this smaller definition difference does not explain the jump to the forecast 39.0%. Thus AMD's first forecast margin of 39.0% contains a definition discontinuity far larger than the documented 1 percentage-point improvement. This cannot be justified as ordinary operating leverage. For NVIDIA, SG&A/revenue declines to 0.6% despite the historical 2.1% level; the hard-coded trajectory does not establish that this expense ratio is attainable. The full arithmetic and overwritten settings are described in `data_valuation_source_audit.md`.

## Rationale for the separately executed assumptions

The first required decision is to choose one earnings definition across actuals and forecasts. An operating measure based on GAAP operating income plus explicitly reconciled D&A would retain R&D expense and avoid including non-operating investment gains in a presumed operating multiple. The currently reported historical FMP EBITDA should be preserved for baseline comparability, with both definitions disclosed. Silently relabeling gross profit less SG&A as EBITDA would be unacceptable. Merely changing `--margin-improvement` cannot repair the later overwrite.

For scenario analysis, the separate analyst-selected deceleration case uses NVIDIA growth of 30%, 20%, 10% and AMD growth of 15%, 10%, 8%, with gross and SG&A margins initially held at the last observed ratios. These are executed scenario assumptions, not SEC guidance or statistically estimated predictions. Their rationale is to model substantial deceleration from the observed annual 65.47% and 34.34% without assuming an immediate common 5% growth path for both businesses. The three-year revenue implications are $370.550 billion for NVIDIA and $47.324 billion for AMD, compared with default $249.953 billion and $40.095 billion. Neither historical growth nor these choices proves the new rates correct.

Before using those scenarios in a portfolio decision, reconcile already published quarterly actuals to the annual forecast, examine segment demand, customer concentration and export-control exposure using information available at the cutoff, and compare against a stated bear case. The available FY2026 NVIDIA filing discusses demand concentration and export controls; AMD's FY2025 filing describes the growth mix and export-related inventory charges. This qualitative context supports wider scenarios and restrained margin assumptions, not any precise rate by itself. No three-year constant path captures these contingencies adequately.

For INTC, AVGO and QCOM, segment-specific historical growth and economics should be reviewed before prescribing new rates. Foundry activity, infrastructure software and licensing make the same margin rule particularly unsuitable. The cohort's business overlap supports research coverage, not identical forecast coefficients. We do not assign unsupported precise alternative estimates to those firms before their frozen financial inputs have been inspected.

The existing local target-price sensitivity remains preserved. The new parameter scenarios separately propagate the executed paths through the same EV/EBITDA method, without changing its earnings definition. Neither calculation establishes the probability or predictive accuracy of a scenario. A repaired earnings model would be a separate future improvement; this parameter exercise is not merged into the Option A before/after experiment, whose writer inputs stay identical.

## Executed offline parameter scenarios

To test the proposed assumptions, I separately invoked the same pinned forecasting functions on frozen NVIDIA and AMD raw statements. Each company has a default replay and an analyst-deceleration case. NVIDIA uses sequential annual revenue growth of 30%, 20%, 10%; AMD uses 15%, 10%, 8%. Both annual contribution/EBITDA-margin input increments and the SG&A-margin change are set to zero. The adjusted contribution and SG&A ratios therefore stay at the processor's rounded latest-actual levels: 71.1% and 2.1% for NVIDIA, 49.5% and 12.0% for AMD. No precise SEC ratio was silently substituted for a rounded pipeline input.

| Company / fiscal forecast | Default revenue, $bn | Adjusted revenue, $bn | Default EBITDA margin | Adjusted EBITDA margin |
|---|---:|---:|---:|---:|
| NVDA 2027E | 226.735 | 280.719 | 70.5% | 69.0% |
| NVDA 2028E | 240.339 | 336.863 | 72.0% | 69.0% |
| NVDA 2029E | 249.953 | 370.550 | 73.5% | 69.0% |
| AMD 2026E | 36.371 | 39.835 | 39.0% | 37.5% |
| AMD 2027E | 38.553 | 43.818 | 40.5% | 37.5% |
| AMD 2028E | 40.095 | 47.324 | 42.0% | 37.5% |


The rationale remains a deceleration from observed annual growth of 65.47% and 34.34%, without assuming the same immediate 5% rate for both businesses or further expense-ratio compression. These are analyst-selected conditional assumptions, not guidance, fitted predictions or validated forecast accuracy. All six forecast periods remain labeled as issuer fiscal years.

The same EV/EBITDA engine uses third-year EBITDA, the original historical mean multiple, original shares and the original rule that assumed net debt equals 10% of each scenario's EV. Debt amounts consequently change; the rule is held fixed, not the dollar debt. Neither discounting nor terminal growth enters this multiple method.

| Company / EBITDA year | Default model target, USD | Adjusted model target, USD | Change |
|---|---:|---:|---:|
| NVDA 2029E | 333.71 | 464.43 | +39.17% |
| AMD 2028E | 377.84 | 398.17 | +5.38% |


NVIDIA's terminal revenue increases 48.25% while its terminal EBITDA margin falls from 73.5% to 69.0%; AMD's revenue increases 18.03% while margin falls from 42.0% to 37.5%. With shares and the multiple unchanged, the target ratio equals the terminal revenue ratio times the EBITDA-margin ratio. Thus stronger growth and lower margins jointly produce the displayed target changes. Those prices are conditional model outputs, not endorsed fair values. Zero margin improvement does not repair the upstream definition: projected EBITDA still omits R&D and does not explicitly reconcile D&A, and AMD's adjusted 37.5% remains far above its 21.002% historical provider measure.

The default replays match all 220 saved table fields. The scenario implementation records 140 internal arithmetic checks; a separately written standard-library Decimal reference passes 670 additional numerical and integrity checks. Full configurations, actual processor CSVs, engine output, annual deltas, assumptions, source hashes and logs are in `audits/forecast_parameter_scenarios/run_v2/`. The run took 1.84 seconds including integrity hashing, with zero API calls, tokens or new API cost. Python network/subprocess rejection probes passed; this is not an OS sandbox. A second check confirms all 9,083 protected original submission files are byte-identical, including snapshots, writer prompts, generated texts, verifier treatments, paired claim audits and the API ledger. This fulfills the forecast-audit parameter exercise and does not introduce a second research improvement or modify the Option A comparison.

