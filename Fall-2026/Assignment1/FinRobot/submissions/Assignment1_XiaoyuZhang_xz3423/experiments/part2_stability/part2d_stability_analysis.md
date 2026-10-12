# Part 2D — NVDA Agent Stability Assessment

## Setup
Three independent generations used the same NVDA financial snapshot and prompt, with GPT-4.1-mini and temperature 0.7. Each run generated one `investment_overview` and one `risks` section. This evaluates narrative consistency rather than reproducible investment decisions or market-price forecasts.

## Results

| Dimension | Run 1 | Run 2 | Run 3 | Interpretation |
|---|---|---|---|---|
| Explicit independent Buy/Hold/Sell recommendation | None | None | None | No independent rating stability can be measured. |
| Independent model-derived price target | None | None | None | No price-target dispersion or standard deviation is defined. |
| External consensus target mentioned | No | No | $339.17, Buy | This exact value was included in the frozen prompt; it is not an independently generated price target. |
| Historical 2026 financials misdescribed as estimates/forecasts | Yes | Yes | Yes | 3/3 runs exhibit an A/E labeling error. |
| Revenue-growth volatility risk in risk list | Yes | Yes | Yes | 3/3 runs. |
| High valuation-multiple risk in risk list | Yes | Yes | Yes | 3/3 runs. |
| SG&A assumptions/expense-trend uncertainty in risk list | Yes | Yes | Yes | 3/3 runs. |

All three `risks` responses have five explicit bullet items. The three risk themes present in every run are revenue growth volatility, valuation multiple sensitivity, and SG&A assumption uncertainty. Thus the intersection is 3 recurring themes; this is a manually defined thematic overlap, not a quantitative semantic similarity score.

## Contradictions and instability

1. The input table labels FY2026 NVIDIA as **2026A**, but every investment overview describes at least one FY2026 amount as *estimated*, *projected*, or *forecast* (e.g., $215.9B revenue or $144.6B EBITDA). Run 2 explicitly says **2026E**, which contradicts the input's **2026A** label.
2. Run 2 also labels the 2026 EV/EBITDA value of 31.4x as **2026E**, whereas it is derived from FY2026 historical ratios.
3. Run 3 summarizes the 2027–2029 forecast revenue growth as **5–6% annually**; the stored scenario is **5%, 6%, 4%**, so the final year is omitted from the claimed range.
4. In Runs 1 and 3, peer EBITDA comparisons mix **NVDA FY2026 actual EBITDA** with **AMD and Intel FY2026 model forecasts**, without consistently marking the difference.
5. Run 3 alone quotes the external analyst consensus target ($339.17) and Buy rating; both were explicitly furnished to the model. No three-run independent price-target comparison is possible.

## Conclusion
The model is comparatively stable in the **broad qualitative risk themes** but unstable in the **presentation of data provenance and historical-versus-forecast labels**. The apparent agreement in some financial figures should not be interpreted as numerical factuality: the same A/E error recurred across all three outputs. The tested prompts did not elicit an independent rating or valuation-derived target, so those requested stability measures are **not available**, not zero variance.

## Methodological limitations
Only three stochastic draws from one company and two selected sections were assessed. No market data was refreshed. The qualitative risk mapping was manually applied. The target price quoted in the third run came from the frozen prompt rather than an independent analysis. The source file used for this review is `stability_all_runs.txt` and corresponding six saved text sections.
