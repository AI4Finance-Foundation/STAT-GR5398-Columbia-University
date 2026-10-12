# Part 3 — Financial Grounding Verifier: Improvement and Evaluation

## Objective

We implemented a financial grounding verifier to check numerical and temporal consistency between generated equity-research narratives and a frozen financial-data table. The improvement focused on three observed failure modes: incorrect scaling of dollar amounts, missed material EBITDA discrepancies, and redundant alerts when a historical actual was described as a forecast.

## Method

We compared the saved baseline verifier (v0.1) with the improved implementation (v0.2), applying both to the same 14 passage-level test cases. Twelve cases were manually constructed to probe known numerical and temporal error patterns. Two cases were previously generated agent passages. Seven passages were manually labeled as containing at least one grounding error and seven as negative. A prediction was considered positive if the verifier emitted one or more findings. Multiple findings on the same passage counted as a single positive prediction. Precision, recall, and F1 were calculated at the passage level, not the individual-claim or alert level.

## Results

| Subset | Verifier | N | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constructed cases | v0.1 | 12 | 4 | 5 | 1 | 2 | 0.444 | 0.800 | 0.571 |
| Constructed cases | v0.2 | 12 | 5 | 0 | 0 | 7 | 1.000 | 1.000 | 1.000 |
| All available cases | v0.1 | 14 | 6 | 5 | 1 | 2 | 0.545 | 0.857 | 0.667 |
| All available cases | v0.2 | 14 | 7 | 0 | 0 | 7 | 1.000 | 1.000 | 1.000 |

On the 14-case evaluation set, v0.2 increased passage-level precision from 54.5% to 100.0%, recall from 85.7% to 100.0%, and F1 from 66.7% to 100.0%. Five false positives and one false negative observed with v0.1 were eliminated on these particular cases.

### Error analysis

- **Unit interpretation:** The baseline falsely flagged correct values such as NVDA's approximately $144.6 billion of FY2026 EBITDA (S01) and $183.7 billion of FY2029 projected EBITDA (S02). The improved version did not flag these examples.
- **Material mismatch detection:** The baseline missed an incorrect NVDA FY2027 EBITDA claim of $382 billion (S03), while v0.2 flagged `EBITDA_VALUE_MISMATCH` and `AMOUNT_EXCEEDS_REVENUE` against the model's approximately $159.85 billion EBITDA value.
- **Temporal grounding:** Both versions detected historical FY2026 actuals described as estimates or projections (S04, S11). The improved version avoided additional spurious numerical alerts in these cases.
- **Previously generated passages:** The two included agent-output passages were flagged by both implementations. This subset is too small to establish performance on naturalistic research reports.

## Interpretation and limitations

These results establish that the implementation fixes specific, reproducible failure cases, **not** that it achieves 100% accuracy on unseen equity reports. The benchmark is small and deliberately focused on previously observed error patterns, and 12 of 14 samples are constructed. The gold labels were assigned with knowledge of the underlying data; they are not a blinded, independently annotated benchmark. In addition, the passage-level binary metric does not capture missed secondary errors within a passage, the severity of a finding, or whether every individual claim is grounded. Future work should include a larger held-out sample of untouched agent reports, claim-level annotation, more companies and data providers, and source-level verification against primary filings.

## Reproducibility

Inputs: `output/NVDA/analysis/financial_metrics_and_forecasts.csv` and the two archived agent passages under `experiments/nvda_grounding_test/`.

Code: `finrobot_equity/core/src/financial_grounding_verifier.v01.py`, `finrobot_equity/core/src/financial_grounding_verifier.py`, and `scripts/evaluate_grounding_verifier.py`.

Outputs: `experiments/part3_verifier/verifier_case_results.csv`, `verifier_metrics.csv`, and `evaluation_method.json`.
