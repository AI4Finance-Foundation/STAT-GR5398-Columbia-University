# Human-in-the-Loop Evaluation Rubric

Rate each generated response independently. Enter whole numbers from 1 to 5 for the six quality
fields, `yes` or `no` for the two binary fields, and optional evidence in `notes`. Do not change the
identifier or evidence columns.

## Rating scale

| Rating | Meaning |
| --- | --- |
| 1 | Incorrect, unsupported, or unusable |
| 2 | Major weaknesses; substantial correction needed |
| 3 | Adequate but incomplete or uneven |
| 4 | Strong; only minor issues |
| 5 | Excellent, specific, and well supported |

## Quality fields

- `positive_development_quality`: relevance, specificity, and support for positive factors.
- `concern_risk_quality`: relevance, specificity, and support for risks or concerns.
- `financial_reasoning_quality`: sound use of news, financial information, and causal reasoning.
- `prediction_justification`: consistency between the predicted direction/magnitude and analysis.
- `factual_grounding`: fidelity to information in the prompt without unsupported claims.
- `clarity_professionalism`: concise organization, readability, and professional tone.

## Binary fields

- `formatting_compliance`: `yes` only when all required headers plus `Prediction:` and `Analysis:`
  appear in the correct order.
- `hallucination_observed`: `yes` when the response asserts a material fact that the supplied prompt
  does not support or that contradicts the prompt.

Use `notes` to identify the specific text supporting a low rating or hallucination decision. Human
ratings must be entered manually; scripts must not populate them.
