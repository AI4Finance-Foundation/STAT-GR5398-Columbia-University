# Verifier V4 coverage validation

V3 remains preserved, including every failed response, repair and charge. Its lexical validator rejected harmless uncovered conjunctions and category headings. Inspection also found a genuine omitted AMD geopolitical-risk sentence. These are different failure classes.

V4 accepts only a finite, explicit list of complete uncovered connector/heading spans (for example `and`, `Additionally,`, and `Market Risks:`). It exports exact skipped spans and source-unit offsets. It does not filter individual words from an otherwise substantive gap. Negation, quantities, causal connectors, and any factual sentence remain subject to rejection. The error now includes the exact unreviewed fragments so a bounded repair can address the actual omission. Numeric checking, unsupported annotations, and contradiction release rules are unchanged.

V4 runs use the same frozen writer inputs and model settings as V3 but are separate verifier experiments. Their manifests bind the validator source SHA-256. They must not resume a V3 run under changed validation. Prior failures and costs remain in the global ledger. This is a development iteration selected after observing pilot failures, not a preregistered held-out experiment.

Regression checks verify that finite connective omissions pass and are recorded, whereas omitted facts, negation and causal qualifiers fail. The separate compatibility patch is unchanged. Clean-apply testing now sets both upstream environment variables used by the regression suite, ensuring that tests import the patched clean checkout.
