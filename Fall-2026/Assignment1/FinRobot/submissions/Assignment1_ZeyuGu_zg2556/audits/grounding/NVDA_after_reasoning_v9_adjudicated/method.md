# Independent V9 delta adjudication

This is machine-assisted review, not human validation. All eight final sections
are covered. Prior corrected judgments transfer only for byte-identical Unicode
source-unit text, in the same section and with the identical original evidence
prompt SHA-256. Sequence alignment preserves occurrence/order and permits no
normalization, warning removal, or paraphrase matching. Inherited labels remain
subject to the limitations and errors of the prior full machine adjudication.

Every changed unit is explicitly decomposed into (1) the supplied CAGR cell,
(2) actual-endpoint CAGR, (3) full-table endpoint CAGR, or (4) forecast status.
Each receives its own classification, numeric status, original prompt citation
and reason. Two independent clauses are not accepted as a single grounded fact.
The independent parser reads the original financial table; the independent
calculator evaluates 100*(exp((ln(end)-ln(start))/elapsed_years)-1) using Decimal
precision 60 and rounds to the precision stated in the report, HALF_UP. It does
not import the production power-based calculator or use its saved outputs.
Positive finite revenue endpoints and a positive year difference are required.
A/E roles must agree with prompt columns; an E endpoint is not realized history.
The report's literal display-cell statement is separate from computed CAGR.
No statistical period is inferred merely from a table-scope description.

Unknown changed prose aborts this narrow offline audit and requires additional
explicit review. A recognized conflicting scalar is Contradicted; absent data
is Unsupported. None of the verifier's own verdicts, citations, or calculated
results is supplied to the evaluator. An unchanged Unsupported annotation stays
Unsupported. The evidence scope is the original writer prompt, not external
truth or later data. The implementation author saw the production rule during
review; this is independent code and evidence checking, not a blind held-out
evaluation. The twelve development examples informed both implementation and
review. Repetition adds claim instances and must not be confused with new facts.

New calculations and forecast clarifications use explicit semantic fact keys;
unchanged lexical keys are inherited. Retention is still a proxy and cannot
measure exact semantic recall across paraphrases. No direct paid API is called;
task-agent authoring costs/tokens remain uninstrumented, not assumed zero.
