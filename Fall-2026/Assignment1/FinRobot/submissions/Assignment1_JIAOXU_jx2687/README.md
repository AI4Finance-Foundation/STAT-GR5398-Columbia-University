# FinRobot Assignment 1 — JIAO XU (jx2687)

**STAT GR5398, Fall 2026.** [Published Medium analysis](https://medium.com/@xujiao0224/auditing-finrobot-when-a-polished-equity-report-outruns-its-evidence-bec92d7e0e49) · [Research report](research_report.pdf) · [Executed notebook](audit_notebook.ipynb)

## What was run

Course-specified `finrobot_equity`, pinned to commit2717499b8e30f242640af08c4ad9afd1113c2d45, with actual GPT-4.1 (`gpt-4.1-2025-04-14`) typed-agent outputs. Formal Technology universe:NVDA,AMD,INTC,AAPL,MSFT; GOOGL also completes the starter set. Two peers per company. Financial sources:FMP stable API,SEC CompanyFacts,original NVIDIA annual report and AMD10K. Data/experiment cutoff:October11,2026.

Six baseline companies have eight actual section outputs and generated HTML reports; NVDA improved report and three frozen-input NVDA runs are included. No API credentials are included. AI assistance was used in code execution,evaluation scaffolding and writing; methodological limits are disclosed.

## Results and limits

- 20 NVDA/AMD metric comparisons:16 matches,4 EBITDA-definition differences needing reconciliation.
- Independent DCF arithmetic checks for five companies and45 complete sensitivity scenarios. These are mechanical scenarios,not investment recommendations.
- Forecast-year validity8/15 to15/15. Separate68-sample SEC backtest MAPE13.74% to15.86%(worse).
- 24 company/section grounding rows,576 provisional claims:177 Supported,397 Unsupported,2 Contradicted,after five recorded review corrections. Model-based extraction/entailment was not fully independently adjudicated; Unsupported is not synonymous with false.
- Three identical-input NVDA runs. Risk-category order is imposed by the prompt; detailed wording differs. No explicit agent rating/target was emitted,so those variances remain undefined.
- Initial two-step runtime35.5–42.0seconds/company. Saved-response cost subtotal$1.0731; some diagnostics and one truncated evaluation response excluded. Rates:$2input,$0.50cachedinput,$8output per milliontokens.

## Direct report and analysis folders

The required files are also directly browsable in this repository:

- `output/{TICKER}/analysis/`: actual CSV/JSON/TXT financial inputs and agent text.
- `output/{TICKER}/report/`: complete generated HTML and chart images.
- `actual_runs/`: baseline, improved and repeated-input agent evidence.
- `audit/`: data, forecast/valuation, grounding and stability CSV/JSON tables.
- `code/`: reproducibility and evaluation scripts.

| Company | Generated report | Analysis |
| --- | --- | --- |
| NVDA | [HTML](output/NVDA/report/Combined_Equity_Report_NVDA.html) | [Files](output/NVDA/analysis/) |
| AMD | [HTML](output/AMD/report/Combined_Equity_Report_AMD.html) | [Files](output/AMD/analysis/) |
| INTC | [HTML](output/INTC/report/Combined_Equity_Report_INTC.html) | [Files](output/INTC/analysis/) |
| AAPL | [HTML](output/AAPL/report/Combined_Equity_Report_AAPL.html) | [Files](output/AAPL/analysis/) |
| MSFT | [HTML](output/MSFT/report/Combined_Equity_Report_MSFT.html) | [Files](output/MSFT/analysis/) |
| GOOGL | [HTML](output/GOOGL/report/Combined_Equity_Report_GOOGL.html) | [Files](output/GOOGL/analysis/) |

## Supplemental archives — full raw evidence is included

Large evidence folders are compressed to keep web-upload batches manageable. Extract once:

```sh
python unpack_evidence.py
```

- `reports_and_audits.zip`:actual_runs/{baseline,improved,evaluation,fmp_cache} with HTML/charts,CSV/JSON/TXT analysis folders,actual agent prompts/outputs; audit/ financial,forecast,valuation,grounding,stability,cost tables; charts/ and companion analysis/.
- `code_and_upstream.zip`:code/ reproducibility scripts; pinned upstream/ source; upstream_snapshot.tar.gz; course originals. Three patches are provided directly and in the evidence archives.
- `raw_source_data.zip`:data/sec/ raw CompanyFacts and original issuer filings,with lineage references in audit/.

For example,the NVIDIA report is `actual_runs/baseline/NVDA/report/Combined_Equity_Report_NVDA.html`; each company's inputs/outputs are under `actual_runs/baseline/TICKER/analysis/`. Full methods and constraints are in `CURRENT_STATUS.md` after extraction.

## Reproduction from this submission directory

Python3.12.14; the exact installed package versions are in requirements.lock.txt. Create a separate environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python code/prepare_upstream.py
```

Create a private config outside the submission folder,containing `[API_KEYS]` with `fmp_api_key`, `openai_api_key`, `openai_base_url=https://api.openai.com/v1`, and `openai_model=gpt-4.1`. Never commit that config. Run:

```sh
.venv/bin/python code/run_data_only.py --config /ABSOLUTE/PRIVATE/config.ini
.venv/bin/python code/analyze_fmp_actual.py
.venv/bin/python code/run_actual_pipeline.py --config /ABSOLUTE/PRIVATE/config.ini
.venv/bin/python code/run_actual_pipeline.py --config /ABSOLUTE/PRIVATE/config.ini --ticker GOOGL
.venv/bin/python code/run_actual_pipeline.py --config /ABSOLUTE/PRIVATE/config.ini --ticker NVDA --improved
.venv/bin/python code/evaluate_agents.py /ABSOLUTE/PRIVATE/config.ini
.venv/bin/python code/review_grounding.py
.venv/bin/python code/summarize_actual.py
```

The last script builds the final PDF through build_actual_pdf.py. Start Jupyter in this directory and execute audit_notebook.ipynb with the isolated environment's kernel. Cached responses and evaluation files avoid repeated calls; remove caches only when intentionally requesting refreshed data or a new experiment. Expected FMP news HTTP402 is disclosed,not replaced by fictitious articles. Fresh financial data may change results.

Dependency check,notebook clean execution,four forecast tests,independent calculations,three clean-applying patches,and exact-key scans passed. Cached repeated-input evidence supports the reported experiment; a completely fresh paid end-to-end reproduction on another machine has not been performed.

[Course assignment](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRobot). Deadline specified asOctober12,2026; exact cutoff time/timezone unconfirmed. This submission reports unresolved audit findings rather than concealing them.
