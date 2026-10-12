from pathlib import Path
import nbformat
root=Path(__file__).resolve().parents[1]
n=nbformat.v4.new_notebook();md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
n.cells=[md('''# FinRobot Assignment 1: reproducible audit companion
## tl;dr
This notebook executes the actual upstream deterministic modules with SEC-normalized inputs. It is **not** an FMP or eight-agent experiment. Missing personal API credentials block the full course pipeline.

Observed results: 68 rolling-origin forecasts; MAPE 13.74% for the 5% baseline versus 15.86% for two-year CAGR. Forecast-year validity improves from 60% to 100%. Twenty original-filing checks match. No agent grounding or stability result is claimed.'''),
md('''## Context & Methods
### Key Assumptions
Data cutoff: 2026-10-11. Annual periods, USD amounts in dollars, fiscal-year labels based on period-end year. Trailing two-year CAGR capped at +/-50%; flat margins. Backtest inputs are filtered to filings available at each origin. Upstream mechanical DCF uses 10% WACC, 2.5% terminal growth and synthetic net debt=10% EV. CFO-capex is not validated FCFF. This notebook evaluates arithmetic and source traceability, not tradeable stock returns.'''),
code('''from pathlib import Path
import sys, subprocess, json
import pandas as pd
ROOT = Path.cwd()
assert (ROOT / 'data/sec/NVDA.json').exists(), 'Run from the evidence package directory'
subprocess.run([sys.executable, 'code/prepare_upstream.py'], check=True)
UPSTREAM = ROOT / 'upstream'
print('Python:', sys.version.split()[0])
print('Pinned upstream:', json.loads((ROOT/'upstream_version.json').read_text())['commit'])'''),
md('## Data\nSEC Company Facts snapshots and original NVIDIA/AMD issuer-hosted annual filings are retained. Per-fact lineage, accessions and source links are in audit/sec_fact_lineage.csv. Hashes in run_manifest.json identify the exact snapshots.'),
code('''sys.path.insert(0,str(ROOT/'code'))
from analyze_sec import main
main(UPSTREAM)'''),
code('''subprocess.run([sys.executable, 'code/crosscheck_filings.py'], check=True)
checks = pd.read_csv(ROOT/'audit/filing_crosscheck.csv')
assert len(checks) == 20 and checks['delta'].abs().max() < .01
checks[['ticker','metric','sec_value','filing_value','delta']]'''),
md('## Results\n### Coverage and sources'),
code('''coverage = pd.read_csv(ROOT/'audit/coverage_summary.csv')
coverage[['ticker','period_end','revenue_bn','revenue_yoy','net_margin','fcf_bn']]'''),
md('### Same-company forecast comparison'),
code('''scorecard = pd.read_csv(ROOT/'audit/forecast_scorecard.csv')
backtest = pd.read_csv(ROOT/'audit/forecast_backtest.csv')
assert (backtest.max_training_filed <= backtest.origin_date).all()
assert (backtest.target_filed > backtest.origin_date).all()
print('Rolling-origin samples:', len(backtest))
print('Pooled MAPE:', backtest[['baseline_ape','improved_ape']].mean().to_dict())
scorecard'''),
md('### Independently reproduce valuation arithmetic'),
code('''valuation = pd.read_csv(ROOT/'audit/valuation_audit.csv')
assert valuation.delta.abs().max() < 1e-8
valuation[['ticker','engine_price','recomputed_price','delta','terminal_share']]'''),
md('### Forecast validity and unit tests'),
code('''subprocess.run([sys.executable,'code/test_forecasts.py',str(UPSTREAM)],check=True)
rows=[]
for _,c in coverage.iterrows():
 for method,file in [('baseline','upstream_default_forecasts.csv'),('improved','financial_metrics_and_forecasts.csv')]:
  frame=pd.read_csv(ROOT/'analysis'/c.ticker/file)
  rows.extend({'ticker':c.ticker,'method':method,'forecast_year':y,'actual_end':c.period_end,'strictly_future':int(y[:-1])>int(c.period_end[:4])} for y in frame.columns if y.endswith('E'))
validity=pd.DataFrame(rows)
validity.to_csv(ROOT/'audit/temporal_validity.csv',index=False)
validity.groupby('method').strictly_future.mean()'''),
md('### Render source-backed findings'),
code('''subprocess.run([sys.executable,'code/build_deliverables.py'],check=True)
from IPython.display import Image, display
display(Image(filename=str(ROOT/'charts/forecast_backtest.png')))'''),
md('''## Takeaways
The patch fixes forecast-year chronology, but the simple CAGR estimator is less accurate on the pooled historical sample. This does not establish forecasting superiority. The upstream forecast EBITDA definition omits R&D, and its synthetic net debt and negative-FCF perpetuity make DCF target prices unsuitable for an investment recommendation.

Full assignment blockers: FMP/OpenAI access, real baseline and improved agent reports, complete claim-level grounding, frozen-input three-run stability, Medium publication and final student submission. Blank values in status tables mean **not run**, not zero error. See README.md for the requirement-by-requirement checklist.''')]
n.metadata={'kernelspec':{'display_name':'FinRobot isolated audit','language':'python','name':'finrobot-audit'},'language_info':{'name':'python','version':'3.12'}}
nbformat.write(n,root/'audit_notebook.ipynb')
