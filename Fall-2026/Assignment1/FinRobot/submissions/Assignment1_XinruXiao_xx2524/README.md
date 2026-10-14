# Assignment 1 - FinRobot Equity Research AI Agent Track

**Student:** Xinru Xiao  
**Folder placeholder:** `Assignment1_XinruXiao_xx2524`
**Model:** `gpt-4.1`  
**Coverage universe:** Technology sector - NVDA, AMD, INTC, AAPL, MSFT.

## Contents

- `reports/`: professional HTML reports (PDF versions are generated alongside them in the final package)
- `artifacts/{TICKER}/analysis/`: CSV/JSON/TXT evidence used to produce each report
- `audits/data_audit.csv`: 22 filing checks across AAPL and MSFT
- `audits/forecast_valuation_audit.csv`: forecast/valuation assumptions and reconciliation
- `audits/grounding_audit.csv`: claim-level Supported/Unsupported/Contradicted audit for all 8 agent sections across NVDA, AAPL and MSFT
- `audits/grounding_summary.csv`: section-level grounding rates
- `audits/stability_test.csv`: three nominally identical NVDA runs
- `audits/nvda_valuation_sensitivity.csv`: valuation target-price sensitivity grid
- `improvement/finrobot_core_improvement.patch`: clean source diff against upstream
- `improvement/before_after_evaluation.csv`: numerical upstream-vs-modified valuation comparison
- `improvement/valuation_reproduction.xlsx`: independent NVDA DCF/valuation reproduction
- `Research_Report.pdf`: research report
- `Medium_Blog_Draft.md`: blog draft

## Environment

```bash
uv venv --python 3.11 .venv-equity
source .venv-equity/bin/activate
uv pip install -r finrobot_equity/requirements.txt
```

`config.ini` is intentionally excluded. Never commit API keys.

## Exact Step 1 commands

```bash
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker NVDA --company-name "NVIDIA Corporation" --config-file finrobot_equity/core/config/config.ini --peer-tickers AMD INTC --generate-text-sections --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker AMD --company-name "Advanced Micro Devices, Inc." --config-file finrobot_equity/core/config/config.ini --peer-tickers NVDA INTC --generate-text-sections --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker INTC --company-name "Intel Corporation" --config-file finrobot_equity/core/config/config.ini --peer-tickers NVDA AMD --generate-text-sections --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker AAPL --company-name "Apple Inc." --config-file finrobot_equity/core/config/config.ini --peer-tickers MSFT CSCO --generate-text-sections --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
python finrobot_equity/core/src/generate_financial_analysis.py --company-ticker MSFT --company-name "Microsoft Corporation" --config-file finrobot_equity/core/config/config.ini --peer-tickers AAPL CSCO --generate-text-sections --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
```

## Step 2 template

Run for each ticker with its corresponding company name and output paths:

```bash
python finrobot_equity/core/src/create_equity_report.py --company-ticker TICKER --company-name "COMPANY" --analysis-csv ./output/TICKER/analysis/financial_metrics_and_forecasts.csv --ratios-csv ./output/TICKER/analysis/ratios_raw_data.csv --peer-ebitda-csv ./output/TICKER/analysis/peer_ebitda_comparison.csv --peer-ev-ebitda-csv ./output/TICKER/analysis/peer_ev_ebitda_comparison.csv --tagline-file ./output/TICKER/analysis/tagline.txt --company-overview-file ./output/TICKER/analysis/company_overview.txt --investment-overview-file ./output/TICKER/analysis/investment_overview.txt --valuation-overview-file ./output/TICKER/analysis/valuation_overview.txt --risks-file ./output/TICKER/analysis/risks.txt --competitor-analysis-file ./output/TICKER/analysis/competitor_analysis.txt --major-takeaways-file ./output/TICKER/analysis/major_takeaways.txt --news-summary-file ./output/TICKER/analysis/news_summary.txt --enable-valuation-analysis --sensitivity-analysis-file ./output/TICKER/analysis/sensitivity_analysis.json --catalyst-analysis-file ./output/TICKER/analysis/catalyst_analysis.json
```

## API cost and runtime disclosure

The provided pipeline artifacts did **not** log OpenAI token usage/cost or end-to-end wall-clock runtime for the five baseline reports. Exact dollar spend and runtime therefore cannot be reconstructed without inventing values. The baseline used 8 GPT-4.1 narrative-generation calls per company (40 calls for five companies); the two additional NVDA stability reruns added 16 calls, for 56 narrative calls in the saved experiment. FMP cost is plan/subscription dependent.

This missing telemetry is treated as a limitation rather than estimated as fact.

## Stability test note

Run 1 produced a valid live rating/target. Runs 2 and 3 used the same command but FMP returned HTTP 429 on live market endpoints, causing `N/A` rating and `$0` target/valuation sentinels. The top three risk narratives were identical across all three runs. This is a reproducibility failure caused by an uncontrolled live-data dependency, not LLM prose variation.

## Medium Blog

https://medium.com/@xx2524/making-an-ai-equity-research-agent-more-reliable-auditing-and-rebuilding-finrobots-valuation-9afaabff4a75
