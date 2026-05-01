# Five-Company Run Commands

This section documents the commands used to generate FinRobot reports for the five required companies:

- AAPL
- NVDA
- AMD
- INTC
- GOOGL

For each company, I generate:

1. FinRobot financial data and text sections.
2. Official FinRobot baseline HTML report.
3. Official FinRobot + ACR enhanced HTML report.

The ACR-enhanced version uses:

text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude


# We take an example for 1. AAPL — Apple Inc.

## Step 1: Generate FinRobot Financial Data and Text Sections
Run on the terminal:
```bash
python generate_financial_analysis.py --company-ticker AAPL --company-name "Apple Inc." --peer-tickers MSFT GOOGL --generate-text-sections
```


## Step 2: Generate Official FinRobot Baseline Report
Run on the terminal:
```bash
python create_equity_report.py --company-ticker AAPL --company-name "Apple Inc." --analysis-csv output\AAPL\analysis\financial_metrics_and_forecasts.csv --ratios-csv output\AAPL\analysis\ratios_raw_data.csv --tagline-file output\AAPL\analysis\tagline.txt --company-overview-file output\AAPL\analysis\company_overview.txt --investment-overview-file output\AAPL\analysis\investment_overview.txt --valuation-overview-file output\AAPL\analysis\valuation_overview.txt --risks-file output\AAPL\analysis\risks.txt --competitor-analysis-file output\AAPL\analysis\competitor_analysis.txt --major-takeaways-file output\AAPL\analysis\major_takeaways.txt --peer-ebitda-csv output\AAPL\analysis\peer_ebitda_comparison.csv --peer-ev-ebitda-csv output\AAPL\analysis\peer_ev_ebitda_comparison.csv --output-dir output\AAPL\official_baseline_report --enable-text-regeneration
```


## (New)Step 3: Generate FinRobot + ACR Enhanced Report
Run on the terminal:
```bash
python create_equity_report.py --company-ticker AAPL --company-name "Apple Inc." --analysis-csv output\AAPL\analysis\financial_metrics_and_forecasts.csv --ratios-csv output\AAPL\analysis\ratios_raw_data.csv --tagline-file output\AAPL\analysis\tagline.txt --company-overview-file output\AAPL\analysis\company_overview.txt --investment-overview-file output\AAPL\analysis\investment_overview.txt --valuation-overview-file output\AAPL\analysis\valuation_overview.txt --risks-file output\AAPL\analysis\risks.txt --competitor-analysis-file output\AAPL\analysis\competitor_analysis.txt --major-takeaways-file output\AAPL\analysis\major_takeaways.txt --peer-ebitda-csv output\AAPL\analysis\peer_ebitda_comparison.csv --peer-ev-ebitda-csv output\AAPL\analysis\peer_ev_ebitda_comparison.csv --output-dir output\AAPL\official_plus_acr_report --enable-text-regeneration --enable-assignment2-enhancement
```