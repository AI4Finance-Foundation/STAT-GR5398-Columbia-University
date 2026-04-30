

# FinRobot Equity Research Enhancement: Frontier LLMs, Model-Role Assignment, and Claude-Inspired Research Modules

## Medium Link 
https://medium.com/@hg2783/enhancing-finrobot-equity-research-with-frontier-llms-model-role-assignment-and-claude-inspired-d581ea8c4a17
## Output 
STAT-GR5398-Spring-2026/FinRobot-Equity-Research-Group/Assignment2/submissions/Assignment2_HanGao_hg2783/finrobot_equity/core/src/output

## Overview

This project extends the FinRobot equity research workflow by upgrading the base model configuration, adding new analyst-style research modules, and testing different model-role assignments across a multi-agent report generation pipeline.

The original FinRobot pipeline provides a strong foundation for automated equity research report generation. It can fetch financial data, process historical financial metrics, generate forecasts, create charts, and assemble professional multi-page HTML/PDF-style equity research reports.

This enhancement makes the workflow more modular, model-provider agnostic, and research-oriented. The pipeline now supports frontier LLM providers including OpenAI GPT-4.1, Claude, and Gemini. It also adds structured research modules for SEC filing analysis, revenue-driver analysis, earnings analysis, sensitivity analysis, catalyst analysis, enhanced news, retail sentiment, and credit/cashflow metrics.

The final selected setup is a hybrid workflow:

```text
Hybrid Report = GPT-4.1 main analyst + Claude critic + deterministic research modules
```

GPT-4.1 is used for coherent report structure and final writing. Claude is used as the critic/reviewer for valuation, risk, and competitor analysis. Gemini is used as an alternative frontier-model baseline.

---

## Project Goals

The project focuses on five main goals:

1. Replace the original base model setup with updated frontier LLMs.
2. Test different model-role assignments across a multi-agent research workflow.
3. Add Claude-inspired equity research modules.
4. Refine prompts and module outputs to produce more analyst-style reasoning.
5. Compare model outputs and select the best final report configuration.

---

## Architecture

```text
finrobot_equity/
├── core/                              # Analysis engine
│   ├── config/
│   │   └── config.ini.template        # API key configuration template
│   ├── assets/                        # Static assets and logos
│   ├── src/
│   │   ├── generate_financial_analysis.py   # Step 1: Data fetching, analysis, module outputs
│   │   ├── create_equity_report.py          # Step 2: HTML report generation
│   │   ├── generate_pdf_report.py           # Step 3: PDF report generation (optional)
│   │   ├── Run.ipynb                        # Jupyter notebook demo
│   │   └── modules/                         # Core modules
│   │       ├── common_utils.py              # Config and API key management
│   │       ├── market_data_api.py           # FMP API client
│   │       ├── financial_data_processor.py  # Metrics extraction and forecasting
│   │       ├── text_generator_agents.py     # LLM text generation orchestrator
│   │       ├── chart_generator.py           # Financial chart rendering
│   │       ├── enhanced_chart_generator.py  # Advanced chart configurations
│   │       ├── html_renderer.py             # HTML report renderer
│   │       ├── html_template_professional.py# HTML template definitions
│   │       ├── report_data_loader.py        # Data loading utilities
│   │       ├── report_structure.py          # Report layout management
│   │       ├── enhanced_text_generator.py   # Advanced text processing
│   │       ├── sensitivity_analyzer.py      # Sensitivity analysis
│   │       ├── catalyst_analyzer.py         # Market catalyst identification
│   │       ├── news_integrator.py           # News data integration
│   │       ├── retail_sentiment_client.py   # Optional retail sentiment insights
│   │       ├── valuation_engine.py          # Valuation modeling
│   │       ├── pdf_generator.py             # PDF report generation
│   │       ├── professional_pdf_report.py   # Professional PDF templates
│   │       └── equity_agents/               # AI agents and research modules
│   │           ├── agent_manager.py
│   │           ├── tagline_agent.py
│   │           ├── company_overview_agent.py
│   │           ├── investment_overview_agent.py
│   │           ├── valuation_overview_agent.py
│   │           ├── risks_agent.py
│   │           ├── competitor_analysis_agent.py
│   │           ├── major_takeaways_agent.py
│   │           ├── news_summary_agent.py
│   │           ├── sec_filing_analyzer.py
│   │           ├── revenue_driver_analyzer.py
│   │           └── earnings_analyzer.py
│   └── tests/
│       ├── test_generate_report.py
│       └── test_modules.py
│
└── web_app/                           # Web interface (FastAPI)
    ├── main.py
    ├── auth.py
    ├── admin_routes.py
    ├── database/
    ├── middleware/
    ├── templates/
    ├── static/
    └── data/
```

---

## Pipeline

The report generation follows a two-step workflow:

```text
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  generate_financial_analysis.py │     │  create_equity_report.py     │
│                                 │     │                              │
│  1. Fetch data from FMP API     │────>│  1. Load analysis outputs    │
│  2. Process financial metrics   │     │  2. Auto-fetch market data   │
│  3. Generate forecasts          │     │  3. Generate charts          │
│  4. Run peer comparison         │     │  4. Inject module sections   │
│  5. Run research modules        │     │  5. Render HTML report       │
│  6. Generate AI text sections   │     │                              │
│                                 │     │                              │
│  Output: CSV + JSON + MD + TXT  │     │  Output: HTML report         │
└─────────────────────────────────┘     └──────────────────────────────┘
```

The first step produces structured financial outputs and module artifacts. The second step assembles those artifacts into a professional equity research report.

---

## Claude-Inspired Research Enhancements

The most important improvement is the addition of structured research modules inspired by Claude-style equity research workflows. Instead of relying only on free-form text generation, the pipeline now produces intermediate JSON and Markdown artifacts that are injected into the final HTML report.

| Enhancement | Output Artifact | Purpose |
|---|---|---|
| SEC Filing Highlights | `sec_filing_analysis.json`, `sec_filing_summary.md` | Extract filing-based risk signals, catalyst signals, and management discussion themes |
| Revenue Driver Analysis | `revenue_driver_analysis.json` | Link revenue growth assumptions to product, services, pricing, demand, and strategic drivers |
| Earnings Analysis | `earnings_analysis.json`, `earnings_summary.md` | Create a post-earnings style read-through around revenue, EPS, EBITDA, margins, and watch items |
| Sensitivity Analysis | `sensitivity_analysis.json` | Evaluate how changes in assumptions may affect the investment outlook |
| Catalyst Analysis | `catalyst_analysis.json` | Identify positive, negative, and neutral catalysts |
| Enhanced News | `enhanced_news.json` | Summarize recent news impact and connect it to the investment thesis |
| Retail Sentiment | `retail_sentiment.json` | Add market-facing sentiment signals such as buzz, bullishness, and platform divergence |
| Credit & Cashflow Metrics | formatted ratio tables | Add liquidity, leverage, profitability, and cashflow depth to the report |

These modules make the report more transparent and reproducible because each new section is tied to a concrete output artifact.

---

## Role-Aware Prompt Engineering

This project treats prompt engineering as a workflow-level design problem rather than a single-prompt editing task. The pipeline uses role-aware prompting, section-specific objectives, and evidence-grounded output constraints.

| Role | Model / Component | Objective |
|---|---|---|
| Main analyst | GPT-4.1 | Build a coherent investment narrative and final report structure |
| Critic / reviewer | Claude | Challenge assumptions, identify downside risks, critique valuation, and improve analytical sharpness |
| Alternative draft generator | Gemini | Provide a comparison baseline and alternative explanatory wording |
| Structured research modules | Rule-based and LLM-assisted modules | Generate consistent JSON/Markdown artifacts for earnings, filings, revenue drivers, catalysts, news, and sentiment |

The prompt and output refinement changed the report from generic summaries into analyst-style reasoning. For example, instead of writing:

```text
EBITDA margin is expected to improve.
```

The refined earnings module produces:

```text
EBITDA margin is projected to expand from 34.7% in 2025A to 44.8% in 2027E; the magnitude of this step-up should be treated as an important execution assumption rather than a mechanical extrapolation.
```

This wording is more useful because it cites the forecast, explains why the margin change matters, and highlights execution risk.

---

## Model-Role Assignment Experiment

The project tested both unified and mixed-model setups.

| Setup | Main Analyst | Critic / Reviewer | Final Writer | Purpose |
|---|---|---|---|---|
| Unified GPT-4.1 | GPT-4.1 | GPT-4.1 | GPT-4.1 | Test whether one strong model can handle the full report consistently |
| Unified Claude | Claude | Claude | Claude | Test whether Claude alone provides deeper reasoning and critique |
| Unified Gemini | Gemini | Gemini | Gemini | Test Gemini as an alternative frontier model baseline |
| Mixed GPT + Claude | GPT-4.1 | Claude | GPT-4.1 | Combine GPT’s structure with Claude’s critical judgment |
| Mixed GPT + Gemini | GPT-4.1 | Gemini | GPT-4.1 | Test whether Gemini adds useful alternative wording |

The final selected configuration is the mixed GPT-4.1 + Claude setup.

---

## Results Summary

| Model Setup | Strengths | Weaknesses | Best Use |
|---|---|---|---|
| GPT-4.1 standalone | Cleanest structure, strongest consistency, best final report readiness | Sometimes less critical than Claude | Main analyst / final writer |
| Claude standalone | Strongest valuation critique, risk analysis, and competitor reasoning | Slightly less clean as a full report generator | Critic / reviewer |
| Gemini standalone | Detailed explanatory output and useful alternative wording | More generic, less selective, and less critical | Baseline / backup draft |
| Hybrid GPT + Claude | Best balance of structure and critical reasoning | Requires merging outputs | Final selected setup |

Qualitative scorecard:

| Setup | Reasoning Quality | Report Consistency | Analytical Depth | Formatting Stability | Overall |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 standalone | 9.0 | 9.1 | 8.6 | 9.2 | 9.0 |
| Claude standalone | 9.2 | 8.4 | 9.3 | 8.5 | 8.9 |
| Gemini standalone | 8.2 | 8.0 | 7.8 | 7.8 | 7.9 |
| Hybrid GPT + Claude | 9.3 | 9.0 | 9.4 | 9.0 | 9.3 |

The hybrid setup performed best because it combined GPT-4.1’s structure and Claude’s critical judgment.

---

## Final Hybrid Report Composition

| Final Report Section | Source |
|---|---|
| Executive Summary | GPT-4.1 |
| Company Overview | GPT-4.1 |
| Investment Update | GPT-4.1 with Claude-style critique |
| Earnings Analysis | Deterministic earnings module |
| SEC Filing Highlights | SEC filing module |
| Revenue Driver Analysis | Revenue driver module |
| Valuation | Claude |
| Risks | Claude |
| Competitor Analysis | Claude |
| Recent News Summary | GPT-4.1 |
| Sensitivity Analysis | Structured module output |
| Catalyst Analysis | Structured module output |
| Enhanced News | Structured module output |
| Retail Sentiment | Structured module output |
| Financial Tables and Charts | Structured CSV outputs |

---

## Quick Start

### 1. Configure API Keys

```bash
cp finrobot_equity/core/config/config.ini.template finrobot_equity/core/config/config.ini
```

Edit `config.ini` with your keys:

```ini
[API_KEYS]
fmp_api_key = YOUR_FMP_API_KEY
openai_api_key = YOUR_OPENAI_API_KEY
anthropic_api_key = YOUR_ANTHROPIC_API_KEY
gemini_api_key = YOUR_GEMINI_API_KEY
adanos_api_key = YOUR_ADANOS_API_KEY  # Optional: Retail Sentiment Insights
openai_base_url = OPTIONAL_OPENAI_BASE_URL
```

If `adanos_api_key` is configured, the report pipeline adds an optional Retail Sentiment Insights layer to the equity research output.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the core dependencies manually:

```bash
pip install pandas numpy requests matplotlib openai anthropic google-genai
```

### 3. Change to the Source Directory

```bash
cd finrobot_equity/core/src
```

---

## Run Analysis

Example for Apple Inc. with GPT-4.1:

```bash
python generate_financial_analysis.py \
  --company-ticker AAPL \
  --company-name "Apple Inc." \
  --peer-tickers MSFT AMZN GOOGL META \
  --enable-sec-filing-analysis \
  --enable-revenue-driver-analysis \
  --enable-sensitivity-analysis \
  --enable-catalyst-analysis \
  --enable-enhanced-news \
  --generate-text-sections \
  --llm-provider openai \
  --output-dir ./output/AAPL/gpt41_analysis \
  --text-output-dir ./output/AAPL/gpt41_analysis
```

Claude run:

```bash
python generate_financial_analysis.py \
  --company-ticker AAPL \
  --company-name "Apple Inc." \
  --peer-tickers MSFT AMZN GOOGL META \
  --enable-sec-filing-analysis \
  --enable-revenue-driver-analysis \
  --enable-sensitivity-analysis \
  --enable-catalyst-analysis \
  --enable-enhanced-news \
  --generate-text-sections \
  --llm-provider anthropic \
  --output-dir ./output/AAPL/claude_analysis \
  --text-output-dir ./output/AAPL/claude_analysis
```

Gemini run:

```bash
python generate_financial_analysis.py \
  --company-ticker AAPL \
  --company-name "Apple Inc." \
  --peer-tickers MSFT AMZN GOOGL META \
  --enable-sec-filing-analysis \
  --enable-revenue-driver-analysis \
  --enable-sensitivity-analysis \
  --enable-catalyst-analysis \
  --enable-enhanced-news \
  --generate-text-sections \
  --llm-provider gemini \
  --llm-model gemini-2.5-flash \
  --output-dir ./output/AAPL/gemini_analysis \
  --text-output-dir ./output/AAPL/gemini_analysis
```

---

## Generate Equity Report

Example for GPT-4.1:

```bash
python create_equity_report.py \
  --company-ticker AAPL \
  --company-name "Apple Inc." \
  --output-dir ./output/AAPL/gpt41_report \
  --analysis-csv ./output/AAPL/gpt41_analysis/financial_metrics_and_forecasts.csv \
  --ratios-csv ./output/AAPL/gpt41_analysis/ratios_raw_data.csv \
  --peer-ebitda-csv ./output/AAPL/gpt41_analysis/peer_ebitda_comparison.csv \
  --peer-ev-ebitda-csv ./output/AAPL/gpt41_analysis/peer_ev_ebitda_comparison.csv \
  --tagline-file ./output/AAPL/gpt41_analysis/tagline.txt \
  --company-overview-file ./output/AAPL/gpt41_analysis/company_overview.txt \
  --investment-overview-file ./output/AAPL/gpt41_analysis/investment_overview.txt \
  --earnings-summary-file ./output/AAPL/gpt41_analysis/earnings_summary.md \
  --valuation-overview-file ./output/AAPL/gpt41_analysis/valuation_overview.txt \
  --risks-file ./output/AAPL/gpt41_analysis/risks.txt \
  --competitor-analysis-file ./output/AAPL/gpt41_analysis/competitor_analysis.txt \
  --major-takeaways-file ./output/AAPL/gpt41_analysis/major_takeaways.txt \
  --news-summary-file ./output/AAPL/gpt41_analysis/news_summary.txt \
  --sensitivity-analysis-file ./output/AAPL/gpt41_analysis/sensitivity_analysis.json \
  --catalyst-analysis-file ./output/AAPL/gpt41_analysis/catalyst_analysis.json \
  --enhanced-news-file ./output/AAPL/gpt41_analysis/enhanced_news.json
```

Key outputs:

```text
output/AAPL/gpt41_report/Combined_Equity_Report_AAPL.html
output/AAPL/gpt41_report/Professional_Equity_Report_AAPL.html
```

---

## Hybrid Report Workflow

Create a hybrid analysis folder:

```bash
mkdir -p ./output/AAPL/hybrid_analysis
cp ./output/AAPL/gpt41_analysis/* ./output/AAPL/hybrid_analysis/
```

Replace selected high-judgment sections with Claude outputs:

```bash
cp ./output/AAPL/claude_analysis/valuation_overview.txt ./output/AAPL/hybrid_analysis/valuation_overview.txt
cp ./output/AAPL/claude_analysis/risks.txt ./output/AAPL/hybrid_analysis/risks.txt
cp ./output/AAPL/claude_analysis/competitor_analysis.txt ./output/AAPL/hybrid_analysis/competitor_analysis.txt
```

Generate the hybrid report:

```bash
python create_equity_report.py \
  --company-ticker AAPL \
  --company-name "Apple Inc." \
  --output-dir ./output/AAPL/hybrid_report \
  --analysis-csv ./output/AAPL/hybrid_analysis/financial_metrics_and_forecasts.csv \
  --ratios-csv ./output/AAPL/hybrid_analysis/ratios_raw_data.csv \
  --peer-ebitda-csv ./output/AAPL/hybrid_analysis/peer_ebitda_comparison.csv \
  --peer-ev-ebitda-csv ./output/AAPL/hybrid_analysis/peer_ev_ebitda_comparison.csv \
  --tagline-file ./output/AAPL/hybrid_analysis/tagline.txt \
  --company-overview-file ./output/AAPL/hybrid_analysis/company_overview.txt \
  --investment-overview-file ./output/AAPL/hybrid_analysis/investment_overview.txt \
  --earnings-summary-file ./output/AAPL/hybrid_analysis/earnings_summary.md \
  --valuation-overview-file ./output/AAPL/hybrid_analysis/valuation_overview.txt \
  --risks-file ./output/AAPL/hybrid_analysis/risks.txt \
  --competitor-analysis-file ./output/AAPL/hybrid_analysis/competitor_analysis.txt \
  --major-takeaways-file ./output/AAPL/hybrid_analysis/major_takeaways.txt \
  --news-summary-file ./output/AAPL/hybrid_analysis/news_summary.txt \
  --sensitivity-analysis-file ./output/AAPL/hybrid_analysis/sensitivity_analysis.json \
  --catalyst-analysis-file ./output/AAPL/hybrid_analysis/catalyst_analysis.json \
  --enhanced-news-file ./output/AAPL/hybrid_analysis/enhanced_news.json
```

Open the report:

```bash
open ./output/AAPL/hybrid_report/Combined_Equity_Report_AAPL.html
```

---

## Extension to Additional Companies

The same workflow can be extended to additional large-cap technology companies, including:

- Microsoft Corporation (MSFT)
- Amazon.com, Inc. (AMZN)
- Alphabet Inc. (GOOGL)
- Meta Platforms, Inc. (META)

This tests whether the enhanced FinRobot workflow and the GPT-4.1 + Claude hybrid setup generalize beyond a single company.

---

## Deployment Commands

| Command | Description |
|:---|:---|
| `./deploy.sh start` | Start the web application |
| `./deploy.sh stop` | Stop the application |
| `./deploy.sh restart` | Restart the application |
| `./deploy.sh status` | Check running status and recent logs |
| `./deploy.sh install` | Install or update dependencies only |

### Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `GITHUB_CLIENT_ID` | — | GitHub OAuth client ID |
| `GITHUB_CLIENT_SECRET` | — | GitHub OAuth client secret |
| `FINROBOT_ADMIN_EMAIL` | `admin@finrobot.com` | Initial admin email |
| `FINROBOT_ADMIN_PASSWORD` | random | Initial admin password |
| `WEB_HOST` | `127.0.0.1` | Server bind address |
| `WEB_PORT` | `8001` | Server port |

---

## API Dependencies

| Service | Required | Purpose |
|:---|:---|:---|
| Financial Modeling Prep | Yes | Financial data, market metrics, peer comparison |
| OpenAI | Yes for GPT-4.1 runs | AI-powered report text generation |
| Anthropic Claude | Yes for Claude runs | Critic/reviewer analysis and standalone Claude report |
| Google Gemini | Yes for Gemini runs | Alternative frontier model baseline |
| Adanos Finance API | No | Optional retail sentiment insights |

---

## Lessons Learned

1. **Model choice matters, but model-role assignment matters more.** GPT-4.1 was best at structure and final writing, while Claude was better at critique and risk analysis.
2. **Structured modules improve reliability.** Intermediate JSON and Markdown outputs made the pipeline easier to debug, reproduce, and evaluate.
3. **Prompt engineering should be workflow-level.** Role-aware prompting, section-specific objectives, and evidence-grounded output constraints improved output quality.
4. **Automated financial research needs both generation and critique.** A fluent model may not be skeptical enough, while a strong critic may not produce the cleanest final report. Combining both improves output quality.

---

## Conclusion

This project enhances the FinRobot equity research pipeline by replacing the original base model setup, adding frontier LLM provider support, implementing Claude-inspired research modules, and testing unified versus mixed-model report generation.

The best standalone model was GPT-4.1 because it produced the most coherent and submission-ready report. Claude was the strongest critic because it provided deeper valuation, risk, and competitor analysis. Gemini was useful as a comparison baseline but was less suitable for the final report.

The best-performing setup was the hybrid GPT-4.1 + Claude configuration. GPT-4.1 served as the main analyst and final writer, while Claude acted as the critic/reviewer for valuation, risks, and competitor analysis. Deterministic modules handled earnings analysis, SEC filing highlights, revenue-driver analysis, sensitivity analysis, catalyst tracking, enhanced news, retail sentiment, and financial tables.

Overall, the experiment shows that role-specialized model assignment can improve AI-generated financial research. The final hybrid workflow produced a stronger balance of report consistency, analytical depth, financial grounding, and investment judgment than any single-model setup.

---

## License

Apache 2.0. See `LICENSE` for details.
