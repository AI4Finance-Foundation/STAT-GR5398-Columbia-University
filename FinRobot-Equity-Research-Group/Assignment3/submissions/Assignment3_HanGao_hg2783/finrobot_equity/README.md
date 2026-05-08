# FinRobot Equity Research Agent

An AI-powered financial research agent that converts investor queries into structured stock research outputs using financial statements, SEC filings, valuation data, peer benchmarks, news signals, retail sentiment, and LLM reasoning.

This project extends the baseline FinRobot workflow with a Claude-style plugin architecture: reusable workflows, task-specific analysis modules, flexible data/tool connections, and role-specialized LLM agents for filing review, revenue-driver extraction, risk summarization, competitor benchmarking, and automated report generation.
## Medium 
https://medium.com/@hg2783/from-finrobot-equity-research-to-portfolio-construction-building-an-ai-assisted-technology-stock-9929a359920e

## Architecture

```
finrobot_equity/
├── core/                              # Analysis engine
│   ├── config/
│   │   └── config.ini.template        # API key configuration template
│   ├── assets/                        # Static assets (logos for PDF reports)
│   ├── src/
│   │   ├── generate_financial_analysis.py   # Step 1: Data fetching & analysis
│   │   ├── create_equity_report.py          # Step 2: HTML report generation
│   │   ├── generate_pdf_report.py           # Step 3: PDF report generation (optional)
│   │   ├── Run.ipynb                        # Jupyter notebook demo
│   │   └── modules/                         # Core modules
│   │       ├── common_utils.py              #   Config & API key management
│   │       ├── market_data_api.py           #   FMP API client
│   │       ├── financial_data_processor.py  #   Metrics extraction & forecasting
│   │       ├── text_generator_agents.py     #   LLM text generation orchestrator
│   │       ├── chart_generator.py           #   Financial chart rendering
│   │       ├── enhanced_chart_generator.py  #   Advanced chart configurations
│   │       ├── html_renderer.py             #   HTML report renderer
│   │       ├── html_template_professional.py#   HTML template definitions
│   │       ├── report_data_loader.py        #   Data loading utilities
│   │       ├── report_structure.py          #   Report layout management
│   │       ├── enhanced_text_generator.py   #   Advanced text processing
│   │       ├── sensitivity_analyzer.py      #   Sensitivity analysis
│   │       ├── catalyst_analyzer.py         #   Market catalyst identification
│   │       ├── news_integrator.py           #   News data integration
│   │       ├── retail_sentiment_client.py   #   Optional retail sentiment insights
│   │       ├── valuation_engine.py          #   Valuation modeling
│   │       ├── pdf_generator.py             #   PDF report generation
│   │       ├── professional_pdf_report.py   #   Professional PDF templates
│   │       └── equity_agents/               #   AI agents for report sections
│   │           ├── agent_manager.py         #     Agent orchestration
│   │           ├── tagline_agent.py         #     Executive tagline
│   │           ├── company_overview_agent.py#     Company overview
│   │           ├── investment_overview_agent.py  # Investment thesis
│   │           ├── valuation_overview_agent.py   # Valuation analysis
│   │           ├── risks_agent.py                # Risk assessment
│   │           ├── competitor_analysis_agent.py  # Competitive landscape
│   │           ├── major_takeaways_agent.py      # Key takeaways
│   │           └── news_summary_agent.py         # News summarization
│   └── tests/                         # Unit tests
│       ├── test_generate_report.py
│       └── test_modules.py
│
└── web_app/                           # Web interface (FastAPI)
    ├── main.py                        # Application entry point & API routes
    ├── auth.py                        # Authentication (local + GitHub OAuth)
    ├── admin_routes.py                # Admin API endpoints
    ├── database/                      # Database layer (SQLAlchemy + SQLite)
    │   ├── connection.py
    │   ├── models.py
    │   └── crud.py
    ├── middleware/
    │   └── request_logger.py          # Request logging
    ├── templates/                     # Jinja2 HTML templates
    ├── static/                        # CSS & images
    └── data/                          # Runtime data (auto-created, gitignored)
```

## Project Highlights

- **AI financial agent**: turns a company ticker or investor query into a complete equity research workflow.
- **Claude-style plugin workflow**: organizes the system into reusable, task-specific modules instead of one monolithic notebook.
- **Financial grounding**: combines FMP financial statements, valuation metrics, peer comparisons, market data, news, and optional retail sentiment.
- **LLM agent orchestration**: uses specialized agents for tagline, company overview, investment thesis, valuation, risks, competitors, takeaways, and news summary.
- **Portfolio signal evaluation**: compares quant-only, simple FinRobot, and risk-aware FinRobot signals using cumulative return, Sharpe ratio, max drawdown, active return, and alpha diagnostics.

## Pipeline

The equity research workflow follows a two-step pipeline:

```
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  generate_financial_analysis.py │     │  create_equity_report.py     │
│                                 │     │                              │
│  1. Fetch financial data        │────>│  1. Load analysis outputs    │
│  2. Process metrics/forecasts   │     │  2. Auto-fetch market data   │
│  3. Run peer comparison         │     │  3. Generate charts          │
│  4. Run plugin-style modules    │     │  4. Render HTML report       │
│  5. Generate LLM sections       │     │  5. Validate & regenerate    │
│                                 │     │                              │
│  Output: CSV + JSON + TXT       │     │  Output: 3-page HTML report  │
└─────────────────────────────────┘     └──────────────────────────────┘
```

## Evaluation Design

The project evaluates both the research-generation workflow and its investment usefulness.

1. **Report quality evaluation** compares different model-role setups, including unified GPT, unified Claude, unified Gemini, GPT + Claude, and GPT + Gemini. The evaluation focuses on analytical depth, financial grounding, risk coverage, consistency, and final report readiness.
2. **Portfolio signal evaluation** converts FinRobot outputs into stock-selection signals and compares quant-only, simple FinRobot, and risk-aware FinRobot strategies on cumulative return, Sharpe ratio, max drawdown, active return, alpha diagnostics, and score differentiation.

This makes the project more than a report generator: it tests whether LLM-enhanced research can produce usable investment signals.

## Quick Start

### 1. Configure API Keys

```bash
cp finrobot_equity/core/config/config.ini.template finrobot_equity/core/config/config.ini
```

Edit `config.ini` with your keys:

```ini
[API_KEYS]
fmp_api_key = YOUR_FMP_API_KEY          # https://financialmodelingprep.com/developer
openai_api_key = YOUR_OPENAI_API_KEY    # https://platform.openai.com/account/api-keys
adanos_api_key = YOUR_ADANOS_API_KEY    # Optional: enables Retail Sentiment Insights
```

If `adanos_api_key` is configured, the report pipeline adds an optional **Retail Sentiment Insights** layer to the equity research output. It supplements the existing news workflow with structured public-retail activity snapshots across Reddit, X.com, and Polymarket.

### 2. Deploy via Script

```bash
chmod +x deploy.sh
./deploy.sh start
```

Access the web interface at `http://127.0.0.1:8001`.

### 3. Or Run via Command Line

```bash
# Step 1: Financial analysis
python finrobot_equity/core/src/generate_financial_analysis.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --config-file finrobot_equity/core/config/config.ini \
    --peer-tickers AMD INTC \
    --generate-text-sections

# Step 2: Generate report
python finrobot_equity/core/src/create_equity_report.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --analysis-csv output/NVDA/analysis/financial_metrics_and_forecasts.csv \
    --ratios-csv output/NVDA/analysis/ratios_raw_data.csv \
    --tagline-file output/NVDA/analysis/tagline.txt \
    --company-overview-file output/NVDA/analysis/company_overview.txt \
    --investment-overview-file output/NVDA/analysis/investment_overview.txt \
    --valuation-overview-file output/NVDA/analysis/valuation_overview.txt \
    --risks-file output/NVDA/analysis/risks.txt \
    --competitor-analysis-file output/NVDA/analysis/competitor_analysis.txt \
    --major-takeaways-file output/NVDA/analysis/major_takeaways.txt \
    --peer-ev-ebitda-csv output/NVDA/analysis/peer_ev_ebitda_comparison.csv \
    --enable-text-regeneration \
    --config-file finrobot_equity/core/config/config.ini
```

## Deployment Commands

| Command | Description |
|:---|:---|
| `./deploy.sh start` | Start the web application (auto-installs dependencies) |
| `./deploy.sh stop` | Stop the application |
| `./deploy.sh restart` | Restart the application |
| `./deploy.sh status` | Check running status and recent logs |
| `./deploy.sh install` | Install/update dependencies only |

### Environment Variables

| Variable | Default | Description |
|:---|:---|:---|
| `GITHUB_CLIENT_ID` | — | GitHub OAuth client ID |
| `GITHUB_CLIENT_SECRET` | — | GitHub OAuth client secret |
| `FINROBOT_ADMIN_EMAIL` | `admin@finrobot.com` | Initial admin email |
| `FINROBOT_ADMIN_PASSWORD` | (random) | Initial admin password |
| `WEB_HOST` | `127.0.0.1` | Server bind address |
| `WEB_PORT` | `8001` | Server port |

## API Dependencies

| Service | Required | Purpose |
|:---|:---|:---|
| [Financial Modeling Prep](https://financialmodelingprep.com/developer) | Yes | Financial data, market metrics, peer comparison |
| [OpenAI](https://platform.openai.com/) | Yes | LLM reasoning, agent orchestration, and text generation |
| Adanos Finance API | No | Optional retail sentiment insights for Reddit, X.com, and Polymarket |

## Resume Summary

Built an AI financial agent that converts investor queries into stock research outputs using financial statements, SEC filings, valuation data, peer benchmarks, and LLM reasoning. Added a Claude-style plugin workflow for filing review, revenue-driver extraction, risk summaries, and report generation, then evaluated it through multi-model role tests and quant-only vs. FinRobot-enhanced portfolio signals.

## License

Apache 2.0 — See [LICENSE](../LICENSE) for details.

# FinRobot-Assisted Equity Research and Portfolio Strategy

This project builds an integrated AI-finance pipeline that connects **FinRobot equity research** with **systematic portfolio construction and backtesting**. The system starts from company-level financial data, SEC filings, news, peer benchmarks, and LLM-generated research outputs, then converts those outputs into stock-level signals for ranking, weighting, backtesting, and attribution analysis.

The project has two connected parts:

1. **Enhanced FinRobot equity research pipeline**: extends the baseline FinRobot workflow with frontier LLMs, structured research modules, and role-specialized report generation.
2. **FinRobot-assisted technology portfolio strategy**: converts FinRobot research outputs into company scores, combines them with quantitative signals, constructs portfolios, and evaluates performance against SPY, QQQ, XLK, All-Tech Equal Weight, and Random Top15 baselines.

The main goal is not only to generate equity research reports, but also to test whether AI-generated research can become **measurable investment signals**.

---

## 1. Project Objective

The final assignment asks for an integrated AI trading strategy that combines AI research, quantitative signals, and portfolio logic, with a required backtest from **2026-01-01 to 2026-04-30** against **SPY**.

This project follows a multi-signal scoring design:

```text
Financial data + filings + news
→ FinRobot hybrid equity research analysis
→ structured research artifacts
→ FinRobot-derived company scores
→ quantitative + AI-enhanced final score
→ portfolio construction
→ backtesting, benchmarking, attribution, and evaluation
```

The core research question is:

> Can FinRobot-generated equity research outputs be transformed into systematic stock-ranking and portfolio-weighting signals, and can those signals improve portfolio construction within a technology-focused investment universe?

---

## 2. System Architecture

```text
finrobot_equity/
├── core/                              # Analysis engine
│   ├── config/
│   │   └── config.ini.template        # API key configuration template
│   ├── assets/                        # Static assets for reports
│   ├── src/
│   │   ├── generate_financial_analysis.py   # Step 1: data fetching, analysis, modules, LLM sections
│   │   ├── create_equity_report.py          # Step 2: HTML equity report generation
│   │   ├── generate_pdf_report.py           # Step 3: PDF report generation (optional)
│   │   ├── Run.ipynb                        # Notebook demo
│   │   ├── modules/                         # Research and report-generation modules
│   │   │   ├── common_utils.py              # Config and API key management
│   │   │   ├── market_data_api.py           # FMP API client
│   │   │   ├── financial_data_processor.py  # Metrics extraction and forecasting
│   │   │   ├── text_generator_agents.py     # LLM text generation orchestrator
│   │   │   ├── sensitivity_analyzer.py      # Sensitivity analysis
│   │   │   ├── catalyst_analyzer.py         # Market catalyst identification
│   │   │   ├── news_integrator.py           # News integration
│   │   │   ├── retail_sentiment_client.py   # Optional retail sentiment insights
│   │   │   ├── valuation_engine.py          # Valuation modeling
│   │   │   └── equity_agents/               # Specialized report-section agents
│   │   └── portfolio_strategy/              # FinRobot-assisted portfolio strategy
│   │       ├── config.py                    # Universe, benchmark, dates, strategy settings
│   │       ├── data_loader.py               # Price data and monthly returns
│   │       ├── signals.py                   # Quantitative momentum / volatility score
│   │       ├── finrobot_score.py            # Simple and risk-aware FinRobot scores
│   │       ├── backtester.py                # Portfolio construction and return calculation
│   │       ├── analytics.py                 # Active return and alpha/beta diagnostics
│   │       ├── baselines.py                 # All-Tech EW and Random Top15 baselines
│   │       ├── attribution.py               # Holdings attribution and active-vs-All-Tech analysis
│   │       ├── plot_results.py              # Plot generation
│   │       └── run_tech_strategy.py         # Main strategy runner
│   └── tests/                              # Unit tests
│
└── web_app/                                # Optional FastAPI web interface
```

---

## 3. Enhanced FinRobot Equity Research Pipeline

The enhanced FinRobot pipeline follows a two-stage report-generation workflow:

```text
Financial Data + News + Filings
→ generate_financial_analysis.py
→ CSV / JSON / Markdown / TXT artifacts
→ create_equity_report.py
→ Professional HTML equity research report
```

### Stage 1: `generate_financial_analysis.py`

This script:

- fetches financial data from Financial Modeling Prep;
- processes historical financial statements and key ratios;
- generates forecasts and peer comparisons;
- runs structured research modules;
- generates LLM-based report sections;
- saves intermediate CSV, JSON, Markdown, and TXT artifacts.

### Stage 2: `create_equity_report.py`

This script:

- loads the analysis outputs;
- fetches market data;
- generates charts;
- injects text and structured sections;
- renders a professional multi-page HTML equity research report.

---

## 4. Frontier LLM and Role-Specialized Research Design

The baseline FinRobot workflow was extended with multiple frontier LLM configurations:

| Configuration | Provider / Model | Purpose |
|---|---|---|
| GPT-4.1 standalone | OpenAI GPT-4.1 | Single-model baseline for structure and final report quality |
| Claude standalone | Claude | Deeper valuation critique, risk framing, and competitor analysis |
| Gemini standalone | Gemini 2.5 Pro / Flash | Alternative frontier-model baseline |
| Hybrid GPT + Claude | GPT-4.1 + Claude | GPT-4.1 acts as main analyst/final writer; Claude acts as critic/reviewer |

The final research workflow uses a role-specialized design:

| Role | Model / Component | Objective |
|---|---|---|
| Main analyst | GPT-4.1 | Build coherent investment narrative and final report structure |
| Critic / reviewer | Claude | Challenge assumptions, identify downside risks, critique valuation and peers |
| Alternative draft generator | Gemini | Provide alternative drafts and comparison baseline |
| Structured modules | Rule-based and LLM-assisted modules | Generate consistent JSON, Markdown, and TXT research artifacts |

The hybrid setup produced the strongest balance of report consistency, analytical depth, financial grounding, and investment judgment.

---

## 5. Structured Research Modules

The enhanced FinRobot pipeline adds analyst-style research modules that generate reusable intermediate artifacts:

| Module | Output Artifacts | Purpose |
|---|---|---|
| SEC Filing Highlights | `sec_filing_analysis.json`, `sec_filing_summary.md` | Extract filing-based risks, catalysts, and management themes |
| Revenue Driver Analysis | `revenue_driver_analysis.json` | Link revenue growth assumptions to product, demand, pricing, and strategy |
| Earnings Analysis | `earnings_analysis.json`, `earnings_summary.md` | Generate post-earnings-style read-throughs |
| Sensitivity Analysis | `sensitivity_analysis.json` | Evaluate how assumption changes affect outlook |
| Catalyst Analysis | `catalyst_analysis.json` | Identify positive, negative, and neutral catalysts |
| Enhanced News | `enhanced_news.json` | Summarize recent news and investment impact |
| Retail Sentiment | `retail_sentiment.json` | Optional public-retail activity and sentiment snapshot |

These artifacts make the research workflow more transparent, debuggable, and reproducible.

---

## 6. Portfolio Strategy Design

The portfolio strategy converts FinRobot research outputs into portfolio signals.

### 6.1 Universe

The strategy uses a large-cap technology and growth universe, including:

```text
NVDA, MSFT, AAPL, AMZN, GOOGL, META, AVGO, AMD,
CRM, ORCL, ADBE, NFLX, TSLA, COST, NOW, PANW,
CRWD, SNOW, INTU, CSCO, QCOM, TXN, MU, AMAT,
LRCX, KLAC, ANET, PLTR, IBM, INTC
```

This is not a full-market stock-selection test. The goal is to test whether FinRobot-enhanced scoring improves ranking and weighting **within a technology-focused universe**.

### 6.2 Backtest Window

```text
2026-01-01 to 2026-04-30
```

This short window is treated as a pilot validation period, not proof of persistent long-term alpha.

---

## 7. Scoring Methods

The project compares three score modes.

### 7.1 Quant Only

```text
Final Score = 100% Quant Score
```

The quant score uses price-based signals such as momentum and low-volatility adjustment.

### 7.2 Simple FinRobot

```text
Final Score = 70% Quant Score + 30% Simple FinRobot Score
```

The simple FinRobot score rewards positive research-language terms and penalizes negative/risk terms. It achieved the best short-window numerical result, but later diagnostics showed that the score saturated at 1.0 for all stocks. Therefore, it should be interpreted as a naive text-score smoothing baseline rather than a true company-level ranking signal.

### 7.3 Risk-Aware FinRobot

```text
Final Score = 70% Quant Score + 30% Risk-Aware FinRobot Score
```

The risk-aware score is defined as:

```text
Risk-Aware FinRobot Score =
25% Growth Quality
+ 20% Competitive Position
+ 20% Earnings Quality
+ 15% Catalyst Strength
+ 10% Valuation Attractiveness
- 10% Business Risk
- 10% Valuation Risk
```

This score is more economically interpretable and preserves cross-sectional variation across companies.

---

## 8. Portfolio Construction

For each score mode, the project constructs three portfolios:

| Strategy | Description | Interpretation |
|---|---|---|
| `Top15_EqualWeight_Tech` | Select Top15 stocks by score and weight equally | Tests stock selection |
| `Top15_ScoreWeighted_Tech` | Select Top15 stocks and weight by final score | Main FinRobot-to-portfolio strategy |
| `50QQQ_50ScoreWeighted_Tech` | 50% QQQ + 50% Top15 ScoreWeighted | Practical core-satellite implementation |

The main project strategy is `Top15_ScoreWeighted_Tech` because it directly converts the final score into portfolio weights.

---

## 9. Benchmarks and Baselines

The strategy is evaluated against:

| Benchmark / Baseline | Purpose |
|---|---|
| SPY | Broad U.S. equity benchmark |
| QQQ | Growth / Nasdaq-heavy benchmark |
| XLK | Technology sector benchmark |
| All-Tech Universe EqualWeight | Controls for the selected technology universe itself |
| Random Top15 portfolios | Tests whether random concentration in tech could produce similar results |

The All-Tech and Random Top15 baselines are especially important because they address the concern that the strategy only worked because the technology universe was strong.

---

## 10. Main Backtest Results

### 10.1 Top15 ScoreWeighted Strategy

| Score Mode | Cumulative Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| `quant_only` | 23.48% | 1.3663 | -4.76% |
| `simple_finrobot` | **23.85%** | **1.3774** | **-4.65%** |
| `risk_aware_finrobot` | 23.64% | 1.3726 | -4.68% |

Both FinRobot-integrated methods modestly improved over the pure quantitative baseline. The improvement was small but directionally consistent: return improved, Sharpe improved, and max drawdown improved.

### 10.2 Benchmark Comparison

| Strategy / Benchmark | Cumulative Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| SPY | 4.63% | 0.7294 | -5.76% |
| QQQ | 7.83% | 0.8661 | -7.07% |
| XLK | 10.65% | 0.9201 | -7.52% |
| All-Tech Universe EqualWeight | 8.74% | 0.8240 | -8.97% |
| Top15 ScoreWeighted, simple | 23.85% | 1.3774 | -4.65% |
| Top15 ScoreWeighted, risk-aware | 23.64% | 1.3726 | -4.68% |

The strategy outperformed SPY, QQQ, XLK, and the All-Tech EqualWeight baseline during the pilot window.

### 10.3 Random Top15 Baseline

Across 500 random Top15 portfolios from the same technology universe:

| Metric | Random Median | Random p75 | Random p95 | Risk-Aware Top15 ScoreWeighted |
|---|---:|---:|---:|---:|
| Cumulative Return | 8.90% | 14.56% | 21.74% | **23.64%** |
| Sharpe | 0.842 | 1.159 | 1.483 | **1.373** |
| Max Drawdown | -8.90% | -7.84% | -6.23% | **-4.68%** |

The risk-aware Top15 ScoreWeighted strategy exceeded the 95th percentile of random Top15 portfolios in cumulative return and had better drawdown control than the random p95 drawdown.

---

## 11. Active Return and Alpha/Beta Diagnostics

### 11.1 Active Return versus SPY

| Strategy | Cumulative Active Return vs SPY | Monthly Win Rate | Information Ratio |
|---|---:|---:|---:|
| Simple Top15 ScoreWeighted | 19.22% | 75% | 1.687 |
| Risk-Aware Top15 ScoreWeighted | 18.97% | 75% | 1.679 |

### 11.2 Active Return versus QQQ

| Strategy | Cumulative Active Return vs QQQ | Monthly Win Rate | Information Ratio |
|---|---:|---:|---:|
| Simple Top15 ScoreWeighted | 16.22% | 75% | 1.900 |
| Risk-Aware Top15 ScoreWeighted | 16.03% | 75% | 1.892 |

### 11.3 Alpha/Beta Diagnostics

| Strategy | Benchmark | Annualized Alpha | Beta |
|---|---|---:|---:|
| Simple Top15 ScoreWeighted | SPY | 44.99% | 2.48 |
| Simple Top15 ScoreWeighted | QQQ | 33.17% | 1.79 |
| Risk-Aware Top15 ScoreWeighted | SPY | 44.28% | 2.47 |
| Risk-Aware Top15 ScoreWeighted | QQQ | 32.60% | 1.79 |

The alpha diagnostics are positive, but the beta exposure is high. The strategy should therefore be interpreted as a concentrated, high-beta active technology strategy rather than a defensive portfolio.

---

## 12. Holdings Attribution

Holdings attribution shows that performance was driven mainly by semiconductor, AI infrastructure, hardware, and networking names.

For the risk-aware version, the largest contributors were approximately:

| Ticker | Total Contribution | Average Weight | Interpretation |
|---|---:|---:|---|
| INTC | 6.91% | 6.29% | Large rebound contributor |
| AMD | 4.35% | 6.44% | Semiconductor / AI infrastructure |
| MU | 2.58% | 7.10% | Semiconductor memory exposure |
| AVGO | 1.83% | 6.64% | AI infrastructure / semiconductor |
| ANET | 1.61% | 6.10% | AI networking infrastructure |
| KLAC | 1.53% | 7.76% | Semiconductor equipment |
| AMZN | 1.51% | 5.94% | Cloud/platform exposure |
| NVDA | 1.18% | 6.46% | AI infrastructure |
| GOOGL | 1.02% | 7.33% | Cloud/platform |
| CSCO | 0.96% | 7.03% | Networking infrastructure |

The simple version slightly outperformed the risk-aware version mainly because it assigned a slightly higher weight to ANET, which performed strongly in April 2026. The performance gap between the two was primarily an allocation effect within the same selected basket, not a different stock-selection effect.

---

## 13. Key Findings

1. **The system successfully connects FinRobot research to portfolio construction.**  
   FinRobot outputs are converted into measurable company-level signals and used in ranking and weighting.

2. **FinRobot-integrated scores modestly improve the score-weighted strategy.**  
   Both `simple_finrobot` and `risk_aware_finrobot` improved cumulative return, Sharpe, and max drawdown versus `quant_only`.

3. **The strategy beats SPY, QQQ, XLK, All-Tech EqualWeight, and most Random Top15 portfolios.**  
   This suggests that results are not explained only by broad technology exposure or random concentration.

4. **The simple score achieved the best short-window numerical result but saturated at 1.0.**  
   It should be interpreted as a smoothing baseline rather than a true company-ranking signal.

5. **The risk-aware score is more defensible.**  
   It preserves cross-sectional variation and uses an economically interpretable structure.

6. **The strategy is high beta.**  
   Beta is approximately 2.5 versus SPY and 1.8 versus QQQ, so results should not be interpreted as low-risk alpha.

7. **Performance was concentrated in April 2026.**  
   The strongest contributions came from semiconductor and AI infrastructure names.

---

## 14. Limitations

- The backtest window is short: only 2026-01-01 to 2026-04-30.
- The universe is restricted to large-cap technology and growth stocks.
- The strategy has high beta versus both SPY and QQQ.
- The FinRobot scores are currently heuristic text-derived proxies.
- The simple FinRobot score saturated at 1.0 across all stocks.
- Some risk-aware score components also showed saturation.
- Much of the strategy performance came from the April 2026 rebound.

These limitations mean the project should be interpreted as successful pipeline validation and preliminary evidence, not proof of persistent long-term alpha.

---

## 15. Future Work

Future extensions include:

- replacing heuristic text scoring with structured LLM-generated company scores;
- extending the backtest over a longer historical period;
- testing broader universes beyond technology stocks;
- adding sector and theme exposure constraints;
- adding transaction cost sensitivity analysis;
- adding rolling-window robustness tests;
- adding structured LLM explanations for top holdings;
- generating an automated portfolio report from the strategy outputs.

---

## 16. How to Run

### 16.1 Configure API Keys

```bash
cp finrobot_equity/core/config/config.ini.template finrobot_equity/core/config/config.ini
```

Edit `config.ini`:

```ini
[API_KEYS]
fmp_api_key = YOUR_FMP_API_KEY
openai_api_key = YOUR_OPENAI_API_KEY
adanos_api_key = YOUR_ADANOS_API_KEY  # optional retail sentiment insights
```

### 16.2 Generate Equity Research Outputs

```bash
python finrobot_equity/core/src/generate_financial_analysis.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --config-file finrobot_equity/core/config/config.ini \
    --peer-tickers AMD INTC \
    --generate-text-sections
```

### 16.3 Generate HTML Research Report

```bash
python finrobot_equity/core/src/create_equity_report.py \
    --company-ticker NVDA \
    --company-name "NVIDIA Corporation" \
    --analysis-csv output/NVDA/analysis/financial_metrics_and_forecasts.csv \
    --ratios-csv output/NVDA/analysis/ratios_raw_data.csv \
    --tagline-file output/NVDA/analysis/tagline.txt \
    --company-overview-file output/NVDA/analysis/company_overview.txt \
    --investment-overview-file output/NVDA/analysis/investment_overview.txt \
    --valuation-overview-file output/NVDA/analysis/valuation_overview.txt \
    --risks-file output/NVDA/analysis/risks.txt \
    --competitor-analysis-file output/NVDA/analysis/competitor_analysis.txt \
    --major-takeaways-file output/NVDA/analysis/major_takeaways.txt \
    --peer-ev-ebitda-csv output/NVDA/analysis/peer_ev_ebitda_comparison.csv \
    --enable-text-regeneration \
    --config-file finrobot_equity/core/config/config.ini
```

### 16.4 Run Portfolio Strategy

```bash
cd finrobot_equity/core/src
python -m portfolio_strategy.run_tech_strategy
```

### 16.5 Generate Plots

```bash
python -m portfolio_strategy.plot_results
```

---

## 17. Main Outputs

The strategy creates outputs under:

```text
output/TECH_STRATEGY/
```

Important files include:

```text
score_method_comparison.csv
random_top15_percentiles.csv
random_top15_summary.csv
quant_only/performance_summary.csv
simple_finrobot/performance_summary.csv
risk_aware_finrobot/performance_summary.csv
simple_finrobot/holdings_attribution.csv
risk_aware_finrobot/holdings_attribution.csv
simple_finrobot/active_vs_spy.csv
simple_finrobot/active_vs_qqq.csv
risk_aware_finrobot/active_vs_spy.csv
risk_aware_finrobot/active_vs_qqq.csv
simple_finrobot/alpha_beta_regression.csv
risk_aware_finrobot/alpha_beta_regression.csv
```

---

## 18. Resume Summary

Built an AI-assisted equity research and portfolio construction pipeline that converts FinRobot-generated research outputs into stock-level ranking and weighting signals. Extended the FinRobot workflow with frontier LLM model-role experiments, structured research modules, and risk-aware scoring, then evaluated quant-only, simple FinRobot, and risk-aware FinRobot strategies through backtesting, benchmark comparison, random baselines, alpha/beta diagnostics, and holdings attribution.

---

## 19. Final Takeaway

This project does not claim to prove persistent long-term market-beating alpha. Instead, it demonstrates a complete and reproducible workflow for translating AI-generated equity research into systematic portfolio signals.

The final result is a strong prototype for connecting:

```text
AI equity research agents
→ structured company scoring
→ systematic portfolio construction
→ professional backtesting and attribution analysis
```

## License

Apache 2.0 — See [LICENSE](../LICENSE) for details.