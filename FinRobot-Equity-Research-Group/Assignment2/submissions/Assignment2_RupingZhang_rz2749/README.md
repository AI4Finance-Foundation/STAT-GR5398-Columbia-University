# Improving FinRobot Equity Research Reports with Frontier Models and a Claude-Inspired ACR Workflow

## 1. Introduction

This project is my Assignment 2 submission for the FinRobot Equity Research AI Agent Track. The assignment asks us to improve FinRobot in two ways: first, by upgrading the base models to stronger frontier LLMs, and second, by adding Claude-inspired equity research capabilities into the multi-agent pipeline.

My main focus is improving the reliability, groundedness, and analytical quality of FinRobot-generated equity research reports. During the project, I observed that an LLM-generated financial report can look fluent but still suffer from two important problems:

1. Some information may be missing from the available data source.
2. The model may still produce confident or generic conclusions even when the supporting data are incomplete.

Therefore, I designed this project around a practical question:

> How can we make FinRobot reports more grounded, more transparent about missing data, and more useful for equity research?

To address this, I built a Claude-inspired **Analyst-Critic-Reviser** workflow and integrated it into FinRobot's report-generation pipeline.

---

## 2. Assignment Requirements and My Submission

This assignment requires both model experimentation and a meaningful Claude-inspired enhancement to the FinRobot equity research workflow. My submission addresses both parts.

| Assignment Requirement | My Implementation |
|---|---|
| Replace original models with newer frontier models | Tested GPT, Claude, Gemini, and mixed-model configurations |
| Experiment with model combinations | Compared unified-model settings against mixed-role settings such as Claude Analyst + GPT Critic + Claude Reviser |
| Add Claude-inspired equity research enhancement | Implemented an Analyst-Critic-Reviser workflow with explicit critique and revision |
| Improve report quality and structure | Added prompts for grounded financial reasoning, missing-data disclosure, and HTML-ready equity research writing |
| Generate reports for required companies | Generated reports for NVDA, AMD, INTC, AAPL, and GOOGL |
| Compare results and identify best setup | Concluded that Claude Analyst + GPT Critic + Claude Reviser produced the best overall balance |
| Provide reproducible implementation | Added `run_commands.md`, code files, report folders, and integrated HTML demos |

The final integrated system keeps FinRobot's original strengths:

- FMP financial data ingestion
- Financial metrics and forecasts
- Peer comparison tables
- Market data fetching
- Professional HTML report rendering

At the same time, it enhances the text-generation layer when the following flag is enabled:

```bash
--enable-assignment2-enhancement
```

---

## 3. What I Observed in the Original FinRobot-Style Output

Before modifying the pipeline, I noticed that FinRobot-style reports can sometimes be less accurate or less complete than expected. The issue is not simply that the model is weak. Instead, the problem comes from the interaction between the data layer and the generation layer.

### 3.1 Data Missingness

Some fields are not always available in the structured input or API output. For example:

- Immediate market reaction
- Current valuation multiples
- Sensitivity analysis assumptions
- Catalyst timing
- Segment-level details
- Peer comparison details
- Management commentary details

When these fields are missing, the report may either leave gaps or generate generic language.

### 3.2 Overconfident Language

LLMs often try to complete a financial narrative even when the input data are incomplete. This may lead to:

- Unsupported claims
- Invented or weakly supported numbers
- Vague investment conclusions
- Promotional language
- Insufficient disclosure of missing information

This was the key problem I wanted to solve.

---

## 4. My Main Enhancement: Analyst-Critic-Reviser Workflow

To improve report reliability, I implemented a Claude-inspired **Analyst-Critic-Reviser** workflow.

The workflow has three stages:

| Stage | Role | Purpose |
|---|---|---|
| Analyst | Drafts the section | Produces the first version of the equity research section |
| Critic | Reviews the draft | Checks missing data, unsupported claims, invented numbers, vague language, and weak reasoning |
| Reviser | Produces final text | Revises the section using the Critic's feedback and outputs grounded HTML-ready text |

This design is inspired by real equity research workflows, where an analyst's first draft is reviewed before publication. Instead of relying on a single-pass generation, the ACR workflow forces the output to go through a critique and revision stage.

The most important role is the Critic. It checks whether the draft is supported by the available financial data and whether the language is too promotional or too speculative. The Reviser then rewrites the section into a more conservative, structured, and grounded final answer.

---

## 5. How I Modified FinRobot

I did not replace the entire FinRobot repository. Instead, I added a modular enhancement layer on top of FinRobot's original pipeline.

The key code files are:

```text
code/
├── assignment2_text_enhancer.py
├── config3_model_experiment.py
├── create_equity_report_modified.py
├── Code_Contribution_Summary.md
└── run_commands.md
```

### 5.1 `assignment2_text_enhancer.py`

This file implements the Analyst-Critic-Reviser workflow.

It contains:

- A unified model-calling interface for GPT, Claude, and Gemini
- Analyst prompt construction
- Critic prompt construction
- Reviser prompt construction
- The full ACR workflow function

The final default integrated configuration is:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

### 5.2 `create_equity_report_modified.py`

This is the modified FinRobot report script.

The main change is the optional flag:

```bash
--enable-assignment2-enhancement
```

When this flag is enabled, FinRobot's text sections are routed through the ACR workflow.

Without the flag, the original FinRobot pipeline can still run normally. This makes the modification modular and non-destructive.

### 5.3 `config3_model_experiment.py`

This script runs controlled model-combination experiments under the same ACR structure.

It compares:

- GPT-only
- Claude-only
- Gemini-only
- Claude Analyst + GPT Critic + Claude Reviser
- Gemini Analyst + GPT Critic + Claude Reviser

### 5.4 `run_commands.md`

This file documents the commands used to reproduce:

- The official FinRobot baseline report
- The FinRobot + ACR enhanced report
- The model-combination experiments
- The five-company HTML report generation process

---

## 6. Official FinRobot Pipeline Integration

One important part of this project is that the final HTML report is not generated by a completely separate script. It is integrated with FinRobot's official report pipeline.

The official pipeline has two main steps.

### Step 1: Generate FinRobot Financial Data and Text Sections

```cmd
python generate_financial_analysis.py --company-ticker NVDA --company-name "NVIDIA Corporation" --peer-tickers AMD INTC --generate-text-sections
```

This step produces FMP-backed data and text-section files under:

```text
output\NVDA\analysis\
```

Example outputs include:

```text
financial_metrics_and_forecasts.csv
ratios_raw_data.csv
peer_ebitda_comparison.csv
peer_ev_ebitda_comparison.csv
tagline.txt
company_overview.txt
investment_overview.txt
valuation_overview.txt
risks.txt
competitor_analysis.txt
major_takeaways.txt
```

### Step 2A: Generate Official FinRobot Baseline Report

```cmd
python create_equity_report.py --company-ticker NVDA --company-name "NVIDIA Corporation" --analysis-csv output\NVDA\analysis\financial_metrics_and_forecasts.csv --ratios-csv output\NVDA\analysis\ratios_raw_data.csv --tagline-file output\NVDA\analysis\tagline.txt --company-overview-file output\NVDA\analysis\company_overview.txt --investment-overview-file output\NVDA\analysis\investment_overview.txt --valuation-overview-file output\NVDA\analysis\valuation_overview.txt --risks-file output\NVDA\analysis\risks.txt --competitor-analysis-file output\NVDA\analysis\competitor_analysis.txt --major-takeaways-file output\NVDA\analysis\major_takeaways.txt --peer-ebitda-csv output\NVDA\analysis\peer_ebitda_comparison.csv --peer-ev-ebitda-csv output\NVDA\analysis\peer_ev_ebitda_comparison.csv --output-dir output\NVDA\official_baseline_report --enable-text-regeneration
```

This produces the official FinRobot baseline report.

### Step 2B: Generate FinRobot + ACR Enhanced Report

```cmd
python create_equity_report.py --company-ticker NVDA --company-name "NVIDIA Corporation" --analysis-csv output\NVDA\analysis\financial_metrics_and_forecasts.csv --ratios-csv output\NVDA\analysis\ratios_raw_data.csv --tagline-file output\NVDA\analysis\tagline.txt --company-overview-file output\NVDA\analysis\company_overview.txt --investment-overview-file output\NVDA\analysis\investment_overview.txt --valuation-overview-file output\NVDA\analysis\valuation_overview.txt --risks-file output\NVDA\analysis\risks.txt --competitor-analysis-file output\NVDA\analysis\competitor_analysis.txt --major-takeaways-file output\NVDA\analysis\major_takeaways.txt --peer-ebitda-csv output\NVDA\analysis\peer_ebitda_comparison.csv --peer-ev-ebitda-csv output\NVDA\analysis\peer_ev_ebitda_comparison.csv --output-dir output\NVDA\official_plus_acr_report --enable-text-regeneration --enable-assignment2-enhancement
```

This produces the integrated Assignment 2 demo:

```text
FinRobot FMP data
+ FinRobot financial tables
+ FinRobot peer comparison
+ FinRobot professional HTML renderer
+ Claude-inspired Analyst-Critic-Reviser text enhancement
```

---

## 7. Model Experiments

I tested multiple model configurations.

### 7.1 Main Model Configurations

| Configuration | Analyst | Critic | Reviser |
|---|---|---|---|
| GPT-only | GPT | GPT | GPT |
| Claude-only | Claude | Claude | Claude |
| Gemini-only | Gemini | Gemini | Gemini |
| Mixed 1 | Claude | GPT/OpenAI | Claude |
| Mixed 2 | Gemini | GPT/OpenAI | Claude |

### 7.2 Controlled Model Experiment

AAPL was used as the main controlled comparison company for model-combination experiments.

The reports are saved under:

```text
reports/config3_model_experiments/
```

Example files:

```text
AAPL_config3_gpt.md
AAPL_config3_claude.md
AAPL_config3_gemini.md
AAPL_config3_mixed_claude_gpt_claude.md
AAPL_config3_mixed_gemini_gpt_claude.md
```

Using the same company for these experiments helps isolate the effect of model choice and model-role assignment. This was important because different companies have different financial profiles; using one controlled company makes the model comparison cleaner.

---

## 8. Reports Generated

The project generates three types of outputs.

### 8.1 Main ACR Reports for Five Required Companies

The final ACR reports are saved under:

```text
reports/config3_acr/
```

The five required companies are:

```text
AAPL
NVDA
AMD
INTC
GOOGL
```

Expected files:

```text
AAPL.md
NVDA.md
AMD.md
INTC.md
GOOGL.md
```

These reports use the final Analyst-Critic-Reviser workflow.

### 8.2 Model-Combination Reports

The model-combination experiment reports are saved under:

```text
reports/config3_model_experiments/
```

These reports compare GPT-only, Claude-only, Gemini-only, and mixed-model ACR settings.

### 8.3 Integrated HTML Demo

The integrated HTML reports are saved under:

```text
integrated_demo/
├── AAPL_official_plus_acr/
├── NVDA_official_plus_acr/
├── AMD_official_plus_acr/
├── INTC_official_plus_acr/
├── GOOGL_official_plus_acr/
└── NVDA_official_baseline/
```

The Markdown reports in `reports/` are the main assignment deliverables. The HTML reports in `integrated_demo/` are additional reproducibility evidence showing that the ACR workflow is connected to FinRobot's official HTML renderer.

---

## 9. AAPL HTML Comparison: Official Baseline vs. ACR-Enhanced Version

To make the effect of the Assignment 2 enhancement more concrete, I also compared two AAPL HTML reports:

- [AAPL ACR-Enhanced Report](https://odysseus11111.github.io/5298_HW2/5298_HW2-main-main/integrated_demo/AAPL_official_plus_acr/Professional_Equity_Report_AAPL.html)
- [AAPL Official FinRobot Baseline Report](https://odysseus11111.github.io/5298_HW2/5298_HW2-main-main/integrated_demo/AAPL/official_baseline_report/Professional_Equity_Report_AAPL.html)

The baseline report uses FinRobot's official data pipeline and HTML renderer without my Assignment 2 ACR enhancement. The ACR-enhanced report uses the same FinRobot pipeline, but routes the text-generation layer through my Analyst-Critic-Reviser workflow.

### 9.1 Main Difference

The main improvement is in the quality and groundedness of the written research sections.

In the original baseline version, several major sections are incomplete or generic. For example, the baseline report shows placeholder-style outputs such as:

```text
Apple Inc. (AAPL) tagline analysis not available.
Apple Inc. (AAPL) company overview analysis not available.
Apple Inc. (AAPL) investment overview analysis not available.
Apple Inc. (AAPL) valuation overview analysis not available.
```

In contrast, the ACR-enhanced version replaces these incomplete sections with full equity research discussion. It provides a more detailed investment thesis, company overview, profitability analysis, valuation discussion, risk discussion, competitive analysis, and recent news summary.

The enhanced version also explicitly identifies itself as:

```text
Assignment 2 ACR Enhanced Report
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

This makes the model configuration transparent and directly connects the HTML output to the Assignment 2 experiment design.

### 9.2 What the ACR Workflow Improves

| Area | Official FinRobot Baseline | ACR-Enhanced Version |
|---|---|---|
| Investment Thesis | Mostly unavailable or generic | Full narrative based on revenue, margins, EPS, and valuation trends |
| Company Overview | Placeholder-style missing output | Detailed discussion of revenue growth, profitability, and operating leverage |
| Valuation Analysis | Often unavailable or shallow | More detailed P/E, EPS, margin, and valuation compression analysis |
| Risk Discussion | Partially missing or generic | More structured discussion of operational, market, regulatory, and valuation risks |
| Groundedness | May contain incomplete sections | Critic stage checks missing data, unsupported claims, and invented numbers |
| Final Writing Quality | Less polished | Reviser produces more professional and HTML-ready prose |

The most important improvement is not just that the enhanced version is longer. The more important improvement is that it is more grounded. The Critic stage checks whether claims are supported by the available data, and the Reviser rewrites the final section to avoid overconfident or unsupported statements.

### 9.3 Why Some Advanced Sections Still Say "Not Available"

Some sections in the HTML report still show:

```text
Sensitivity analysis not available.
Catalyst analysis not available.
Advanced charts not available.
```

This is expected.

These sections correspond to optional advanced FinRobot modules, such as:

- Sensitivity analysis
- Catalyst analysis
- Technical / advanced chart analysis

I did not make these optional modules the core of my Assignment 2 submission, because they require additional structured inputs or module-specific JSON files that are separate from the stable FinRobot two-step report pipeline.

For example:

- `Sensitivity Analysis` requires a structured sensitivity-analysis input file.
- `Key Catalysts` requires a structured catalyst-analysis file.
- `Technical & Advanced Analysis` requires additional technical or advanced chart inputs.

Since these inputs were not the focus of the assignment and were not consistently available through the standard FMP/text-section pipeline, I chose not to rely on them as the core contribution.

### 9.4 What I Focused on Instead

The main goal of my project is to improve FinRobot's text-generation reliability and report groundedness.

Therefore, I focused on the more stable and assignment-relevant part of the pipeline:

```text
FinRobot FMP data
+ FinRobot financial tables
+ FinRobot peer comparison
+ FinRobot professional HTML renderer
+ Claude-inspired Analyst-Critic-Reviser text enhancement
```

This means the report still uses FinRobot's official data and rendering pipeline, while my enhancement improves the written research sections.

### 9.5 Why This Still Meets the Assignment Goal

The assignment asks us to improve FinRobot by experimenting with stronger models and adding Claude-inspired equity research enhancements.

My project does this by:

1. Testing GPT, Claude, Gemini, and mixed-model role assignments.
2. Adding a Claude-inspired Analyst-Critic-Reviser workflow.
3. Integrating the workflow into FinRobot's official HTML report pipeline.
4. Improving incomplete or weak text sections in the original baseline.
5. Making the final report more grounded, transparent, and professional.

Therefore, the unavailable advanced sections are not a failure of the main project. They simply indicate that those optional advanced modules were not enabled with the required structured inputs. The core Assignment 2 enhancement is the ACR workflow, which directly improves the reliability and quality of the report text.

---

## 10. Comparison of Results

### 10.1 Overall Comparison

| Version | Data Source | Model Workflow | Strength | Weakness | Verdict |
|---|---|---|---|---|---|
| Official FinRobot Baseline | FMP via FinRobot | FinRobot default text regeneration | Uses real financial data and professional HTML renderer | Less explicit about missing-data issues | Good baseline |
| GPT-only ACR | Facts package / structured input | GPT as Analyst, Critic, and Reviser | Concise and stable | Less rich analytical writing | Useful baseline |
| Claude-only ACR | Facts package / structured input | Claude as Analyst, Critic, and Reviser | Strong long-form writing and structure | May be less aggressive as a critic | Strong writing quality |
| Gemini-only ACR | Facts package / structured input | Gemini as Analyst, Critic, and Reviser | Useful comparison model | Less consistent in final structure | Useful robustness check |
| Mixed Claude-GPT-Claude | Facts package + FinRobot demo | Claude Analyst, GPT Critic, Claude Reviser | Best balance of analysis, critique, and revision | Higher API complexity | Best final configuration |
| Official FinRobot + ACR | FMP via FinRobot | Claude-GPT-Claude ACR | Real data + professional HTML + grounded text | Advanced optional modules require additional structured inputs | Main integrated demo |

### 10.2 Best Configuration

The best overall setup is:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

### Why This Worked Best

Claude produced stronger analyst-style writing, especially for structured financial discussion. GPT/OpenAI worked well as a Critic because it was concise and direct in identifying unsupported claims, missing data, and weak reasoning. Claude then produced a polished final revision.

This mixed configuration produced the best balance between:

- Analytical depth
- Critique quality
- Groundedness
- Final writing quality
- HTML-ready output

---

## 11. Answering the Key Issue: Why Are Some FinRobot Outputs Missing or Less Accurate?

A key observation from this project is that missing or inaccurate FinRobot-style output is caused by two layers.

### 11.1 Data Layer

Some information is not always available from the facts package or the FMP output. Examples include:

- Immediate stock price reaction
- Exact valuation multiples
- Sensitivity analysis assumptions
- Catalyst timing
- Detailed segment commentary
- Some management commentary details

If these inputs are missing, the report cannot fully support certain claims.

### 11.2 Generation Layer

Even when data are missing, LLMs may still try to complete the report with fluent but generic language.

This can produce:

- Unsupported conclusions
- Overconfident claims
- Generic investment takeaways
- Weak risk discussion
- Unclear distinction between facts and interpretation

### 11.3 My Solution

The Analyst-Critic-Reviser workflow directly addresses this issue.

The Critic explicitly checks:

- Whether claims are supported by available data
- Whether numbers are invented
- Whether missing data are disclosed
- Whether the reasoning is too vague
- Whether the language is too promotional

The Reviser then rewrites the report to be more conservative, transparent, and grounded.

This is the main improvement of my Assignment 2 pipeline.

---

## 12. Engineering Design

The enhancement is designed to be modular.

### 12.1 Non-Destructive Integration

The original FinRobot pipeline can still run without my enhancement.

Without the flag:

```bash
--enable-assignment2-enhancement
```

FinRobot runs the baseline pipeline.

With the flag enabled, FinRobot routes report sections through the ACR workflow.

### 12.2 Reproducibility

The key commands are documented in:

```text
code/run_commands.md
```

### 12.3 Code Organization

The submitted code files are:

```text
code/
├── assignment2_text_enhancer.py
├── config3_model_experiment.py
├── create_equity_report_modified.py
├── Code_Contribution_Summary.md
└── run_commands.md
```

### 12.4 Security

API keys are not included in the repository.

The following files should not be committed:

```text
.env
config.ini
OAI_CONFIG_LIST
config_api_keys
```

---

## 13. Limitations

There are still some limitations.

First, the quality of the final report depends on the quality and completeness of the financial data. If the input data do not include certain fields, the model should not invent them.

Second, some advanced FinRobot modules, such as sensitivity analysis, catalyst analysis, and enhanced news, require specific structured inputs. I explored these modules, but my main contribution focuses on the more stable and assignment-relevant part: improving the text-generation layer.

Third, the ACR workflow requires multiple model calls, so it is more expensive and slower than a single-pass generation pipeline.

---

## 14. Conclusion

This project improves FinRobot by adding a model-upgraded, Claude-inspired Analyst-Critic-Reviser workflow to the report-generation process.

The main contributions are:

1. Testing multiple frontier models and model-role assignments.
2. Adding a Claude-inspired critique and revision layer.
3. Integrating the enhancement into FinRobot's official report pipeline.
4. Improving the groundedness and reliability of generated equity research reports.
5. Producing both Markdown reports and integrated FinRobot HTML demos.

The final best configuration is:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

This setup produced the strongest balance of financial reasoning, missing-data awareness, critique quality, and polished final writing.

Most importantly, the project shows that improving FinRobot is not only about making reports look better. It is about making the analysis more reliable when data are incomplete.
