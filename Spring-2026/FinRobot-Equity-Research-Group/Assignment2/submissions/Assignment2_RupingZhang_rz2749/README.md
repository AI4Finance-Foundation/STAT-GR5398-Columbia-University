# Improving FinRobot Equity Research Reports with Frontier Models and an Analyst-Critic-Reviser Workflow

## Summary of My Contribution

This repository contains my Assignment 2 submission for the FinRobot Equity Research AI Agent Track.

My main contribution is an **Analyst-Critic-Reviser (ACR) enhancement layer** for FinRobot equity research reports. Instead of generating a report in one pass, my workflow adds a quality-control process:

1. **Analyst**: generates the first draft of each equity research section.
2. **Critic**: checks the draft for missing data, unsupported claims, invented numbers, vague language, and weak reasoning.
3. **Reviser**: rewrites the section into a more grounded, conservative, and professional final version.

I tested this workflow with GPT, Claude, Gemini, and mixed-model configurations. The best final configuration was:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

The final submission includes source code, generated Markdown reports, integrated FinRobot HTML demos, and a Medium-style reflection. In the integrated AAPL HTML demo, the ACR-enhanced report also incorporates earnings analysis, valuation reasoning, margin-sensitivity discussion, catalyst/risk-style analysis, and missing-data disclosure inside the main research narrative.

---

## 1. Project Motivation

The assignment asks us to improve FinRobot in two ways:

1. Upgrade the original model setup with stronger frontier LLMs.
2. Add Claude-inspired equity research capabilities to the multi-agent pipeline.

During this project, I observed that FinRobot-style LLM-generated financial reports can look fluent but still have two important problems:

- Some information may be missing from the available financial data.
- The model may still produce confident or generic conclusions even when the supporting data are incomplete.

Therefore, I designed this project around the following question:

> How can we make FinRobot reports more grounded, more transparent about missing data, and more useful for equity research?

My answer is to add a Claude-inspired **Analyst-Critic-Reviser** workflow to the report-generation process.

---

## 2. Assignment Requirements and My Implementation

| Assignment Requirement | My Implementation |
|---|---|
| Replace original models with newer frontier models | Tested GPT, Claude, Gemini, and mixed-model configurations |
| Experiment with model combinations | Compared single-model settings and mixed-role settings |
| Add Claude-inspired equity research enhancement | Implemented an Analyst-Critic-Reviser workflow with earnings-style analysis, sensitivity-aware valuation reasoning, and catalyst/risk-style discussion |
| Improve report quality and structure | Added prompts for grounded reasoning, missing-data disclosure, and HTML-ready writing |
| Incorporate equity research skills | Integrated earnings, valuation, sensitivity, catalyst, and risk reasoning into the core report narrative |
| Generate reports for required companies | Generated reports for NVDA, AMD, INTC, AAPL, and GOOGL |
| Compare results and identify best setup | Found that Claude Analyst + GPT Critic + Claude Reviser gave the best balance |
| Provide reproducible implementation | Added code files, reports, integrated demos, and run commands |

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

## 3. Repository Structure

```text
Assignment2_RupingZhang_rz2749/
├── README.md
├── blog_post.md
├── code/
│   ├── assignment2_text_enhancer.py
│   ├── config3_model_experiment.py
│   ├── create_equity_report_modified.py
│   ├── Code_Contribution_Summary.md
│   └── run_commands.md
├── reports/
│   ├── config3_acr/
│   └── config3_model_experiments/
└── integrated_demo/
    ├── AAPL_official_plus_acr/
    ├── NVDA_official_plus_acr/
    ├── AMD_official_plus_acr/
    ├── INTC_official_plus_acr/
    ├── GOOGL_official_plus_acr/
    └── NVDA_official_baseline/
```

---

## 4. Main Enhancement: Analyst-Critic-Reviser Workflow

The ACR workflow has three stages.

| Stage | Role | Purpose |
|---|---|---|
| Analyst | Drafts the section | Produces the first version of the equity research section |
| Critic | Reviews the draft | Checks missing data, unsupported claims, invented numbers, vague language, and weak reasoning |
| Reviser | Produces final text | Revises the section using the Critic's feedback and outputs grounded HTML-ready text |

This design is inspired by real equity research workflows, where an analyst's first draft is reviewed before publication.

The most important role is the **Critic**. It checks whether the draft is supported by the available financial data and whether the language is too promotional or speculative. The **Reviser** then rewrites the section into a more conservative, structured, and grounded final answer.

---

## 5. Key Code Files

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

This is the modified FinRobot report-generation script.

The main change is the optional flag:

```bash
--enable-assignment2-enhancement
```

When this flag is enabled, FinRobot's text sections are routed through the ACR workflow. Without the flag, the original FinRobot pipeline can still run normally. This makes the modification modular and non-destructive.

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

## 6. Model Experiments

I tested multiple model configurations.

| Configuration | Analyst | Critic | Reviser |
|---|---|---|---|
| GPT-only | GPT | GPT | GPT |
| Claude-only | Claude | Claude | Claude |
| Gemini-only | Gemini | Gemini | Gemini |
| Mixed 1 | Claude | GPT/OpenAI | Claude |
| Mixed 2 | Gemini | GPT/OpenAI | Claude |

AAPL was used as the main controlled comparison company for the model-combination experiment. The reports are saved under:

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

Using the same company helps isolate the effect of model choice and model-role assignment.

---

## 7. Reports Generated

### 7.1 Main ACR Reports for Five Required Companies

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

### 7.2 Integrated HTML Demo

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

The Markdown reports in `reports/` are the main assignment deliverables. The HTML reports in `integrated_demo/` provide additional reproducibility evidence showing that the ACR workflow is connected to FinRobot's official HTML renderer.

---

## 8. Integrated FinRobot Pipeline

One important part of this project is that the final HTML report is not generated by a completely separate script. It is integrated with FinRobot's official report pipeline.

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

## 9. Baseline vs. ACR-Enhanced Comparison

To make the effect of the Assignment 2 enhancement more concrete, I compared the official FinRobot baseline report with the ACR-enhanced report.

Example AAPL HTML reports:

- [AAPL ACR-Enhanced Report](https://odysseus11111.github.io/5298_HW2/5298_HW2-main-main/integrated_demo/AAPL_official_plus_acr/Professional_Equity_Report_AAPL.html)
- [AAPL Official FinRobot Baseline Report](https://odysseus11111.github.io/5298_HW2/5298_HW2-main-main/integrated_demo/AAPL/official_baseline_report/Professional_Equity_Report_AAPL.html)

The baseline report uses FinRobot's official data pipeline and HTML renderer without my Assignment 2 ACR enhancement. The ACR-enhanced report uses the same FinRobot pipeline but routes the text-generation layer through my Analyst-Critic-Reviser workflow.

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

### 9.2 Summary Comparison

| Area | Official FinRobot Baseline | ACR-Enhanced Version |
|---|---|---|
| Investment Thesis | Mostly unavailable or generic | Full narrative based on revenue, margins, EPS, and valuation trends |
| Company Overview | Placeholder-style missing output | More detailed discussion of revenue growth, profitability, and operating leverage |
| Valuation Analysis | Often unavailable or shallow | More detailed discussion of P/E, EPS, margin, and valuation compression |
| Risk Discussion | Partially missing or generic | More structured discussion of operational, market, regulatory, and valuation risks |
| Groundedness | May contain incomplete sections | Critic stage checks missing data, unsupported claims, and invented numbers |
| Final Writing Quality | Less polished | Reviser produces more professional and HTML-ready prose |

The most important improvement is not only that the enhanced version is longer. The more important improvement is that it is more grounded. The Critic stage checks whether claims are supported by the available data, and the Reviser rewrites the final section to avoid overconfident or unsupported statements.

---

## 10. Overall Evaluation

### 10.1 Evaluation Criteria

I evaluate the baseline and enhanced workflows using four qualitative criteria.

| Criterion | Description |
|---|---|
| Groundedness | Whether the report avoids unsupported or invented claims |
| Missing Data Disclosure | Whether the report clearly states when data is unavailable |
| Specificity | Whether the report uses concrete company-specific discussion |
| Professional Structure | Whether the report reads like a sell-side research update |

### 10.2 Overall Comparison

| Version | Data Source | Model Workflow | Strength | Weakness | Verdict |
|---|---|---|---|---|---|
| Official FinRobot Baseline | FMP via FinRobot | FinRobot default text regeneration | Uses real financial data and professional HTML renderer | Less explicit about missing-data issues | Good baseline |
| GPT-only ACR | Facts package / structured input | GPT as Analyst, Critic, and Reviser | Concise and stable | Less rich analytical writing | Useful baseline |
| Claude-only ACR | Facts package / structured input | Claude as Analyst, Critic, and Reviser | Strong long-form writing and structure | May be less aggressive as a critic | Strong writing quality |
| Gemini-only ACR | Facts package / structured input | Gemini as Analyst, Critic, and Reviser | Useful comparison model | Less consistent in final structure | Useful robustness check |
| Mixed Claude-GPT-Claude | Facts package + FinRobot demo | Claude Analyst, GPT Critic, Claude Reviser | Best balance of analysis, critique, and revision | Higher API complexity | Best final configuration |
| Official FinRobot + ACR | FMP via FinRobot | Claude-GPT-Claude ACR | Real data + professional HTML + grounded text | Advanced optional modules require additional structured inputs | Main integrated demo |

### 10.3 Best Configuration

The best overall setup is:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

Claude produced stronger analyst-style writing, especially for structured financial discussion. GPT/OpenAI worked well as the Critic because it was concise and direct in identifying unsupported claims, missing data, and weak reasoning. Claude then produced a polished final revision.

This mixed configuration produced the best balance between:

- Analytical depth
- Critique quality
- Groundedness
- Final writing quality
- HTML-ready output

---

## 11. How the ACR Report Handles Earnings, Sensitivity, and Catalyst-Style Analysis

Some standalone optional FinRobot HTML template slots, such as `Sensitivity Analysis`, `Key Catalysts`, and `Technical & Advanced Analysis`, may still show:

```text
Sensitivity analysis not available.
Catalyst analysis not available.
Advanced charts not available.
```

This does **not** mean that these research ideas are absent from my project. The reason is that these specific standalone HTML slots require separate structured input files, such as a dedicated sensitivity-analysis file, a catalyst-analysis file, or additional technical/advanced chart inputs.

Instead of forcing the model to fill those standalone slots without reliable structured inputs, I incorporated earnings analysis, sensitivity-aware valuation reasoning, and catalyst/risk-style discussion into the main ACR-enhanced research narrative.

This design choice is intentional: when structured data are incomplete, the report should avoid hallucinating unsupported numbers or pretending that a full standalone module exists. The ACR workflow instead improves the parts of the report that are supported by the available FinRobot/FMP data.

### 11.1 Earnings-Style Analysis

The ACR-enhanced report discusses earnings and profitability through:

- revenue growth trends,
- EBITDA and contribution margin changes,
- EPS growth,
- operating leverage,
- SG&A efficiency,
- projected earnings power.

For example, in the AAPL integrated HTML demo, the report discusses revenue growth, EPS growth, EBITDA margin expansion, SG&A margin compression, and P/E compression as part of the investment thesis and valuation narrative.

### 11.2 Sensitivity-Aware Valuation Reasoning

The ACR-enhanced report also includes sensitivity-aware reasoning inside the valuation and risk discussion.

Instead of producing a separate sensitivity table without structured assumptions, the report highlights which forecast assumptions are important and where the valuation could be pressured. For example, it discusses:

- whether projected EBITDA margin expansion is realistic,
- how P/E compression may create return headwinds,
- how revenue growth or margin shortfalls could affect valuation,
- why large margin-expansion assumptions should be monitored.

This means the ACR workflow adds sensitivity-style thinking in a cautious and data-grounded way.

### 11.3 Catalyst and Risk-Style Discussion

The report also incorporates catalyst and risk-style analysis through sections such as:

- growth outlook,
- strategic positioning,
- competitive landscape,
- regulatory risk,
- supply-chain exposure,
- AI competition,
- market saturation,
- macroeconomic sensitivity,
- valuation risk.

These are not presented as a separate standalone `Key Catalysts` module because the required catalyst-specific structured input file was not consistently available. However, the investment narrative still discusses business drivers, risks, and uncertainty in an analyst-style format.

### 11.4 Why This Is Better Than Filling Every Slot Mechanically

My goal was not to mechanically fill every optional HTML section. My goal was to improve the reliability of FinRobot's core equity research narrative.

The ACR workflow therefore focuses on:

```text
FinRobot FMP data
+ FinRobot financial tables
+ FinRobot peer comparison
+ FinRobot professional HTML renderer
+ ACR-enhanced earnings, valuation, sensitivity, catalyst/risk, and missing-data reasoning
```

This makes the final report more useful while reducing the risk of unsupported claims or invented financial details.
---

## 12. How to Reproduce

The main implementation files are located in the `code/` folder.

- `assignment2_text_enhancer.py`: implements the text enhancement logic.
- `config3_model_experiment.py`: runs the Analyst-Critic-Reviser model experiment.
- `create_equity_report_modified.py`: generates modified FinRobot equity reports.
- `run_commands.md`: contains the exact commands used in my experiments.

Generated reports are saved in the `reports/` folder. Integrated HTML demo outputs are saved in the `integrated_demo/` folder.

For the exact commands, see:

```text
code/run_commands.md
```

---

## 13. Engineering Design

### 13.1 Modular and Non-Destructive Integration

The original FinRobot pipeline can still run without my enhancement.

Without the flag:

```bash
--enable-assignment2-enhancement
```

FinRobot runs the baseline pipeline.

With the flag enabled, FinRobot routes report sections through the ACR workflow.

### 13.2 Security

API keys are not included in the repository.

The following files should not be committed:

```text
.env
config.ini
OAI_CONFIG_LIST
config_api_keys
```

---

## 14. Limitations

There are still some limitations.

First, the quality of the final report depends on the quality and completeness of the financial data. If the input data do not include certain fields, the model should not invent them.

Second, some standalone advanced FinRobot template slots, such as sensitivity analysis, catalyst analysis, and advanced charts, require specific structured inputs. Rather than filling these slots without reliable data, I incorporated earnings, valuation, sensitivity-aware, and catalyst/risk-style reasoning into the ACR-enhanced research narrative.

Third, the ACR workflow requires multiple model calls, so it is more expensive and slower than a single-pass generation pipeline.

Fourth, the comparison is mainly qualitative. I did not use human expert scoring or external financial databases to verify every claim. Therefore, the results should be interpreted as evidence that the ACR workflow improves report structure and grounding, rather than as a complete financial accuracy benchmark.

---

## 15. Medium-Style Reflection

The Medium-style reflection is provided in:

```text
blog_post.md
```

---

## 16. Conclusion

This project improves FinRobot by adding a frontier-model, Claude-inspired Analyst-Critic-Reviser workflow to the report-generation process.

The main contributions are:

1. Testing multiple frontier models and model-role assignments.
2. Adding a Claude-inspired critique and revision layer.
3. Integrating the enhancement into FinRobot's official report pipeline.
4. Improving the groundedness and reliability of generated equity research reports, including earnings, valuation, sensitivity-aware, and catalyst/risk-style discussion.
5. Producing both Markdown reports and integrated FinRobot HTML demos.

The final best configuration is:

```text
Analyst = Claude
Critic = GPT/OpenAI
Reviser = Claude
```

This setup produced the strongest balance of financial reasoning, missing-data awareness, critique quality, and polished final writing.

Most importantly, the project shows that improving FinRobot is not only about making reports look better. It is about making the analysis more reliable when data are incomplete.
