# GR5398 26 Fall: FinRobot Equity Research AI Agent Track

## Assignment 1

### 0. Target

This assignment is built on the **[`finrobot_equity`](https://github.com/AI4Finance-Foundation/FinRobot/tree/master/finrobot_equity)**
module — the production equity-research pipeline behind [finrobot.ai](https://finrobot.ai). It
fetches financial data, runs a team of LLM agents over it, and renders a multi-page HTML/PDF
equity research report from a single command.

Running it is the easy part, and it is not the assignment. **The assignment is to run it, then
prove whether you can trust what it produced, and then make one part of it measurably better.**

You will:

1. **Run** the two-step pipeline end to end and produce research reports for a coverage universe
   of at least **5 companies in one sector** (start with `NVDA`, `AMD`, `INTC`, `AAPL`, `GOOGL`,
   then pick a sector of your own).
2. **Audit** every number and every claim the system produced — against filings, against the data
   that was actually in the agent's prompt, and against a valuation you compute yourself.
3. **Improve** one component of the pipeline, and show a before/after comparison that a reader can
   check.
4. **Evaluate** the whole thing quantitatively, and write it up as a research report plus a
   [Medium](https://medium.com/) blog.

Submit everything to a new folder in `/submissions` named `Assignment1_Name_UNI` (NOT a new branch!).

Assignment 1 Submission Due Day: **Oct 12, 2026**

> [!NOTE]
> This replaces the old `agent_annual_report.ipynb` tutorial assignment. That notebook is still a
> useful 30-minute warm-up if you have never used AutoGen — find it in
> [`tutorials_advanced/`](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/tutorials_advanced/agent_annual_report.ipynb)
> upstream, or in this repo under `Spring-2026/FinRobot-Equity-Research-Group/Assignment1/`. It is
> optional and it is not graded.

---

### 1. The System You Are Working On

```
finrobot_equity/
├── core/
│   ├── config/config.ini.example      # API keys (copy to config.ini)
│   └── src/
│       ├── generate_financial_analysis.py   # Step 1: data + analysis + AI text
│       ├── create_equity_report.py          # Step 2: charts + HTML report
│       ├── generate_pdf_report.py           # Step 3: PDF (optional)
│       ├── Run.ipynb                        # notebook demo
│       └── modules/
│           ├── market_data_api.py           # FMP client
│           ├── financial_data_processor.py  # metrics extraction + forecasts
│           ├── valuation_engine.py          # EV/EBITDA, peer comps, DCF
│           ├── sensitivity_analyzer.py      # forecast sensitivity tables
│           ├── catalyst_analyzer.py         # event/catalyst identification
│           ├── news_integrator.py           # news retrieval + categorization
│           ├── chart_generator.py           # charts
│           ├── html_renderer.py             # report rendering
│           └── equity_agents/               # ← the agent team
└── web_app/                                 # FastAPI front end
```

**The agent team** (`equity_agents/agent_manager.py`) — eight agents, each built on the
`openai-agents` SDK with a **Pydantic `output_type`**, so every section returns a typed object
rather than free text:

| Agent | Output field | Writes |
| --- | --- | --- |
| `tagline_agent` | `tagline` | One-line thesis |
| `company_overview_agent` | `overview` | Company overview |
| `investment_overview_agent` | `investment_update` | Investment thesis |
| `valuation_overview_agent` | `valuation_analysis` | Valuation discussion |
| `risks_agent` | `risk_analysis` | 600–800 word risk breakdown |
| `competitor_analysis_agent` | `competitive_analysis` | Competitive landscape |
| `major_takeaways_agent` | `takeaways` | Key takeaways |
| `news_summary_agent` | `news_summary` | News summary |

Every agent receives **the same prompt**: `_prepare_financial_data_prompt()` dumps the financial
metrics table, the peer EBITDA and EV/EBITDA tables, and recent news into markdown. That is the
entire evidence base. Anything in the output that is not derivable from that prompt is, by
definition, either prior knowledge or a hallucination — and telling those apart is Part 2 of this
assignment.

> [!IMPORTANT]
> **The course rule for this track**: separate *deterministic calculations* from *LLM
> interpretations*. Numbers come from the data layer and must be reproducible; the LLM's job is to
> explain them, never to invent them. Your audit is where you check that this actually holds.

---

### 2. Part 0 — Setup and Baseline Run

#### 2.1 Install

```bash
git clone https://github.com/AI4Finance-Foundation/FinRobot.git
cd FinRobot
pip install -r requirements-equity.txt
cp finrobot_equity/core/config/config.ini.example finrobot_equity/core/config/config.ini
```

> [!NOTE]
> The module README says the template is `config.ini.template`, but the file in the repo is
> `config.ini.example`. Small upstream inconsistencies like this are normal — a PR fixing one is a
> legitimate (and welcome) open-source contribution.

#### 2.2 Keys

| Service | Required | Purpose |
| --- | --- | --- |
| [Financial Modeling Prep](https://financialmodelingprep.com/developer) | **Yes** | Statements, market metrics, peers, news |
| [OpenAI](https://platform.openai.com/) | **Yes** | The eight agents (`openai_model`, default `gpt-4.1`) |
| Adanos Finance | No | Optional retail sentiment (Reddit / X / Polymarket) |

`config.ini` also accepts `openai_base_url` and `openai_model`, so a proxy or a non-OpenAI
endpoint works. Record which model you used — it is part of your results.

> [!WARNING]
> ⚠️ **DO NOT SUBMIT YOUR OWN API KEY ONTO GITHUB!!!** `config.ini` must stay out of your commit.
> Check your notebook outputs too — a key echoed into a cell and saved is still a leaked key.

#### 2.3 Run the pipeline

**Step 1 — data, forecasts, and AI text**

```bash
python finrobot_equity/core/src/generate_financial_analysis.py \
    --company-ticker NVDA --company-name "NVIDIA Corporation" \
    --config-file finrobot_equity/core/config/config.ini \
    --peer-tickers AMD INTC \
    --generate-text-sections \
    --enable-sensitivity-analysis --enable-catalyst-analysis --enable-enhanced-news
```

| Input | Meaning |
| --- | --- |
| `--company-ticker`, `--company-name` | The subject company |
| `--peer-tickers` | Peers for the EBITDA / EV/EBITDA comparison |
| `--years-limit` (default 5) | Years of history to fetch |
| `--period` (`annual` / `quarterly`) | Statement periodicity |
| `--revenue-growth-2025/2026/2027` | **Forecast assumptions**, defaults `0.05 / 0.06 / 0.04` |
| `--margin-improvement` (default `0.01`) | Annual margin improvement assumed |
| `--sga-margin-improvement` (default `-0.005`) | SG&A efficiency assumed |
| `--news-days-back`, `--news-limit` | News window |
| `--enable-*` flags | Turn on sensitivity / catalyst / enhanced-news analysis |

**Output** — `output/{TICKER}/analysis/`:

| File | Contents |
| --- | --- |
| `financial_metrics_and_forecasts.csv` | Historical metrics + 3-year forecast — the spine of the report |
| `ratios_raw_data.csv`, `*_raw_data.csv` | Raw statements and ratios |
| `peer_ebitda_comparison.csv`, `peer_ev_ebitda_comparison.csv` | Peer tables |
| `company_news.json`, `enhanced_news.json`, `news_summary.md` | News layer |
| `sensitivity_analysis.json`, `sensitivity_summary.md` | Sensitivity output |
| `catalyst_analysis.json`, `catalyst_summary.md` | Catalysts |
| `retail_sentiment.json` | Only if an Adanos key is configured |
| `tagline.txt`, `company_overview.txt`, `investment_overview.txt`, `valuation_overview.txt`, `risks.txt`, `competitor_analysis.txt`, `major_takeaways.txt`, `news_summary.txt` | One file per agent |
| `analysis_summary.json` | Run manifest |

**Step 2 — charts and report**

```bash
python finrobot_equity/core/src/create_equity_report.py \
    --company-ticker NVDA --company-name "NVIDIA Corporation" \
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

**Output** — `output/{TICKER}/report/`: a multi-page HTML report with charts, plus the generated
chart images. Run `generate_pdf_report.py` for the PDF.

Note what Step 2 does *on its own*: it **auto-fetches** share price, target price, rating, market
cap, forward P/E, P/B, dividend yield, free float, ROE and net debt/equity from FMP unless you pass
them explicitly (`--skip-auto-fetch` disables this). So the headline numbers on page 1 do not come
from the agents at all. Know which numbers come from where — you are about to audit them.

The FastAPI front end (`./deploy.sh start`, then `http://127.0.0.1:8001`) runs the same pipeline
through a browser. Use whichever you prefer; the CLI is easier to script for 5+ companies.

---

### 3. Part 1 — Build a Coverage Universe

Produce reports for **at least 5 companies in one sector**, each with **at least 2 peers**.

+ Start with `NVDA`, `AMD`, `INTC`, `AAPL`, `GOOGL` to get the pipeline working.
+ Then choose **one sector of your own** and cover it properly — ideally including names with
  *different* price trends (a winner, a laggard, a turnaround). A pipeline that only ever sees
  good news is untested.
+ Keep every intermediate artifact. The `analysis/` folder is your evidence; the HTML/PDF is just
  the presentation layer.

---

### 4. Part 2 — Audit: Can You Trust This Report?

This is the core of the assignment. Three audits, each producing a table you submit.

#### 4.1 Data audit — do the numbers match the filings?

For **at least 2 companies**, pick 10 line items from `financial_metrics_and_forecasts.csv`
(revenue, EBITDA, net income, margins, EV/EBITDA, …) and verify each against the company's own
10-K/10-Q on [SEC EDGAR](https://www.sec.gov/edgar).

| Metric | Year | Pipeline value | Filing value | Δ | Verdict | Explanation |
| --- | --- | --- | --- | --- | --- | --- |

Differences are not automatically bugs — FMP may use a different fiscal-year convention, a
restated figure, or a different EBITDA definition. Your job is to **explain** each gap, not just
flag it.

#### 4.2 Forecast & valuation audit — are the assumptions defensible?

+ The 3-year forecast is driven by **command-line constants** (`--revenue-growth-2025 0.05`,
  `--margin-improvement 0.01`, …) passed into
  `financial_data_processor.calculate_growth_and_forecasts()`. Those defaults are not a model —
  they are a guess that applies equally to a hypergrowth chip designer and a mature utility.
  Document what the defaults imply for *your* companies, and what you changed them to and why.
+ `valuation_engine.py` offers `calculate_ev_ebitda_valuation()`, `calculate_peer_comparison_valuation()`,
  `calculate_dcf_valuation()`, `generate_football_field_data()` and `synthesize_valuation()`.
  **Reproduce one valuation by hand** (notebook or spreadsheet) and reconcile it with the engine's
  output. State every assumption the engine makes that it does not print — discount rate, terminal
  growth, target multiple, which year's EBITDA.
+ Run `sensitivity_analyzer.py` and report how much the target price moves across the revenue and
  margin ranges. If a ±5% revenue swing moves fair value by 40%, the point estimate in the report
  is theater — say so.

#### 4.3 Grounding audit — is the prose supported by the evidence?

For **each of the 8 agent sections**, across **at least 3 companies**, extract every factual and
quantitative claim and classify it:

| Class | Meaning |
| --- | --- |
| **Supported** | Directly derivable from the data in `_prepare_financial_data_prompt()` |
| **Unsupported** | Plausible, but nothing in the prompt establishes it (model prior knowledge) |
| **Contradicted** | The prompt's data says otherwise |

Report a **grounding rate** per section: `supported / total claims`. Then answer: which agents
hallucinate most, and why? Look at the prompts (`risks_agent.py` asks for 600–800 words about
risks; the prompt supplies financial tables and news, nothing about regulation or supply chains)
and explain the structural reason, not just the symptom.

Also measure **stability**: run the same ticker **3 times** and compare the rating, target price,
and the top-3 risks. Report the variance. A research process that returns a different answer each
time on the same inputs is not a research process.

---

### 5. Part 3 — Make One Thing Better

Pick **one** and implement it. Depth beats breadth; a single well-evaluated improvement is worth
more than four half-finished ones.

| Option | What it means |
| --- | --- |
| **A. Verifier agent** | Add a ninth agent that runs *after* the others, re-checks every number in the generated prose against the source tables, and flags or rewrites unsupported claims. Report the grounding rate before and after. |
| **B. Data-driven forecasts** | Replace the hard-coded growth constants with an actual forecast — historical CAGR, segment build-up, analyst consensus, or a small model. Backtest it: forecast an earlier year and compare against what actually happened. |
| **C. A real valuation** | Implement a DCF with explicit, printed WACC and terminal-growth assumptions, a football-field chart across methods, and a sensitivity grid. Explain where and why the methods disagree. |
| **D. Evaluation harness** | Build a reusable scorer: rubric + automated checks (numeric consistency, section length, claim grounding) that runs over N reports and outputs a scorecard. This is the option that makes everyone else's work measurable. |
| **E. A new analytical section** | Add an agent + data path the pipeline lacks — segment analysis, ESG, short-interest/ownership, supply-chain concentration. It must be evidence-grounded: new section *and* the new data it consumes. |

Requirements for whichever you pick:

+ A clean diff against upstream (a branch or patch file), not a pile of copied files.
+ A **before/after comparison on the same companies**, with numbers.
+ An honest account of what it still gets wrong.

Genuinely good work here is a pull request to
[AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot). That is the
point of the course.

---

### 6. Part 4 — Report and Blog

Your research report should cover:

+ **Setup**: model used, API costs, runtime per report, total spend
+ **Coverage universe**: which sector, which companies, why
+ **Audit results**: the three tables from Part 2, with the grounding rate and stability numbers
+ **Your improvement**: design, diff, before/after evaluation, remaining limitations
+ **Judgment**: would you hand one of these reports to a portfolio manager? Which sections are
  decision-grade today, which are not, and what would have to change?

Then publish the analysis as a [Medium](https://medium.com/) blog — your analysis, not an
AI-generated report pasted into a post.

---

### 7. Background: Multi-Agent Systems

A **Multi-Agent System (MAS)** is a computational framework in which multiple autonomous agents
interact to accomplish a task that would be difficult or inefficient for a single agent. Each
agent has a well-defined role, its own reasoning process, and a structured way to communicate.

![Multi-Agent System](https://www.kdnuggets.com/wp-content/uploads/Building-Your-First-Multi-Agent-System-A-Beginners-Guide_2.png)

A MAS decomposes a complex problem into role-specific subtasks. Agents may collaborate, verify
each other's outputs, or run under a central controller. Compared with single-agent systems this
buys robustness, interpretability and scalability — which is exactly why it is used in finance.

`finrobot_equity` is a **parallel specialist** design: eight agents share one evidence prompt,
each owns one section, each returns a typed Pydantic object, and `EquityResearchAgentManager`
collects the results. There is no debate step and no verifier — which is a deliberate trade-off
for speed and cost, and also the most obvious place to improve it (Part 3, option A).

Compare that with the **Expert / Shadow / UserProxy** pattern in the older AutoGen tutorial: an
Expert agent produces the analysis, a Shadow agent independently handles isolated long-context
Q&A over the filing so it never pollutes the main chat, and a UserProxy executes functions and
controls termination. Separating execution, verification and control is what made that pipeline
terminate reliably. Knowing both patterns — and when each is the right one — is part of what you
should take away from this assignment.

---

### 8. What to Submit

Create `submissions/Assignment1_Name_UNI/` containing:

| # | Deliverable |
| --- | --- |
| 1 | **Reports**: the HTML/PDF reports for your 5+ companies |
| 2 | **Artifacts**: the `output/{TICKER}/analysis/` folders (CSV / JSON / TXT) that produced them |
| 3 | **Audit tables**: data audit, forecast & valuation audit, grounding audit, stability test — as CSV or a notebook, not prose |
| 4 | **Your improvement**: the diff/patch or branch link, plus the before/after evaluation |
| 5 | **Research report** (PDF) covering section 6 |
| 6 | **`README.md`**: exact commands to reproduce every run, the model and config you used, the API cost, and the link to your Medium blog |

> [!WARNING]
> ⚠️ **DO NOT SUBMIT YOUR OWN API KEY ONTO GITHUB!!!** Keep `config.ini` out of the commit, and
> clear notebook outputs that echo keys.
