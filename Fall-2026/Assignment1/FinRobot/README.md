# GR5398 26 Fall: FinRobot Equity Research AI Agent Track

## Assignment 1

### 0. Target

In this assignment, we would like you to run a tutorial of FinRobot ( `/source_code/agent_annual_report.ipynb`) to automatically generate an annual financial report, and basically learn what you will do in this semester.

+ You will learn basic methods of designing a **Multi-Agent System** (**MAS**)
+ You will learn to use Expert/Shadow/UserProxy structure to get a better result
+ You should generate no less than 5 financial reports using different stocks in a same industry field (better if they have different stock trends)
  + You should start with `NVDA`, `AMD`, `INTC`, `AAPL`, `GOOGL`, and then you can try some other fields
+ After generating these 5 reports, write a basic analysis on their performance according to their historical performance and current market information, and publish your analysis report (not AI-generated reports) as a blog onto [medium](https://medium.com/)
+ All of your code files and financial reports should be uploaded into a new folder in `/submissions` named as `Assignment1_Name_UNI` (NOT a new branch!)

To find for more detailed informations, please refer to [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) and specific notebook file [agent_annual_report.ipynb](https://github.com/AI4Finance-Foundation/FinRobot/blob/master/tutorials_advanced/agent_annual_report.ipynb).

Assignment 1 Financial Reports Submission Due Day: Oct 12, 2026

### 0.1 Inputs and Outputs

Everything below happens inside `source_code/agent_annual_report.ipynb`.

**Input — configuration**

| What | Where | Notes |
| --- | --- | --- |
| OpenAI API key | `OAI_CONFIG_LIST` (rename from `OAI_CONFIG_LIST_sample`) | The notebook filters for `gpt-4-0125-preview`; a newer model is fine, but say which one you used in your report |
| SEC API key | `config_api_keys` (rename from `config_api_keys_sample`) | Used to locate the 10-K filing |
| FMP API key | `config_api_keys` | Financial Modeling Prep — statements and metrics |
| `work_dir` | notebook cell | Where all intermediate files and the final PDF are written |

**Input — the task itself**

```python
company     = "NextEra"              # the company to analyze
competitors = ["DUK", "CEG", "AEP"]  # peers for the comparison table
fyear       = "2024"                 # fiscal year of the 10-K
```

These three variables are the *only* things you change per report. Everything else — which
sections exist, how long each paragraph must be, what the last sentence of the competitor
analysis must discuss — is fixed by the task prompt, and the agents are required to satisfy it
before the PDF is allowed to be generated.

**What runs**

| Agent | Role |
| --- | --- |
| `user_proxy` | Executes the Python functions and controls the conversation |
| `expert` | "Expert Investor" — plans the report and writes the analysis |
| `expert_shadow` | Handles isolated long-context Q&A over the 10-K in a muted nested chat, so the filing never pollutes the main chat history |

Tools registered to the expert: `FMPUtils.get_sec_report` (find the filing),
`ReportAnalysisUtils` and `ReportChartUtils` (analysis and plotting knowledge),
`TextUtils.check_text_length` (enforce the word counts), `IPythonUtils.display_image`,
`ReportLabUtils.build_annual_report` (render the PDF). The chat is capped at `max_turns=50`.

**Output** (all in `work_dir`)

+ Instruction / resource `.txt` files — the hand-offs between expert and shadow
+ Generated charts (PNG), displayed in the chat as they are produced
+ **`{Company}_Annual_Report_{fyear}.pdf`** — the deliverable, a two-page report:
  + Page 1 — business overview, market position, operating results; **each paragraph 150–160 words**
  + Page 2 — risk assessment, competitors analysis; **each paragraph 500–600 words**, ending with
    whether the company's multi-year performance justifies or contradicts its current EV/EBITDA

> [!TIP]
> If the PDF never appears, it is almost always the word-count gate: the expert is told not to
> render until `check_text_length` passes. Read the chat log to see which paragraph is failing.

### 1. Multi-Agent System

A **Multi-Agent System (MAS)** is a computational framework in which multiple autonomous agents interact with each other to accomplish a task that would be difficult or inefficient for a single agent to complete alone. Each agent has a well-defined role, its own reasoning process, and the ability to communicate with other agents through structured messages.

![Building Your First Multi-Agent System: A Beginner's Guide ...](https://www.kdnuggets.com/wp-content/uploads/Building-Your-First-Multi-Agent-System-A-Beginners-Guide_2.png)

In practice, a MAS decomposes a complex problem into smaller, role-specific subtasks and assigns them to different agents. These agents may collaborate, verify each other’s outputs, or operate under a central controller to ensure stability and reliability. Compared with single-agent systems, multi-agent systems offer improved robustness, interpretability, and scalability, making them especially suitable for high-stakes domains such as finance, healthcare, and decision support systems.

### 2. Expert/Shadow/UserProxy Structure

The **Expert / Shadow / UserProxy structure** is a controlled multi-agent design pattern for building reliable and automated reasoning systems.

In this structure, the **Expert agent** is responsible for generating the primary analysis or decision, acting as the main problem solver. The **Shadow agent** operates independently to review or validate the Expert’s reasoning, helping to identify potential errors, biases, or missing considerations. The **UserProxy agent** serves as a controller that manages the interaction flow, monitors completion criteria, and determines when the task should be terminated.

By separating execution, verification, and control into distinct agents, this structure improves robustness, interpretability, and stability compared to single-agent systems. It is particularly suitable for high-stakes applications such as financial analysis, where correctness and controlled termination are critical.

### 3. Optimization (Optional)

You can add some useful parts or financial ratios to let report reader have a better understanding of this company's performance.

### 4. What to Submit

Create `submissions/Assignment1_Name_UNI/` and include:

1. **The 5+ generated PDF reports** — same sector, ideally with different price trends
2. **Your notebook**, with the `company` / `competitors` / `fyear` settings you used for each run
   (and the config file names you renamed, *without* the keys in them)
3. **Your own analysis** (PDF) — not the AI-generated reports, but your comparison of these
   companies against their historical performance and current market information: where the agent
   was right, where it was shallow, where it was wrong
4. **`README.md`** listing which companies you covered, which LLM you used, roughly what each run
   cost in API tokens, and the link to your Medium blog

> [!WARNING]
> ⚠️ **DO NOT SUBMIT YOUR OWN API KEY ONTO GITHUB!!!** `OAI_CONFIG_LIST` and `config_api_keys`
> hold live keys — keep them out of the commit, and clear notebook outputs that echo them.


