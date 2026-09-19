# GR5398 26 Fall — Assignment 1

Assignment 1 covers all three research tracks. Read the instructions for the track(s) you are
doing, and submit into that track's `submissions/` folder.

## The three tracks

| Track | Instructions | Starter code |
| --- | --- | --- |
| [FinRL-Trading](./FinRL-Trading) — Quantitative Trading Strategy | [README](./FinRL-Trading/README.md) | [`source_code/`](./FinRL-Trading/source_code) |
| [FinGPT](./FinGPT) — Financial Large Language Models | [README](./FinGPT/README.md) | [`source_code/`](./FinGPT/source_code) |
| [FinRobot](./FinRobot) — Equity Research AI Agent | [README](./FinRobot/README.md) | [`source_code/`](./FinRobot/source_code) |

## What you will learn

Assignment 1 is your hands-on onboarding: you run a real pipeline end to end before proposing
your own research. Read this first so you know what each track asks of you.

### FinRL-Trading — the quantitative trading pipeline

- Run a simplified but complete FinRL-Trading workflow end to end, so you know what the rest of
  the semester looks like.
- Build a portfolio from a selected universe, and learn the fundamentals of quantitative
  trading — especially **stock selection**.
- Implement a **full backtest** on real historical data, instead of stopping at model metrics.
- Judge a strategy on more than cumulative return.

### FinGPT — fine-tuning and evaluating financial LLMs

- The math and the engineering of **LoRA / PEFT**: the low-rank update `ΔW = BA`, and where
  adapters are injected (`q/k/v/o_proj` in attention, `gate/up/down_proj` in the feed-forward
  layers) while the pretrained weights stay frozen.
- The full training stack: Dataset → Tokenizer → DataCollator → PEFT model → Trainer, run with
  DeepSpeed, 8-bit quantization, and Wandb monitoring.
- Trading off hyperparameters under a VRAM budget (`max_length`, `batch_size`,
  `gradient_accumulation_steps`, `torch_dtype`).
- *Optional*: build your own dataset — pull news from Finnhub, generate reference answers with
  an LLM, and assemble prompts for a universe such as NASDAQ-100.
- **Compare your fine-tuned model against the teacher model quantitatively**: binary accuracy,
  MSE, ROUGE-1/2/L, and inference time — not "it looks reasonable".

### FinRobot — multi-agent equity research

- How to design a **Multi-Agent System (MAS)**: decompose a complex task into role-specific
  agents that communicate through structured messages.
- The **Expert / Shadow / UserProxy** pattern — execution, verification, and control separated
  into distinct agents, which is what makes the system reliable and makes it terminate.
- Generate annual research reports for at least 5 companies in one sector, automatically.
- Then do the part the agents cannot: analyze those reports yourself against real historical
  performance and current market information.

### Across all three tracks

- **Engineering discipline**: reproducible submissions, a README that lets someone else run your
  code, and a normal PR workflow.
- **Secret hygiene**: API keys live in `.env` or environment variables, never in the repo.
- **Technical writing**: turn your experiments into a Medium blog post written for an outside
  reader.

These skills map directly onto how the course evaluates work later in the semester — prediction
quality (FinGPT), return and risk (FinRL-Trading), agent capability (FinRobot), and efficiency
(all of them).

## How many tracks should I do?

It depends on how many credits you are registered for:

| Registered credits | Expectation |
| --- | --- |
| **0–2 credits** | Pick **any one** of the three tracks and complete it. |
| **2–3 credits** | You are strongly encouraged to complete **all three** tracks. |

If you do more than one track, submit each one separately under its own track's `submissions/`
folder (same `Assignment1_Name_UNI` naming in each).

## How to submit

1. Create a **new folder** (not a new branch) under the track's `submissions/` directory,
   named `Assignment1_Name_UNI` — e.g. `Assignment1_JaneDoe_jd1234`.
2. Put your code, notebooks, and report in that folder, plus a short `README.md` describing
   what you did and how to run it.
3. Include the link to your Medium blog post in that `README.md`.
4. Open a pull request against `master`.

> [!WARNING]
> ⚠️ **DO NOT SUBMIT YOUR OWN API KEY ONTO GITHUB!!!**

**Due: Oct 20, 2026.**
