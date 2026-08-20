# Assignment 2 — Xian Zhang (xz3447)

GR5398 Spring 2026 · FinRobot Equity Research AI Agent Track

## What's in this folder

```
Assignment2_Xian Zhang_xz3447/
├── README.md                    ← you are here
├── requirements.txt             ← Python deps for reproducing the runs
├── comparison.md                ← cross-profile comparison stats
├── reports/                     ← 20 PDF reports (5 companies × 4 model configs)
│   ├── gpt_baseline/            all-OpenAI baseline
│   ├── claude_all/              all-Claude (Opus + Haiku)
│   ├── gemini_all/              all-Gemini (2.5 Pro + Flash)
│   └── mixed_critic/            Claude author + GPT critic + Gemini light tasks
└── code/                        ← my new modules + scripts
    ├── llm_providers/           multi-provider abstraction (OpenAI/Anthropic/Gemini)
    ├── equity_prompts.py        rich analyst prompts (incl. critic prompt)
    ├── model_routing.py         the 4 routing profiles
    ├── text_generator_agents.py main pipeline entry, with critique-revise
    └── scripts/
        ├── run_experiments.py       batch runner (5 companies × 4 profiles)
        ├── compare_outputs.py       cross-profile comparison stats
        └── generate_all_pdfs.py     batch PDF generation
```

## What I did

**Track A — Model upgrade.** Built a provider-agnostic abstraction (`llm_providers/`) so the FinRobot pipeline can route each section's LLM call to OpenAI, Anthropic, or Google Gemini based on the model name. Tested four routing profiles on the five required companies (NVDA, AMD, INTC, AAPL, GOOGL):

| Profile | Heavy sections | Light sections | Critic |
|---|---|---|---|
| `gpt_baseline`  | gpt-4.1         | gpt-4o-mini       | — |
| `claude_all`    | claude-opus-4-5 | claude-haiku-4-5  | claude-opus-4-5 |
| `gemini_all`    | gemini-2.5-pro  | gemini-2.5-flash  | gemini-2.5-pro |
| `mixed_critic`  | claude-opus-4-5 | gemini-2.5-flash  | gpt-4.1 (cross-family) |

**Track B — Claude-inspired enhancement.** Added a Critique-Revise stage on top of the pipeline, modeled on Anthropic's Financial Services equity-research workflow. For the three highest-stakes sections (`investment_overview`, `risks`, `valuation_overview`), a Senior Reviewer agent (potentially in a different model family) issues a structured verdict (`ACCEPT` / `REVISE` / `MAJOR_REVISE`) plus a numbered list of specific issues. The original analyst then rewrites the section conditioned on the critique. Every step writes a JSON audit log so the pipeline is fully reproducible and debuggable.

## Headline finding

Across 20 reports, Claude-authored profiles cite **~1.8× more concrete numbers per section** than GPT or Gemini (29.0 vs ~15.6 figures/section on average). On the `risks` section specifically, the cross-family `mixed_critic` profile (Claude author + GPT critic) hits **39.2 figures vs gpt_baseline's 12.6 — a 3× lift**.

Full per-section / per-cell stats are in `comparison.md`.

## How to reproduce

1. Install dependencies (Python 3.10+; tested on 3.11 / conda):

   ```bash
   pip install -r requirements.txt
   ```

2. Drop the files in `code/` into the FinRobot Equity project tree:
   - `code/llm_providers/`, `code/equity_prompts.py`, `code/model_routing.py`, `code/text_generator_agents.py` → `finrobot_equity/core/src/modules/`
   - `code/scripts/*.py` → `finrobot_equity/scripts/`

3. Populate `finrobot_equity/core/config/config.ini` with API keys for FMP, OpenAI, Anthropic, and Gemini.

4. Run the full experiment matrix:

   ```bash
   cd finrobot_equity
   python scripts/run_experiments.py            # generates all 20 reports' analysis + HTML
   python scripts/generate_all_pdfs.py          # converts to PDF
   python scripts/compare_outputs.py            # cross-profile comparison stats
   ```

A profile is selected per-call via the `FINROBOT_PROFILE` environment variable; the runner takes care of setting it.

## Notes for the reviewer

- All 20 reports were generated from real FMP API data (no mocks).
- Each cell also has a per-section audit JSON capturing draft, critique, verdict, and final text. I kept these in the source repo (`finrobot_equity/core/output/experiments/<profile>/<TICKER>/audit/`) but they are not in this submission folder to keep size down.
- The Medium-style blog post (Deliverable B) is not included in this submission; only Deliverable A (reports) and the supporting code are submitted here.

— Xian Zhang (xz3447)
