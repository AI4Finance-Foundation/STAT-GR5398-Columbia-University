# FinRobot Integrated Research Overlay

## Overview

This repository contains the FinRobot side of an integrated portfolio-research workflow. FinRobot generates point-in-time equity research snapshots, extracts structured research flags with an LLM, applies deterministic portfolio-adjustment rules, and writes auditable benchmark outputs under a single integration run folder.

The integrated workflow is the main entry point for the current project:

```text
external base weights
  -> FinRobot point-in-time research snapshots
  -> structured research flags
  -> deterministic overlay rules
  -> FinRL-only, FinRL+FinRobot, and SPY benchmark outputs
```

The base weights are treated as an input file. FinRobot does not depend on how those weights were produced; its responsibility is research generation, flag extraction, rule-based adjustment, benchmarking, and auditability.

Related Medium Blog Post: https://medium.com/@yy3590/beating-spy-with-an-integrated-ai-trading-strategy-finrl-trading-finrobot-and-fingpt-b65c91f29bdd

---

## What Was Added

The main project additions are:

1. **Integrated backtest runner**: `finrobot_equity/core/src/integrated_backtest.py`.
2. **Automated research snapshots**: generate, reuse, or regenerate ticker/date snapshots inside one integration run.
3. **Active research setup registry**: one JSON file selects the production LLM setup used by integrated runs.
4. **Structured research flags**: fixed labels for risk, valuation, catalyst, sentiment, and thesis strength.
5. **Deterministic overlay rules**: configurable guardrails convert flags into portfolio weight adjustments.
6. **FMP-aware caching**: repeated runs reuse exact endpoint/ticker/parameter cache hits unless explicitly refreshed.
7. **Enhanced news/catalyst support**: optional point-in-time news inputs for the research and flagging workflow.

The older model-comparison scripts are still included, but they are supporting utilities. They can be used to decide which LLM setup should be written into `active_research_setup.json`; they are not required every time the integrated workflow is run.

---

## Repository Structure

```text
FinRobot/
├── README.md
├── run_experiments.py              # Supporting model-comparison workflow
├── evaluate_sections.py            # Supporting section-level evaluation
├── build_mixed_reports.py          # Supporting mixed-report builder
├── run_role_mixed.py               # Supporting analyst/critic comparison
├── output/
│   ├── cache/fmp/                  # Shared low-level FMP cache
│   ├── storage/<run_id>/           # Model-comparison artifacts
│   └── integration/<run_id>/       # Integrated workflow artifacts
└── finrobot_equity/
    └── core/
        ├── config/
        │   ├── config.ini
        │   └── active_research_setup.json
        └── src/
            ├── integrated_backtest.py
            ├── generate_financial_analysis.py
            ├── create_equity_report.py
            ├── integrations/
            │   ├── active_setup.py
            │   ├── backtest_runner.py
            │   ├── research_automation.py
            │   ├── research_flags.py
            │   ├── flag_rules.py
            │   └── weights.py
            └── modules/
                ├── market_data_api.py
                ├── news_integrator.py
                ├── llm_gateway.py
                └── text_generator_agents.py
```

---

## Configure FinRobot

### API config

Provide real API keys in:

```text
finrobot_equity/core/config/config.ini
```

This file is used by the research generator, report builder, LLM gateway, FMP data fetches, and news integration.

### Active research setup

The integrated workflow reads the production model setup from:

```text
finrobot_equity/core/config/active_research_setup.json
```

This is the normal place to specify which model FinRobot should use for production research generation. The CLI also has `--active-research-setup` for experiments with an alternate registry file, but the default project workflow uses the single config file above.

Supported setup types:

- `single_model`: one provider/model generates all research sections.
- `role_mixed`: an analyst model generates the draft, then a critic model revises configured sections.

Example single-model setup:

```json
{
  "setup_type": "single_model",
  "selected_setup_id": "claude_production",
  "analyst": {
    "provider": "claude",
    "model": "claude-opus-4-6"
  },
  "source_comparison_run": "manual",
  "selected_at_utc": "2026-05-07T00:00:00Z",
  "notes": "Production setup for integrated backtests."
}
```

Example role-mixed setup:

```json
{
  "setup_type": "role_mixed",
  "selected_setup_id": "role_mixed_b",
  "analyst": {
    "provider": "gemini",
    "model": "gemini-3.1-pro-preview"
  },
  "critic": {
    "provider": "claude",
    "model": "claude-opus-4-6"
  },
  "critic_sections": ["investment_overview", "risks", "major_takeaways"],
  "source_comparison_run": "output/storage/20260501_005147",
  "selected_at_utc": "2026-05-07T00:00:00Z",
  "notes": "Selected by the comparison workflow."
}
```

Each integrated run snapshots this registry into:

```text
output/integration/<integration_run_id>/config/active_research_setup_snapshot.json
```

That makes the run reproducible even if the active setup is changed later.

---

## Prepare Base Weights

Pass the base-weight CSV with `--finrl-weights`. Relative paths are supported when running from the repository root.

The loader accepts the FinRL-style wide format:

```csv
date,AAPL,NVDA,GOOGL,SPY,CASH
2026-01-02,0.20,0.25,0.15,0.30,0.10
2026-01-09,0.18,0.27,0.14,0.31,0.10
```

It also accepts long format:

```csv
date,ticker,weight
2026-01-02,AAPL,0.20
2026-01-02,NVDA,0.25
2026-01-02,SPY,0.30
2026-01-02,CASH,0.10
```

If the input has weekly weights and `--research-frequency monthly`, the FinRobot overlay is refreshed monthly and then applied until the next research date. With `--research-frequency weekly`, the workflow attempts to generate or reuse research snapshots weekly.

---

## Run The Integrated Workflow

From the FinRobot repo root:

```powershell
python finrobot_equity\core\src\integrated_backtest.py `
  --start 2026-01-01 `
  --end 2026-04-30 `
  --research-frequency monthly `
  --finrl-weights output\finrl_weights.csv `
  --integration-run-id prod_2026q1_attempt7 `
  --research-mode generate-missing `
  --enable-enhanced-news `
  --enable-catalyst-analysis `
  --news-days-back 7 `
  --news-limit 100
```

Useful options:

- `--integration-run-id`: names the folder under `output/integration/`. Reusing the same id reuses or overwrites artifacts inside that run depending on `--research-mode` and cache flags.
- `--research-mode existing`: use existing snapshots only and fail if any required snapshot is missing.
- `--research-mode generate-missing`: reuse existing snapshots and generate only missing ticker/date snapshots.
- `--research-mode regenerate`: rebuild all required research snapshots for the run.
- `--force-refresh-fmp`: bypass existing FMP cache entries and fetch fresh data.
- `--force-refresh-flags`: rerun LLM flag extraction instead of using `flags_cache`.
- `--stock-boost-cash-funding-limit`: allow a limited amount of cash to fund positive stock boosts.
- `--stock-boost-etf-funding-limit`: allow a limited amount of ETF exposure to fund positive stock boosts.

For a full rebuild of the latest research snapshots and FMP data:

```powershell
python finrobot_equity\core\src\integrated_backtest.py `
  --start 2026-01-01 `
  --end 2026-04-30 `
  --research-frequency monthly `
  --finrl-weights output\finrl_weights.csv `
  --integration-run-id prod_2026q1_attempt7 `
  --research-mode regenerate `
  --force-refresh-fmp `
  --force-refresh-flags `
  --enable-enhanced-news `
  --enable-catalyst-analysis
```

---

## Integrated Output Layout

Each run writes to:

```text
output/integration/<integration_run_id>/
├── config/
│   └── active_research_setup_snapshot.json
├── research_snapshots/
│   └── <as_of_date>/<ticker>/
│       ├── analysis/
│       ├── report/
│       ├── config/
│       └── logs/
├── flags_cache/
├── weights/
│   ├── base_weights_normalized.csv
│   ├── adjusted_weights.csv
│   └── weight_history.csv
├── audit/
│   ├── research_snapshots.json
│   ├── research_flags.json
│   └── decision_audit.json
├── equity_curves.csv
├── metrics.json
└── integrated_backtest_summary.json
```

The most useful files for reviewing a run are:

- `metrics.json`: headline benchmark metrics.
- `equity_curves.csv`: FinRL-only, FinRL+FinRobot, and SPY curves.
- `audit/research_flags.json`: the extracted flags by ticker/date.
- `audit/decision_audit.json`: flags, multipliers, guardrails, and final weight decisions.
- `weights/weight_history.csv`: base and adjusted weights through time.
- `config/active_research_setup_snapshot.json`: exact model setup used for the run.

Do not upload full `output/integration/` trees directly to a public repository. Snapshot `config/` folders can contain copied API config files, logs are not needed, and older attempts are bulky generated artifacts. For sharing results, copy only sanitized summary files such as `metrics.json`, `equity_curves.csv`, `audit/*.json`, and `weights/*.csv`.

---

## Research Flags And Overlay Rules

The flagging LLM reads each ticker snapshot and emits a fixed schema:

```text
risk_flag
valuation_flag
catalyst_flag
sentiment_flag
thesis_strength
```

The LLM does not directly set portfolio weights. Instead:

1. `research_flags.py` extracts the structured labels.
2. `flag_rules.py` maps each label set to a multiplier.
3. `weights.py` applies deterministic portfolio guardrails.
4. `decision_audit.json` records the final decision.

The current guardrails include:

- no-op band for small multiplier changes
- maximum relative adjustment cap
- defensive-base behavior when the base portfolio is already highly defensive
- cash release from reduced positions
- optional cash or ETF funding for stock boosts
- final budget normalization

This design keeps the LLM role explainable and bounded: it classifies research, while portfolio impact is controlled by deterministic code.

---

## FMP Cache And News Behavior

Integrated backtests default to:

```text
--fmp-cache-policy reuse-until-forced
```

This reuses exact endpoint, ticker, and parameter cache hits across repeated runs until `--force-refresh-fmp` is passed. The shared FMP cache remains outside individual integration runs:

```text
output/cache/fmp
```

News cache keys remain exact-window. The key includes endpoint, ticker, exact `from`/`to`, and `limit` parameters. This avoids using a wide cached news window that may have been capped by the data provider and may not contain the articles needed for a specific as-of date.

Enhanced news and catalyst analysis are enabled with:

```text
--enable-enhanced-news
--enable-catalyst-analysis
```

The news window can be adjusted with:

```text
--news-days-back <days>
--news-limit <count>
```

---

## Supporting Model-Comparison Workflow

The following scripts are retained as supporting research tools:

- `run_experiments.py`: batch report generation across providers/models/tickers.
- `evaluate_sections.py`: section-level scoring and mapping.
- `build_mixed_reports.py`: mixed-report construction from selected sections.
- `run_role_mixed.py`: analyst/critic report comparison.

Their main purpose in the current project is to help choose the active setup recorded in:

```text
finrobot_equity/core/config/active_research_setup.json
```

They write artifacts under `output/storage/<run_id>/`. The integrated workflow does not rerun these comparisons; it only consumes the selected setup.

The accepted comparison run used during development was:

```text
run_id = 20260501_005147
```

That run selected `role_mixed_b` as the strongest analyst/critic setup, but integrated runs can also be executed explicitly with a single model by editing `active_research_setup.json`.

---

## Current Result Reference

The current assignment-facing integrated result is:

```text
output/integration/prod_2026q1_attempt7
```

Key benchmark result from that run:

| Strategy | Total Return |
|---|---:|
| SPY | 5.19% |
| FinRL-only | 17.09% |
| FinRL + FinRobot | 17.87% |

The run also confirmed that the enhanced news path was active: 36 research snapshots were processed, 418 total news articles were seen by the workflow, and 11 snapshots contained catalyst/news files.

---

## Notes

- Keep real credentials out of Git. `finrobot_equity/core/config/config.ini` should remain local.
- Keep generated caches out of Git unless a small sanitized result bundle is intentionally prepared.
- Use `generate-missing` for normal iteration because it preserves existing snapshots and avoids unnecessary FMP and LLM calls.
- Use `regenerate` only when changing research-generation prompts, model setup, or data-refresh assumptions.
