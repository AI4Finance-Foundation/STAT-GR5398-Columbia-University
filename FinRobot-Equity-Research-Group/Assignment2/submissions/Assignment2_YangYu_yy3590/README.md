# FinRobot Equity Research Enhancement: Frontier LLM Experiments, FMP-Aware Caching, Two-Stage Evaluation, Section Mixing, and Role-Mixed Analyst/Critic Workflow

## Overview

This project extends the FinRobot equity research workflow with a reproducible experiment framework across frontier models, plus workflow-level enhancements inspired by practical analyst/reviewer pipelines.

The baseline FinRobot flow already supports:

1. Financial data collection and metric processing.
2. Forecast generation and peer comparison.
3. Multi-section equity report assembly.

This enhancement adds three major capabilities on top of the baseline:

1. **Scalable multi-model batch experiments** over required companies.
2. **FMP-aware caching and fallback** to reduce API bottlenecks and stabilize runs.
3. **Evaluation and report synthesis layers** (two-stage scoring, section-level mapping, and role-mixed analyst/critic construction).

The final accepted benchmark run in this repository is:

- `run_id = 20260501_005147`
- universe: `NVDA, AMD, INTC, AAPL, GOOGL`
- base models: `openai_gpt-5.4`, `claude_claude-opus-4-6`, `gemini_gemini-3.1-pro-preview`

---

## Project Goals

This work targets both assignment tracks in `requirements.md`.

1. Replace base models with frontier LLMs and compare outcomes.
2. Add meaningful equity-research workflow improvements beyond model swap.
3. Build a reproducible, batchable, and evaluable pipeline.
4. Compare unified-model, section-mixed, and role-mixed strategies.
5. Select and justify the best-performing setup using structured evidence.

---

## Repository Structure

```text
FinRobot/
├── run_experiments.py          # Batch generation + baseline two-stage evaluation
├── evaluate_sections.py        # Section-level scoring and global section mapping
├── build_mixed_reports.py      # Build mixed reports from section mapping + optional re-eval
├── run_role_mixed.py           # Build/evaluate analyst-critic role-mixed reports
├── output/
│   ├── cache/fmp/              # FMP endpoint-level cache store
│   └── storage/<run_id>/       # Reproducible run artifacts
└── finrobot_equity/
    └── core/src/
        ├── generate_financial_analysis.py
        ├── create_equity_report.py
        └── modules/
            ├── market_data_api.py
            ├── llm_gateway.py
            ├── text_generator_agents.py
            └── ...
```

---

## Pipeline

### 1) Baseline report generation (3 models x 5 tickers)

`run_experiments.py` runs task-queued model x ticker jobs and archives outputs under a single run id.

Each task executes:

1. `generate_financial_analysis.py`
2. `create_equity_report.py`

Then baseline evaluation is applied with two stages:

1. **Stage1** independent scoring for every report (default judge: `gpt-4o-mini`).
2. **Stage2** pairwise review for close competitors only (default judge: `gpt-5-nano`).

Final scores are merged with blend logic where stage2 is executed.

### 2) Section-level model selection

`evaluate_sections.py` scores section artifacts across models and writes:

- `section_scores.json/.csv`
- `global_section_mapping.json`

Selection rule:

- highest section mean score
- tie-break by lower stddev
- then provider cost priority

### 3) Mixed report assembly by section

`build_mixed_reports.py` copies section files by global mapping into `mixed_analysis/`, rebuilds full reports, and evaluates them into:

- `evaluation_mixed_summary.json`

### 4) Role-mixed analyst/critic construction

`run_role_mixed.py` builds two role combos from existing reports:

1. `role_mixed_a`: analyst=`claude`, critic=`gemini`
2. `role_mixed_b`: analyst=`gemini`, critic=`claude`

The critic revises selected sections (`investment_overview`, `risks`, `major_takeaways` by default), then reports are rebuilt and evaluated into:

- `role_mixed_manifest.json`
- `evaluation_role_mixed_summary.json`
- updated `final_scoreboard.csv/.md`

---

## FMP Caching and API-Quota Resilience

A key engineering bottleneck was FMP free-tier call limits. The pipeline was upgraded with endpoint-level caching in `market_data_api.py`.

Implemented cache behavior:

1. **Same-day cache hit**: if endpoint+ticker+params were fetched on the same local date, read from cache (`cache_hit`) instead of refetching.
2. **Fresh fetch**: if no valid same-day cache, query network and write cache (`fresh_fetch`).
3. **Stale fallback**: on fetch failure, fallback to existing cached payload when available (`fallback_stale_cache`).

Run-level aggregation in `run_experiments.py` tracks cache-origin counts and cache-hit ratio, improving throughput and reproducibility under quota pressure.

---

## Experiments and Results (Run: 20260501_005147)

### Baseline model averages (from `final_scoreboard.csv`)

| Setup | Mean Final Score |
|---|---:|
| `openai_gpt-5.4` | 49.192 |
| `claude_claude-opus-4-6` | 53.464 |
| `gemini_gemini-3.1-pro-preview` | 53.732 |

Per-ticker baseline winners:

- `AAPL`: gemini
- `AMD`: gemini
- `GOOGL`: claude
- `INTC`: claude
- `NVDA`: openai

### Section-mixed results

| Setup | Mean Final Score |
|---|---:|
| `mixed` | 47.132 |

Observation: section-mixed underperformed all major baselines on mean score.

Likely reason observed in practice:

- Section-level local optimality did not guarantee report-level global coherence.
- Cross-section transitions, thesis continuity, and narrative consistency degraded after composition.

### Role-mixed analyst/critic results

| Setup | Mean Final Score |
|---|---:|
| `role_mixed_a` (claude analyst, gemini critic) | 47.948 |
| `role_mixed_b` (gemini analyst, claude critic) | 54.828 |

Role-mixed winner frequency:

- `role_mixed_b` won 5/5 tickers inside role-mixed comparisons.

This run’s strongest overall mean came from `role_mixed_b`.

---

## Assignment Alignment

This implementation covers the required tasks in `requirements.md`:

1. **Base model replacement**: OpenAI/Claude/Gemini all integrated and tested.
2. **Model combinations**: unified baselines, section-mixed, and role-mixed analyst/critic.
3. **Workflow enhancement**: FMP-aware cache/fallback, staged evaluation, section-level mapping.
4. **Prompt/role refinement**: role-specific critic revision flow for key sections.
5. **Result comparison and analysis**: structured scoreboards and per-run summaries.

---

## Reproducibility

### 1) Configure credentials

Use `config_template.txt` as template and provide real keys in:

- `finrobot_equity/core/config/config.ini`

### 2) Core commands

Baseline batch run:

```bash
python run_experiments.py --mode full --enable-two-stage-eval --run-id <RUN_ID>
```

Section evaluation:

```bash
python evaluate_sections.py --run-id <RUN_ID>
```

Build and evaluate section-mixed reports:

```bash
python build_mixed_reports.py --run-id <RUN_ID> --evaluate-mixed --mixed-eval-stage2 --skip-auto-fetch
```

Build and evaluate role-mixed reports:

```bash
python run_role_mixed.py --run-id <RUN_ID> --skip-auto-fetch
```

### 3) Main artifacts

Under `output/storage/<RUN_ID>/`:

1. `reports_index.json`
2. `evaluation_summary.json`
3. `global_section_mapping.json`
4. `evaluation_mixed_summary.json`
5. `evaluation_role_mixed_summary.json`
6. `final_scoreboard.csv`

---

## Limitations and Notes

1. Findings are conditioned on current prompt budgets and high token ceilings.
2. Cross-provider token accounting is not directly comparable without normalization.
3. API bottlenecks are dominated by FMP constraints rather than LLM inference limits.
4. Section-mixed strategy should be treated as a controlled experiment, not an automatic quality booster.

---

## Final Takeaway

For this accepted run, the best practical configuration was:

```text
Role-mixed B = Gemini analyst + Claude critic
```

The project demonstrates that workflow design (caching, staged evaluation, and role assignment) can have as much impact as raw base-model choice.
