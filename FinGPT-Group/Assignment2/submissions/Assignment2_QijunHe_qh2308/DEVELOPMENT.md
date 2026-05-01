# FinGPT Two-Agent Signal Pipeline — Development Guide

## Current State

- Inference runs on a local vLLM instance (DeepSeek-R1-Distill-Llama-8B).
- Agent 1 extracts a `NewsFingerprint` (facts + 3-class sentiment) using real model log-probabilities.
- Agent 2 converts the fingerprint into a `TradingSignal` using real model log-probabilities + PMI prior correction.
- Both agents use a two-phase vLLM approach: Phase 1 generates chain-of-thought (CoT) reasoning, Phase 2 scores choices via `prompt_logprobs`.
- Backtest mode processes articles in batches of 10 (5 vLLM calls per batch).
- `resume_backtest.ipynb` allows re-processing a previous result CSV without re-running Agent 2.

---

## 1) Architecture

### Signal generation (per article)

```
News article text
    │
    ▼
Agent 1 — extractor.py
  ├─ Call 1: Guided decoding (StructuredOutputParams)
  │           → raw facts: source, headline, companies, keywords
  ├─ Call 2: CoT generation  stop=["</think>"]
  │           → chain-of-thought sentiment reasoning
  └─ Call 3: prompt_logprobs scoring × 3 (POSITIVE / NEGATIVE / NEUTRAL)
              softmax(log_probs / CALIBRATION_T)  ← Python post-processing
              → NewsFingerprint (sentiment_label, confidence, probabilities)
    │
    ▼
Agent 2 — reasoner.py
  ├─ Call 4: CoT generation  stop=["</think>"]
  │           → chain-of-thought trading strategy reasoning
  └─ Call 5: prompt_logprobs scoring × 3  ("A"=BUY / "B"=HOLD / "C"=SELL)
              PMI correction: pmi[i] = logp[i] − null_logp[i]
              softmax(pmi / CALIBRATION_T)  ← Python post-processing
              → TradingSignal (direction, confidence, signal_probabilities)
    │
    ▼
Price fetch — price_fetcher.py
  yfinance interval="1d"  (5 trading days / 7-day window)
  close-to-close realized return
    │
    ▼
Metrics: direction_accuracy, Sharpe, total_pnl, vs_fingpt_accuracy
```

**Batch processing:** articles are grouped in batches of 10; each batch makes 5 `engine.generate()` calls (not 5 × 10).

### Backtest flow

```
HF dataset / local parquet / local CSV
  → backtest/dataset_parser.py    (parses multi-headline blocks)
  → backtest/backtester.py        (orchestrates batches)
      ├─ extract_fingerprint_batch()
      ├─ generate_signal_batch()
      └─ get_realized_return()
  → output/backtest_<timestamp>.csv + metrics dict
```

### Resume flow (no Agent 2 re-run)

```
backtest_resume_*.csv  (has signal_logits column with A/B/C logprobs)
  → notebooks/resume_backtest.ipynb  Track A
      ├─ Empirical PMI correction  (mean-centre logits across rows)
      ├─ Re-derive signal_direction / confidence / probabilities
      ├─ Inline yfinance fetch  (interval="1d", cache cleared)
      └─ Metrics + backtest_pmi_<timestamp>.csv
```

---

## 2) Key Files

```
FinGPT_Part2/
├── config.py                    # Global constants, env-var loading
├── pipeline.py                  # Live news → signal entry point
├── vllm_logits_client.py        # vLLM wrapper: two-phase real logprobs + legacy
│
├── agent1/
│   ├── extractor.py             # extract_fingerprint() / extract_fingerprint_batch()
│   ├── prompt.py                # EXTRACTION_PROMPT, SENTIMENT_COT_PROMPT, SENTIMENT_DECISION_PREFIX
│   └── schema.py                # NewsFingerprint (Pydantic)
│
├── agent2/
│   ├── reasoner.py              # generate_signal() / generate_signal_batch() + PMI correction
│   ├── prompt.py                # STRATEGY_COT_PROMPT, STRATEGY_DECISION_PREFIX, STRATEGY_SCORE_TOKENS
│   └── schema.py                # TradingSignal (Pydantic)
│
├── backtest/
│   ├── __init__.py
│   ├── dataset_parser.py        # HF / parquet / CSV loading
│   ├── price_fetcher.py         # yfinance daily-interval fetch + dual cache
│   ├── backtester.py            # run_backtest() / compute_metrics()
│   └── run_backtest.py          # CLI entry point
│
├── ingestion/
│   └── news_fetcher.py          # Alpaca / Finnhub live news
│
├── evaluation/
│   ├── evaluator.py
│   └── collect_batch.py
│
├── notebooks/
│   ├── demo.ipynb               # Full pipeline Colab demo
│   └── resume_backtest.ipynb    # Post-process existing CSV (no GPU for Track A)
│
└── DEVELOPMENT.md
```

---

## 3) Real Logprobs — Two-Phase Design

### Why not self-reported logits?

Asking a local LLM to output `{"logits": [f1, f2, f3]}` produces numbers that correlate poorly with the model's actual token probabilities, because the model is predicting tokens in a conversational context, not introspecting its weights.

### Two-phase approach

**Phase 1 — CoT generation**

```python
SamplingParams(max_tokens=1024, temperature=0.0, stop=["</think>"])
```

The model reasons freely inside `<think>…</think>` and stops before producing a commitment.

**Phase 2 — Scoring via `prompt_logprobs`**

For each candidate choice (e.g. "POSITIVE"):

```python
scoring_prompt = original_prompt + cot_text + "</think>\n" + decision_prefix + choice
SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
```

vLLM always includes the actual prompt token's log-probability, so we read `log P(choice_token | context)` directly from the model's computation graph, not from the model's text generation.

For multi-token choices (e.g. "POSITIVE" → ["POS", "ITIVE"]), individual token logprobs are summed:

```
log P("POSITIVE" | ctx) = log P("POS" | ctx) + log P("ITIVE" | ctx + "POS")
```

### Agent 1 scoring tokens

`SENTIMENT_DECISION_PREFIX = "Sentiment: "`  
Choices: `["POSITIVE", "NEGATIVE", "NEUTRAL"]`

The word "POSITIVE" tokenizes cleanly and has sufficiently distinct priors at this position.

### Agent 2 scoring tokens and PMI correction

**Problem:** After `"The answer is ("`, the token `"B"` has an unconditional LM prior of ~0.75 regardless of article content.  
This swamps the news signal and causes HOLD to dominate.

**Solution — A/B/C tokens + PMI correction:**

```python
STRATEGY_DECISION_PREFIX = "The answer is ("
STRATEGY_SCORE_TOKENS    = ["A", "B", "C"]   # A=BUY, B=HOLD, C=SELL
```

Single ASCII letters have lower and more balanced priors than the words BUY/HOLD/SELL.  
PMI correction removes whatever prior remains:

```python
# computed once per engine session with a neutral null article
null_logprobs = _compute_null_logprobs()          # [logP(A|null), logP(B|null), logP(C|null)]

# per article
pmi_logits    = [raw[i] - null_logprobs[i] for i in range(3)]
probs         = softmax(pmi_logits, T=CALIBRATION_T)
strategy      = STRATEGY_SET[argmax(probs)]       # BUY / HOLD / SELL
```

Empirical PMI (Track A in `resume_backtest.ipynb`) estimates the null prior as the mean logprob across all articles in the CSV, requiring no model re-run.

---

## 4) Agent Behavior

### Agent 1 (`agent1/extractor.py`)

- `extract_fingerprint(article_text) → Optional[NewsFingerprint]`
- `extract_fingerprint_batch(articles) → list[Optional[NewsFingerprint]]`
- Phase 1: guided JSON extraction (facts only, no sentiment).
- Phase 2: sentiment CoT + `prompt_logprobs` scoring.
- `set_shared_vllm_engine(engine)` injects a pre-loaded engine.
- `get_shared_vllm_engine()` lets Agent 2 borrow the same engine.

### Agent 2 (`agent2/reasoner.py`)

- `generate_signal(fingerprint) → Optional[TradingSignal]`
- `generate_signal_batch(fingerprints) → list[Optional[TradingSignal]]`
- PMI null logprobs computed once on first call, cached in `_null_logprobs`.
- `set_shared_vllm_engine(engine)` resets `_null_logprobs` (re-computed for new engine).
- Strict mode: `None` returned if `parse_success=False`; diagnostic markdown saved to `FINGPT_DIAG_MD_DIR/agent2`.

### `vllm_logits_client.py`

Public functions:

| Function | Purpose |
|---|---|
| `softmax(logits, temperature)` | Numerically stable softmax with temperature scaling |
| `extract_thinking(raw_output)` | Extracts CoT text from `<think>…</think>` |
| `get_real_choice_logits(...)` | Single-article two-phase logprob extraction (PRIMARY) |
| `get_real_choice_logits_batch(...)` | Batch version — 2 `engine.generate()` calls for N articles (PRIMARY) |
| `get_choice_logits(...)` | Legacy self-reported JSON logits (kept for reference) |
| `get_choice_logits_batch(...)` | Legacy batch version (kept for reference) |

---

## 5) Price Fetching

`backtest/price_fetcher.py` computes close-to-close realized returns.

**Key settings:**

| Parameter | Value | Reason |
|---|---|---|
| `interval` | `"1d"` | A 7-day date window with `"1wk"` returns at most 1 bar; daily gives 5 trading days |
| MultiIndex flatten | `hist.columns.get_level_values(0)` | yfinance ≥0.2.38 returns `("Close","AXP")` columns for a single ticker |

**Two-level cache:**

- `_RETURN_CACHE` — in-memory `dict[tuple, float]`, lives for the Python process.
- `_DISK_CACHE` — JSON file at `FINGPT_YF_CACHE_PATH` (default `output/yfinance_return_cache.json`).

**Cache pitfall:** `_RETURN_CACHE` is module-level and persists across Jupyter cells in the same kernel session. If a previous run stored `None` values, subsequent calls return `None` from memory before fetching. Clear it explicitly when reusing a kernel:

```python
import backtest.price_fetcher as _pf
_pf._RETURN_CACHE.clear()
_pf._DISK_CACHE.clear()
_pf._DISK_CACHE_LOADED = False
```

Or use the fresh cache path env var in the resume notebook:

```python
os.environ["FINGPT_YF_CACHE_PATH"] = "/tmp/yfinance_return_cache_resume.json"
```

---

## 6) vLLM Colab / Jupyter Compatibility

**Problem:** vLLM ≥0.6 (v1 engine) spawns an `EngineCore` subprocess that calls `sys.stdout.fileno()`. IPython's `OutStream` has no real file descriptor → crash:

```
io.UnsupportedOperation: fileno
```

**Fix 1 — Disable multiprocessing (primary, set before any `LLM()` call):**

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

Forces the v1 engine to use `InprocClient` (no subprocess, no `fileno` call).

**Fix 2 — stdout redirect (belt-and-suspenders):**

```python
_nb_stdout = sys.stdout
try:
    sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)   # real fd
    engine = LLM(...)
finally:
    sys.stdout = _nb_stdout   # restore notebook stdout
```

If a subprocess is still spawned, it inherits a real file object and `fileno()` succeeds.

Both fixes are applied in `demo.ipynb` Cell 3 (env) and Cell 4 (engine load), and in `resume_backtest.ipynb` Cell B2/B4.

---

## 7) Config Constants (`config.py`)

| Constant | Default | Purpose |
|---|---|---|
| `FINGPT_MODEL_PATH` | `""` (env var) | Path to local HF-format model weights |
| `SHARE_SINGLE_LLM_BETWEEN_AGENTS` | `False` | One vLLM engine shared by both agents |
| `CALIBRATION_T` | `1.2` | Softmax temperature for logprob → probability |
| `SENTIMENT_CLASSES` | `["POSITIVE","NEGATIVE","NEUTRAL"]` | Agent 1 scoring labels |
| `STRATEGY_SET` | `["BUY","HOLD","SELL"]` | Agent 2 strategy labels (display) |
| `LOGITS_MAX_TOKENS` | `1024` | CoT token budget per article |
| `FINGPT_YF_CACHE_PATH` | `output/yfinance_return_cache.json` | Price cache disk path |

---

## 8) Troubleshooting Playbook

### A) `io.UnsupportedOperation: fileno` on `LLM()` init

**Cause:** vLLM v1 engine subprocess calls `sys.stdout.fileno()` in a Jupyter/Colab kernel.

**Fix:** set `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"` before importing vLLM, and wrap `LLM()` in the `sys.stdout` redirect pattern (see §6).

---

### B) Agent 2 always returns HOLD (neutral)

**Cause A:** Scoring words "BUY"/"HOLD"/"SELL" at `"Strategy: "` — the word "HOLD" has a strong LM prior at this position.

**Fix:** Use `STRATEGY_SCORE_TOKENS = ["A","B","C"]` with `STRATEGY_DECISION_PREFIX = "The answer is ("`. ✓ Already applied.

**Cause B:** Even A/B/C has a residual B-prior (~0.75). News signal cannot overcome it.

**Fix:** PMI correction — subtract null-context logprobs. ✓ Applied in `agent2/reasoner.py` (`_compute_null_logprobs` + `_null_logprobs` subtraction in `_process_signal_result`).

**Post-hoc fix (no model re-run):** Use Track A in `resume_backtest.ipynb` — empirical PMI from existing CSV logits.

---

### C) All price fetches return `None` (`price_fetch_failed`)

**Cause A:** `interval="1wk"` over a 7-day window → 1 weekly bar → `len < 2` guard.

**Fix:** `interval="1d"` in `price_fetcher.py`. ✓ Applied.

**Cause B:** `_RETURN_CACHE` stale from a previous kernel session run.

**Fix:** Clear the three module globals before fetching (see §5).

**Cause C:** yfinance ≥0.2.38 MultiIndex columns — `hist["Close"]` returns a DataFrame.

**Fix:** `hist.columns = hist.columns.get_level_values(0)` before indexing. ✓ Applied.

---

### D) `ValueError: Unsupported dataset format`

**Cause:** Old `dataset_parser` module cached in kernel.

**Fix:** `importlib.reload(backtest.dataset_parser)` or restart runtime.

---

### E) `ValueError: The truth value of a Series is ambiguous`

**Cause:** HF→pandas row value is `Series` (duplicate column labels), not scalar.

**Fix:** Current parser normalises duplicate columns — ensure the latest `dataset_parser.py` is loaded.

---

### F) Agent 2 diagnostic markdown contains only reasoning, no structured output

**Cause:** Legacy path (self-reported logits) or model output cut off by `max_tokens`.

**Fix:** The real-logprobs path (`get_real_choice_logits`) does not require the model to produce structured output — diagnosis is different. Check `parse_success` field in the result dict; if `False`, vLLM `prompt_logprobs` indexing may have failed (BPE boundary issue). Increase `LOGITS_MAX_TOKENS` or inspect `_sum_choice_logprob` warnings in the log.

---

## 9) Operational Recommendations

- Use `SHARE_SINGLE_LLM_BETWEEN_AGENTS=true` on Colab to halve GPU memory pressure.
- Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before any `LLM()` call in notebooks.
- Keep `CALIBRATION_T=1.2`; values < 1 sharpen distributions but can over-commit.
- Keep PMI correction enabled — turning it off restores the HOLD-dominance bias.
- Track skip reasons (`fingerprint_failed`, `signal_failed`, `price_fetch_failed`) as pipeline health signals.
- Use a fresh `FINGPT_YF_CACHE_PATH` per resume run to avoid stale None entries.

---

## 10) Quick Commands

Run full backtest from CLI:

```bash
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --metrics
```

Run a subset:

```bash
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --max-rows 50 --metrics
```

Reload modules in a live Colab kernel after code changes:

```python
import importlib
import agent1.prompt, agent1.extractor
import agent2.prompt, agent2.reasoner
import backtest.dataset_parser, backtest.price_fetcher, backtest.backtester
import vllm_logits_client

for m in [agent1.prompt, agent1.extractor, agent2.prompt, agent2.reasoner,
          backtest.dataset_parser, backtest.price_fetcher, backtest.backtester,
          vllm_logits_client]:
    importlib.reload(m)
```
