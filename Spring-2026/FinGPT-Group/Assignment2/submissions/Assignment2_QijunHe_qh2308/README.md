# FinGPT Two-Agent Signal Pipeline

A local, inference-only pipeline that reads financial news and produces calibrated
trading signals. Two specialised agents run sequentially on the same vLLM engine:
Agent 1 extracts structured sentiment from the article; Agent 2 reasons about the
appropriate trading action and returns a directional signal with a confidence score.

All decision logic lives in deterministic Python — the LLM never outputs numbers
or JSON for the signal itself.  Instead, the model's genuine token-level
log-probabilities are read directly from vLLM and converted to probabilities via
softmax.

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM runtime** | [vLLM](https://github.com/vllm-project/vllm) | Local high-throughput LLM serving, `prompt_logprobs`, guided decoding |
| **Foundation model** | DeepSeek-R1-Distill-Llama-8B-finetuned | Chain-of-thought reasoning model (Llama architecture) |
| **Schema / validation** | [Pydantic v2](https://docs.pydantic.dev/) | `NewsFingerprint`, `TradingSignal` data models |
| **Tokenisation** | [HuggingFace Transformers](https://github.com/huggingface/transformers) | Chat-template formatting, BPE token ID resolution |
| **Backtest dataset** | [HuggingFace Datasets](https://github.com/huggingface/datasets) | `FinGPT/fingpt-forecaster-dow30-202305-202405` |
| **Price data** | [yfinance](https://github.com/ranaroussi/yfinance) | Daily OHLC close-to-close realized returns |
| **Data wrangling** | [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) | DataFrame I/O, PMI correction, metrics |
| **Secrets / config** | [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env`-based environment variable loading |
| **Execution environment** | Google Colab (A100 / T4 GPU) | Recommended runtime for model loading |

---

## Project Structure

```
FinGPT_Part2/
├── config.py                    # Global constants, env-var loading
├── pipeline.py                  # Live news → signal entry point
├── vllm_logits_client.py        # vLLM wrapper: real two-phase logprobs
│
├── agent1/
│   ├── extractor.py             # Fact extraction + sentiment scoring
│   ├── prompt.py                # EXTRACTION_PROMPT, SENTIMENT_COT_PROMPT
│   └── schema.py                # NewsFingerprint (Pydantic)
│
├── agent2/
│   ├── reasoner.py              # Trading signal generation + PMI correction
│   ├── prompt.py                # STRATEGY_COT_PROMPT, A/B/C scoring tokens
│   └── schema.py                # TradingSignal (Pydantic)
│
├── backtest/
│   ├── dataset_parser.py        # HF / parquet / CSV dataset loading
│   ├── price_fetcher.py         # yfinance daily-interval fetch + dual cache
│   ├── backtester.py            # run_backtest() / compute_metrics()
│   └── run_backtest.py          # CLI entry point
│
├── ingestion/
│   └── news_fetcher.py          # Alpaca / Finnhub live news ingestion
│
├── notebooks/
│   ├── demo.ipynb               # Full pipeline Colab demo
│   └── resume_backtest.ipynb    # Re-process an existing CSV (no GPU for Track A)
│
├── README.md
└── DEVELOPMENT.md               # Architecture deep-dive and troubleshooting
```

---

## How Sentiment Analysis Works

Sentiment is not extracted by asking the LLM to output a label or number.
Instead, the pipeline uses a two-phase approach that reads the model's genuine
token-level log-probabilities from vLLM.

### Phase 1 — Chain-of-thought reasoning

The article is fed to DeepSeek-R1 with a prompt that instructs it to reason about
market sentiment inside `<think>…</think>` tags.  vLLM stops generation the moment
the model emits `</think>`:

```
Prompt:
  You are a financial news sentiment analyst. Reason step by step about
  the market sentiment conveyed by the following article. Write your
  analysis inside <think>...</think> tags. Consider:
  • Which companies or sectors are affected and how?
  • Is the news fundamentally positive, negative, or neutral for investors?
  • Are there conflicting signals or ambiguity in the article?

  [article text]

Model output (stopped at </think>):
  <think>
  The article reports record quarterly revenue, beating analyst estimates
  by 8 %.  iPhone demand remains strong and guidance was raised.  This
  is clearly positive for Apple shareholders ...
  </think>
```

The CoT reasoning is preserved as an audit trail but plays no direct role in the
final probability calculation.

### Phase 2 — Scoring via `prompt_logprobs`

Three scoring prompts are constructed, one per candidate sentiment class:

```
[Phase 1 prompt] + [generated CoT] + </think>
Sentiment: POSITIVE
```
```
[Phase 1 prompt] + [generated CoT] + </think>
Sentiment: NEGATIVE
```
```
[Phase 1 prompt] + [generated CoT] + </think>
Sentiment: NEUTRAL
```

All three are submitted in a single batched `engine.generate()` call with
`SamplingParams(prompt_logprobs=1, max_tokens=1)`.  vLLM always includes the
log-probability of the actual prompt token at every position, so we directly read:

```
log P("POSITIVE" | full context)
log P("NEGATIVE" | full context)
log P("NEUTRAL"  | full context)
```

For choices that span multiple BPE tokens (e.g. "POSITIVE" → ["▁POS", "ITIVE"]),
the per-token logprobs are summed to obtain the full-string joint log-probability.

### Calibrated probabilities

The three log-probabilities are converted to a calibrated probability vector via
temperature-scaled softmax:

```python
probs = softmax(log_probs, temperature=CALIBRATION_T)   # CALIBRATION_T = 1.2
label = ["POSITIVE", "NEGATIVE", "NEUTRAL"][argmax(probs)]
confidence = max(probs)
```

`CALIBRATION_T > 1` softens the distribution, preventing the model from being
over-confident on ambiguous articles.

### Output — `NewsFingerprint`

```python
NewsFingerprint(
    source = "Reuters",
    published_at = "2024-02-25T14:30:00Z",
    headline = "Apple reports record Q4 revenue ...",
    companies_named = ["Apple", "AAPL"],
    event_keywords = ["revenue", "iphone", "earnings"],
    sentiment_label = "POSITIVE", # argmax of softmax
    sentiment_score = 1.0, # +1 / 0 / -1 scalar
    sentiment_confidence = 0.91,
    sentiment_probabilities = {"POSITIVE": 0.91, "NEGATIVE": 0.05, "NEUTRAL": 0.04},
    calibration_T = 1.2,
    article_text = "...", # full text passed to Agent 2
)
```

---

## How Trading Signal Generation Works

Agent 2 takes a `NewsFingerprint` and produces a directional trading signal using
the same two-phase real-logprobs approach, with one additional correction.

### Phase 1 — Strategy CoT

The fingerprint (article text + sentiment vector) is fed to the model with a prompt
that presents three lettered options and asks for reasoning:

```
You are a quantitative trading strategist. Based on the news article and
sentiment analysis below, reason step by step about the appropriate
one-week trading action.  Write ALL of your analysis inside
<think>...</think> tags.

At the end you will select one lettered option:
  (A) BUY  — bullish signal; expect the price to rise
  (B) HOLD — insufficient or ambiguous signal; stay flat
  (C) SELL — bearish signal; expect the price to fall

Sentiment analysis: { ... }
News article: [text]
```

### Phase 2 — A/B/C scoring with PMI correction

Single ASCII letters are used as scoring tokens rather than the words
BUY / HOLD / SELL, because:

- Single letters are always single BPE tokens — no boundary effects.
- The words "HOLD" and "SELL" have strong unconditional LM priors at
  typical financial-text decision points, causing the model to default
  to HOLD regardless of news content.

Even with letters, the token "B" retains a residual prior of ~0.75 at
`"The answer is ("`.  **PMI (Pointwise Mutual Information) correction**
removes this bias:

```
PMI(choice, article) = logP(choice | article) − logP(choice | null article)
```

The null-context logprobs are computed once per engine session by running the
same pipeline on a completely neutral article ("No specific news available.").
Subtracting them leaves only the component driven by actual news content.

After PMI correction and softmax, the result is a calibrated probability vector
over BUY / HOLD / SELL.

### Output — `TradingSignal`

```python
TradingSignal(
    ticker = "AXP",
    direction = "long", # "long" | "neutral" | "short"
    strategy_tag = "event_driven",
    confidence = 0.71,
    cot = "<think>The article reports ...",
    signal_logits  = [-1.38, -0.28, -6.26], # raw logprobs [A, B, C]
    signal_probabilities = {"BUY": 0.63, "HOLD": 0.35, "SELL": 0.02},  # after PMI
    calibration_T = 1.2,
)
```

---

## Backtest Pipeline

The backtest runs over the Dow 30 news dataset (`FinGPT/fingpt-forecaster-dow30-202305-202405`)
and measures how well the generated signals predict one-week price moves.

### Metrics

| Metric | Description |
|---|---|
| `direction_accuracy` | Fraction of signals where predicted direction matches realized price move |
| `long_accuracy` | Accuracy on BUY signals specifically |
| `short_accuracy` | Accuracy on SELL signals specifically |
| `mean_strategy_return` | Average return of the simulated long/short/flat positions |
| `annualized_sharpe` | Weekly Sharpe ratio annualised by √52 |
| `total_pnl` | Sum of all strategy returns across the sample |
| `vs_fingpt_accuracy` | Agreement rate with the FinGPT dataset's own labels |

### Skip reasons

| Reason | Meaning |
|---|---|
| `fingerprint_failed` | Agent 1 could not extract a valid `NewsFingerprint` |
| `signal_failed` | Agent 2 returned `None` (parse error or validation failure) |
| `price_fetch_failed` | yfinance returned no data for the ticker + date window |

### Run from CLI

```bash
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --metrics

# Subset run
python -m backtest.run_backtest \
  --dataset "FinGPT/fingpt-forecaster-dow30-202305-202405" \
  --max-rows 50 --metrics
```

---

## Notebooks

### `notebooks/demo.ipynb` — Full pipeline demo (Colab)

Runs the complete two-agent pipeline end-to-end.  Requires a GPU runtime (A100 recommended, T4 works with `float16`).

Cells:
1. GPU check + install
2. Mount Drive and clone / pull repo
3. Load secrets and configure environment
4. Load vLLM engine (shared between both agents)
5. Single-article smoke test
6. Batch smoke test
7–end. Full backtest + metrics

---

## Quick Start (Colab)

1. Open `notebooks/demo.ipynb` in Google Colab with an A100 or T4 GPU runtime.
2. Upload your model weights to Google Drive at the path set in Cell 3 (`FINGPT_MODEL_PATH`).
3. Add your API keys to Colab Secrets (`FINNHUB_API_KEY`, `ALPACA_API_KEY`, etc.).
4. Run all cells in order.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `FINGPT_MODEL_PATH` | Yes | Path to local HuggingFace-format model weights |
| `SHARE_SINGLE_LLM_BETWEEN_AGENTS` | Recommended | `true` — share one vLLM engine to save VRAM |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | Yes (Colab) | Set to `0` to avoid `fileno` crash in Jupyter |
| `FINGPT_CALIBRATION_T` | No | Softmax temperature (default `1.2`) |
| `FINGPT_LOGITS_MAX_TOKENS` | No | CoT token budget per article (default `1024`) |
| `FINGPT_YF_CACHE_PATH` | No | Price cache JSON path (default `output/yfinance_return_cache.json`) |
| `NEWS_PROVIDER` | Live only | `finnhub` or `alpaca` |
| `FINNHUB_API_KEY` | Live only | Finnhub API key |
| `ALPACA_API_KEY` | Live only | Alpaca Markets API key |
| `ALPACA_API_SECRET` | Live only | Alpaca Markets API secret |

---

## Key Design Decisions

**Real logprobs over self-reported numbers.**  
Local LLMs are unreliable at introspecting their own confidence numerically.  Reading `prompt_logprobs` from vLLM gives the model's actual beliefs as computed by its forward pass.

**Chain-of-thought separated from scoring.**  
The model reasons in Phase 1 (unlimited within `max_tokens`) and is scored in Phase 2 (a single deterministic lookup).  This preserves the CoT audit trail while keeping the decision mechanism noise-free.

**PMI prior correction.**  
Language model next-token probabilities are confounded by the unconditional prior of each word at a given syntactic position.  PMI subtracts that prior so only the news-driven information determines the signal.

**Batch inference.**  
All vLLM calls are batched (10 articles per batch, 5 engine calls per batch total).  This saturates GPU throughput without exhausting KV-cache memory on a single A100.

**Strict failure handling.**  
Every agent returns `None` on any parsing or validation failure rather than manufacturing a fallback signal.  Skip reasons are recorded in the output CSV so pipeline health can be monitored.
