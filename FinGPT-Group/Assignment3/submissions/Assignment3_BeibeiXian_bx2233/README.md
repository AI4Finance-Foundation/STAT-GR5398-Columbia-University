# Assignment 3 — ContrarianFusion
**GR5398 MA Mentored Research · Columbia University · Spring 2026**
Medium Link: https://medium.com/@bx2233/contrarianfusion-when-my-llm-was-wrong-in-the-right-direction-part-2-9469dc4998a4?postPublishedType=initial


---

## What This Is

A three-signal weekly long-short equity strategy built on a key discovery from Assignment 2: the fine-tuned FinGPT adapter exhibits a statistically significant but directionally inverted Information Coefficient (IC = −0.2122, p = 0.034). Rather than discarding the model, this work inverts its predictions as a contrarian signal and fuses it with technical momentum and fundamental quality filters.

**Out-of-sample results (2026 Q1, 17 weeks):**

| | ContrarianFusion | SPY |
|---|---|---|
| Total Return | **+7.35%** | +5.77% |
| Sharpe Ratio | **1.570** | 1.292 |
| Max Drawdown | −6.03% | — |

2,580 FinGPT inference predictions · 0 parse errors · 1,058.8 min on A100 80GB GPU

---

## Repository Structure

```
Assignment3_BeibeiXian_bx2233/
│
├── code/
│   ├── a3_data_pipeline.py        
│   ├── a3_backtest.py             
│   └── a3_signal_diagnostics.py  
│
├── notebooks/
│   ├── FinGPT_A3_Inference.ipynb    
│   └── FinGPT_A3_DataPipeline.ipynb 
│
├── outputs/
│   ├── signal_diagnostics.png     
│   └── equity_curve_real.png      
│
├── data/                          
│   ├── fingpt_inputs.csv          
│   ├── fingpt_signals.csv         
│   ├── prices_2026q1.csv          
│   ├── features_2026q1.csv        
│   └── macro_regime.csv           
│
├── report/
│   └── GR5398_Report_BeibeiXian_bx2233.pdf
│
├── requirements.txt
└── .gitignore
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate data files
```bash
python code/a3_data_pipeline.py
```
Writes `fingpt_inputs.csv`, `prices_2026q1.csv`, `features_2026q1.csv`, `macro_regime.csv` to `data/`.

### Step 3 — Run FinGPT inference *(requires A100 GPU in Google Colab Pro)*
Open `notebooks/FinGPT_A3_Inference.ipynb`. This took **17.65 hours** to complete.
Output: `fingpt_signals.csv` (2,580 rows) → save to `data/`.

### Step 4 — Run the backtest
```bash
python code/a3_backtest.py
```
Reads `fingpt_signals.csv` + data files, prints ablation results, saves equity curve.

### Step 5 — Regenerate signal diagnostics
```bash
python code/a3_signal_diagnostics.py
```

---

## The Contrarian Inversion

The core insight is a single line of code:

```python
contrarian_score = -composite_score(fingpt_signal)
```

The FinGPT adapter says "Bullish" 90% of the time in 2026 Q1. This is not noise — it is a structural artifact of training on Dow30 blue-chip corporate news, which is dominated by earnings beats and analyst upgrades. A model that is *systematically* wrong in a predictable direction is an edge, not a failure.

Validated by: Spearman(contrarian_score, Massive_sentiment) = **−0.3608** (p < 0.0001, N = 2,580).

---

## What You Need to Add

> **`data/fingpt_signals.csv`** — download from your Google Drive:
> `A3_ContrarianFusion/outputs/fingpt_signals.csv`
> This is the 2,580-row FinGPT inference output. Without it the backtest cannot run.

> **`signal_module.py`** (optional for re-running inference) — copy from your Assignment 2 folder. Not needed if you only want to run the backtest on the existing signals.

---

## Signal Architecture

| Layer | Weight | Description |
|---|---|---|
| FinGPT Contrarian (β) | 0.40 | `−composite_score` — inverted LLM prediction |
| Technical Momentum (α) | 0.35 | FinRL-style cross-sectional Z-score features |
| Quality Filter (γ) | 0.25 | FinRobot-inspired fundamental screening |

Macro regime overlay (FRED: VIX, T10Y2Y, FEDFUNDS, BAMLH0A0HYM2):
- **RISK_ON**: full long + short exposure
- **NEUTRAL**: 70% exposure
- **RISK_OFF**: 40% long, 0% short (prevents short-squeeze losses)
