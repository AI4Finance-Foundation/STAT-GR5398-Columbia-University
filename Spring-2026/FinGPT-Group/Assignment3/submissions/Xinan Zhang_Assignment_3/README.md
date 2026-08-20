# Weekly Alpha Pulse: Dual-Engine Quant Strategy 🚀

> **"Let AI pick the stocks, and let Math size the bets."**
> A Dow 30 rotation strategy powered by Llama-3 sentiment analysis and Risk Parity portfolio management.

---

## 📌 Executive Summary

Weekly Alpha Pulse is a "Dual-Engine" quantitative trading system designed to navigate volatile markets by combining fundamental stability with AI-driven momentum. The strategy integrates:

1.  **Fundamental Engine:** Utilizing high-dimension Quality Scores to filter for financially robust blue-chip assets.
2.  **AI Sentiment Engine:** Leveraging **FinGPT ** to interpret weekly financial news and capture institutional sentiment inflection points.

During a 17-week stress test (encompassing high volatility and V-shaped reversals), the strategy achieved a **10.01% cumulative return** with a maximum drawdown of only **3.95%**, significantly outperforming the S&P 500 benchmark.

---

## 🛠️ Key Features

- **NLP Signal Extraction:** Employs a RAG-based FinGPT architecture with a custom **"Bulletproof Parser"** to transform unstructured news into high-fidelity trading signals.
- **Dynamic Risk Parity:** Moves beyond fixed-weight allocations. It utilizes volatility-inverse weighting to dynamically adjust position sizes based on real-time market risk.
- **Automated Hyperparameter Tuning:** Features a built-in **Grid Search** engine to optimize Target Volatility and Volatility Lookback periods (Vol Period).
- **Granular P/L Attribution:** Generates professional-grade reports monitoring weekly profit contribution, AI confidence levels, and holdings details for transparent performance tracking.

---

## 📈 Performance

| Metric | Strategy (Risk Balanced) | Benchmark (SPY) |
| :--- | :--- | :--- |
| **Total Cumulative Return (%)** | **10.01%** | 4.79% |
| **Max Drawdown (%)** | **3.95%** | > 8.00% |
| **Sharpe Ratio (Weekly)** | **0.36** | N/A |



---

## 📂 Project Structure

```text
├── data/                  # Historical price and financial news datasets
├── notebooks/             # Core experiments and research (Final_Assignment.ipynb)
├── src/
│   ├── strategy.py        # AlphaPulseRiskBalanced strategy logic
│   ├── engine.py          # Backtesting orchestrator and Grid Search functions
│   └── report_gen.py      # P/L reporting and visualization tools
├── signal_cache_all.pkl   # Serialized AI sentiment signal cache
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
