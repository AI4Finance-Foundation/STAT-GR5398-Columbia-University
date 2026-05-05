from pathlib import Path
def write_reports(root:Path,config,metrics):
    b=root/"blog"; b.mkdir(exist_ok=True); note=_data_note(root)
    mo=metrics.loc["modified","sharpe_ratio"]>metrics.loc["original","sharpe_ratio"]; mq=metrics.loc["modified","cumulative_return"]>metrics.loc["qqq","cumulative_return"]
    table=_md_table(metrics)
    md=f"""# Adaptive Rotation Strategy with QQQ Rolling Selection Enhancement

## Strategy Overview
This report compares the original Adaptive Rotation Strategy, a modified version with Rui's Assignment 1 QQQ rolling selection enhancement, QQQ, SPY, and an equal-weight benchmark. Multiple groups diversify leadership regimes; Information Ratio selects the active group; residual momentum ranks assets; weekly rebalancing controls turnover; and a drawdown overlay responds to losses.

## Original Strategy Methodology
The original strategy uses 12-week group IR, SPY 200-day moving-average regime filtering, top-2 residual momentum selection, equal weights, and a drawdown risk overlay.

## Modified Strategy Methodology
Rui's Assignment 1 logic was recovered from `FinRL-Trading-Full-Workload.ipynb`, `outputs_step2/stock_selected.csv`, and `strategy_framework/strategies/equal_weight_topk.py`. Assignment 1 selects the top 25 percent of ML-predicted names by trade date and improves allocation with sector-neutral predicted-return residuals, robust z-scores, dynamic Top-N, and hysteresis. Here it is integrated as a Growth Tech pre-filter using history-only QQQ rolling momentum plus Assignment 1 predicted-return scores where available.

## Backtest Setup
Data source note: {note}
The final run requires the full official Adaptive Rotation universe: AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA, XOM, CVX, FCX, BHP, GLD, SLV, TLT, IEF, XLU, XLV, IAU, SHY, UUP, SPY, and QQQ. Date range starts at `{config.get('start_date')}`. Rebalance timing is weekly. Signals use data before the holding week, and daily returns apply weights with a one-day lag. Transaction cost is `{config.get('transaction_cost_bps')}` bps per turnover.

## Backtest Results
{table}

![Equity Curve](../figures/equity_curve_original_modified_qqq.png)
![Drawdown](../figures/drawdown_original_modified_qqq.png)

## Research Questions
Does QQQ rolling selection improve the Growth Tech sleeve? The modified strategy isolates that layer, so differences versus original show the effect when Growth Tech is selected.
Does the modified strategy improve Sharpe ratio? {'Yes' if mo else 'No'}.
Does the modified strategy reduce max drawdown? {'Yes' if metrics.loc['modified','max_drawdown']>metrics.loc['original','max_drawdown'] else 'No'}.
Is improvement stable or regime-dependent? It is regime-dependent because the enhancement only affects Growth Tech.
Which layer matters most? This experiment isolates intra-group ranking; group selection still controls broad exposure.
What trade-off exists? Top-2 concentration can increase upside capture and idiosyncratic drawdown.
If better, the mechanism is stronger Growth Tech pre-filtering. If worse, the pre-filter removed names residual momentum would have preferred.

## Failure Analysis
Lookback sensitivity, regime lag, residual momentum instability, top-2 concentration, transaction costs, QQQ benchmark dominance, and Assignment 1 overfitting risk are the main weaknesses.

## Conclusion
Modified beat original on Sharpe: {'yes' if mo else 'no'}. Modified beat QQQ on cumulative return: {'yes' if mq else 'no'}. Future work should test independent OOS periods and less concentrated sizing.
"""
    (b/"medium_blog_assignment2.md").write_text(md,encoding="utf-8"); (b/"medium_blog_assignment2.html").write_text("<html><body>"+md.replace("\n","<br>\n")+"</body></html>",encoding="utf-8")
def write_readme(root:Path,config,metrics):
    note=_data_note(root)
    table=_md_table(metrics)
    txt=f"""# GR5398 Assignment 2: Adaptive Rotation Strategy Enhancement

## Goal
Reproduce Adaptive Rotation, enhance it with Rui's Assignment 1 QQQ rolling selection logic, and compare original, modified, QQQ, SPY, and equal-weight benchmarks.

## Folder Structure
`run_assignment2.py`, `src/`, `configs/`, `data/cache/`, `results/`, `figures/`, and `blog/`.

## Strategy Summary
Original: 12-week IR group selection, SPY moving-average regime filter, residual momentum top-2 equal weighting, and drawdown risk control. Modified: same framework, but Growth Tech is pre-filtered with Rui's Assignment 1 Top-N ranking idea.

## Assignment 1 Integration
Inspected `FinRL-Trading-Full-Workload.ipynb`, `ray_blog.md`, `outputs_step2/stock_selected.csv`, `stock_selected_analysis.py`, and `strategy_framework/strategies/equal_weight_topk.py`. Recovered logic: top 25 percent ML-predicted stock selection, sector-neutral predicted-return residual score, robust z-score ranking, dynamic Top-N, and hysteresis.

## Installation
`pip install -r requirements.txt`

## Run
`python run_assignment2.py --config configs/assignment2_config.yaml --run-all`

Optional modes:

`python run_assignment2.py --mode original`

`python run_assignment2.py --mode modified`

`python run_assignment2.py --mode benchmarks`

## Outputs
CSV outputs are in `results/`, figures in `figures/`, and the report in `blog/`.

## Main Results
{table}

## Assumptions
{note}
Full ticker universe: AAPL, MSFT, NVDA, META, AMZN, GOOGL, TSLA, XOM, CVX, FCX, BHP, GLD, SLV, TLT, IEF, XLU, XLV, IAU, SHY, UUP, SPY, QQQ.

Signals are history-only, weights are applied with a one-day lag, and transaction cost is configurable.

## Known Limitations
The pipeline now fails loudly if any required ticker is missing or has too few observations. If `data/cache/missing_tickers.csv` exists after a failed run, those tickers must be resolved before treating the backtest as complete.
"""
    (root/"README.md").write_text(txt,encoding="utf-8")

def _data_note(root:Path):
    note_path=root/"data/cache/data_source_note.txt"
    missing_path=root/"data/cache/missing_tickers.csv"
    note=note_path.read_text(encoding="utf-8").strip() if note_path.exists() else "yfinance adjusted close data."
    if missing_path.exists():
        missing=missing_path.read_text(encoding="utf-8").strip().replace("\n", ", ")
        note += f" Missing ticker file currently exists: {missing}."
    return note

def _md_table(df):
    cols=list(df.columns)
    lines=["| strategy | "+" | ".join(cols)+" |","|---|"+"|".join(["---:"]*len(cols))+"|"]
    for idx,row in df.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            vals.append(f"{float(v):.4f}" if isinstance(v,(int,float)) else str(v))
        lines.append(f"| {idx} | "+" | ".join(vals)+" |")
    return "\n".join(lines)
