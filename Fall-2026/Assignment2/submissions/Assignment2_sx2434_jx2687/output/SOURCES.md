# Results and provenance

The four course portfolios use the existing growth-tech scores. A uses selection alone, B adds the frozen valuation filter, C adds full exposure to A, and D combines the filter with full exposure. A and C have identical results. B and D have identical results. Each split starts with $100,000. The course computes CAGR from the first recorded closing NAV, using 252 sessions per year, and turnover as annualized absolute traded notional divided by average NAV.

## Official portfolio results

| File | Origin and interpretation |
| --- | --- |
| `ablation_dev.csv`, `ablation_val.csv`, `ablation_test.csv` | Original course output for each period. Test ran once with the frozen two-vote rule. |
| `test_runs.jsonl` | Unmodified log of the official test, including the implementation commit and command line. |
| `portfolio_comparison.csv` | A–D plus passive SPY and QQQ in each period. The four strategy rows are taken directly from the corresponding ablation. |
| `formal/passive_benchmarks.csv` | Six published aggregate ETF rows for the same splits. Within every period, all four local ablation rows independently confirm the ETF date range, total return and Sharpe. The supplement supplies CAGR, volatility and maximum drawdown. The passive price series has no simulated trading or course turnover statistic. |

The ETF aggregates are from Xinru Xiao's archived course submission. The analysis applies the course `performance` function to each ETF over the evaluation dates. The included file selects development, validation and test rows and omits the repeated cost-description field. [Published benchmark table](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/de5ddae93b832a5b68ef64b39b07ca5a35623c6e/Fall-2026/Assignment2/submissions/Assignment2_XinruXiao_xx2524/analysis/benchmark_metrics.csv) · [Calculation](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/blob/de5ddae93b832a5b68ef64b39b07ca5a35623c6e/Fall-2026/Assignment2/submissions/Assignment2_XinruXiao_xx2524/reproduction/outputs/retraining/analyze_benchmarks.py)

## Formal filter and standalone valuation

| Directory | Contents |
| --- | --- |
| `evaluation/` | `team_eval_dev.csv` and `course_pe_eval_dev.csv` are the development evaluation CSVs from the course script. They include price labels that mature in the following year. `val_matured_signals.csv` restricts validation labels to dates through December 2023. `separate_model_scores.csv` summarizes the signals available to each model. |
| `evaluation/matched_signals.csv` | Compares the team and course P/E estimate on the same ticker, as-of date and realized outcome. Both sets of labels stop at the split boundary. The historical course convention takes the latest print inside the forward horizon. The exact-close variant requires a print on the horizon session. `inside_team` and `inside_pe` are counts. Divide them by `paired_outcomes` for the corresponding 252-session hit rates. |
| `formal/` | The rule comparisons, complete annual-return and ticker-attribution tables, daily A/B NAVs for development and validation, and the independent implementation reconciliation. Annual returns use the prior December close, with $100,000 as the initial capital in the first year of each split. Incremental net P&L is the difference in actual holding-path P&L, including fees. |
| `formal/cases/` | Same-week NVDA and ASML filter inputs, ranks, replacement basket, subsequent returns and development-only case replays. `official_episode_trades.csv` contains the actual A/B ledger rows for these two names from November 4, 2019 through February 4, 2020. `teacher_trade.csv` isolates the two fills for the actual ASML round trip. The equity files describe hypothetical veto exemptions. |

## Weighted extension

Every result in `extension/` is from development or historical validation. The experiment was designed after the official test. Validation had already been used to compare earlier choices in this rule family.

| Directory | Contents |
| --- | --- |
| [DCF weight experiment](extension/weights/) | Equal weights versus a reduced DCF weight on sensitive weeks, before applying any disagreement cutoff. The summary reports net CAGR in percent. The intervals report paired differences in percentage points from a 20-session block bootstrap. The protocol specifies both weighting rules. |
| `extension/protocol.json`, `extension/val_metrics.csv` | Time-stamped 1.983 specification and course-convention validation metrics. |
| `extension/gate/` | Net metrics and daily NAVs for each fixed cutoff, including the ungated weighted filter (`K_inf`). The annual comparisons use the same initial-capital convention as the formal annual table. |
| `extension/threshold/` | Full development curve, its leave-NVDA-out counterpart, and the MPWR/INTU events defining the peak. Each curve measures the difference against its own selection-only portfolio. |
| `extension/placebo/` | Weekly summaries and aggregated outcomes from 500 random paths in each period. Each draw preserves weekly veto counts within selector ranks 1–10, 11–20 and 21 onward. The gross-return proxy applies 95% exposure and weekly target weights. The summary retains the distribution statistics. The weekly files retain observed and average placebo returns, not all 500 individual paths. |
| `extension/anet/` | Actual and counterfactual validation fills, weekly holdings, activity, contribution and NAV. `disck_unpriced.csv` derives unavailable holding value from the execution trace on four sample rebalances. `execution_gap_by_start.csv` uses both the first close and common initial capital as CAGR denominators. In these files, the filtered or strategy arm is the weighted extension. |

## Rebuild the reported tables

From the team submission directory, run the following commands with Python 3 and the course environment for pandas. They use the archived outputs in this folder and do not invoke the course runner.

```bash
python analysis/build_portfolio_comparison.py
python analysis/summarize_evaluation.py
python analysis/summarize_execution.py
python analysis/check_report_evidence.py
```

The first three commands regenerate the portfolio table, separate-model signal scores and execution tables. The last reconciles annual performance, stock attribution, the selected threshold, weekly attribution and observed placebo-proxy CAGR against the archived results. The main reproduction instructions describe how the formal course inputs and development/validation portfolios are built.
