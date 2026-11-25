# Valuation filter selection protocol

The official growth model and full timing exposure stay fixed. Screen the first 20 growth candidates by course rank among decision-date members. Keep names with fewer than two valid positive method estimates or a largest-to-smallest ratio above 2.5. For the others, exclude the stock when median fair value divided by the current price falls strictly below 1 minus the discount threshold. Lower-ranked names can enter at the weekly rebalance.

The only eligible thresholds are 15%, 25% and 35%. Use the development period to inspect the three variants. On validation, choose the highest official portfolio Sharpe rounded to three decimals, then the lowest reported annual turnover, then the threshold closest to 25%, then the lower threshold. No forward valuation label enters this choice. The interquartile valuation band has a nominal 50% one-year price coverage target to be checked empirically.

Research returns, cash buffer, costs, universe, stock weights and rebalance frequency follow the course engine. All four formal ablation cells use the course scores. C has full timing exposure, so C must equal A and D must equal B. The experimental timing rules and retrained scores are independent extensions. Compute both the module evaluation and all four portfolio cells in a single registered holdout batch. Archive the log and do not tune afterward.
