# FinRobot Module Export

This folder is a GitHub-uploadable export of our FinRobot module for the combined `finrl-finrobot-fingpt` pipeline.

## What Is Included

- `scripts/`: our FinRobot wrapper and backtest scripts
- `results/`: final CSV outputs and backtest artifacts

## Important Note

This export intentionally keeps the current script logic and many of the current default paths **unchanged**.

That means some scripts may still assume local paths such as:

- `finrobot_tracka/...`
- `finrl2/...`
- `external/FinRobot/...`

This version is meant to be a clean upload package of our current work, not yet the fully normalized final pipeline interface version.

## Original External Dependency

The original FinRobot repository is **not included** here.

If someone wants to rerun the full FinRobot generation step, they still need a separate local copy of the original FinRobot dependency and any required API keys.

## Folder Layout

```text
finrobot/
  README.md
  scripts/
  results/
    backtest_results/
```

## Main Files

### Scripts

- `scripts/run_full_finrobot.py`
- `scripts/build_historical_news_signals.py`
- `scripts/build_retail_social_signals.py`
- `scripts/build_finrobot_multipliers.py`
- `scripts/backtest_finrobot_comparison.py`

### Core Result Files

- `results/finrobot2_multipliers.csv`
- `results/finrobot2_rebalanced_panel.csv`
- `results/historical_news_signals.csv`
- `results/company_peer_map.csv`

### Backtest Outputs

- `results/backtest_results/performance_summary.csv`
- `results/backtest_results/equity_curve.csv`
- `results/backtest_results/equity_curve.png`
- `results/backtest_results/finrl_vs_no_lookahead_finrobot_attribution.csv`
- `results/backtest_results/old_vs_improved_weight_attribution.csv`
- `results/backtest_results/strategy_daily_returns.csv`
- `results/backtest_results/prices.csv`

