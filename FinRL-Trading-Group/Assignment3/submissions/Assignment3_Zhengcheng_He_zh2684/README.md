# Assignment 3 Submission
## Overview

We built an integrated AI trading strategy combining FinRL, FinRobot, and FinGPT.  
FinRL generates the base portfolio using an adaptive rotation strategy (enhanced with a QQQ rolling momentum filter), FinRobot refines weights using research-based signals, and FinGPT tests news sentiment as a risk filter.  

The strategy was backtested from 2026-01-01 to 2026-04-30 and outperformed SPY.
## File Description

- `New story-Medium`
  Medium Blog

- `adaptive_rotation_engine_qqq.py`  
  Core implementation of the modified adaptive rotation strategy.

- `alpaca_weekly_backtest.ipynb`  
  FinGPT weekly backtest.

- `finrobot_integrated_v3.ipynb`  
  Strategy with FinRL components.

- `backtest_modified_qqq_2026-01-01_to_2026-04-30.png`  
  Visualization of backtest performance.

- `finrobot_report.pdf`  
  Detailed FinRobot report describing the strategy, modification and results.
