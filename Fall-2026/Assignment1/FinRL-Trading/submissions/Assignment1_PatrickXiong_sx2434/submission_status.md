# Submission status - 9 October 2026

Author: Patrick Xiong (sx2434).

Medium is published: https://medium.com/@sx2434/a-finrl-backtest-after-repairing-prices-identities-and-financial-dates-1771f8ed587d

Course PR: https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/pull/9 (created and verified in draft state).

The 8 October results are preserved: 242 rolling windows, 10,204 forecasts; top-10 cumulative return 175.22% and maximum drawdown -34.54%, versus SPY 135.84% and -24.50%. Top-25 returned 127.08%. These are coverage-limited development results.

Unresolved: 62 symbols have historical decision-date price gaps; terminal proceeds, original financial vintages and historical sectors are not certified. WRDS account approval is pending. No WRDS repair or new repaired-data training has been performed.

The draft PR must not be marked ready until these data issues are resolved and all linked outputs and claims are updated. See data_access/README.md for extraction and acceptance criteria.
