# A FinRL backtest after repairing prices, identities and financial dates

Patrick Xiong | Columbia STAT GR5398 | Research results from 8 October 2026.

I used the course's supervised stock-selection workflow: financial features predict the next quarter's return with Random Forest, XGBoost and LightGBM. The proposed strategy keeps the highest-ranked 10% of eligible stocks, equally weights them and rebalances quarterly. I selected this candidate after the 7 October development experiments, before the repaired results below. Top 25% remains the course baseline; this is not an untouched holdout.

## What the audit changed

Point-in-time index membership did not restore companies absent from the daily price table. The new recovery accepted 47 missing ticker series and excluded 59 unresolved or ambiguous cases. Current SEC issuer identities, official rename announcements, provider metadata and adjusted-return consistency support these decisions. They do not certify a complete survivorship-free sample.

A second problem was financial-period matching. Fiscal quarter labels do not always denote the same calendar quarter. Matching the actual income filing and fiscal period restored consistency in the fixed 48-row spot check: 30 income EPS values and 29 current ratios matched their supplied payloads. The aggregate PE uses quarter-date price divided by quarterly EPS and consequently differs from the provider's PE definition. Original unrevised filing vintages remain unverified.

Canonical security identities now avoid duplicate company/quarter observations and choose one complete price series for both model labels and portfolio execution. Missing held prices stop the simulation rather than being silently forward-filled indefinitely.

## What the model actually learns

The target is the next quarter's log price return. Training compares Random Forest, XGBoost and LightGBM within 242 rolling sector-quarter windows; the revised run produced 10,204 stock forecasts. Random Forest won 173 windows, XGBoost 39 and LightGBM 30 on validation mean squared error. A win count describes this validation setup; it is not a significance test.

Top 10% means sorting eligible stocks by predicted return at each quarterly decision and buying the highest-ranked tenth at equal initial weights. Weights then drift with prices until the next rebalance. For example, from 400 eligible stocks the rule selects about 40; it does not select the stocks whose future returns were already known.

## The measured result

Across the full available trading window, 2020-06-01 to 2025-09-30, top 10% returned 175.22% cumulatively, top 25% 127.08%, SPY 135.84%, and QQQ 164.92%. SPY had annualized volatility of 17.47%, Sharpe of 1.01, and maximum drawdown of -24.50%. The candidate therefore had higher realized return and a deeper drawdown, with a lower Sharpe ratio than SPY. The candidate's annualized return was 20.92%, volatility 22.92%, zero-risk-free-rate Sharpe 0.94, maximum drawdown -34.54%, and annual two-sided turnover 4.85x.

Each dollar bought or sold costs 10 basis points, including initial purchases. Trades occur at the designated close; new positions earn returns from the next session. Fees are paid from portfolio assets. The report retains the original 2020-09-01 common-window comparison, tests costs and execution delays, and explains stock/sector contributions.

## What remains unresolved

The initial decision has usable quotes for 450 of 505 historical members. Across all 22 decisions, the current audit lists 62 distinct historical symbols with price-coverage gaps. This count differs from the 59 rejected names in the original recovery batch because the scopes differ. Filing timestamps are screened, but that does not prove the numbers are the versions originally disclosed. Historical sector labels also remain unverified.

As of 9 October, my WRDS account is awaiting approval. The next data step is to verify access to CRSP terminal-event histories and Compustat historical vintages, reconcile their identifiers and definitions, and refit the same rules. The results below have not been rerun on those data.

## What this result establishes

The fixed candidate beats SPY within this available full-window dataset. Missing failed/acquired-company histories, incomplete financial vintages and the development choices limit any broader claim of alpha. Top-10% annualized arithmetic excess has a descriptive bootstrap interval of -2.67% to 10.58%; the interval does not account for all those data and selection issues.

The deliverable is a reproducible, coverage-limited research experiment with recorded failures and source decisions. Completing the assignment's bias-free sample requirement remains a separate unresolved item. The accompanying README specifies the commands, source snapshots, recovery evidence and limitations.

Sources: [course instructions](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/master/Fall-2026/Assignment1/FinRL-Trading), [pinned data](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/dd1fdcf5db69b4cf42e90a185211defa8d5ae273/data), [SEC company tickers](https://www.sec.gov/files/company_tickers.json), issuer announcements in the evidence manifest, and Yahoo Finance adjusted prices.


## Reproduction and assistance

The course submission folder is `Fall-2026/Assignment1/FinRL-Trading/submissions/Assignment1_PatrickXiong_sx2434/` in my [course fork](https://github.com/pattybepatient/STAT-GR5398-Columbia-University). Its README records reproduction instructions and the research status. The draft PR remains a work in progress while data coverage and vintage checks are unresolved. AI tools assisted implementation and review; automated checks and source reconstruction support the reported arithmetic, not a certification of the source data.
