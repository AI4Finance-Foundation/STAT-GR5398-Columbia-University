# Historical data repair - 9 October 2026

Status: WRDS account approval pending, confirmed by Patrick. No WRDS data have been retrieved.

## Scope and exports

`historical_price_requests.csv` lists 62 symbols with missing historical-member price observations across 22 decisions (615 member-date gaps). The earlier 106-symbol recovery list and its 47 accepted / 59 rejected outcomes refer to the original recovery batch; these are different universes/counting units.

Request 2015-06-15 through 2025-09-30 for training and trading, including security-name history. First resolve ticker histories to stable CRSP identifiers and their effective intervals. A current ticker match is insufficient for merger predecessors, reused symbols and share classes.

Export daily total returns, unadjusted prices, splits/dividends, effective trading dates, delisting codes/dates, delisting returns and cash/security consideration. Keep raw WRDS column names and export metadata. Determine whether the subscribed CRSP format already embeds delisting returns before combining fields. Do not double-count terminal returns, infer proceeds from a final quote, or silently set missing proceeds to zero. Verify the accessible data actually reach September 2025; the library catalogue's stated years may lag the current subscription.

`financial_vintage_requests.csv` records 22,909 audit rows requiring historical provenance (not 22,909 unique filings). For each issuer, fiscal period and decision date, retrieve the version available by that decision. Preserve original filing/accession identifiers, public availability timestamps, version timestamps, units, fiscal period and source fields. Use Compustat Snapshot / a documented point-in-time product, or the original SEC filing; ordinary latest-restated quarterly fundamentals do not establish original vintages. Retrieve dated identifier links and historical sector classifications where available.

Columbia lists CRSP, Compustat and Snapshot in its subscriptions. An individual account and the actual subscribed libraries must be verified after approval:
https://guides.library.columbia.edu/WRDS/collection
https://guides.library.columbia.edu/WRDS/overview

## Financial reconciliation

Reconcile the 52 model features against dated source values, including quarterly versus year-to-date cash flow, quarterly versus trailing earnings, fiscal/calendar periods, units, shares and split adjustments. Preserve both raw and recomputed features. Ratios need matching price and per-share adjustment bases.

A concrete check: Apple's 30 April 2020 release reports diluted EPS of $2.55 for the quarter ended 28 March 2020. The local provider payload for that period reports diluted EPS of $0.64. The difference requires a share-adjustment/vintage reconciliation; it alone does not demonstrate an earnings restatement or predictive leakage. Do not compare basic EPS with diluted EPS.
Source: https://www.apple.com/newsroom/2020/04/apple-reports-second-quarter-results/

## Acceptance and rerun

1. Every historical index member at each decision has a documented security identity and either usable history or a specific unresolved reason. Include failed/acquired issuers. Membership alone does not establish price coverage.
2. Every holding through a terminal event has verified proceeds; unknown terminal proceeds block an unbiased completion claim.
3. Every model feature's value and availability are traced to a contemporaneous vintage. A long filing lag does not fix later revisions.
4. Rebuild features, labels and returns together under one documented data basis; preserve the old results as a separate experiment. Refit all 242 windows under the fixed top-10 and top-25 rules. Report changed sample counts and metrics even if performance deteriorates.
5. Rerun arithmetic, timing, corporate-action, coverage and source reconstruction checks, then update PDF, Medium and PR consistently.

Licensed raw exports stay outside the public repository. Publish extraction instructions, provenance summaries and permitted derived results. If Snapshot or terminal data are unavailable, record the actual limitation and seek course guidance; do not label an incomplete sample bias-free.
