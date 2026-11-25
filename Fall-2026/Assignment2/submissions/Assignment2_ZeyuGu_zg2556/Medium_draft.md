# When a valuation filter tells a growth portfolio to sell

**Zeyu Gu · Columbia University · STAT GR5398**

In the 2022–23 validation period, a weekly portfolio of growth stocks lost 6.24%. Adding a valuation filter reduced the loss to 3.82% and cut the maximum drawdown by 3.53 percentage points. Both portfolios held Discovery (DISCK). After its last quote on April 8, 2022, the course backtester valued those shares at the same $24.42 closing price through the end of 2023. Both drawdowns include that assumption. The same valuation filter cost 24.02 percentage points of cumulative return over 2017–21. What did the filter actually learn about expensive stocks?

## Three ways to price the same business

Consider Adobe at the close of December 31, 2021. A discounted cash-flow model put its stock at $900.36 per share. Its own historical EV/EBITDA multiples implied $541.23. Same-sector peers’ trailing earnings multiples implied $428.15. The middle estimate was only 4.6% below the adjusted closing price. The filter kept Adobe eligible for the next rebalance.

Each estimate uses information dated before the trade. The cash-flow projection begins with four reported quarters of revenue per share. Historical growth fades toward 2.5% over ten forecast years. Weekly returns against SPY determine beta. For the risk-free rate, I used the latest 10-year Treasury yield (DGS10) with an observation date on or before the decision date. The dataset gives no release timestamp, so I could not verify whether investors had that value when the portfolio decision was made. The second method reconstructs the company’s own historical multiple after the relevant filings were available. The third compares trailing earnings with contemporaneous index peers in the same sector. No historical stock price is estimated by a language model.

On each weekly decision, the rule examines the top 20 growth candidates. An exclusion requires at least two valid methods, a largest-to-smallest estimate ratio no higher than 2.5, and a median estimate more than 15% below price. The course ranking supplies the replacement. A company with too little filing history remains eligible. This matters for growth stocks, where missing earnings history can otherwise become an unintended exclusion rule.

## Better downside protection depended on the period

| Period | Portfolio | Cumulative return | Sharpe | Max drawdown | Annual turnover |
|---|---|---:|---:|---:|---:|
| 2017–21 development | Selection | +226.02% | 0.980 | −32.99% | 10.48× |
| 2017–21 development | With valuation | +202.00% | 0.928 | −33.93% | 13.37× |
| 2022–23 validation | Selection | −6.24% | 0.028 | −42.52% | 9.71× |
| 2022–23 validation | With valuation | −3.82% | 0.063 | −38.99% | 11.05× |

All four runs use the same stock scores, weekly execution, cash buffer and 6-basis-point trading cost. Turnover counts buys and sells. The penalty varied materially by year. In 2019 the filter lifted the return from 22.61% to 29.83%. In 2020 it reduced the return from 37.96% to 27.55%. The threshold was chosen from three predeclared candidates on validation, so the validation gain also reflects configuration selection.

## A fair-value range needs to earn its confidence level

Across the 480 validation candidate-months, at least one method produced a value for 348. Among the 188 valued observations with an exact one-year follow-up price inside the same split, only 81 finished inside the estimated interval. That is 43.1% against a nominal target of 50%. Development coverage was lower still at 32.6%. A range formed by varying assumptions is a useful description of model uncertainty, but future prices can move far beyond it.

Among validation candidates, the average monthly rank correlation between estimated upside and subsequent returns was −0.082 at three months, −0.107 at six months and −0.108 at one year. Apparent bargains tended to lag more expensive peers within the model’s growth shortlist. The one-year comparison changes when both methods must value the same stocks. On the 175 matched validation observations with a realized return, the composite’s average monthly rank correlation was −0.199, compared with −0.118 for the course P/E example. Different coverage can look like better forecasting. The improvement in one portfolio drawdown therefore provides limited evidence of a repeatable valuation premium.

## The frozen rule on the holdout

On January 2, 2024 through October 9, 2026, the selection portfolio returned 167.33% cumulatively. The same portfolio with the frozen valuation rule returned 190.11%. Sharpe rose from 1.301 to 1.434, while maximum drawdown narrowed from 30.53% to 28.54%. Higher turnover was part of the trade: annual gross turnover rose from 12.88× to 14.76× and the number of trades rose from 704 to 817. The gains include the course’s execution costs.

The improvement in the portfolio coincided with weak calibration of prices. The one-year valuation interval contained 98 of 355 observed prices (27.6%), compared with a 50% target. On the 326 mature observations that both models could value, the composite’s mean monthly rank correlation was +0.063 and the P/E example’s was +0.038. Stock ordering and price accuracy answer different questions. The holdout also does not erase the loss of return in development.

The test used the validation-selected 15% rule from local commit `5af60c20f1e93da8f6350aec9001ebfbafea57cf`. The ledger records one holdout run. I froze the configuration before that run and posted its hash to the course repository afterward. I disclosed the timing and asked whether the late posting is acceptable. A TA response is pending. [Course issue #21](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/issues/21) · [Course design and evaluation protocol](https://github.com/AI4Finance-Foundation/STAT-GR5398-Columbia-University/tree/43eeda7b011270814fa28cc38e9081e36128cf8a/Fall-2026/Assignment2)

## What changed when I retrained the selector or varied exposure?

I ran a separate selection experiment on January through June 2018. Two stocks can finish a year with the same gain even if one rose steadily and the other repeatedly reversed direction. I added a feature that divides the net change in log price over four quarters by the total absolute movement over those quarters. Each month the original model and the model with this feature used the same data cutoff. The feature raised six-month cumulative return from 10.93% to 14.71% and annualized turnover from 9.45× to 10.52×. In May it changed three of the ten top-ranked names.

I then retrained both selectors in the same software environment every month from 2017 through 2023, 84 fits per model. Adding the feature raised annualized net return from 24.89% to 27.99% over 2017–21. In the 2022–23 course replay, the two-year loss narrowed from 5.11% to 3.25%, maximum drawdown fell from 41.38% to 40.76%, and annual two-way turnover fell from 9.60× to 9.26×. The new selector beat the original in 13 of 24 months and trailed it by 1.30 percentage points in 2023. I had already studied these validation years. This retrospective comparison ends before the formal holdout.

For the news test, a fine-tuned GPT-2 model scored Federal Reserve headlines for the course engine’s 12% sentiment input. Validation stock returns moved from 3.79% with neutral news to 3.66% with those scores. I tried averaging only the days with a release in a separate development-period replay. The mean change in weekly exposure almost tripled, from 1.09 to 3.23 basis points, while the cumulative return slipped from 127.70% to 127.40%.

I then collected the full statements and compared the two inputs on the same 38 scheduled meetings for which the model was eligible. Every title received the same score of 0.44046. In May 2019 the Fed held its target rate steady. In July it cut the rate by a quarter of a percentage point. Even with the full text, the model gave those actions nearly identical scores of 0.50226 and 0.50139. [May statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190501a.htm) · [July statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190731a.htm)

Full text increased development stock returns from 127.43% to 127.97% relative to titles. During 2022–2023 it changed the target exposure at 13 of the 16 meetings, yet stock returns edged down from 3.77% to 3.76%, compared with 3.79% for a neutral news input. The deepest drawdown widened by 0.005 percentage points. The fuller text changed the model’s scores without improving either validation outcome. These markets had already been examined in earlier experiments. The formal holdout used course selection scores and fixed exposure throughout.

I also compared successive statements. In February and March 2023 the Fed raised its target by the same quarter point. February retained the prospect of successive increases. March made further tightening conditional. I gave one vote to the rate decision and one to the change in guidance, so the March hike and its softer wording canceled out. On the same 16 validation meetings the rule returned 3.30%, compared with 3.76% for the full-statement model. Maximum drawdown deepened from 27.70% to 27.98%. The rule produced ten hawkish signals, six neutral signals and no dovish signals. This sample tells us little about how it would trade a cycle of rate cuts. [February statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm) · [March statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm)

I kept each weekly policy signal and reduced how much it could change target exposure. On development meeting weeks, keeping 33.88% of the policy rule’s original shift matched the full-text model’s average change in exposure. Using that fraction in validation raised the rule’s return from 3.30% to 3.67% and narrowed its maximum drawdown from 27.98% to 27.71%. The full-text model still finished slightly ahead. Across the nine meetings when the rule cut exposure, the tradable stock basket fell four times and rose five times by the following weekly open. Shrinking the trades helped, while these nine decisions provided little evidence of skill in anticipating weekly losses. Only two development signals had actually moved the target, and in validation the scaled rule still moved it more than four times as much as the full-text model on average.

Finally, I used the full-text model’s actual trade size at each meeting. The policy signal could point that trade up or down. On a neutral call the ordinary neutral-news portfolio target applied. The original full-text model returned 3.76451%. Keeping its direction but skipping neutral calls returned 3.76896%. Switching the direction on the same active dates returned 3.78483%. The six actual reversals therefore added about 1.59 basis points of return, while maximum drawdown deepened by about one tenth of a basis point. The stocks fell after three of the six new cuts and rose after the other three. These six decisions provide little evidence of a consistent directional advantage.

## Investment judgment

The smaller drawdown in validation makes me willing to keep testing the filter, but the lower return in development gives me pause. Before committing capital, I would paper-trade the two portfolios side by side to judge whether the downside protection is worth the upside I may forgo.
