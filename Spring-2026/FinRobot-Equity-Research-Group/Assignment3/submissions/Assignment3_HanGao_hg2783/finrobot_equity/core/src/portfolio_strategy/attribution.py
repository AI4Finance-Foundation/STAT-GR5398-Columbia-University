# portfolio_strategy/attribution.py

import os
import pandas as pd


def export_holdings_attribution(
    output_dir,
    mode,
    weights,
    final_scores,
    quant_scores,
    finrobot_scores_df,
    monthly_returns,
):
    """
    Export holdings-level attribution for each rebalance/holding month.

    Output columns:
    date, score_mode, ticker, weight, final_score, quant_score,
    simple_finrobot_score, risk_aware_finrobot_score, next_month_return,
    return_contribution
    """
    rows = []

    finrobot_lookup = {}
    if finrobot_scores_df is not None and not finrobot_scores_df.empty:
        finrobot_lookup = finrobot_scores_df.set_index("ticker").to_dict(orient="index")

    for date in weights.index:
        active_weights = weights.loc[date]
        active_weights = active_weights[active_weights > 0]

        for ticker, weight in active_weights.items():
            ticker_info = finrobot_lookup.get(ticker, {})

            stock_return = None
            if ticker in monthly_returns.columns:
                stock_return = monthly_returns.loc[date, ticker]

            rows.append({
                "date": date,
                "score_mode": mode,
                "ticker": ticker,
                "weight": weight,
                "final_score": final_scores.loc[date, ticker] if ticker in final_scores.columns else None,
                "quant_score": quant_scores.loc[date, ticker] if ticker in quant_scores.columns else None,
                "simple_finrobot_score": ticker_info.get("simple_finrobot_score"),
                "risk_aware_finrobot_score": ticker_info.get("risk_aware_finrobot_score"),
                "growth_quality_score": ticker_info.get("growth_quality_score"),
                "competitive_position_score": ticker_info.get("competitive_position_score"),
                "earnings_quality_score": ticker_info.get("earnings_quality_score"),
                "catalyst_strength_score": ticker_info.get("catalyst_strength_score"),
                "valuation_attractiveness_score": ticker_info.get("valuation_attractiveness_score"),
                "business_risk_score": ticker_info.get("business_risk_score"),
                "valuation_risk_score": ticker_info.get("valuation_risk_score"),
                "stock_return": stock_return,
                "return_contribution": weight * stock_return if stock_return is not None else None,
            })

    attribution = pd.DataFrame(rows)
    attribution.to_csv(os.path.join(output_dir, "holdings_attribution.csv"), index=False)

    return attribution


def compute_active_vs_all_tech(strategy_returns_df):
    """
    Compute active returns versus All-Tech Universe Equal Weight baseline.
    """
    benchmark = strategy_returns_df["All_Tech_Universe_EqualWeight"]
    rows = []

    for col in strategy_returns_df.columns:
        if col == "All_Tech_Universe_EqualWeight":
            continue

        active = strategy_returns_df[col] - benchmark

        rows.append({
            "strategy": col,
            "benchmark": "All_Tech_Universe_EqualWeight",
            "average_monthly_active_return": active.mean(),
            "cumulative_active_return": (
                strategy_returns_df[col].add(1).prod()
                - benchmark.add(1).prod()
            ),
            "monthly_win_rate": (active > 0).mean(),
            "tracking_error_annualized": active.std() * (12 ** 0.5),
            "information_ratio": (
                (active.mean() * 12) / (active.std() * (12 ** 0.5))
                if active.std() != 0 else None
            ),
        })

    return pd.DataFrame(rows)