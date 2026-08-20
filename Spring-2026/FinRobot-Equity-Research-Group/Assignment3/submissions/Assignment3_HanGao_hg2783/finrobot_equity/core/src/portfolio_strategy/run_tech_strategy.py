# portfolio_strategy/run_tech_strategy.py

import os
import pandas as pd

from portfolio_strategy.config import (
    DATA_START_DATE,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    BENCHMARKS,
    TECH_UNIVERSE,
    TOP_N,
    REBALANCE_FREQ,
    TRANSACTION_COST,
)
from portfolio_strategy.data_loader import load_or_download_prices, to_monthly_returns
from portfolio_strategy.signals import compute_quant_score
from portfolio_strategy.backtester import (
    build_equal_weight_portfolio,
    build_score_weighted_portfolio,
    compute_portfolio_returns,
    compute_core_satellite_returns,
    summarize_all_strategies,
)

from portfolio_strategy.analytics import active_return_summary, regression_summary
from portfolio_strategy.finrobot_score import build_finrobot_scores, load_finrobot_scores
from portfolio_strategy.baselines import (
    compute_all_tech_equal_weight_returns,
    compute_random_topn_equal_weight_baselines,
)
from portfolio_strategy.attribution import (
    export_holdings_attribution,
    compute_active_vs_all_tech,
)


def build_final_scores_for_mode(quant_scores, finrobot_scores_df, mode):
    """
    Build final ranking scores for one ablation mode.

    Modes:
    - quant_only: 100% Quant Score
    - simple_finrobot: 70% Quant Score + 30% Simple FinRobot Score
    - risk_aware_finrobot: 70% Quant Score + 30% Risk-aware FinRobot Score
    """
    final_scores = quant_scores.copy()

    if mode == "quant_only":
        return final_scores

    score_column_by_mode = {
        "simple_finrobot": "simple_finrobot_score",
        "risk_aware_finrobot": "risk_aware_finrobot_score",
    }

    if mode not in score_column_by_mode:
        raise ValueError(f"Unknown score mode: {mode}")

    score_column = score_column_by_mode[mode]

    if finrobot_scores_df.empty or score_column not in finrobot_scores_df.columns:
        print(f"No valid {score_column} found. Falling back to Quant Score only for mode={mode}.")
        return final_scores

    finrobot_score_map = finrobot_scores_df.set_index("ticker")[score_column]

    for ticker in final_scores.columns:
        if ticker in finrobot_score_map.index:
            final_scores[ticker] = (
                0.7 * final_scores[ticker]
                + 0.3 * finrobot_score_map.loc[ticker]
            )

    return final_scores


def run_backtest_for_score_mode(
    mode,
    mode_output_dir,
    final_scores,
    quant_scores,
    finrobot_scores_df,
    monthly_returns,
    available_universe,
):
    """
    Run portfolio construction, backtest, analytics, and CSV export for one score mode.
    """
    os.makedirs(mode_output_dir, exist_ok=True)

    ew_weights = build_equal_weight_portfolio(
        scores=final_scores,
        monthly_returns=monthly_returns,
        universe=available_universe,
        top_n=TOP_N,
        rebalance_freq=REBALANCE_FREQ
    )

    score_weights = build_score_weighted_portfolio(
        scores=final_scores,
        monthly_returns=monthly_returns,
        universe=available_universe,
        top_n=TOP_N,
        rebalance_freq=REBALANCE_FREQ,
        max_weight=0.10
    )

    ew_result = compute_portfolio_returns(
        weights=ew_weights,
        monthly_returns=monthly_returns,
        transaction_cost=TRANSACTION_COST
    )

    score_result = compute_portfolio_returns(
        weights=score_weights,
        monthly_returns=monthly_returns,
        transaction_cost=TRANSACTION_COST
    )

    qqq_core_satellite = compute_core_satellite_returns(
        qqq_returns=monthly_returns["QQQ"],
        active_returns=score_result["net_return"],
        qqq_weight=0.5,
        active_weight=0.5
    )

    all_tech_equal_weight = compute_all_tech_equal_weight_returns(
        monthly_returns=monthly_returns,
        universe=available_universe,
    )

    strategy_returns = {
        "SPY": monthly_returns["SPY"],
        "QQQ": monthly_returns["QQQ"],
        "XLK": monthly_returns["XLK"],
        "All_Tech_Universe_EqualWeight": all_tech_equal_weight,
        "Top15_EqualWeight_Tech": ew_result["net_return"],
        "Top15_ScoreWeighted_Tech": score_result["net_return"],
        "50QQQ_50ScoreWeighted_Tech": qqq_core_satellite,
    }

    summary = summarize_all_strategies(strategy_returns)
    strategy_returns_df = pd.DataFrame(strategy_returns)

    active_vs_spy = active_return_summary(strategy_returns_df, benchmark_name="SPY")
    active_vs_qqq = active_return_summary(strategy_returns_df, benchmark_name="QQQ")
    regression_results = regression_summary(strategy_returns_df, benchmarks=("SPY", "QQQ"))

    active_vs_all_tech = compute_active_vs_all_tech(strategy_returns_df)

    holdings_attribution = export_holdings_attribution(
        output_dir=mode_output_dir,
        mode=mode,
        weights=score_weights,
        final_scores=final_scores,
        quant_scores=quant_scores,
        finrobot_scores_df=finrobot_scores_df,
        monthly_returns=monthly_returns,
    )

    final_scores.to_csv(os.path.join(mode_output_dir, "final_scores.csv"))
    ew_weights.to_csv(os.path.join(mode_output_dir, "equal_weight_weights.csv"))
    score_weights.to_csv(os.path.join(mode_output_dir, "score_weighted_weights.csv"))
    ew_result.to_csv(os.path.join(mode_output_dir, "equal_weight_returns.csv"))
    score_result.to_csv(os.path.join(mode_output_dir, "score_weighted_returns.csv"))
    strategy_returns_df.to_csv(os.path.join(mode_output_dir, "strategy_returns.csv"))
    summary.to_csv(os.path.join(mode_output_dir, "performance_summary.csv"), index=False)
    active_vs_spy.to_csv(os.path.join(mode_output_dir, "active_vs_spy.csv"), index=False)
    active_vs_qqq.to_csv(os.path.join(mode_output_dir, "active_vs_qqq.csv"), index=False)
    active_vs_all_tech.to_csv(os.path.join(mode_output_dir, "active_vs_all_tech.csv"), index=False)
    regression_results.to_csv(os.path.join(mode_output_dir, "alpha_beta_regression.csv"), index=False)

    summary_with_mode = summary.copy()
    summary_with_mode.insert(0, "score_mode", mode)

    return {
        "mode": mode,
        "summary": summary,
        "summary_with_mode": summary_with_mode,
        "active_vs_spy": active_vs_spy,
        "active_vs_qqq": active_vs_qqq,
        "active_vs_all_tech": active_vs_all_tech,
        "holdings_attribution": holdings_attribution,
        "regression_results": regression_results,
    }


def main():
    output_dir = "./output/TECH_STRATEGY"
    os.makedirs(output_dir, exist_ok=True)

    all_tickers = sorted(list(set(TECH_UNIVERSE + BENCHMARKS)))
    cache_path = os.path.join(output_dir, "prices.csv")

    prices = load_or_download_prices(
        tickers=all_tickers,
        start_date=DATA_START_DATE,
        end_date=BACKTEST_END_DATE,
        cache_path=cache_path
    )

    monthly_prices = prices.resample("M").last()
    monthly_returns_full = to_monthly_returns(prices)

    available_universe = [t for t in TECH_UNIVERSE if t in monthly_returns_full.columns]
    print(f"Available universe size: {len(available_universe)}")
    print(available_universe)

    quant_scores_full = compute_quant_score(
        monthly_prices=monthly_prices[available_universe],
        monthly_returns=monthly_returns_full[available_universe]
    )

    quant_scores_full.to_csv(os.path.join(output_dir, "quant_scores_full.csv"))
    
    finrobot_scores_path = os.path.join(output_dir, "finrobot_scores.csv")

    if os.path.exists(finrobot_scores_path):
        finrobot_scores_df = load_finrobot_scores(finrobot_scores_path)
    else:
        finrobot_scores_df = build_finrobot_scores(
            universe=available_universe,
            output_root="./output",
            output_csv=finrobot_scores_path
        )
    monthly_returns = monthly_returns_full.loc[
        BACKTEST_START_DATE:BACKTEST_END_DATE
    ].copy()

    random_returns, random_summary, random_percentiles = compute_random_topn_equal_weight_baselines(
        monthly_returns=monthly_returns,
        universe=available_universe,
        top_n=TOP_N,
        n_simulations=500,
        seed=42,
    )

    random_returns.to_csv(os.path.join(output_dir, "random_top15_returns.csv"))
    random_summary.to_csv(os.path.join(output_dir, "random_top15_summary.csv"), index=False)
    random_percentiles.to_csv(os.path.join(output_dir, "random_top15_percentiles.csv"), index=False)

    quant_scores = quant_scores_full.loc[
        BACKTEST_START_DATE:BACKTEST_END_DATE
    ].copy()

    quant_scores.to_csv(os.path.join(output_dir, "quant_scores_backtest_window.csv"))

    score_modes = [
        "quant_only",
        "simple_finrobot",
        "risk_aware_finrobot",
    ]

    all_summaries = []
    ablation_results = {}

    for mode in score_modes:
        print(f"\nRunning score mode: {mode}")

        final_scores = build_final_scores_for_mode(
            quant_scores=quant_scores,
            finrobot_scores_df=finrobot_scores_df,
            mode=mode,
        )

        mode_output_dir = os.path.join(output_dir, mode)
        result = run_backtest_for_score_mode(
            mode=mode,
            mode_output_dir=mode_output_dir,
            final_scores=final_scores,
            quant_scores=quant_scores,
            finrobot_scores_df=finrobot_scores_df,
            monthly_returns=monthly_returns,
            available_universe=available_universe,
        )

        all_summaries.append(result["summary_with_mode"])
        ablation_results[mode] = result

    score_method_comparison = pd.concat(all_summaries, ignore_index=True)
    score_method_comparison.to_csv(
        os.path.join(output_dir, "score_method_comparison.csv"),
        index=False,
    )

    print("\nScore Method Comparison:")
    print(score_method_comparison)

    print("\nBacktest window:")
    print(f"{BACKTEST_START_DATE} to {BACKTEST_END_DATE}")

    print("\nSaved separate CSV outputs for each score mode:")
    for mode in score_modes:
        print(f"- {os.path.join(output_dir, mode)}")


if __name__ == "__main__":
    main()