# portfolio_strategy/plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "./output/TECH_STRATEGY"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_score_method_comparison():
    path = os.path.join(OUTPUT_DIR, "score_method_comparison.csv")
    return pd.read_csv(path)


def plot_score_method_comparison():
    """
    Compare quant_only, simple_finrobot, and risk_aware_finrobot
    for Top15 ScoreWeighted strategy.
    """
    df = load_score_method_comparison()

    target = df[df["strategy"] == "Top15_ScoreWeighted_Tech"].copy()

    metrics = [
        ("cumulative_return", "Cumulative Return"),
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown"),
    ]

    for metric, title in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(target["score_mode"], target[metric])
        plt.title(f"Score Method Comparison: {title}")
        plt.xlabel("Score Mode")
        plt.ylabel(title)
        plt.xticks(rotation=20)
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, f"score_method_{metric}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_cumulative_returns(mode="risk_aware_finrobot"):
    """
    Plot cumulative returns for benchmark and strategy comparison.
    """
    path = os.path.join(OUTPUT_DIR, mode, "strategy_returns.csv")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)

    cumulative = (1 + returns).cumprod() - 1

    selected_cols = [
        "SPY",
        "QQQ",
        "XLK",
        "All_Tech_Universe_EqualWeight",
        "Top15_EqualWeight_Tech",
        "Top15_ScoreWeighted_Tech",
        "50QQQ_50ScoreWeighted_Tech",
    ]

    selected_cols = [col for col in selected_cols if col in cumulative.columns]

    plt.figure(figsize=(10, 6))
    for col in selected_cols:
        plt.plot(cumulative.index, cumulative[col], label=col)

    plt.title(f"Cumulative Returns - {mode}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, f"cumulative_returns_{mode}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_drawdowns(mode="risk_aware_finrobot"):
    """
    Plot drawdowns for benchmark and strategy comparison.
    """
    path = os.path.join(OUTPUT_DIR, mode, "strategy_returns.csv")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)

    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1

    selected_cols = [
        "SPY",
        "QQQ",
        "XLK",
        "All_Tech_Universe_EqualWeight",
        "Top15_ScoreWeighted_Tech",
        "50QQQ_50ScoreWeighted_Tech",
    ]

    selected_cols = [col for col in selected_cols if col in drawdown.columns]

    plt.figure(figsize=(10, 6))
    for col in selected_cols:
        plt.plot(drawdown.index, drawdown[col], label=col)

    plt.title(f"Drawdown Comparison - {mode}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, f"drawdowns_{mode}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_random_top15_percentiles():
    """
    Compare strategy performance against random Top15 distribution.
    """
    random_path = os.path.join(OUTPUT_DIR, "random_top15_percentiles.csv")
    comparison_path = os.path.join(OUTPUT_DIR, "score_method_comparison.csv")

    random_df = pd.read_csv(random_path)
    comparison_df = pd.read_csv(comparison_path)

    strategy = comparison_df[
        (comparison_df["score_mode"] == "risk_aware_finrobot")
        & (comparison_df["strategy"] == "Top15_ScoreWeighted_Tech")
    ].iloc[0]

    metrics = ["cumulative_return", "sharpe", "max_drawdown"]

    for metric in metrics:
        row = random_df[random_df["metric"] == metric].iloc[0]

        labels = ["Random p25", "Random Median", "Random p75", "Random p95", "Strategy"]
        values = [
            row["p25"],
            row["median"],
            row["p75"],
            row["p95"],
            strategy[metric],
        ]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.title(f"Random Top15 Baseline vs Strategy: {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=25)
        plt.tight_layout()

        output_path = os.path.join(PLOTS_DIR, f"random_top15_{metric}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_top_holdings_contribution(mode="risk_aware_finrobot", top_n=10):
    """
    Plot top positive holdings by total contribution.
    """
    path = os.path.join(OUTPUT_DIR, mode, "holdings_attribution.csv")
    attribution = pd.read_csv(path)

    contribution = (
        attribution.groupby("ticker")["return_contribution"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(contribution.index, contribution.values)
    plt.title(f"Top {top_n} Holdings by Return Contribution - {mode}")
    plt.xlabel("Ticker")
    plt.ylabel("Total Return Contribution")
    plt.xticks(rotation=25)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, f"top_holdings_contribution_{mode}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    ensure_dir(PLOTS_DIR)

    plot_score_method_comparison()

    for mode in ["quant_only", "simple_finrobot", "risk_aware_finrobot"]:
        plot_cumulative_returns(mode)
        plot_drawdowns(mode)

    plot_random_top15_percentiles()
    plot_top_holdings_contribution(mode="risk_aware_finrobot", top_n=10)

    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()