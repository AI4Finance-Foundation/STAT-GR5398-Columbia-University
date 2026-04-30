"""
Trend Exposure Overlay for Assignment 2
======================================

This script implements an optional improvement on top of the modified
Adaptive Rotation equity curve: a no-lookahead trend exposure overlay.

The overlay uses the QQQ trend signal from the previous rebalance
period. If the benchmark is above its moving average, the strategy keeps full
exposure. If it is below the moving average, the strategy keeps partial exposure
instead of going fully risk-off.

This is designed as a risk-management improvement rather than a replacement for
the original Adaptive Rotation selection logic.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "submission_final" / "outputs"
INPUT_CANDIDATES = [
    OUTPUT_DIR / "assignment2_line_chart_data.csv",
    PROJECT_ROOT / "analysis_output" / "assignment2_line_chart_data.csv",
]

BASE_COL = "Modified Adaptive Rotation"
BENCHMARK_COL = "QQQ"
IMPROVED_COL = "Modified + QQQ Trend Exposure Overlay"
MA_WEEKS = 8
RISK_OFF_EXPOSURE = 0.60


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find assignment2_line_chart_data.csv")


def calculate_metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    final_nav = float(nav.iloc[-1])
    annual_return = final_nav ** (1 / years) - 1
    annual_volatility = float(returns.std() * np.sqrt(52))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    drawdown = nav / nav.cummax() - 1
    return {
        "Final NAV": final_nav,
        "Total Return": final_nav - 1,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": float(drawdown.min()),
        "Win Rate": float((returns > 0).mean()),
    }


def apply_overlay(df: pd.DataFrame) -> pd.DataFrame:
    base_nav = df[BASE_COL].dropna()
    benchmark_nav = df[BENCHMARK_COL].dropna()
    base_returns = base_nav.pct_change().dropna()

    moving_average = benchmark_nav.rolling(MA_WEEKS, min_periods=MA_WEEKS).mean()
    risk_on = benchmark_nav > moving_average
    # Keep full exposure until the moving-average signal has enough history.
    risk_on = risk_on.mask(moving_average.isna(), True)
    # Shift one rebalance period to avoid look-ahead bias.
    risk_on = risk_on.shift(1).reindex(base_returns.index).ffill().fillna(True).infer_objects(copy=False)
    exposure = risk_on.map({True: 1.0, False: RISK_OFF_EXPOSURE}).astype(float)

    improved_returns = base_returns * exposure
    improved_nav = (1 + improved_returns).cumprod()
    improved_nav = pd.concat([pd.Series([1.0], index=[base_nav.index[0]]), improved_nav])
    improved_nav.name = IMPROVED_COL

    result = df.copy()
    result[IMPROVED_COL] = improved_nav.reindex(result.index).ffill()
    result["Trend Overlay Exposure"] = exposure.reindex(result.index).ffill().fillna(1.0)
    return result


def save_metrics(result: pd.DataFrame) -> pd.DataFrame:
    strategies = [
        IMPROVED_COL,
        BASE_COL,
        "Original Adaptive Rotation",
        "QQQ",
        "S&P 500 Proxy",
    ]
    rows = []
    seen = set()
    for strategy in strategies:
        if strategy in seen:
            continue
        seen.add(strategy)
        if strategy in result.columns:
            metrics = calculate_metrics(result[strategy])
            metrics = {"Strategy": strategy, **metrics}
            rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "trend_overlay_comparison_metrics.csv", index=False)
    return metrics_df


def save_plot(result: pd.DataFrame) -> None:
    plot_cols = [
        IMPROVED_COL,
        BASE_COL,
        "Original Adaptive Rotation",
        "QQQ",
        "S&P 500 Proxy",
    ]
    plot_cols = list(dict.fromkeys([col for col in plot_cols if col in result.columns]))

    plt.figure(figsize=(13, 7))
    for col in plot_cols:
        plt.plot(result.index, result[col], linewidth=2, label=f"{col} ({result[col].dropna().iloc[-1]:.2f}x)")

    plt.title("Assignment 2: Trend Exposure Overlay Improvement", fontsize=15, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("NAV, Initial = 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trend_overlay_equity_curve.png", dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_file = find_input_file()
    df = pd.read_csv(input_file, parse_dates=["Date"]).set_index("Date")

    # Keep a copy of the source curve data inside the final submission folder.
    df.to_csv(OUTPUT_DIR / "assignment2_line_chart_data.csv")

    result = apply_overlay(df)
    result.to_csv(OUTPUT_DIR / "trend_overlay_equity_curve.csv")
    metrics_df = save_metrics(result)
    save_plot(result)

    print("Saved trend overlay outputs to:", OUTPUT_DIR)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
