import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STOCK_SELECTED_PATH = "/Users/pengyueer/PycharmProjects/5398/Assign2/stock_selected.csv"
PRICE_PATH = "/Users/pengyueer/PycharmProjects/5398/Assign2/Comparing_prices.csv"
OUTPUT_DIR = "assign2_outputs"

LOOKBACK_WEEKS = 12
TOP_N = 2
INITIAL_CAPITAL = 1.0

REAL_ASSETS = ["XOM", "CVX", "FCX", "BHP", "GLD", "SLV"]
DEFENSIVE_ASSETS = ["TLT", "IEF", "XLU", "XLV", "IAU", "SHY", "UUP"]
DEFAULT_ASSETS = ["SPY", "QQQ", "IAU", "XLU", "XLV"]
STATIC_GROWTH = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]
BENCHMARK = "QQQ"


def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_stock_selected(path):
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["tic"] = df["tic"].astype(str).str.upper().str.strip()
    df["predicted_return"] = pd.to_numeric(df["predicted_return"], errors="coerce")
    df = df.dropna(subset=["tic", "predicted_return", "trade_date"])
    return df


def load_prices(path):
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices.columns = [str(c).upper().strip() for c in prices.columns]
    prices = prices.ffill().dropna(how="all")
    return prices


def information_ratio(ret):
    ret = ret.dropna()
    if len(ret) < 5:
        return -999
    std = ret.std()
    if std == 0 or np.isnan(std):
        return -999
    return ret.mean() / std


def get_growth_pool(stock_selected, date):
    available = stock_selected[stock_selected["trade_date"] <= date]
    if available.empty:
        return STATIC_GROWTH

    latest_date = available["trade_date"].max()
    pool = available[available["trade_date"] == latest_date]
    pool = pool.sort_values("predicted_return", ascending=False)

    return pool["tic"].drop_duplicates().tolist()


def select_by_residual_momentum(prices_window, candidates, top_n=2):
    valid = [t for t in candidates if t in prices_window.columns]

    scores = {}

    for tic in valid:
        s = prices_window[tic].dropna()
        if len(s) < 20:
            continue

        total_ret = s.iloc[-1] / s.iloc[0] - 1
        vol = s.pct_change().std()

        if pd.isna(vol) or vol == 0:
            score = total_ret
        else:
            score = total_ret / vol

        scores[tic] = score

    if len(scores) == 0:
        return valid[:top_n]

    selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return selected


def max_drawdown(curve):
    return (curve / curve.cummax() - 1).min()


def run_backtest(prices, stock_selected):
    daily_ret = prices.pct_change().fillna(0)

    rebalance_dates = prices.resample("W-FRI").last().index
    rebalance_dates = [d for d in rebalance_dates if d in prices.index]

    result_records = []
    weight_records = []
    holding_records = []

    for i in range(LOOKBACK_WEEKS, len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]
        lookback_start = rebalance_dates[i - LOOKBACK_WEEKS]

        price_window = prices.loc[lookback_start:reb_date]

        growth_pool = get_growth_pool(stock_selected, reb_date)

        groups = {
            "Growth_QQQ_Rolling": growth_pool,
            "Real_Assets": REAL_ASSETS,
            "Defensive_Assets": DEFENSIVE_ASSETS,
            "Default_Assets": DEFAULT_ASSETS
        }

        group_ir = {}

        for name, tickers in groups.items():
            valid = [t for t in tickers if t in daily_ret.columns]
            if len(valid) == 0:
                group_ir[name] = -999
            else:
                group_ret = daily_ret.loc[lookback_start:reb_date, valid].mean(axis=1)
                group_ir[name] = information_ratio(group_ret)

        best_group = max(group_ir, key=group_ir.get)
        candidates = groups[best_group]

        selected_assets = select_by_residual_momentum(
            price_window,
            candidates,
            top_n=TOP_N
        )

        selected_assets = [t for t in selected_assets if t in daily_ret.columns]

        if len(selected_assets) == 0:
            selected_assets = ["QQQ"]

        weight = 1 / len(selected_assets)

        for asset in selected_assets:
            weight_records.append({
                "rebalance_date": reb_date,
                "asset": asset,
                "weight": weight,
                "selected_group": best_group
            })

        holding_records.append({
            "rebalance_date": reb_date,
            "selected_group": best_group,
            "selected_assets": ",".join(selected_assets),
            "growth_pool_size": len(growth_pool),
            "growth_ir": group_ir["Growth_QQQ_Rolling"],
            "real_assets_ir": group_ir["Real_Assets"],
            "defensive_ir": group_ir["Defensive_Assets"],
            "default_ir": group_ir["Default_Assets"]
        })

        period_ret = daily_ret.loc[
            (daily_ret.index > reb_date) & (daily_ret.index <= next_reb_date),
            selected_assets
        ].mean(axis=1)

        for date, r in period_ret.items():
            qqq_r = daily_ret.loc[date, BENCHMARK] if BENCHMARK in daily_ret.columns else 0

            result_records.append({
                "trade_date": date,
                "strategy_return": r,
                "qqq_return": qqq_r,
                "selected_group": best_group,
                "holdings": ",".join(selected_assets)
            })

    result = pd.DataFrame(result_records)
    weights = pd.DataFrame(weight_records)
    holdings = pd.DataFrame(holding_records)

    result["equity_curve"] = INITIAL_CAPITAL * (1 + result["strategy_return"]).cumprod()
    result["qqq_curve"] = INITIAL_CAPITAL * (1 + result["qqq_return"]).cumprod()

    return result, weights, holdings


def calculate_metrics(result):
    strategy_ret = result["strategy_return"]
    qqq_ret = result["qqq_return"]
    excess_ret = strategy_ret - qqq_ret

    metrics = {
        "Strategy Total Return": result["equity_curve"].iloc[-1] - 1,
        "QQQ Total Return": result["qqq_curve"].iloc[-1] - 1,
        "Strategy Annual Return": strategy_ret.mean() * 252,
        "QQQ Annual Return": qqq_ret.mean() * 252,
        "Strategy Annual Volatility": strategy_ret.std() * np.sqrt(252),
        "QQQ Annual Volatility": qqq_ret.std() * np.sqrt(252),
        "Strategy Sharpe": strategy_ret.mean() / strategy_ret.std() * np.sqrt(252),
        "QQQ Sharpe": qqq_ret.mean() / qqq_ret.std() * np.sqrt(252),
        "Information Ratio": excess_ret.mean() / excess_ret.std() * np.sqrt(252),
        "Max Drawdown Strategy": max_drawdown(result["equity_curve"]),
        "Max Drawdown QQQ": max_drawdown(result["qqq_curve"])
    }

    return pd.DataFrame(metrics.items(), columns=["Metric", "Value"])


def plot_equity_curve(result):
    plt.figure(figsize=(12, 6))
    plt.plot(result["trade_date"], result["equity_curve"], label="Adaptive Rotation + QQQ Rolling Selection")
    plt.plot(result["trade_date"], result["qqq_curve"], label="QQQ Benchmark")
    plt.title("Assignment 2 Backtest Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "assignment2_line_chart.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_drawdown(result):
    strategy_dd = result["equity_curve"] / result["equity_curve"].cummax() - 1
    qqq_dd = result["qqq_curve"] / result["qqq_curve"].cummax() - 1

    plt.figure(figsize=(12, 5))
    plt.plot(result["trade_date"], strategy_dd, label="Strategy Drawdown")
    plt.plot(result["trade_date"], qqq_dd, label="QQQ Drawdown")
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "drawdown_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_group_distribution(holdings):
    counts = holdings["selected_group"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Selected Asset Group Distribution")
    plt.xlabel("Asset Group")
    plt.ylabel("Number of Rebalances")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "group_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    make_output_dir()

    stock_selected = load_stock_selected(STOCK_SELECTED_PATH)
    prices = load_prices(PRICE_PATH)

    result, weights, holdings = run_backtest(prices, stock_selected)
    metrics = calculate_metrics(result)

    result.to_csv(os.path.join(OUTPUT_DIR, "backtest_result.csv"), index=False)
    weights.to_csv(os.path.join(OUTPUT_DIR, "portfolio_weights.csv"), index=False)
    holdings.to_csv(os.path.join(OUTPUT_DIR, "weekly_holdings.csv"), index=False)
    metrics.to_csv(os.path.join(OUTPUT_DIR, "comparison_metrics.csv"), index=False)

    chart_data = result[["trade_date", "equity_curve", "qqq_curve"]]
    chart_data.to_csv(os.path.join(OUTPUT_DIR, "assignment2_line_chart_data.csv"), index=False)

    plot_equity_curve(result)
    plot_drawdown(result)
    plot_group_distribution(holdings)

    print("Done.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(metrics)


if __name__ == "__main__":
    main()