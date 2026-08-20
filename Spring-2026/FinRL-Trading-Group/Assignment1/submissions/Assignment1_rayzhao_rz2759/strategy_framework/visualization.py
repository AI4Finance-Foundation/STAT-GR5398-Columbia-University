from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker as mtick

LAYOUT_VERSION = "dashboard_layout_v2_2026-02-20"


def _read_valid_cache(path: str | os.PathLike | None) -> pd.DataFrame | None:
    if path and os.path.exists(path):
        cache_df = pd.read_csv(path, index_col=0, parse_dates=True)
        if {"QQQ", "SPY"}.issubset(set(cache_df.columns)):
            return cache_df[["QQQ", "SPY"]].sort_index()
    return None


def _covers_requested_window(df: pd.DataFrame | None, start_dt, end_dt) -> bool:
    if df is None or df.empty:
        return False
    start_req = pd.to_datetime(start_dt).normalize()
    end_req = pd.to_datetime(end_dt).normalize()
    idx = df.index.sort_values()
    return (idx.min() <= start_req) and (idx.max() >= end_req)


def safe_download_benchmark(start_dt, end_dt, cache_file, fallback_cache=None, retries=5):
    """Download QQQ/SPY with retry + cache fallback."""
    cached_primary = _read_valid_cache(cache_file)
    cached_fallback = _read_valid_cache(fallback_cache) if fallback_cache is not None else None

    start_req = pd.to_datetime(start_dt).normalize()
    end_req = pd.to_datetime(end_dt).normalize()

    # Prefer caches only when they fully cover the requested backtest window.
    if _covers_requested_window(cached_primary, start_req, end_req):
        return cached_primary.loc[(cached_primary.index >= start_req) & (cached_primary.index <= end_req)]
    if _covers_requested_window(cached_fallback, start_req, end_req):
        return cached_fallback.loc[(cached_fallback.index >= start_req) & (cached_fallback.index <= end_req)]

    try:
        import yfinance as yf
    except Exception as e:
        for cached in [cached_primary, cached_fallback]:
            if cached is not None:
                clipped = cached.loc[(cached.index >= start_req) & (cached.index <= end_req)]
                if not clipped.empty:
                    print("Using partial benchmark cache due unavailable yfinance; date range may be shorter than portfolio.")
                    return clipped
        raise RuntimeError("yfinance is required for benchmark download when cache is unavailable") from e

    start_str = pd.to_datetime(start_dt).strftime("%Y-%m-%d")
    end_str = (pd.to_datetime(end_dt) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    last_err = None
    for k in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=["QQQ", "SPY"],
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )

            if isinstance(df.columns, pd.MultiIndex):
                qqq_close = df["Close"]["QQQ"].rename("QQQ")
                spy_close = df["Close"]["SPY"].rename("SPY")
                bench = pd.concat([qqq_close, spy_close], axis=1)
            else:
                bench = df[["QQQ", "SPY"]].copy()

            bench = bench.dropna(how="all")
            if not bench.empty:
                bench.to_csv(cache_file)
                return bench.loc[(bench.index >= start_req) & (bench.index <= end_req)]

            last_err = RuntimeError("Benchmark download returned empty dataframe")
        except Exception as e:
            last_err = e

        time.sleep(2 * k)

    for cached_path, cached in [(cache_file, cached_primary), (fallback_cache, cached_fallback)]:
        if cached is not None:
            clipped = cached.loc[(cached.index >= start_req) & (cached.index <= end_req)]
            if not clipped.empty:
                print(f"Using partial benchmark cache: {cached_path}")
                return clipped

    raise RuntimeError(f"Unable to download benchmark data: {last_err}")


def perf_stats(nav_series: pd.Series) -> pd.Series:
    r = nav_series.pct_change().dropna()
    ann_ret = nav_series.iloc[-1] ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = (nav_series / nav_series.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return pd.Series(
        {
            "Total Return": nav_series.iloc[-1] - 1,
            "Ann Return": ann_ret,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Calmar": calmar,
        }
    )


def build_strategy_dashboard(
    result: pd.DataFrame,
    strategy_output_dir: str | os.PathLike,
    output_path_step2: str | os.PathLike,
    config_name: str,
    roll_window: int = 63,
):
    print(f"Visualization layout version: {LAYOUT_VERSION}")

    strategy_output_dir = str(strategy_output_dir)
    output_path_step2 = str(output_path_step2)

    cache_file = os.path.join(strategy_output_dir, "benchmark_qqq_spy_cache.csv")
    fallback_cache = os.path.join(output_path_step2, "benchmark_qqq_spy_cache.csv")
    bench = safe_download_benchmark(result.index.min(), result.index.max(), cache_file, fallback_cache=fallback_cache)
    bench = bench.loc[(bench.index >= result.index.min()) & (bench.index <= result.index.max())]
    if not bench.empty and ((bench.index.min() > result.index.min()) or (bench.index.max() < result.index.max())):
        print(
            "Benchmark coverage is shorter than portfolio: "
            f"benchmark {bench.index.min()} to {bench.index.max()}, "
            f"portfolio {result.index.min()} to {result.index.max()}."
        )

    qqq_nav = bench["QQQ"] / bench["QQQ"].iloc[0]
    spy_nav = bench["SPY"] / bench["SPY"].iloc[0]
    comparison_df = pd.DataFrame({"Portfolio": result["nav"].sort_index()})
    comparison_df = comparison_df.join(pd.DataFrame({"QQQ": qqq_nav, "SPY": spy_nav}).sort_index(), how="left")
    comparison_df[["QQQ", "SPY"]] = comparison_df[["QQQ", "SPY"]].ffill()
    comparison_df = comparison_df.dropna(subset=["Portfolio", "QQQ", "SPY"])
    comparison_df = comparison_df / comparison_df.iloc[0]

    ret = comparison_df.pct_change().dropna()
    stats_table = pd.DataFrame(
        {
            "Portfolio": perf_stats(comparison_df["Portfolio"]),
            "QQQ": perf_stats(comparison_df["QQQ"]),
            "SPY": perf_stats(comparison_df["SPY"]),
        }
    ).T

    stats_print = stats_table.copy()
    for col in ["Total Return", "Ann Return", "Ann Vol", "Max Drawdown"]:
        stats_print[col] = stats_print[col].map(lambda x: f"{x:.2%}")
    stats_print["Sharpe"] = stats_table["Sharpe"].map(lambda x: f"{x:.2f}")
    stats_print["Calmar"] = stats_table["Calmar"].map(lambda x: f"{x:.2f}")

    print("Performance Summary:")
    print(stats_print[["Total Return", "Ann Return", "Ann Vol", "Sharpe", "Max Drawdown", "Calmar"]])

    roll_ret = (1 + ret).rolling(roll_window).apply(np.prod, raw=True) - 1
    roll_excess_spy = roll_ret["Portfolio"] - roll_ret["SPY"]
    roll_excess_qqq = roll_ret["Portfolio"] - roll_ret["QQQ"]
    dd_df = comparison_df / comparison_df.cummax() - 1

    yearly = comparison_df.resample("Y").last().pct_change().dropna()
    yearly.index = yearly.index.year

    viz_df = result.reindex(comparison_df.index).ffill()
    if "exposure" not in viz_df.columns:
        viz_df["exposure"] = 1.0
    if "regime" not in viz_df.columns:
        viz_df["regime"] = 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(19, 12), facecolor="#f5f7fb", constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9], width_ratios=[1.0, 1.0])

    top = outer[0, :].subgridspec(1, 2, width_ratios=[3.4, 1.2], wspace=0.05)
    ax_nav = fig.add_subplot(top[0, 0])
    ax_nav_tbl = fig.add_subplot(top[0, 1])
    ax_dd = fig.add_subplot(outer[1, 0])
    ax_excess = fig.add_subplot(outer[1, 1])
    ax_year = fig.add_subplot(outer[2, 0])
    ax_exp = fig.add_subplot(outer[2, 1])

    col = {
        "Portfolio": "#1f3b73",
        "QQQ": "#e07a1f",
        "SPY": "#2a9d8f",
        "pos": "#5cb85c",
        "neg": "#e76f51",
        "riskoff": "#f94144",
        "transition": "#f9c74f",
    }

    ax_nav.plot(comparison_df.index, comparison_df["Portfolio"], label="Portfolio", linewidth=2.8, color=col["Portfolio"])
    ax_nav.plot(comparison_df.index, comparison_df["QQQ"], label="QQQ", linewidth=2.0, color=col["QQQ"], alpha=0.9)
    ax_nav.plot(comparison_df.index, comparison_df["SPY"], label="SPY", linewidth=2.0, color=col["SPY"], alpha=0.9)
    ax_nav.fill_between(comparison_df.index, comparison_df["Portfolio"], comparison_df["SPY"], where=(comparison_df["Portfolio"] >= comparison_df["SPY"]), color=col["pos"], alpha=0.10)
    ax_nav.fill_between(comparison_df.index, comparison_df["Portfolio"], comparison_df["SPY"], where=(comparison_df["Portfolio"] < comparison_df["SPY"]), color=col["neg"], alpha=0.08)
    ax_nav.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=1)
    ax_nav.set_title("Cumulative NAV Comparison", fontsize=13, fontweight="bold")
    ax_nav.set_ylabel("Normalized NAV")
    ax_nav.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax_nav.legend(loc="upper left", ncol=3, frameon=True)
    ax_nav.grid(alpha=0.25, linestyle="--")

    summary_for_table = stats_print.loc[
        ["Portfolio", "QQQ", "SPY"],
        ["Total Return", "Ann Return", "Ann Vol", "Sharpe", "Max Drawdown"],
    ]
    ax_nav_tbl.axis("off")
    ax_nav_tbl.set_title("Performance Summary", fontsize=11, fontweight="bold", pad=8)
    tbl = ax_nav_tbl.table(
        cellText=summary_for_table.values,
        rowLabels=summary_for_table.index,
        colLabels=summary_for_table.columns,
        cellLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.96],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    ax_dd.plot(dd_df.index, dd_df["Portfolio"], label="Portfolio", linewidth=2.3, color=col["Portfolio"])
    ax_dd.plot(dd_df.index, dd_df["QQQ"], label="QQQ", linewidth=1.8, color=col["QQQ"], alpha=0.9)
    ax_dd.plot(dd_df.index, dd_df["SPY"], label="SPY", linewidth=1.8, color=col["SPY"], alpha=0.9)
    ax_dd.fill_between(dd_df.index, dd_df["Portfolio"], 0, color=col["Portfolio"], alpha=0.12)
    ax_dd.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
    ax_dd.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax_dd.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_dd.legend(frameon=True)
    ax_dd.grid(alpha=0.25, linestyle="--")

    ax_excess.plot(roll_excess_spy.index, roll_excess_spy, label="Portfolio - SPY", color="#d62839", linewidth=2.2)
    ax_excess.plot(roll_excess_qqq.index, roll_excess_qqq, label="Portfolio - QQQ", color="#6d597a", linewidth=2.0)
    ax_excess.fill_between(roll_excess_spy.index, roll_excess_spy, 0, where=(roll_excess_spy >= 0), color=col["pos"], alpha=0.12)
    ax_excess.fill_between(roll_excess_spy.index, roll_excess_spy, 0, where=(roll_excess_spy < 0), color=col["neg"], alpha=0.10)
    ax_excess.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
    ax_excess.set_title(f"Rolling {roll_window}-Day Excess Return", fontsize=12, fontweight="bold")
    ax_excess.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_excess.legend(frameon=True)
    ax_excess.grid(alpha=0.25, linestyle="--")

    if not yearly.empty:
        ax_year.plot(yearly.index, yearly["Portfolio"], marker="o", linewidth=2.2, color=col["Portfolio"], label="Portfolio")
        ax_year.plot(yearly.index, yearly["QQQ"], marker="o", linewidth=1.9, color=col["QQQ"], label="QQQ")
        ax_year.plot(yearly.index, yearly["SPY"], marker="o", linewidth=1.9, color=col["SPY"], label="SPY")
        ax_year.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
        ax_year.set_title("Calendar-Year Returns", fontsize=12, fontweight="bold")
        ax_year.set_xlabel("Year")
        ax_year.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_year.legend(ncol=3, frameon=True)
        ax_year.grid(alpha=0.25, linestyle="--")
    else:
        ax_year.text(0.5, 0.5, "Not enough yearly data", ha="center", va="center")
        ax_year.set_title("Calendar-Year Returns", fontsize=12, fontweight="bold")
        ax_year.axis("off")

    exp_series = viz_df["exposure"].clip(lower=0.0, upper=1.0)
    regime_series = viz_df["regime"].fillna(1).astype(int)

    ax_exp.plot(exp_series.index, exp_series.values, color="#277da1", linewidth=2.3, label="Equity Exposure")

    riskoff_mask = regime_series == 2
    transition_mask = regime_series == 1

    if riskoff_mask.any():
        ax_exp.fill_between(exp_series.index, 0, 1, where=riskoff_mask.values, color=col["riskoff"], alpha=0.10, label="RiskOff")
    if transition_mask.any():
        ax_exp.fill_between(exp_series.index, 0, 1, where=transition_mask.values, color=col["transition"], alpha=0.10, label="Transition")

    ax_exp.set_ylim(0, 1.05)
    ax_exp.set_title("Dynamic Exposure with Regime Overlay", fontsize=12, fontweight="bold")
    ax_exp.set_ylabel("Exposure")
    ax_exp.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_exp.grid(alpha=0.25, linestyle="--")

    handles, labels = ax_exp.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_exp.legend(
        unique.values(),
        unique.keys(),
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        fontsize=9,
    )

    for ax in [ax_nav, ax_dd, ax_excess, ax_year, ax_exp]:
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(f"Dynamic Asset Allocation Dashboard | {config_name}", fontsize=17, fontweight="bold")

    fig_file = os.path.join(strategy_output_dir, "strategy_dashboard.png")
    plt.savefig(fig_file, dpi=320, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved visualization dashboard to: {fig_file}")
    print(f"Aligned comparison data shape: {comparison_df.shape}")
    print(f"Comparison date range: {comparison_df.index.min()} to {comparison_df.index.max()}")

    plt.show()
    return comparison_df, stats_table, fig_file
