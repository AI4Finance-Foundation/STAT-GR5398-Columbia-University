from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/finrobot_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = ROOT / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
ORANGE = "#D97706"
GREEN = "#2E7D32"
RED = "#B23A48"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "legend.frameon": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#172033",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_equity(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    series = pd.to_numeric(frame["equity"], errors="coerce").dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def rank_ic(frame: pd.DataFrame, horizon: int) -> tuple[float, float, int, int]:
    col = f"ret_{horizon}"
    values: list[float] = []
    valid = frame.dropna(subset=[col])
    for _, group in valid.groupby("as_of"):
        if len(group) >= 5:
            corr = group["mispricing"].corr(group[col], method="spearman")
            if pd.notna(corr):
                values.append(float(corr))
    series = pd.Series(values, dtype=float)
    mean = float(series.mean())
    t_stat = float(mean / series.std() * np.sqrt(len(series))) if len(series) > 1 and series.std() > 0 else np.nan
    return mean, t_stat, len(series), int(valid[col].notna().sum())


def plot_development_rank_ic(ax: plt.Axes) -> None:
    frames = {
        "Three-method model": pd.read_csv(OUTPUT / "valuation_eval_dev_student.csv"),
        "Course P/E example": pd.read_csv(OUTPUT / "valuation_eval_dev_example.csv"),
    }
    horizons = [63, 126, 252]
    colors = [NAVY, GRAY]
    x = np.arange(len(horizons))
    width = 0.30
    for idx, ((label, frame), color) in enumerate(zip(frames.items(), colors)):
        vals = [rank_ic(frame, h)[0] for h in horizons]
        bars = ax.bar(
            x + (idx - 0.5) * width,
            vals,
            width=width,
            label=label,
            color=color,
            alpha=0.94,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.006,
                f"{value:+.3f}",
                ha="center",
                va="top",
                fontsize=8,
            )
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x, ["63 sessions", "126 sessions", "252 sessions"])
    ax.set_ylabel("Mean monthly Spearman rank IC")
    ax.set_title("Rank IC in development", fontweight="bold")
    ax.legend(ncol=1, loc="lower right", fontsize=8.5)
    ax.set_ylim(-0.15, 0.01)


def figure_rank_ic() -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    plot_development_rank_ic(ax)
    save(fig, "rank_ic_by_period.png")


def plot_development_interval_coverage(ax: plt.Axes) -> None:
    frames = [
        ("Three-method model", pd.read_csv(OUTPUT / "valuation_eval_dev_student.csv"), NAVY),
        ("Course P/E example", pd.read_csv(OUTPUT / "valuation_eval_dev_example.csv"), GRAY),
    ]
    items = []
    for label, frame, color in frames:
        eligible = frame.dropna(subset=["price_252", "low", "high"])
        inside = ((eligible["price_252"] >= eligible["low"]) & (eligible["price_252"] <= eligible["high"]))
        items.append((label, float(inside.mean()) * 100, int(inside.sum()), len(eligible), color))
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    colors = [x[4] for x in items]
    bars = ax.bar(labels, values, color=colors, width=0.64)
    for bar, (_, value, count, n, _) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%\n({count}/{n})", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 45)
    ax.set_ylabel("252-session endpoint inside [low, high] (%)")
    ax.set_title("One-year band coverage in development", fontweight="bold")


def figure_interval_coverage() -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    plot_development_interval_coverage(ax)
    save(fig, "interval_coverage.png")


def figure_development_quality() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.25))
    plot_development_rank_ic(axes[0])
    plot_development_interval_coverage(axes[1])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.88, wspace=0.22)
    save(fig, "development_rank_ic_and_coverage.png")


def figure_valuation_vs_price() -> None:
    frame = pd.read_csv(OUTPUT / "valuation_eval_dev_student.csv")
    frame = frame.dropna(subset=["price", "fair_value", "method_spread"]).copy()
    frame = frame[(frame["price"] > 0) & (frame["fair_value"] > 0)]

    pooled_values = np.concatenate(
        [frame["price"].to_numpy(), frame["fair_value"].to_numpy()]
    )
    lower, upper = np.quantile(pooled_values, [0.005, 0.995])
    visible = frame[
        frame["price"].between(lower, upper)
        & frame["fair_value"].between(lower, upper)
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    scatter = ax.scatter(
        visible["price"],
        visible["fair_value"],
        c=visible["method_spread"].clip(lower=1.0, upper=10.0),
        cmap="viridis",
        norm=LogNorm(vmin=1.0, vmax=10.0),
        s=26,
        alpha=0.68,
        linewidths=0,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=RED,
        linestyle="--",
        linewidth=1.4,
        label="Fair value = market price",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Market price per share (log scale)")
    ax.set_ylabel("Aggregated fair value per share (log scale)")
    ax.set_title(
        f"Valuation estimates versus market prices in development (n={len(frame):,})",
        fontweight="bold",
    )
    ax.legend(loc="upper left")

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.055)
    colorbar.minorticks_off()
    colorbar.set_ticks([1, 2, 3, 5, 10])
    colorbar.set_ticklabels(["1x", "2x", "3x", "5x", "10x"])
    colorbar.set_label(
        "Method spread (highest / lowest fair value; capped at 10x)"
    )
    fig.text(
        0.5,
        0.015,
        "Axes use the pooled 0.5th-99.5th percentiles of positive development prices and fair values; "
        "observations outside these limits are omitted from display only.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.16, top=0.90)
    save(fig, "valuation_vs_price.png")


def period_equities() -> list[tuple[str, pd.Series, pd.Series]]:
    return [
        (
            "Development",
            read_equity(OUTPUT / "configuration_study" / "dev_grid" / "baseline_A" / "equity.csv"),
            read_equity(OUTPUT / "configuration_study" / "dev_grid" / "absolute_m30_s2_n3" / "equity.csv"),
        ),
        (
            "Validation",
            read_equity(OUTPUT / "configuration_study" / "val_grid" / "baseline_A" / "equity.csv"),
            read_equity(OUTPUT / "configuration_study" / "val_grid" / "absolute_m30_s2_n3" / "equity.csv"),
        ),
        (
            "Test",
            read_equity(OUTPUT / "test_A_selection" / "equity.csv"),
            read_equity(OUTPUT / "test_B_valuation_absolute" / "equity.csv"),
        ),
    ]


def figure_cumulative_returns() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for ax, (label, baseline, valuation) in zip(axes, period_equities()):
        nav_a = baseline / 100000.0 * 100.0
        nav_b = valuation / 100000.0 * 100.0
        ax.plot(nav_a.index, nav_a, color=NAVY, linewidth=1.6, label="A: selection")
        ax.plot(nav_b.index, nav_b, color=ORANGE, linewidth=1.6, label="B: valuation")
        ax.set_title(label)
        ax.set_ylabel("Portfolio value (initial = 100)")
        ax.tick_params(axis="x", rotation=30)
        if ax is axes[0]:
            ax.legend(loc="upper left")
    fig.suptitle("Cumulative portfolio value under the frozen configuration", y=1.02, fontsize=12, fontweight="bold")
    save(fig, "cumulative_returns.png")


def figure_drawdowns() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for ax, (label, baseline, valuation) in zip(axes, period_equities()):
        dd_a = baseline / baseline.cummax() - 1.0
        dd_b = valuation / valuation.cummax() - 1.0

        ax.plot(
            dd_a.index,
            dd_a * 100,
            color=NAVY,
            linewidth=1.35,
            zorder=2,
            label="A: selection",
        )
        ax.plot(
            dd_b.index,
            dd_b * 100,
            color=ORANGE,
            linewidth=1.7,
            zorder=3,
            label="B: valuation",
        )
        ax.set_title(label)
        ax.set_ylabel("Drawdown (%)")
        ax.tick_params(axis="x", rotation=30)
        if ax is axes[0]:
            ax.legend(loc="lower left")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.77, wspace=0.32)
    fig.suptitle("Portfolio drawdowns", y=0.98, fontsize=12, fontweight="bold")
    save(fig, "drawdowns.png")


def shortened_config(config: str) -> str:
    mapping = {
        "baseline_A": "Baseline A",
        "absolute_m20_s3_n2": "Abs -20 / 2m / 3x",
        "absolute_m30_s3_n2": "Abs -30 / 2m / 3x",
        "absolute_m40_s3_n2": "Abs -40 / 2m / 3x",
        "absolute_m20_s2_n2": "Abs -20 / 2m / 2x",
        "absolute_m20_s3_n3": "Abs -20 / 3m / 3x",
        "absolute_m30_s2_n3": "Abs -30 / 3m / 2x",
        "quantile10_s3_n2": "Q10 / 2m / 3x",
        "quantile20_s3_n2": "Q20 / 2m / 3x",
        "quantile10_s2_n3": "Q10 / 3m / 2x",
    }
    return mapping.get(config, config)


def figure_configuration_selection() -> None:
    dev = pd.read_csv(OUTPUT / "configuration_study" / "dev_filter_grid_summary.csv")
    val = pd.read_csv(OUTPUT / "configuration_study" / "validation_filter_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for ax, frame, label in [(axes[0], dev, "Development grid"), (axes[1], val, "Preregistered validation shortlist")]:
        for row in frame.itertuples(index=False):
            frozen = row.config_id == "absolute_m30_s2_n3"
            baseline = row.config_id == "baseline_A"
            color = GREEN if frozen else NAVY if baseline else ORANGE
            size = 86 if frozen or baseline else 46
            ax.scatter(row.total_return_pct, row.sharpe, s=size, color=color, alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
            show_label = label.startswith("Preregistered") or frozen or baseline or row.config_id == "quantile10_s2_n3"
            if show_label:
                offset = (4, 4)
                if label.startswith("Development") and baseline:
                    offset = (-58, 5)
                elif label.startswith("Development") and frozen:
                    offset = (6, -10)
                elif label.startswith("Development") and row.config_id == "quantile10_s2_n3":
                    offset = (-70, -10)
                ax.annotate(
                    shortened_config(row.config_id),
                    (row.total_return_pct, row.sharpe),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=6.7,
                    color="#273244",
                )
        ax.axhline(0, color=LIGHT_GRAY, linewidth=0.8)
        ax.set_xlabel("Total return (%)")
        ax.set_ylabel("Sharpe ratio")
        ax.set_title(label)
    fig.suptitle("Configuration selection: development evidence and preregistered validation", y=1.02, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.01, "Green marks the frozen absolute -30%, three-method, 2x-spread configuration; full development labels are in the report table.", ha="center", fontsize=8, color=GRAY)
    save(fig, "configuration_selection.png")


def annual_returns(series: pd.Series) -> pd.Series:
    year_end = series.groupby(series.index.year).last()
    initial = pd.Series({int(year_end.index.min()) - 1: 100000.0})
    return pd.concat([initial, year_end]).pct_change().dropna() * 100.0


def figure_yearly_attribution() -> None:
    records = []
    for _, baseline, valuation in period_equities():
        a = annual_returns(baseline)
        b = annual_returns(valuation)
        for year in a.index:
            records.append((int(year), float(b.loc[year] - a.loc[year])))
    years = [x[0] for x in records]
    gaps = [x[1] for x in records]
    colors = [GREEN if x >= 0 else RED for x in gaps]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(years, gaps, color=colors, width=0.72)
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.axvline(2021.5, color=GRAY, linestyle="--", linewidth=0.9)
    ax.axvline(2023.5, color=GRAY, linestyle="--", linewidth=0.9)
    for bar, value in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.35 if value >= 0 else -0.55), f"{value:+.2f}", ha="center", va="center", fontsize=7.6)
    ax.text(2019, 1.01, "Development", transform=ax.get_xaxis_transform(), ha="center", color=GRAY, fontsize=8)
    ax.text(2022.5, 1.01, "Validation", transform=ax.get_xaxis_transform(), ha="center", color=GRAY, fontsize=8)
    ax.text(2025, 1.01, "Test", transform=ax.get_xaxis_transform(), ha="center", color=GRAY, fontsize=8)
    ax.set_ylabel("Calendar-year B-A return gap (percentage points)")
    ax.set_title("Frozen filter contribution varies sharply by year", fontweight="bold", pad=36)
    ax.set_xticks(years, [str(y) if y != 2026 else "2026 YTD" for y in years])
    save(fig, "yearly_attribution.png")


def figure_dcf_funnel() -> None:
    funnel = pd.read_csv(OUTPUT / "dev_analysis" / "report_evidence" / "dcf_funnel.csv")
    selected = funnel[funnel["count"] > 0].copy()
    label_map = {
        "insufficient_quarters": "Insufficient quarters",
        "engine_rejected": "Engine rejected",
        "valued": "Valued",
        "invalid_margin": "Invalid margin",
        "growth_history_missing": "Growth history missing",
        "nonpositive_price": "Nonpositive/missing price",
        "beta_insufficient_pairs": "Insufficient beta pairs",
        "missing_component": "Missing component",
    }
    selected["label"] = selected["status"].map(label_map)
    selected = selected.sort_values("count")
    colors = [GREEN if x == "valued" else RED if x == "engine_rejected" else ORANGE for x in selected["status"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.barh(selected["label"], selected["count"], color=colors)
    for bar, value in zip(bars, selected["count"]):
        ax.text(value + 7, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=8.5)
    ax.set_xlim(0, 700)
    ax.set_xlabel("Ticker-dates")
    ax.set_title("Locked DCF funnel across 1,200 development ticker-dates", fontweight="bold")
    ax.text(0.99, 0.03, "448 pre-engine exclusions; 752 engine calls = 626 valued + 126 rejected", transform=ax.transAxes, ha="right", fontsize=8, color=GRAY)
    save(fig, "dcf_funnel.png")


def main() -> None:
    set_style()
    figure_rank_ic()
    figure_interval_coverage()
    figure_development_quality()
    figure_valuation_vs_price()
    figure_cumulative_returns()
    figure_drawdowns()
    figure_configuration_selection()
    figure_yearly_attribution()
    figure_dcf_funnel()
    print(f"Wrote 9 figures to {FIGURES}")


if __name__ == "__main__":
    main()
