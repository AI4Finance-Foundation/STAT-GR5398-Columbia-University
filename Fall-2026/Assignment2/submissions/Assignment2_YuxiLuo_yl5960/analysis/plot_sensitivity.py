"""Figures for the sensitivity section, drawn from sens/table_*.csv (written by summarize_sensitivity.py).

    python plot_sensitivity.py   (any Python with pandas + matplotlib; the course venv has no matplotlib)

    sens/fig_sharpe_vs_A.png   portfolio B Sharpe minus portfolio A Sharpe, per variant, dev | val
    sens/fig_ic126.png         combined-value rank IC (126 sessions), per variant, dev | val
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

OUT = Path(__file__).resolve().parent / "sens"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
SERIES, HIGHLIGHT = "#2a78d6", "#eb6834"      # reference palette slots 1 and 2
TITLES = {"dev": "dev 2017–2021", "val": "val 2022–2023"}


def load(kind):
    tabs = {p: pd.read_csv(OUT / f"table_{kind}_{p}.csv") for p in TITLES}
    keep = [t for t in tabs["dev"]["tag"] if t != "A"]
    return {p: t.set_index("tag").loc[keep].reset_index() for p, t in tabs.items()}


def panel(ax, df, col, title, xlabel, fmt):
    y = range(len(df))[::-1]
    colors = [HIGHLIGHT if t == "base" else SERIES for t in df["tag"]]
    ax.barh(list(y), df[col], height=0.55, color=colors, zorder=3)
    ax.axvline(0, color=INK2, lw=1, zorder=4)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    base = df.index[df["tag"] == "base"][0]
    v = df.loc[base, col]
    ax.annotate(fmt.format(v), (v, list(y)[base]), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK)


def figure(kind, col, name, suptitle, xlabel, fmt, note):
    tabs = load(kind)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True, sharex=True, facecolor=SURFACE)
    for ax, (p, df) in zip(axes, tabs.items()):
        panel(ax, df, col, TITLES[p], xlabel, fmt)
    axes[0].set_yticks(range(len(tabs["dev"]))[::-1])
    axes[0].set_yticklabels(tabs["dev"]["variant"], color=INK)
    fig.suptitle(suptitle, x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, note, color=INK2, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    print("wrote", OUT / name)


def main():
    figure("portfolio", "Sharpe - A", "fig_sharpe_vs_A.png",
           "Valuation filter: Sharpe of portfolio B minus selection-only A, one assumption changed at a time",
           "Sharpe(B) − Sharpe(A)", "{:+.3f}",
           "Orange = committed config. 0 = selection only (no filter). PYTHONHASHSEED=0. Range-only variants (bands, DCF grid steps) cannot move B "
           "and are omitted.")
    figure("diag", "IC126", "fig_ic126.png",
           "Rank IC of mispricing vs next-126-session return, one assumption changed at a time",
           "rank IC, 126 sessions", "{:+.3f}",
           "Orange = committed config. Mean monthly Spearman IC over the top-20 growth_tech candidates. Negative IC: names the module calls undervalued went on to underperform.")


if __name__ == "__main__":
    main()
