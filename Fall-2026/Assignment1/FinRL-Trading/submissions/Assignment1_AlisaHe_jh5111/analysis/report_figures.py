import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import strategy_backtest as sb

OUT = "report/figs"
os.makedirs(OUT, exist_ok=True)
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "font.family": "serif"})
SECTORS = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Cons. cyclical", 30: "Cons. defensive",
           35: "Healthcare", 40: "Financials", 45: "Technology", 50: "Comm. services", 55: "Utilities",
           60: "Real estate"}


def nav_figure():
    files = {"Strategy": "nav_Top_25__rank_weighted__monthly.csv",
             "No-model control": "nav_control__All_live_stocks_EW__monthly.csv",
             "Course baseline": "nav_Baseline.csv"}
    navs = {k: pd.read_csv(f"outputs/{f}", index_col=0, parse_dates=True)["nav"] for k, f in files.items()}
    px = sb.benchmark_prices()["SPY"].loc["2020-05-01":sb.BACKTEST_END]
    navs["SPY"] = px / px.iloc[0]
    style = {"Strategy": dict(color=BLUE, lw=2.0), "SPY": dict(color=INK2, lw=1.6, ls="--"),
             "No-model control": dict(color=AQUA, lw=1.3), "Course baseline": dict(color=ORANGE, lw=1.3)}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    for k, n in navs.items():
        a1.plot(n.index, n, label=k, **style[k])
        dd = n / n.cummax() - 1
        a2.plot(dd.index, dd, **{**style[k], "lw": style[k]["lw"] * 0.7})
    ends = sorted(((n.iloc[-1], k) for k, n in navs.items()), reverse=True)
    last_y = None
    for v, k in ends:
        y = v if last_y is None or last_y - v > 0.12 else last_y - 0.12
        a1.annotate(f"{k} {v:.2f}", (navs[k].index[-1], y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
        last_y = y
    a1.axvspan(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-08-03"), color=GRID, alpha=0.6, lw=0)
    a1.text(pd.Timestamp("2020-06-17"), 1.55, "warm-up: SPY", fontsize=6.5, color=INK2, rotation=90,
            ha="center", va="bottom")
    a1.set_ylabel("NAV (start = 1)")
    a1.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, fontsize=7.5)
    a2.set_ylabel("Drawdown")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    a1.set_xlim(navs["SPY"].index[0], navs["SPY"].index[-1] + pd.Timedelta(days=330))
    fig.tight_layout(h_pad=0.4)
    fig.savefig(f"{OUT}/fig_nav.pdf")


def versions_figure():
    rows = [("Course signal (look-ahead)", 2.560, False),
            ("Fix 1 (price timing)", 2.522, True),
            ("Fix 2 (lagged choice)", 2.475, True),
            ("Fix 1 + 2", 2.502, True),
            ("Fix 1 + avg. of 3", 2.519, True),
            ("Fix 1 + 2 + lag", 2.527, True),
            ("Fix 1 + avg. + lag (used)", 2.563, True)]
    spy = sb.benchmark_prices()["SPY"]
    spy_nav = spy.loc[sb.BACKTEST_END] / spy.loc["2020-05-01"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for i, (name, v, leak_free) in enumerate(rows[::-1]):
        used = name.endswith("(used)")
        ax.plot([v], [i], "o", ms=7 if used else 5.5, color=BLUE if leak_free else INK2,
                mfc=BLUE if (leak_free and used) else ("white" if not leak_free else BLUE),
                mec="white" if leak_free else INK2, mew=1.2 if leak_free else 1.4, zorder=3)
        right = v > spy_nav
        ax.text(v + (0.004 if right else -0.004), i, f"{v:.3f}", va="center", ha="left" if right else "right",
                fontsize=7, color=INK)
    ax.axvline(spy_nav, color=INK2, ls="--", lw=1.2)
    ax.text(spy_nav - 0.002, len(rows) - 0.45, f"SPY {spy_nav:.3f}", fontsize=7, color=INK2, va="bottom", ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
    ax.set_xlim(2.44, 2.59)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Final NAV")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_versions.pdf")


def sector_figure():
    daily, close, c2c, pred, sector_of, _ = sb.load_inputs()
    targets = {"Strategy": sb.top_quantile_rank_weighted(pred, 0.75),
               "Sector-neutral scoring": sb.top_quantile_rank_weighted(sb.sector_zscore(pred, sector_of), 0.75),
               "Universe (equal weight)": sb.top_quantile_equal(pred, 0.0)}
    w = pd.DataFrame({k: {SECTORS[s]: sb.avg_sector_weight(t, c2c, sector_of, sector=s) for s in SECTORS}
                      for k, t in targets.items()}).sort_values("Strategy")
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    y = np.arange(len(w))
    h = 0.26
    for j, (k, c) in enumerate(zip(w.columns, [BLUE, AQUA, "#b7b5ad"])):
        ax.barh(y + (1 - j) * (h + 0.02), w[k], height=h, color=c, label=k, edgecolor="white", linewidth=0.5)
    ax.text(w.loc["Technology", "Strategy"] + 0.01, y[list(w.index).index("Technology")] + h + 0.02,
            f"{w.loc['Technology', 'Strategy']:.0%}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(w.index, fontsize=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Average portfolio weight after warm-up")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sectors.pdf")


if __name__ == "__main__":
    nav_figure()
    versions_figure()
    sector_figure()
    print(f"wrote {OUT}/fig_nav.pdf, fig_versions.pdf, fig_sectors.pdf")
