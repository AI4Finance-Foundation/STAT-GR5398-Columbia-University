"""Summarize saved development and validation NAVs. Run after replay.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "final_report_material"
GATED = ["K1.5", "K1.75", "K1.983", "K2.0", "K2.25", "K2.5"]


def summarize():
    annual = []
    rows = []
    for split in ("dev", "val"):
        nav = pd.read_csv(HERE / f"{split}_equity.csv", index_col="date", parse_dates=True)
        log_returns = np.log(nav).diff().dropna()
        cagr = 100 * np.expm1(252 * log_returns.mean())
        for rule in GATED:
            rows.append((split, rule, cagr[rule] - cagr.K_inf))
        closes = nav.groupby(nav.index.year).last()
        returns = 100 * (closes / closes.shift().fillna(100_000) - 1)
        for year, values in returns.iterrows():
            for rule in GATED:
                annual.append((split, year, rule, values[rule], values.K_inf, values[rule] - values.K_inf))
    pd.DataFrame(annual, columns=["split", "year", "rule", "gated_return_pct",
                                  "ungated_return_pct", "difference_pp"]).to_csv(HERE / "annual.csv", index=False)
    return pd.DataFrame(rows, columns=["split", "rule", "difference_pp"])


def plot(differences):
    table = differences.pivot(index="rule", columns="split", values="difference_pp").reindex(GATED)
    fig, ax = plt.subplots(figsize=(7.5, 3.9), layout="constrained")
    x = np.arange(len(table))
    ax.axhline(0, color="#353535", linewidth=1)
    for shift, key, label, color in [(-0.19, "dev", "Development (2017–2021)", "#20766a"),
                                     (0.19, "val", "Validation (2022–2023)", "#a25931")]:
        bars = ax.bar(x + shift, table[key], width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=8)
    ax.set_xticks(x, [rule[1:] for rule in GATED])
    ax.set_xlabel("Maximum ratio of highest to lowest valuation")
    ax.set_ylabel("CAGR difference (percentage points)")
    ax.set_title("Effect of adding a disagreement cutoff to the same weighted filter", fontsize=12, loc="left")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    ax.margins(y=.18)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
    fig.savefig(REPORT / "disagreement_cutoff_effect.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    plot(summarize())
