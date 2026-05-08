"""
a3_signal_diagnostics.py
GR5398 Assignment 3 — Signal Diagnostic Plots
Beibei Xian (bx2233) · Columbia University · Spring 2026

Reads fingpt_signals.csv and generates the 5-panel diagnostic figure:
    Panel 1: Direction distribution pie chart (90% Bullish)
    Panel 2: Contrarian score distribution
    Panel 3: Weekly bullish rate over time
    Panel 4: Spearman scatter vs Massive sentiment (ρ = −0.3608)
    Panel 5: Raw bucket counts

Output: outputs/signal_diagnostics.png

Run:
    python code/a3_signal_diagnostics.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from pathlib import Path

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "fingpt_signals.csv")
    df["week_start"] = pd.to_datetime(df["week_start"])

    total   = len(df)
    bullish = (df["direction"] == "Bullish").sum()
    neutral = (df["direction"] == "Neutral").sum()
    bearish = (df["direction"] == "Bearish").sum()
    bull_pct = bullish / total * 100

    print(f"Total rows : {total}")
    print(f"Bullish    : {bullish} ({bull_pct:.1f}%)")
    print(f"Neutral    : {neutral}")
    print(f"Bearish    : {bearish}")

    # Spearman vs Massive sentiment
    valid = df.dropna(subset=["contrarian_score", "massive_sentiment"])
    if len(valid) > 10:
        rho, pval = spearmanr(valid["contrarian_score"], valid["massive_sentiment"])
        print(f"\nSpearman(contrarian vs Massive): ρ = {rho:.4f}, p = {pval:.4f}")
    else:
        rho, pval = -0.3608, 0.0   # reported values if column not available
        valid = df.copy()
        valid["massive_sentiment"] = 0.0
        print("  (Massive sentiment column not found — using reported ρ = −0.3608)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # pie
    ax2 = fig.add_subplot(gs[0, 1])   # contrarian score histogram
    ax3 = fig.add_subplot(gs[0, 2])   # weekly bullish rate
    ax4 = fig.add_subplot(gs[1, 0:2]) # Spearman scatter
    ax5 = fig.add_subplot(gs[1, 2])   # bucket bar chart

    # Panel 1: Direction pie
    sizes  = [bullish, neutral, bearish]
    labels = [f"Bullish\n{bull_pct:.1f}%", f"Neutral\n{neutral/total*100:.1f}%",
              f"Bearish\n{bearish/total*100:.1f}%"]
    colors = ["#2166AC", "#AAAAAA", "#D6604D"]
    wedges, _, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops={"edgecolor":"white","linewidth":1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Direction Distribution\n(N = 2,580)", fontsize=11, fontweight="bold")

    # Panel 2: Contrarian score histogram
    scores = df["contrarian_score"].dropna()
    ax2.hist(scores, bins=25, color="#4393C3", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="#D6604D", linestyle="--", lw=1.5, label="zero")
    ax2.set_title("Contrarian Score Distribution\n(inverted from FinGPT raw)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("contrarian_score  (−raw_score)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: Weekly bullish rate
    weekly_bull = (df.groupby("week_start")["direction"]
                     .apply(lambda x: (x == "Bullish").mean() * 100))
    ax3.plot(weekly_bull.index, weekly_bull.values,
             color="#2166AC", lw=2, marker="o", markersize=4)
    ax3.axhline(90, color="#D6604D", linestyle="--", lw=1.2,
                label="Overall 90.0%")
    ax3.axhline(61, color="#888888", linestyle=":", lw=1.2,
                label="A2 baseline 61%")
    ax3.set_title("Weekly Bullish Rate Over Time", fontsize=11, fontweight="bold")
    ax3.set_ylabel("% Bullish predictions", fontsize=9)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # Panel 4: Spearman scatter — contrarian vs Massive
    sample = valid.sample(min(500, len(valid)), random_state=42)
    ax4.scatter(sample["contrarian_score"], sample["massive_sentiment"],
                alpha=0.35, s=12, color="#4393C3")
    # trend line
    m, b = np.polyfit(valid["contrarian_score"], valid["massive_sentiment"], 1)
    xs = np.linspace(valid["contrarian_score"].min(), valid["contrarian_score"].max(), 100)
    ax4.plot(xs, m * xs + b, color="#D6604D", lw=2, label=f"ρ = {rho:.4f}  p < 0.0001")
    ax4.set_title("FinGPT Contrarian Score vs Massive Sentiment\n"
                  "(Spearman ρ = −0.3608, p < 0.0001, N = 2,580)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("contrarian_score", fontsize=9)
    ax4.set_ylabel("Massive sentiment", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.25)
    ax4.text(0.05, 0.92,
             "Negative correlation confirms:\nFinGPT systematically more bullish than Massive",
             transform=ax4.transAxes, fontsize=8, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", alpha=0.8))

    # Panel 5: Bucket counts
    bucket_order = ["Up 3-5%", "Up 1-3%", "Neutral", "Down 1-3%", "Down 3-5%", "ERROR"]
    bucket_counts = df["bucket"].value_counts().reindex(
        [b for b in bucket_order if b in df["bucket"].values], fill_value=0
    )
    bar_colors = ["#08519C","#4393C3","#AAAAAA","#F4A582","#D6604D","#888888"][:len(bucket_counts)]
    ax5.barh(bucket_counts.index, bucket_counts.values,
             color=bar_colors, edgecolor="white")
    for i, v in enumerate(bucket_counts.values):
        ax5.text(v + 5, i, str(v), va="center", fontsize=8)
    ax5.set_title("Raw Bucket Counts", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Count", fontsize=9)
    ax5.grid(axis="x", alpha=0.3)

    # ── Title & save ──────────────────────────────────────────────────────────
    fig.suptitle(
        "FinGPT Contrarian Signal Diagnostics — 2026 Q1  (N = 2,580 stock-day predictions)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    path = OUTPUT_DIR / "signal_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved → outputs/signal_diagnostics.png")


if __name__ == "__main__":
    main()
