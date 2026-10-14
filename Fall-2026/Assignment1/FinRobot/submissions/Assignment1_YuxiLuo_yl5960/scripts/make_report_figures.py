"""Figures for the research report (static, light, print). Writes PNGs to submissions/.../report_figures/.

    python scripts/make_report_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submissions" / "Assignment1_YuxiLuo_yl5960" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"
STATUS = {"Supported": "#0ca30c", "Unsupported": "#fab219", "Contradicted": "#d03b3b"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def tidy(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


# Figure 1: verdict mix by section (Step 1 texts, 5 starter companies)
c = pd.read_csv(ROOT / "audit" / "grounding_claims.csv")
c = c[c.variant == "step1"]
order = ["company_overview", "investment_overview", "valuation_overview", "tagline", "risks",
         "competitor_analysis", "major_takeaways"]
labels = {"major_takeaways": "major_takeaways*"}
mix = c.groupby("section")["verdict"].value_counts(normalize=True).unstack(fill_value=0).loc[order[::-1]]
n = c.groupby("section").size()
fig, ax = plt.subplots(figsize=(7.2, 3.3))
left = np.zeros(len(mix))
for v in ("Supported", "Unsupported", "Contradicted"):
    vals = mix[v].values
    ax.barh(range(len(mix)), vals, left=left, color=STATUS[v], height=0.62, edgecolor=SURFACE, linewidth=2, label=v)
    for i, (l, w) in enumerate(zip(left, vals)):
        if w >= 0.07:
            ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v != "Unsupported" else INK)
    left += vals
ax.set_yticks(range(len(mix)), [f"{labels.get(s, s)}  (n={n[s]})" for s in mix.index])
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.set_title("Claims by verdict, Step 1 sections", pad=24, loc="left")
tidy(ax)
fig.text(0.01, 0.01, "* canned fallback text in all five runs", fontsize=7.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT / "fig1_grounding_by_section.png", dpi=220)

# Figure 2: contradicted rate by Part 3 condition, 95% bootstrap CI over company-sections
S = pd.read_csv(ROOT / "audit" / "part3_sections.csv")
conds = ["C0", "C0+V", "C1", "C1+V"]
names = ["C0\noriginal prompt", "C0+V\n+ verifier", "C1\nlabelled evidence", "C1+V\nlabelled + verifier"]
rng = np.random.default_rng(0)
rates, lo, hi = [], [], []
for cond in conds:
    d = S[S.condition == cond].reset_index(drop=True)
    rates.append(d.contradicted.sum() / d.claims.sum())
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(d), len(d))
        boots.append(d.contradicted.values[idx].sum() / d.claims.values[idx].sum())
    lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
rates, lo, hi = map(np.array, (rates, lo, hi))
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(range(4), rates, width=0.5, color=BLUE, edgecolor=SURFACE, linewidth=2)
ax.errorbar(range(4), rates, yerr=[rates - lo, hi - rates], fmt="none", ecolor=INK2, elinewidth=1, capsize=4)
for i, r in enumerate(rates):
    ax.text(i + 0.3, r, f"{r:.1%}", va="center", ha="left", fontsize=9, color=INK)
ax.set_xticks(range(4), names)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.set_title("Claims contradicted by the writer's own data (95% CI)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig2_part3_contradicted.png", dpi=220)

# Figure 3: valuation engine target vs market price
r = pd.read_csv(ROOT / "notes" / "run_summary.csv")
r = r[~r.run.str.contains("/")].copy()
r["move"] = r.engine_up_down.str.rstrip("%").astype(float) / 100
# keep companies the engine could not value (UNH) as their own row, so the gap is visible rather than missing
r = r.sort_values("move", na_position="first").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for i, m in enumerate(r.move):
    if pd.isna(m):
        ax.text(0.03, i, "no value: engine returned NaN (no SG&A line, so no forecast EBITDA)", va="center", ha="left",
                fontsize=8, color=INK2, style="italic")
        continue
    ax.barh(i, m, color=BLUE if m > 0 else RED, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.text(m + (0.03 if m > 0 else -0.03), i, f"{m:+.0%}", va="center", ha="left" if m > 0 else "right", fontsize=8,
            color=INK)
ax.axvline(0, color=INK2, linewidth=0.8)
ax.set_ylim(-0.6, len(r) - 0.4)
ax.set_yticks(range(len(r)), r.run)
ax.set_xlim(-1.25, 1.95)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Valuation engine target vs. market price", loc="left")
tidy(ax)
fig.tight_layout()
fig.savefig(OUT / "fig3_engine_vs_price.png", dpi=220)
print("figures written to", OUT)
