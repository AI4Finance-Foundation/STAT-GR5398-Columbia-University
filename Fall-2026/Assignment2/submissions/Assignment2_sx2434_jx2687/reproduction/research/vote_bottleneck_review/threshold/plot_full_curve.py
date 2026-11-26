"""Render the full development sensitivity figure for the Assignment 2 report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator

here = Path(__file__).resolve().parent
report = here.parents[1] / "final_report_material"
data = pd.read_csv(here / "dev_full_steps.csv").sort_values("K")
assert data.K.min() == 1 and data.K.max() == 2.5 and data.K.is_unique
assert abs(data.iloc[0].delta_cagr_pp) < 1e-8
report.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42})
fig, ax = plt.subplots(figsize=(8.4, 4.05))
fig.subplots_adjust(left=.111, right=.969, top=.858, bottom=.183)
teal, ink, grey = "#125e72", "#192f36", "#58686b"
ax.axhline(0, color=grey, lw=1, ls=(0, (4, 3)))
ax.step(data.K, data.delta_cagr_pp, where="post", color=teal, lw=1.8)
ax.set_xlim(.99, 2.52)
y = data.delta_cagr_pp
span = max(y.max()-y.min(), 1)
ax.set_ylim(y.min() - .14*span, y.max() + .15*span)
for k, offset, align in [(1,(0,-16),"left"), (2,(7,10),"left"), (2.3,(0,-18),"center"), (2.5,(-1,-18),"right")]:
    v = data.loc[data.K.eq(k),"delta_cagr_pp"].item()
    ax.scatter(k, v, s=25, color=teal, zorder=3)
    ax.annotate(f"{k:.1f}×  {v:+.2f}" if k>1 else "1.0×  0.00",
                (k, v), xytext=offset, textcoords="offset points", ha=align, color=ink, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=.88))
peak = data.loc[data.delta_cagr_pp.idxmax()]
ax.annotate(f"Sample high  {peak.delta_cagr_pp:+.2f} pp", (peak.K, peak.delta_cagr_pp),
            xytext=(-16, 2), textcoords="offset points", ha="right", va="center", fontsize=9,
            color=ink, arrowprops={"arrowstyle":"-", "color":grey, "lw":.75})
ax.set(title="Valuation disagreement and annualized return, 2017–2021",
       xlabel="Maximum allowed ratio of highest to lowest central valuation",
       ylabel="CAGR minus selection-only baseline (pp)")
ax.set_title(ax.get_title(), loc="left", pad=11, color=ink, fontsize=14)
ax.set_title("", loc="center")
ax.xaxis.set_major_locator(MultipleLocator(.1))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.grid(axis="y", color="#dee4e4", lw=.65)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
ax.spines[["bottom","left"]].set_color("#93a0a1")
ax.tick_params(length=3, color=grey)
for fmt in ["png","pdf"]:
    fig.savefig(report / f"dev_threshold_1_to_2p5.{fmt}", dpi=200, facecolor="white")
plt.close(fig)
