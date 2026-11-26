"""Show the development threshold curves against their respective selection baselines."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

here = Path(__file__).resolve().parent
out = here.parents[2] / "final_report_material"
curves = [
    ("NVDA included", here.parent / "dev_full_steps.csv", "#606577", "--"),
    ("NVDA excluded", here / "dev_full_steps.csv", "#006e76", "-"),
]
fig, (full, detail) = plt.subplots(1, 2, figsize=(10.1, 4.12),
                                    gridspec_kw={"width_ratios":[2.18, 1]}, sharey=True)
fig.subplots_adjust(left=.076, bottom=.17, top=.75, right=.98, wspace=.16)
fig.suptitle("Does the valuation threshold peak survive without NVDA?", x=.075, y=.98,
             ha="left", fontsize=15, color="#1c3441")
for axis in (full, detail):
    axis.axhline(0, color="#939a9e", lw=.85)
    axis.grid(axis="y", color="#e2e7e7", lw=.65)
    axis.set_axisbelow(True)
    axis.spines[["top","right"]].set_visible(False)
    axis.spines[["left","bottom"]].set_color("#969e9e")
    axis.tick_params(length=3, color="#969e9e", labelsize=9)
    axis.yaxis.set_major_locator(MaxNLocator(6))

for label, path, color, style in curves:
    s = pd.read_csv(path).sort_values("K")
    assert s.K.is_unique and s.K.min()==1 and s.K.max()==2.5
    assert abs(s.delta_cagr_pp.iloc[0]) < 1e-8
    baseline = s.cagr_pct.iloc[0]
    for axis in (full, detail):
        axis.step(s.K, s.delta_cagr_pp, where="post", color=color,
                  linestyle=style, lw=1.8, label=f"{label} (baseline {baseline:.2f}%)")
    if style == "-":
        peak = s.loc[s.delta_cagr_pp.idxmax()]
        full.annotate(f"Ex-NVDA high  +{peak.delta_cagr_pp:.2f} pp",
                      (peak.K, peak.delta_cagr_pp), xytext=(7, 14), textcoords="offset points",
                      fontsize=8, color=color, ha="left", va="bottom",
                      arrowprops={"arrowstyle":"-", "color":color, "lw":.8})
    for k in (1.983, 2.0):
        row = s.loc[s.K.le(k)].iloc[-1]
        detail.scatter(k, row.delta_cagr_pp, s=25, edgecolors=color,
                       facecolors=color if style=="-" else "white", zorder=4)

full.set(xlim=(.995,2.505), ylabel="CAGR minus own baseline (percentage points)")
full.set_xticks([1,1.25,1.5,1.75,2,2.25,2.5], ["1.0","1.25","1.5","1.75","2.0","2.25","2.5"])
full.axvspan(1.956,2.024, color="#eaf0ef", zorder=-1)
detail.set_xlim(1.956,2.024)
detail.set_xticks([1.96,1.983,2.0,2.02],["1.96","1.983","2.0","2.02"])
detail.axvline(1.983, color="#adb7b9", lw=.8, zorder=0)
full.set_title("Development period, 2017–2021", loc="left", fontsize=10)
detail.set_title("Detail around the original peak", loc="left", fontsize=10)
fig.supxlabel("Disagreement cutoff (highest / lowest central valuation)", x=.53, y=.036, fontsize=10)
fig.legend(*full.get_legend_handles_labels(), loc="upper left", ncol=2, frameon=False,
           bbox_to_anchor=(.068,.92), fontsize=9)
plt.rcParams["pdf.fonttype"]=42
for fmt in ("png","pdf"):
    fig.savefig(out / f"dev_threshold_ex_nvda.{fmt}", dpi=190, facecolor="white")
plt.close(fig)
