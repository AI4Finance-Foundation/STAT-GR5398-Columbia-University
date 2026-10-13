import pandas as pd, matplotlib.pyplot as plt
O = "outputs/"
series = {
    "Strategy: top 25% rank-weighted, monthly": "nav_Top_25__rank_weighted__monthly.csv",
    "Control: all live stocks EW, monthly (no model)": "nav_control__All_live_stocks_EW__monthly.csv",
    "Baseline (notebook)": "nav_Baseline.csv",
}
navs = {k: pd.read_csv(O + f, index_col=0, parse_dates=True)["nav"] for k, f in series.items()}
start, end = navs["Baseline (notebook)"].index[0], navs["Baseline (notebook)"].index[-1]
import os, sys
sys.path.insert(0, os.getcwd())
import strategy_backtest as sb
bpx = sb.benchmark_prices()
for t in ["SPY", "QQQ"]:
    px = bpx[t].loc[start:end]
    navs[t] = px / px.iloc[0]
colors = {"Strategy: top 25% rank-weighted, monthly": "#c0392b", "Control: all live stocks EW, monthly (no model)": "#7f8c8d",
          "Baseline (notebook)": "#2980b9", "SPY": "#27ae60", "QQQ": "#e67e22"}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
for k, n in navs.items():
    ls = "--" if k in ("SPY", "QQQ") else "-"
    a1.plot(n.index, n, ls=ls, lw=2.2 if k.startswith("Strategy") else 1.5, color=colors[k], label=f"{k}  ({n.iloc[-1]:.2f})")
    dd = n / n.cummax() - 1
    a2.plot(dd.index, dd, ls=ls, lw=1.2, color=colors[k])
a1.axhline(1, color="gray", lw=0.7); a1.set_ylabel("NAV (start = 1.0)"); a1.grid(alpha=.3); a1.legend(fontsize=9, loc="upper left")
a1.set_title("Net of 0.1% costs, 2020-05-01 → 2025-09-30 (final NAV in brackets)")
a2.set_ylabel("Drawdown"); a2.grid(alpha=.3); a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
fig.tight_layout(); fig.savefig(O + "main_comparison.png", dpi=200)
print("saved", O + "main_comparison.png")
