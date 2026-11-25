"""Build the sensitivity tables from the files run_sensitivity.py wrote.

    python summarize_sensitivity.py            -> prints markdown, writes sens/summary.md

Every number comes from sens/diag/*.json, sens/diag/*.csv, sens/portfolio.jsonl and the
course's output/ablation_<period>.csv (portfolio A). Nothing is typed in by hand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_sensitivity import OUT, PERIODS, SRC, VARIANTS

LABELS = {
    "base": "committed config",
    "erp_040": "ERP 5% -> 4%", "erp_060": "ERP 5% -> 6%",
    "tg_020": "terminal g 2.5% -> 2.0%", "tg_030": "terminal g 2.5% -> 3.0%",
    "gw_000": "year-1 growth = 3y CAGR only", "gw_100": "year-1 growth = last YoY only",
    "gcap_025": "growth cap 35% -> 25%", "gcap_050": "growth cap 35% -> 50%",
    "nwc_000": "NWC/revenue 10% -> 0%", "nwc_020": "NWC/revenue 10% -> 20%",
    "yrs_05": "projection 10y -> 5y", "yrs_15": "projection 10y -> 15y",
    "beta_3y": "beta window 5y -> 3y",
    "hist_3y": "price window 5y -> 3y (beta + EV/EBITDA history)",
    "peer_cap100": "peer P/E cap 200 -> 100", "peers_10": "min peers 5 -> 10",
    "no_align": "no share-basis alignment",
    "bands_10_90": "multiple bands 25-75% -> 10-90%", "dcf_step_x2": "DCF grid steps x2",
}


def diag_rows(period):
    base_csv = pd.read_csv(OUT / "diag" / f"{period}_base.csv")
    base_csv["flag"] = flagged(base_csv)
    key = ["as_of", "ticker"]
    rows = []
    for tag in VARIANTS:
        js = OUT / "diag" / f"{period}_{tag}.json"
        if not js.exists():
            continue
        d = json.loads(js.read_text())
        c, dcf = d["rows"]["combined"], d["rows"]["dcf"]
        v = pd.read_csv(OUT / "diag" / f"{period}_{tag}.csv")
        v["flag"] = flagged(v)
        m = base_csv[key + ["mispricing", "flag"]].merge(v[key + ["mispricing", "flag"]], on=key,
                                                          how="outer", suffixes=("_b", "_v"))
        both = m.dropna(subset=["mispricing_b", "mispricing_v"])
        rank_corr = both.groupby("as_of").apply(
            lambda g: g["mispricing_b"].corr(g["mispricing_v"], method="spearman") if len(g) >= 5 else np.nan,
            include_groups=False).mean()
        fb, fv = m["flag_b"].eq(True), m["flag_v"].eq(True)
        jacc = (fb & fv).sum() / max((fb | fv).sum(), 1)
        rows.append({
            "tag": tag, "variant": LABELS[tag], "cover": d["rows"]["combined"]["coverage"],
            "DCF cover": dcf["coverage"],
            "IC63": c.get("IC63"), "IC126": c.get("IC126"), "t126": c.get("t126"), "IC252": c.get("IC252"),
            "DCF IC126": dcf.get("IC126"),
            "in range": d["range_inside"], "med mispr": d["median_mispricing"],
            "rank corr vs base": round(float(rank_corr), 3), "flag share": d["flag_share"],
            "flag overlap": round(float(jacc), 3),
            "flagged r126": d["flagged_ret126"], "rest r126": d["rest_ret126"],
        })
    return pd.DataFrame(rows)


def flagged(df):
    fairs = df[["fair_dcf", "fair_ev_ebitda", "fair_peer_multiple"]]
    # same rule as student_valuation.overvalued with the committed FILTER_* values (2 methods, premium 0)
    return (df["n_methods"] >= 2) & (df["price"] > fairs.max(axis=1))


def portfolio_rows(period):
    a = pd.read_csv(SRC / "output" / f"ablation_{period}.csv").set_index("run").loc[f"{period}_A_selection"]
    log = OUT / "portfolio.jsonl"
    recs = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    recs = {r["tag"][len("sens_"):]: r for r in recs if r["period"] == period}
    rows = [{"tag": "A", "variant": "A: selection only (no filter)", "return %": a["total_return_pct"], "CAGR %": a["cagr_pct"],
             "Sharpe": a["sharpe"], "MDD %": a["max_drawdown_pct"], "removed/wk": 0.0, "Sharpe - A": 0.0}]
    for tag, (_, range_only) in VARIANTS.items():
        if range_only or tag not in recs:
            continue
        r = recs[tag]
        rows.append({"tag": tag, "variant": LABELS[tag], "return %": r["total_return_pct"], "CAGR %": r["cagr_pct"],
                     "Sharpe": r["sharpe"], "MDD %": r["max_drawdown_pct"],
                     "removed/wk": r["avg_removed_per_week"], "Sharpe - A": round(r["sharpe"] - a["sharpe"], 3)})
    return pd.DataFrame(rows)


def md(df):
    """Markdown table without the optional tabulate dependency."""
    def cell(x):
        if isinstance(x, float):
            return "" if np.isnan(x) else f"{x:g}"
        return str(x)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def robustness(period, d, p):
    """Headline spread across the variants that change a fair value (range-only variants excluded)."""
    fv = [t for t, (_, r) in VARIANTS.items() if not r]
    dd, pp = d[d["tag"].isin(fv)], p[p["tag"].isin(fv)]
    delta = pp["Sharpe - A"]
    rho = pp["removed/wk"].corr(pp["Sharpe"], method="spearman")
    return (f"- {period}: combined IC126 from {dd['IC126'].min():+.3f} to {dd['IC126'].max():+.3f} over {len(dd)} "
            f"variants (t126 {dd['t126'].min():+.2f} to {dd['t126'].max():+.2f}); mispricing rank corr vs base "
            f">= {dd['rank corr vs base'].min():.3f}; flag overlap >= {dd['flag overlap'].min():.3f}. "
            f"Portfolio B Sharpe - A from {delta.min():+.3f} to {delta.max():+.3f}; B beats A in "
            f"{(delta > 0).sum()} of {len(pp)} variants; Spearman(removed/wk, Sharpe) = {rho:+.2f}.")


def main():
    parts, notes = [], []
    for period in PERIODS:
        d, p = diag_rows(period), portfolio_rows(period)
        d.to_csv(OUT / f"table_diag_{period}.csv", index=False)
        p.to_csv(OUT / f"table_portfolio_{period}.csv", index=False)
        parts.append(f"### Valuation quality, {period}\n\n" + md(d.drop(columns="tag")))
        parts.append(f"### Portfolio B, {period} (PYTHONHASHSEED=0)\n\n" + md(p.drop(columns="tag")))
        notes.append(robustness(period, d, p))
    text = "\n\n".join(parts) + "\n\n### Spread across variants\n\n" + "\n".join(notes) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
