"""Why does the filter rule without the DCF do better? A diagnostic, not a new variant. Dev and val only.

Run from source_code/ with PYTHONHASHSEED=0:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/dcf_diagnostic.py

Sample: the committed configuration's evaluation sample (sens/diag/<period>_base.csv: top 20 growth_tech
on the first session of each month, with each method's fair value and the next 63/126-session returns).
For every row it recomputes the stock's revenue growth estimate on as_of (the DCF's year-1 estimate before
its clip, valuation_v2.revenue_growth), from a DataView limited to as_of. Then:

  1. Each method's value / price by growth quintile (pooled across months): is the DCF low on fast growers?
  2. Among rows the committed rule flags, the binding method (the highest value, which the price still
     exceeds), by growth group.
  3. Rows flagged only by the committed rule, only by the EV/EBITDA + peer P/E rule, or by both: how many,
     their growth, and their next 63/126-session returns against the unflagged rest.

Writes study/dcf_diagnostic_<period>.csv (one row per sample row with growth and flags) and prints tables.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import valuation_v2 as v2
from shared.data_view import DataBundle, DataView
from valuation import student_valuation as sv

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def add_growth(d, bundle):
    out = []
    for as_of, g in d.groupby("as_of", sort=True):
        view = DataView(bundle, as_of)
        tick = sorted(g["ticker"])
        q = sv._prepare(view.fundamentals(tick), view.close(tick))
        by = dict(tuple(q.groupby("ticker", sort=False)))
        gg = g.copy()
        gg["growth"] = [v2.revenue_growth(by.get(t)) for t in gg["ticker"]]
        out.append(gg)
    return pd.concat(out)


def main():
    bundle = DataBundle.load()
    for period in ("dev", "val"):
        d = pd.read_csv(HERE / "sens" / "diag" / f"{period}_base.csv")
        d = add_growth(d, bundle)
        d["growth"] = d["growth"].astype(float)
        for m in sv.METHODS:
            d[f"r_{m}"] = d[f"fair_{m}"] / d["price"]
        fairs = d[[f"fair_{m}" for m in sv.METHODS]]
        d["flag_committed"] = (d["n_methods"] >= 2) & (d["price"] > fairs.max(axis=1))
        d["flag_evpeer"] = d[["fair_ev_ebitda", "fair_peer_multiple"]].notna().all(axis=1) & \
            (d["price"] > d[["fair_ev_ebitda", "fair_peer_multiple"]].max(axis=1))
        d["binding"] = fairs.idxmax(axis=1).str.replace("fair_", "", regex=False).where(fairs.notna().any(axis=1))
        d.to_csv(HERE / "study" / f"dcf_diagnostic_{period}.csv", index=False)

        g = d.dropna(subset=["growth"]).copy()
        g["growth_q"] = pd.qcut(g["growth"], 5, labels=["Q1 slow", "Q2", "Q3", "Q4", "Q5 fast"])
        print(f"\n===== {period}: {len(d)} rows, growth known for {len(g)}")
        print("\n1) median value / price by growth quintile (rows where the method values the stock)")
        t1 = g.groupby("growth_q", observed=True).agg(
            growth_median=("growth", "median"),
            dcf=("r_dcf", "median"), ev_ebitda=("r_ev_ebitda", "median"), peer_pe=("r_peer_multiple", "median"),
            dcf_n=("r_dcf", "count"), flagged=("flag_committed", "mean"))
        print(t1.round(3).to_string())

        f = g[g["flag_committed"]]
        print(f"\n2) binding method among {len(f)} committed-rule flags, by growth group (share of flags)")
        f = f.assign(fast=np.where(f["growth"] >= g["growth"].quantile(0.6), "top 40% growth", "bottom 60% growth"))
        print(pd.crosstab(f["fast"], f["binding"], normalize="index", margins=True).round(2).to_string())
        print(pd.crosstab(f["fast"], f["binding"], margins=True).to_string())

        print("\n3) flagged by which rule: count, median growth, mean next-63 / next-126 return")
        grp = np.select([d["flag_committed"] & d["flag_evpeer"], d["flag_committed"], d["flag_evpeer"]],
                        ["both", "committed only", "EV+peer only"], "neither")
        d["group"] = grp
        t3 = d.groupby("group").agg(n=("ticker", "size"), growth_median=("growth", "median"),
                                    dcf_present=("fair_dcf", lambda s: s.notna().mean()),
                                    ret63=("ret_63", "mean"), ret126=("ret_126", "mean"))
        print(t3.round(3).to_string())
        co = d[d["group"] == "committed only"]
        print("committed-only flags, methods present:",
              co[[f"fair_{m}" for m in sv.METHODS]].notna().value_counts().to_dict())
        eo = d[d["group"] == "EV+peer only"]
        print("EV+peer-only flags: median DCF value / price", round(float(eo["r_dcf"].median()), 3),
              "share with DCF above price", round(float((eo["r_dcf"] > 1).mean()), 3))


if __name__ == "__main__":
    main()
