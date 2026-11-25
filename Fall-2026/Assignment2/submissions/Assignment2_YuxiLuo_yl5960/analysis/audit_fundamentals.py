"""Data audit for the valuation module: what is in fundamentals.db and is it usable?

Run from source_code/:
    python "../submissions/Assignment2_YuxiLuo_yl5960/analysis/audit_fundamentals.py"
"""
import sys
from pathlib import Path

SRC = Path.cwd()
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from shared.data_view import DataBundle, DataView

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

bundle = DataBundle.load()
as_of = bundle.prices.last_session_on_or_before(pd.Timestamp("2021-12-31").date())
view = DataView(bundle, as_of)
members = view.members()
f = view.fundamentals(sorted(members))
print("as_of", as_of, "members", len(members), "rows", len(f), "tickers", f["ticker"].nunique())
print("\n== columns and null rates (members, filings known by as_of)")
nulls = f.isna().mean().sort_values()
print(pd.DataFrame({"dtype": f.dtypes.astype(str), "null": nulls.round(3)}).loc[nulls.index].to_string())

print("\n== numeric distributions (last filing per ticker)")
last = f.sort_values("datadate").groupby("ticker").tail(1)
num = last.select_dtypes("number")
print(num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(3).to_string())

print("\n== sector codes")
print(last["gsector"].value_counts(dropna=False).to_string())

print("\n== NVDA last 8 quarters, selected columns")
cols = [c for c in ["datadate", "accepted_date", "filing_date", "EPS", "revenue_per_share", "ebitda_margin",
                    "operating_margin", "net_margin", "capex_per_share", "cash_per_share",
                    "interest_debt_per_share", "debt_to_capital", "effective_tax_rate", "gsector"] if c in f.columns]
print(f[f["ticker"] == "NVDA"].sort_values("datadate")[cols].tail(8).to_string())

print("\n== filing lag (accepted_date - datadate), days")
if "accepted_date" in f.columns:
    lag = (pd.to_datetime(f["accepted_date"], errors="coerce") - pd.to_datetime(f["datadate"])).dt.days
    print(lag.describe().round(1).to_string())

print("\n== quarter spacing: are per-share values single-quarter? (consecutive datadate gaps)")
g = f.sort_values(["ticker", "datadate"]).groupby("ticker")["datadate"]
gaps = pd.to_datetime(f.sort_values(["ticker", "datadate"])["datadate"]).groupby(
    f.sort_values(["ticker", "datadate"])["ticker"]).diff().dt.days.dropna()
print(gaps.describe().round(1).to_string())
print("share of gaps outside 80-100 days:", round(((gaps < 80) | (gaps > 100)).mean(), 3))

print("\n== sanity: TTM revenue_per_share x price vs EPS-based P/E across members")
f2 = f.sort_values(["ticker", "datadate"])
last4 = f2.groupby("ticker").tail(4)
n4 = last4.groupby("ticker")["ticker"].transform("size") == 4
ttm = last4[n4].groupby("ticker").agg(eps=("EPS", "sum"), rps=("revenue_per_share", "sum"))
px = pd.Series(view.last_close(ttm.index))
ttm["pe_ttm"] = px / ttm["eps"]
ttm["ps_ttm"] = px / ttm["rps"]
print(ttm[["pe_ttm", "ps_ttm"]].describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_string())
print("EPS<=0 share:", round((ttm["eps"] <= 0).mean(), 3), " missing price:", int(px.isna().sum()))

print("\n== ebitda_margin - operating_margin (implied D&A % revenue), last 4q median")
da = (last4[n4]["ebitda_margin"] - last4[n4]["operating_margin"]).groupby(last4[n4]["ticker"]).median()
print(da.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_string())
print("negative implied D&A share:", round((da < 0).mean(), 3))
