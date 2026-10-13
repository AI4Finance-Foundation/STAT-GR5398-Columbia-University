import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.getcwd())
import strategy_backtest as sb

O = "outputs/"
STRATS = {
    "Main: top 25% rank-weighted": "nav_Top_25__rank_weighted__monthly.csv",
    "Top 25% EW": "nav_Top_25__EW__monthly_rebal.csv",
    "Sector-neutral top 25%": "nav_Sector_neutral_top_25___monthly.csv",
    "Control: all live EW (no model)": "nav_control__All_live_stocks_EW__monthly.csv",
}
PERIODS = {"Full 2020-05→2025-09": ("2020-05-01", "2025-09-30"),
           "Validation 2020-05→2022-12": sb.VALIDATION, "Test 2023-01→2025-09": sb.TEST}

rets = {k: pd.read_csv(O + f, index_col=0, parse_dates=True).net_ret for k, f in STRATS.items()}
bpx = sb.benchmark_prices()
for t in ["SPY", "QQQ", "RSP"]:
    rets[t] = bpx[t].loc["2020-05-01":].pct_change().fillna(0.0)
R = pd.DataFrame(rets).loc["2020-05-01":"2025-09-30"].dropna(subset=["SPY"]).fillna(0.0)


def stats(r):
    nav = (1 + r).cumprod()
    return {"CAGR": sb.cagr(r), "Vol": r.std() * np.sqrt(252), "Sharpe": sb.sharpe(r), "MaxDD": (nav / nav.cummax() - 1).min()}


rows = []
for p, (a, b) in PERIODS.items():
    for c in R:
        rows.append({"period": p, "strategy": c, **stats(R.loc[a:b, c])})
sub = pd.DataFrame(rows)
sub.to_csv("analysis/subperiod_performance.csv", index=False)
pd.set_option("display.width", 200)
print("Sub-period performance (net of 0.1% costs)")
print(sub.pivot_table(index="strategy", columns="period", values=["CAGR", "Sharpe"], sort=False)
      .map(lambda x: f"{x:.2f}").to_string(), "\n")

sig = []
for c in STRATS:
    ex = R[c] - R.SPY
    m1 = sm.OLS(R[c], sm.add_constant(R[["SPY"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    m2 = sm.OLS(R[c], sm.add_constant(R[["SPY", "RSP"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    sig.append({"strategy": c,
                "excess vs SPY (ann.)": ex.mean() * 252, "tracking error": ex.std() * np.sqrt(252),
                "info ratio": ex.mean() / ex.std() * np.sqrt(252), "t(excess)": ex.mean() / ex.std() * np.sqrt(len(ex)),
                "beta SPY": m1.params.SPY, "alpha vs SPY (ann.)": m1.params.const * 252, "t(alpha SPY)": m1.tvalues.const,
                "beta SPY | 2f": m2.params.SPY, "beta RSP | 2f": m2.params.RSP,
                "alpha SPY+RSP (ann.)": m2.params.const * 252, "t(alpha SPY+RSP)": m2.tvalues.const, "R2 2f": m2.rsquared})
sig = pd.DataFrame(sig).set_index("strategy")
sig.to_csv("analysis/significance.csv")
print("Excess return and factor regressions (daily, HAC t-stats)")
print(sig.T.map(lambda x: f"{x:.3f}").to_string(), "\n")

daily, close, c2c, pred, sector_of, _ = sb.load_inputs()
targets = {"Main: top 25% rank-weighted": sb.top_quantile_rank_weighted(pred, 0.75),
           "Top 25% EW": sb.top_quantile_equal(pred, 0.75),
           "Sector-neutral top 25%": sb.sector_neutral_top(pred, sector_of, 0.75),
           "Control: all live EW (no model)": sb.top_quantile_equal(pred, 0.0)}
spy = stats(R.SPY)
cost_rows = []
for fee in [0.0, 0.001, 0.002, 0.005]:
    for k, w in targets.items():
        r = sb.drift_backtest(w, c2c, "M", fee=fee).net_ret.loc[:"2025-09-30"]
        st = stats(r)
        cost_rows.append({"fee": f"{fee:.1%}", "strategy": k, "NAV": (1 + r).prod(), "CAGR": st["CAGR"],
                          "Sharpe": st["Sharpe"], "beats SPY NAV": (1 + r).prod() > (1 + R.SPY).prod()})
cost = pd.DataFrame(cost_rows)
cost.to_csv("analysis/cost_sensitivity.csv", index=False)
print(f"Cost sensitivity (SPY: NAV {(1 + R.SPY).prod():.2f}, CAGR {spy['CAGR']:.2%}, Sharpe {spy['Sharpe']:.2f})")
print(cost.pivot_table(index="strategy", columns="fee", values=["NAV", "Sharpe"], sort=False).round(2).to_string())
