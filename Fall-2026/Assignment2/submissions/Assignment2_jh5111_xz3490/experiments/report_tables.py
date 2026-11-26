"""Small report tables from saved files and prices; no backtests.

    .venv/bin/python experiments/report_tables.py dev val          (add test after the single test run)

benchmarks  SPY and QQQ buy-and-hold over each period, same formulas as portfolio/backtester.py stats()
ic          monthly rank IC of value() from output/valuation_eval_<period>.csv, with Newey-West t
            (lags 2 / 5 / 11 for 63 / 126 / 252 sessions, since monthly forward windows overlap), and by year.
            The course evaluator lets forward returns run past the end of the period (late-val labels reach
            into test); the *_censored columns drop every label whose window ends after the period end,
            and range_coverage_censored does the same for the 252-session range check.
Writes output/experiments/benchmarks.csv and ic_overlap.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from shared.periods import resolve

LAGS = {63: 2, 126: 5, 252: 11}


def benchmark_row(close, symbol, period, start, end):
    s = close[symbol].loc[str(start):str(end)].dropna()
    r = s.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    years = len(r) / 252
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    return {"period": period, "benchmark": symbol, "total_return_pct": round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
            "cagr_pct": round(cagr * 100, 2), "volatility_pct": round(vol * 100, 2),
            "sharpe": round(r.mean() * 252 / vol, 3), "max_drawdown_pct": round(dd * 100, 2),
            "calmar": round(cagr / abs(dd), 3)}


def newey_west_t(x, lags):
    x = np.asarray(x, dtype=float)
    d, n = x - x.mean(), len(x)
    s = d @ d / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return x.mean() / np.sqrt(s / n)


def monthly_ic(ev, col):
    g = ev.dropna(subset=["mispricing", col]).groupby("as_of")
    return g.apply(lambda m: m["mispricing"].rank().corr(m[col].rank()) if len(m) >= 5 else np.nan).dropna()


def ic_rows(period, sessions, end):
    ev = pd.read_csv(Path("output") / f"valuation_eval_{period}.csv", parse_dates=["as_of"])
    pos = sessions.searchsorted(ev["as_of"].values, side="left")
    rows = []
    for h, lags in LAGS.items():
        col = f"ret_{h}"
        label_end = pd.Series([sessions[i + h] if i + h < len(sessions) else pd.NaT for i in pos], index=ev.index)
        inside = label_end <= pd.Timestamp(end)
        ic, icc = monthly_ic(ev, col), monthly_ic(ev[inside], col)
        row = {"period": period, "horizon": h, "months": len(ic), "mean_ic": round(ic.mean(), 3),
               "t_naive": round(ic.mean() / (ic.std() / np.sqrt(len(ic))), 2),
               "t_newey_west": round(newey_west_t(ic.values, lags), 2),
               "labels_past_period_end": int((ev[col].notna() & ~inside).sum()),
               "months_censored": len(icc), "mean_ic_censored": round(icc.mean(), 3),
               "t_newey_west_censored": round(newey_west_t(icc.values, lags), 2) if len(icc) > lags + 2 else None}
        if h == 252:
            ok = ev[inside].dropna(subset=["price_252"])
            row["range_coverage_censored"] = round(((ok["low"] <= ok["price_252"]) & (ok["price_252"] <= ok["high"])).mean(), 3)
            row["range_obs_censored"] = len(ok)
        row.update({f"ic_{y}": round(v, 3) for y, v in ic.groupby(ic.index.year).mean().items()})
        rows.append(row)
    return rows


def main():
    periods = sys.argv[1:] or ["dev", "val"]
    close = pd.read_parquet("data/prices/daily_adjusted.parquet", columns=["symbol", "date", "close"])
    close = close[close["symbol"].isin(["SPY", "QQQ"])].pivot(index="date", columns="symbol", values="close")
    close.index = pd.to_datetime(close.index)
    sessions = pd.DatetimeIndex(close["SPY"].dropna().index)
    bench, ic = [], []
    for p in periods:
        start, end = resolve(p, None, None)
        bench += [benchmark_row(close, b, p, start, end) for b in ("SPY", "QQQ")]
        ic += ic_rows(p, sessions, end)
    out = Path("output") / "experiments"
    pd.DataFrame(bench).to_csv(out / "benchmarks.csv", index=False)
    pd.DataFrame(ic).to_csv(out / "ic_overlap.csv", index=False)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(bench).to_string(index=False))
        print(pd.DataFrame(ic).to_string(index=False))


if __name__ == "__main__":
    main()
