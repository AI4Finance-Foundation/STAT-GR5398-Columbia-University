import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "./prepared"
RESULTS_DIR = "./results"
EXTRA_PRICES = "./analysis/yf_missing_prices.csv"
OUT_DIR = "./outputs"
FAILURES = {"SIVB": "2023-03-10", "FRC": "2023-05-01"}
FEE_RATE = 0.001 
BACKTEST_START = "2020-05-01"
BACKTEST_END = "2025-09-30"
SECTORS = range(10, 65, 5)
VALIDATION = ("2020-05-01", "2022-12-31")
TEST = ("2023-01-01", "2025-09-30")
MOMENTUM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  
WARMUP_COVERAGE = 0.5
MEMBER_COUNTS = "./analysis/index_member_counts.csv"  
_WARMUP = {} 
SIGNAL_MODELS = ("rf", "xgb", "gbm")
EXTRA_LAG_DAYS = 1
BENCHMARKS = "./analysis/benchmark_prices.csv"
os.makedirs(OUT_DIR, exist_ok=True)


def quarter_reportday_ffill(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    out = df.reindex(pd.bdate_range(df.index.min(), df.index.max())).sort_index()
    idx = out.index
    for col in out.columns:
        s = out[col]
        dates = s.dropna().index
        for i, t0 in enumerate(dates):
            t1 = dates[i + 1] if i + 1 < len(dates) else None
            cap = (t0.to_period("Q") + 1).end_time.normalize()
            mask = (idx > t0) & (idx < (t1 if t1 and t1 <= cap else cap + pd.Timedelta("1ns")))
            out.loc[mask, col] = s.at[t0]
    return out


def usable_date(reportdate, quarter_date, extra_days=EXTRA_LAG_DAYS):
    first = np.maximum(pd.to_datetime(reportdate), pd.to_datetime(quarter_date) + pd.offsets.BDay(1))
    return pd.DatetimeIndex(first) + pd.offsets.BDay(extra_days)


def load_predictions(models=SIGNAL_MODELS, leak_free=True):
    key = pd.read_csv(os.path.join(DATA_DIR, "final_ratios.csv"), usecols=["tic", "date", "reportdate"],
                      parse_dates=["date", "reportdate"]).drop_duplicates(["tic", "reportdate"])
    preds, sector_of = [], {}
    for sec in SECTORS:
        files = [os.path.join(RESULTS_DIR, f"sector{sec}", f"df_predict_{m}.csv") for m in models]
        if not all(os.path.exists(f) for f in files):
            continue
        frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
        df = sum(frames) / len(frames)
        if leak_free:
            long = df.stack().rename("pred").reset_index()
            long.columns = ["reportdate", "tic", "pred"]
            long = long.merge(key, on=["tic", "reportdate"], how="left")
            long["live"] = usable_date(long.reportdate, long.date)
            df = long.pivot_table(index="live", columns="tic", values="pred", aggfunc="last")
        df = quarter_reportday_ffill(df)
        sector_of.update({t: sec for t in df.columns})
        preds.append(df)
    out = pd.concat(preds, axis=1).sort_index()
    out = out.reindex(pd.bdate_range(min(pd.Timestamp(BACKTEST_START), out.index.min()), out.index.max()))
    return out, pd.Series(sector_of)


def top_quantile_equal(pred, q):
    keep = pred.ge(pred.quantile(q, axis=1), axis=0) & pred.notna()
    return keep.div(keep.sum(axis=1), axis=0).fillna(0.0)


def top_quantile_rank_weighted(pred, q):
    keep = pred.ge(pred.quantile(q, axis=1), axis=0) & pred.notna()
    rank = pred.where(keep).rank(axis=1)  
    return rank.div(rank.sum(axis=1), axis=0).fillna(0.0)


def sector_neutral_top(pred, sector_of, q):
    w = pd.DataFrame(0.0, index=pred.index, columns=pred.columns)
    for sec in sector_of.unique():
        cols = [c for c in sector_of.index[sector_of == sec] if c in pred.columns]
        w[cols] = top_quantile_equal(pred[cols], q)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)


def momentum_12_1(close):
    return close.shift(21) / close.shift(252) - 1


def blended_rank_weighted(pred, mom, lam, q=0.75):
    live = pred.notna()
    mom = mom.reindex(pred.index, method="ffill").reindex(columns=pred.columns).where(live)
    score = (1 - lam) * pred.rank(axis=1, pct=True) + lam * mom.rank(axis=1, pct=True).where(mom.notna(), 0.5).where(live)
    return top_quantile_rank_weighted(score, q)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def cagr(r):
    return (1 + r).prod() ** (252 / len(r)) - 1


def choose_momentum_weight(pred, close, c2c):
    mom = momentum_12_1(close)
    grid, targets = {}, {}
    for lam in MOMENTUM_GRID:
        targets[lam] = blended_rank_weighted(pred, mom, lam)
        r = drift_backtest(targets[lam], c2c, "M").net_ret
        v, t = r.loc[VALIDATION[0]:VALIDATION[1]], r.loc[TEST[0]:TEST[1]]
        grid[lam] = {"val_CAGR": cagr(v), "val_Sharpe": sharpe(v), "test_CAGR": cagr(t), "test_Sharpe": sharpe(t)}
    grid = pd.DataFrame(grid).T
    best = grid.val_Sharpe.idxmax()
    return best, grid, targets[best]


# backtest engines
def notebook_baseline(target, daily):
    ret = daily.pivot(index="date", columns="tic", values="intraday")
    dates = target.index.intersection(ret.index)
    w, r = target.loc[dates], ret.reindex(index=dates, columns=target.columns).fillna(0.0)
    gross = (w * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    cost = FEE_RATE * turnover
    cost.iloc[0] = 0.0
    return pd.DataFrame({"gross_ret": gross, "turnover": turnover, "cost": cost, "net_ret": gross - cost})


def drift_backtest(target, c2c, rebalance="M", fee=FEE_RATE, warmup=True):
    dates = target.index.intersection(c2c.index)
    dates = dates[dates <= BACKTEST_END]
    target, r = target.loc[dates], c2c.reindex(index=dates, columns=target.columns).fillna(0.0)
    n = len(target.columns)
    hold_spy = np.zeros(len(dates), dtype=bool)
    spy_r = np.zeros(len(dates))
    if warmup and _WARMUP:
        hold_spy = (_WARMUP["coverage"].reindex(dates) < WARMUP_COVERAGE).fillna(False).values
        spy_r = _WARMUP["spy"].reindex(dates).fillna(0.0).values
    spy_only = np.r_[np.zeros(n), 1.0]
    if rebalance == "D":
        reb = pd.Series(True, index=dates)
    else:
        p = dates.to_period(rebalance)
        reb = pd.Series(np.r_[True, p[1:] != p[:-1]], index=dates)

    w = np.zeros(n + 1)
    rows = []
    for i, d in enumerate(dates):
        rd = np.r_[r.iloc[i].values, spy_r[i]]
        gross = float(w @ rd) if i > 0 else 0.0
        if i > 0 and (1 + gross) != 0:
            w = w * (1 + rd) / (1 + gross)
        turnover = 0.0
        if reb.iloc[i]:
            tgt = spy_only if hold_spy[i] else np.r_[target.iloc[i].values, 0.0]
            turnover = float(np.abs(tgt - w).sum())
            w = tgt.copy()
        cost = fee * turnover
        rows.append((d, gross, turnover, cost, gross - cost))
    return pd.DataFrame(rows, columns=["date", "gross_ret", "turnover", "cost", "net_ret"]).set_index("date")


# metrics
def metrics(net_ret, turnover=None, cost=None):
    nav = (1 + net_ret).cumprod()
    years = (net_ret.index[-1] - net_ret.index[0]).days / 365.25
    dd = nav / nav.cummax() - 1
    vol = net_ret.std() * np.sqrt(252)
    m = {
        "Final NAV": nav.iloc[-1],
        "Cum. return": nav.iloc[-1] - 1,
        "CAGR": nav.iloc[-1] ** (1 / years) - 1,
        "Ann. vol": vol,
        "Sharpe (rf=0)": net_ret.mean() * 252 / vol,
        "Max drawdown": dd.min(),
    }
    if turnover is not None:
        m["Avg daily turnover"] = turnover.mean()
        m["Total cost drag"] = cost.sum()
    return m


def coverage_of(panel):
    members = pd.read_csv(MEMBER_COUNTS, parse_dates=["date"]).set_index("date").n_members
    members = members.reindex(members.index.union(panel.index)).ffill().reindex(panel.index)
    return panel.notna().sum(axis=1) / members


def benchmark_prices():
    if not os.path.exists(BENCHMARKS):
        px = yf.download(["SPY", "QQQ", "RSP"], start="2019-01-01", end="2025-10-01", auto_adjust=True, progress=False)["Close"]
        px.to_csv(BENCHMARKS)
    return pd.read_csv(BENCHMARKS, index_col=0, parse_dates=True)


def load_inputs():
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily.csv"),
                        usecols=["tic", "datadate", "open_adj", "close_adj"], parse_dates=["datadate"])
    daily = daily.rename(columns={"datadate": "date"}).sort_values(["tic", "date"])
    daily["intraday"] = daily.close_adj / daily.open_adj - 1
    close = daily.pivot(index="date", columns="tic", values="close_adj").sort_index()
    db_tickers = set(close.columns)
    if os.path.exists(EXTRA_PRICES):
        extra = pd.read_csv(EXTRA_PRICES, index_col=0, parse_dates=True)
        close = close.join(extra[[c for c in extra.columns if c not in db_tickers]], how="left")
    c2c = close.pct_change(fill_method=None)
    for t, d in FAILURES.items():
        if t not in c2c.columns:
            c2c[t] = np.nan
        c2c.loc[pd.Timestamp(d), t] = -1.0

    pred, sector_of = load_predictions()

    _WARMUP["coverage"] = coverage_of(pred)
    _WARMUP["spy"] = benchmark_prices()["SPY"].pct_change()
    return daily, close, c2c, pred, sector_of, db_tickers


def sector_zscore(pred, sector_of):
    z = pd.DataFrame(np.nan, index=pred.index, columns=pred.columns)
    for sec in sector_of.unique():
        cols = [c for c in sector_of.index[sector_of == sec] if c in pred.columns]
        p = pred[cols]
        z[cols] = p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1), axis=0)
    return z


def avg_sector_weight(target, c2c, sector_of, sector=45, start="2020-08-01"):
    days = target.index.intersection(c2c.index)
    days = days[(days >= start) & (days <= BACKTEST_END)]
    p = days.to_period("M")
    reb = days[np.r_[True, p[1:] != p[:-1]]]
    W = target.loc[reb]
    return W[[c for c in W.columns if sector_of.get(c) == sector]].sum(axis=1).mean()


def main():
    daily, close, c2c, pred, sector_of, db_tickers = load_inputs()
    survivors = [c for c in pred.columns if c in db_tickers]
    print(f"Predictions: {pred.shape[0]} dates x {pred.shape[1]} tickers, "
          f"{pred.index.min().date()} -> {pred.index.max().date()}")

    w_base = top_quantile_equal(pred, 0.75)
    lam, grid, w_mom = choose_momentum_weight(pred, close, c2c)
    print(f"\nMomentum weight grid (Sharpe; chosen on validation {VALIDATION[0]}..{VALIDATION[1]} only):")
    print(grid.round(2).to_string())
    print(f"-> chosen lam = {lam}" + ("  (= the main strategy, momentum adds nothing on validation)" if lam == 0 else ""))
    strategies_extra = {} if lam == 0 else {f"Momentum overlay (lam={lam}, chosen on 2020-22)": w_mom}
    grid.to_csv(os.path.join(OUT_DIR, "momentum_grid.csv"))
    w_main = top_quantile_rank_weighted(pred, 0.75)
    targets = { 
        "Baseline selection, close-to-close, daily rebal": (w_base, "D", True),
        "Top 25% EW, monthly rebal": (w_base, "M", True),
        "Top 10% EW, monthly rebal": (top_quantile_equal(pred, 0.90), "M", True),
        "Top 25% rank-weighted, monthly": (w_main, "M", True),
        "Sector-neutral top 25%, monthly": (sector_neutral_top(pred, sector_of, 0.75), "M", True),
        "Top 25% rank-weighted, sector-neutral scoring": (top_quantile_rank_weighted(sector_zscore(pred, sector_of), 0.75), "M", True),
        "[control] All live stocks EW, monthly (no model)": (top_quantile_equal(pred, 0.0), "M", True),
        "[biased] Top 25% EW monthly, survivors only": (top_quantile_equal(pred[survivors], 0.75), "M", True),
        "[no warm-up rule] Top 25% rank-weighted, monthly": (w_main, "M", False),
    }
    targets.update({k: (w, "M", True) for k, w in strategies_extra.items()})
    pred_course, _ = load_predictions(models=("best",), leak_free=False)
    w_base_course = top_quantile_equal(pred_course, 0.75)
    w_course = top_quantile_rank_weighted(pred_course, 0.75)
    strategies = {"Baseline (notebook replica)": notebook_baseline(w_base_course, daily)}
    cov_leakfree = _WARMUP["coverage"]
    _WARMUP["coverage"] = coverage_of(pred_course)
    strategies["[look-ahead] Course signal, top 25% rank-weighted, monthly"] = drift_backtest(w_course, c2c, "M")
    _WARMUP["coverage"] = cov_leakfree
    tech = {"Baseline (notebook replica)": avg_sector_weight(w_base_course, c2c, sector_of),
            "[look-ahead] Course signal, top 25% rank-weighted, monthly": avg_sector_weight(w_course, c2c, sector_of)}
    for k, (w, freq, wu) in targets.items():
        strategies[k] = drift_backtest(w, c2c, freq, warmup=wu)
        tech[k] = avg_sector_weight(w, c2c, sector_of)

    start = min(s.index.min() for s in strategies.values())
    end = min(max(s.index.max() for s in strategies.values()), pd.Timestamp(BACKTEST_END))
    bench = {}
    bpx = benchmark_prices()
    for t in ["SPY", "QQQ"]:
        bench[t] = bpx[t].loc[start:end].pct_change().fillna(0.0)

    rows, navs = {}, {}
    for name, res in strategies.items():
        res = res.loc[:end]
        rows[name] = metrics(res.net_ret, res.turnover, res.cost)
        navs[name] = (1 + res.net_ret).cumprod()
        res.assign(nav=navs[name]).to_csv(os.path.join(OUT_DIR, f"nav_{''.join(ch if ch.isalnum() else '_' for ch in name.split(' (')[0]).strip('_')}.csv"))
    for t, r in bench.items():
        r = r.loc[start:end]
        rows[t] = metrics(r)
        navs[t] = (1 + r).cumprod()

    table = pd.DataFrame(rows).T
    table["Avg tech weight"] = pd.Series(tech)
    pd.set_option("display.width", 200)
    fmt = table.copy()
    for c in fmt.columns:
        fmt[c] = table[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}" if c in ("Final NAV", "Sharpe (rf=0)") else
                              f"{x:.2%}" if c != "Avg daily turnover" else f"{x:.4f}")
    print(f"\nBacktest {start.date()} -> {end.date()}, fee {FEE_RATE:.1%} per unit turnover\n")
    print(fmt.to_string())
    table.to_csv(os.path.join(OUT_DIR, "performance_table.csv"))

    print(f"\nSaved {OUT_DIR}/performance_table.csv and one NAV series per strategy")


if __name__ == "__main__":
    main()
