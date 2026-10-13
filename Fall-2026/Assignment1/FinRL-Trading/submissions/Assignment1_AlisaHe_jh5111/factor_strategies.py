import os
import numpy as np
import pandas as pd

import strategy_backtest as sb

OUT = "outputs"
ACCOUNTING = ["sue", "fscore", "gpa", "accruals", "value"]
PRICE = ["mom", "lowvol"]
COMPOSITES = {
    "Quality (F + GP/A + accruals)": ["fscore", "gpa", "accruals"],
    "QVM: all 7 factors": ACCOUNTING + PRICE,
}

def accounting_factors():
    fr = pd.read_csv(os.path.join(sb.DATA_DIR, "final_ratios.csv"), parse_dates=["date", "reportdate"])
    fr = fr.sort_values(["tic", "date"]).drop_duplicates(["tic", "date"], keep="last")
    prev = fr[["tic", "date", "EPS", "net_income_ratio", "asset_turnover", "debt_to_assets", "cur_ratio",
               "gross_margin"]].copy()
    prev["date"] = prev["date"] + pd.offsets.QuarterEnd(4)
    fr = fr.merge(prev, on=["tic", "date"], how="left", suffixes=("", "_ly"))

    roa = fr.net_income_ratio * fr.asset_turnover
    roa_ly = fr.net_income_ratio_ly * fr.asset_turnover_ly
    cfo_ta = fr.ocf_to_sales * fr.asset_turnover

    fr["eps_yoy"] = fr.EPS - fr.EPS_ly
    sd = fr.groupby("tic").eps_yoy.transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    fr["sue"] = fr.eps_yoy / sd.where(sd > 0)

    sig = pd.DataFrame({
        "roa_pos": roa > 0,
        "cfo_pos": cfo_ta > 0,
        "droa_pos": roa > roa_ly,
        "accrual": cfo_ta > roa,
        "dlever": fr.debt_to_assets < fr.debt_to_assets_ly,
        "dliquid": fr.cur_ratio > fr.cur_ratio_ly,
        "dmargin": fr.gross_margin > fr.gross_margin_ly,
        "dturn": fr.asset_turnover > fr.asset_turnover_ly,
    })
    fr["fscore"] = sig.sum(axis=1).where(fr.EPS_ly.notna())

    fr["gpa"] = fr.gross_margin * fr.asset_turnover
    fr["accruals"] = -(fr.net_income_ratio - fr.ocf_to_sales) * fr.asset_turnover
    ep = 1 / fr.pe.replace(0, np.nan)
    bp = 1 / fr.pb.replace(0, np.nan)
    fr["value"] = (ep.groupby([fr.date, fr.gsector]).rank(pct=True) +
                   bp.groupby([fr.date, fr.gsector]).rank(pct=True)) / 2
    return fr


def to_daily(fr, col, index):
    d = fr.dropna(subset=[col, "reportdate"]).copy()
    d["live"] = sb.usable_date(d.reportdate, d.date)
    p = d.pivot_table(index="live", columns="tic", values=col, aggfunc="last")
    return sb.quarter_reportday_ffill(p).reindex(index)


def sector_rank(panel, sector_of):
    out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for sec in sector_of.unique():
        cols = [c for c in sector_of.index[sector_of == sec] if c in panel.columns]
        out[cols] = panel[cols].rank(axis=1, pct=True)
    return out


def build_signals(close, c2c, pred, sector_of):
    live = pred.notna()
    fr = accounting_factors()
    ranks = {}
    for f in ACCOUNTING:
        raw = to_daily(fr, f, pred.index).reindex(columns=pred.columns).where(live)
        ranks[f] = sector_rank(raw, sector_of)
    mom = sb.momentum_12_1(close)
    vol = c2c.rolling(252, min_periods=126).std()
    for f, raw in [("mom", mom), ("lowvol", -vol)]:
        raw = raw.reindex(pred.index, method="ffill").reindex(columns=pred.columns).where(live)
        ranks[f] = raw.rank(axis=1, pct=True)
    coverage = {f: (r.notna().sum(axis=1) / live.sum(axis=1)).loc["2020-05-01":sb.BACKTEST_END].mean()
                for f, r in ranks.items()}
    return ranks, coverage


def composite(ranks, names, live):
    stack = np.stack([ranks[n].values for n in names])
    n_ok = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(stack, axis=0)
    m[n_ok < max(1, len(names) / 2)] = np.nan
    return pd.DataFrame(m, index=live.index, columns=live.columns).where(live)


def pre_backtest_ic(fr, end="2019-12-31"):
    out = {}
    for f in ACCOUNTING:
        d = fr[fr.date <= end].dropna(subset=[f, "y_return"])
        ic = d.groupby(["date", "gsector"]).apply(lambda g: g[f].corr(g.y_return, method="spearman"),
                                                  include_groups=False).groupby("date").mean()
        out[f] = {"IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)), "quarters": len(ic)}
    return pd.DataFrame(out).T


def evaluate(name, score, c2c, rows):
    res = sb.drift_backtest(sb.top_quantile_rank_weighted(score, 0.75), c2c, "M")
    top = sb.drift_backtest(sb.top_quantile_equal(score, 0.80), c2c, "M").net_ret
    bot = sb.drift_backtest(sb.top_quantile_equal(-score, 0.80), c2c, "M").net_ret
    ls = (top - bot).loc["2020-05-01":sb.BACKTEST_END]
    r = res.net_ret.loc["2020-05-01":sb.BACKTEST_END]
    row = {}
    for tag, (a, b) in [("val", sb.VALIDATION), ("test", sb.TEST)]:
        rr, l = r.loc[a:b], ls.loc[a:b]
        row[f"{tag}_CAGR"] = sb.cagr(rr)
        row[f"{tag}_Sharpe"] = sb.sharpe(rr)
        row[f"{tag}_LS_ann"] = l.mean() * 252
        row[f"{tag}_LS_t"] = l.mean() / l.std() * np.sqrt(len(l))
    nav = (1 + r).cumprod()
    row.update({"full_NAV": nav.iloc[-1], "full_Sharpe": sb.sharpe(r), "full_MaxDD": (nav / nav.cummax() - 1).min(),
                "avg_daily_turnover": res.turnover.mean()})
    rows[name] = row
    return row


def main():
    pd.set_option("display.width", 220)
    daily, close, c2c, pred, sector_of, _ = sb.load_inputs()
    live = pred.notna()
    ranks, coverage = build_signals(close, c2c, pred, sector_of)
    prior = pre_backtest_ic(accounting_factors())
    prior.to_csv(os.path.join(OUT, "factor_pre2020_ic.csv"))
    print("Pre-backtest (2016-2019) within-sector IC of accounting factors:\n", prior.round(3).to_string(), "\n")
    pre_chosen = list(prior.index[prior.IC > 0])
    print("Average coverage of the live universe per factor:",
          {k: f"{v:.0%}" for k, v in coverage.items()}, "\n")

    rows = {}
    evaluate("ML model (main, reference)", pred, c2c, rows)
    for f in ACCOUNTING + PRICE:
        evaluate(f"factor: {f}", ranks[f], c2c, rows)
    single = pd.DataFrame(rows).T

    chosen = [f for f in ACCOUNTING + PRICE if single.loc[f"factor: {f}", "val_LS_ann"] > 0]
    comps = dict(COMPOSITES)
    comps[f"Val-selected ({' + '.join(chosen)})"] = chosen
    comps[f"Pre-2020-selected ({' + '.join(pre_chosen)})"] = pre_chosen
    for name, names in comps.items():
        evaluate(f"composite: {name}", composite(ranks, names, live), c2c, rows)

    table = pd.DataFrame(rows).T
    table.to_csv(os.path.join(OUT, "factor_strategies.csv"))
    fmt = table.copy()
    for c in fmt.columns:
        fmt[c] = table[c].map(lambda x: f"{x:.2f}" if ("Sharpe" in c or "_t" in c or "NAV" in c) else f"{x:.1%}"
                              if "turnover" not in c else f"{x:.3f}")
    print(fmt.to_string())
    cands = table[table.index.str.startswith(("composite", "ML"))]
    best = cands.val_Sharpe.idxmax()
    print(f"\n-> chosen on validation Sharpe among ML + pre-specified composites: {best}")
    print(f"   test: CAGR {table.loc[best, 'test_CAGR']:.1%}, Sharpe {table.loc[best, 'test_Sharpe']:.2f}"
          f"  | full NAV {table.loc[best, 'full_NAV']:.2f}")
    pre = [i for i in table.index if "Pre-2020" in i][0]
    print(f"-> fixed before the backtest from 2016-2019 data: {pre}")
    print(f"   2020-22: CAGR {table.loc[pre, 'val_CAGR']:.1%}, Sharpe {table.loc[pre, 'val_Sharpe']:.2f} | "
          f"2023-25: CAGR {table.loc[pre, 'test_CAGR']:.1%}, Sharpe {table.loc[pre, 'test_Sharpe']:.2f} | "
          f"full NAV {table.loc[pre, 'full_NAV']:.2f}, Sharpe {table.loc[pre, 'full_Sharpe']:.2f}, MaxDD {table.loc[pre, 'full_MaxDD']:.1%}")
    return table, best


if __name__ == "__main__":
    main()
