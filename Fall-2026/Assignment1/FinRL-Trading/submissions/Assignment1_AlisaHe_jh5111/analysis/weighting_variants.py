import os, sys
import numpy as np, pandas as pd, yfinance as yf

sys.path.insert(0, os.getcwd())
import strategy_backtest as sb

VOL_FILE = "analysis/yf_missing_volume.csv"
RENAMES = {"BRK.B": "BRK-B", "ABC": "COR", "ANTM": "ELV", "FLT": "CPAY", "PKI": "RVTY", "RE": "EG",
           "WLTW": "WTW", "FBHS": "FBIN", "CDAY": "DAY", "NLOK": "GEN", "GPS": "GAP"}


def dollar_volume(close):
    d = pd.read_csv(os.path.join(sb.DATA_DIR, "daily.csv"), usecols=["tic", "datadate", "prccd", "volume"],
                    parse_dates=["datadate"])
    dv = (d.assign(dv=d.prccd * d.volume).pivot(index="datadate", columns="tic", values="dv"))
    extra = [c for c in close.columns if c not in dv.columns]
    if not os.path.exists(VOL_FILE):
        got = {}
        for _ in range(3): 
            todo = [t for t in extra if t not in got]
            if not todo:
                break
            raw = yf.download([RENAMES.get(t, t) for t in todo], start="2019-01-01", end="2025-10-01",
                              auto_adjust=True, progress=False, group_by="ticker")
            for t in todo:
                y = RENAMES.get(t, t)
                if y in raw.columns.get_level_values(0) and raw[y]["Volume"].notna().any():
                    got[t] = raw[y]["Volume"]
        pd.DataFrame(got).to_csv(VOL_FILE)
    vol = pd.read_csv(VOL_FILE, index_col=0, parse_dates=True)
    vol = vol[[c for c in vol.columns if c in close.columns]]
    dv = dv.join(close[vol.columns] * vol.reindex(close.index), how="outer")
    return dv.reindex(close.index).rolling(63, min_periods=20).mean()


def weights_from_scores(pred, extra_score, q=0.75):
    keep = pred.ge(pred.quantile(q, axis=1), axis=0) & pred.notna()
    s = extra_score.where(keep)
    s = s.where(s.notna() | ~keep, s.median(axis=1), axis=0)
    s = s.where(keep)
    return s.div(s.sum(axis=1), axis=0).fillna(0.0)


def main():
    daily, close, c2c, pred, sector_of, _ = sb.load_inputs()
    dv = dollar_volume(close).reindex(pred.index, method="ffill").reindex(columns=pred.columns)
    vol63 = c2c.rolling(63, min_periods=20).std().reindex(pred.index, method="ffill").reindex(columns=pred.columns)
    rank = pred.where(pred.ge(pred.quantile(0.75, axis=1), axis=0)).rank(axis=1)

    cands = {
        "Rank-weighted (main)": sb.top_quantile_rank_weighted(pred, 0.75),
        "Size-weighted (dollar volume)": weights_from_scores(pred, dv),
        "Rank x sqrt(size)": weights_from_scores(pred, rank * np.sqrt(dv)),
        "Rank / volatility": weights_from_scores(pred, rank / vol63),
    }
    rows = {}
    for k, w in cands.items():
        res = sb.drift_backtest(w, c2c, "M")
        r = res.net_ret
        v, t, f = (r.loc[a:b] for a, b in [sb.VALIDATION, sb.TEST, ("2020-05-01", sb.BACKTEST_END)])
        nav = (1 + f).cumprod()
        rows[k] = {"val_CAGR": sb.cagr(v), "val_Sharpe": sb.sharpe(v), "test_CAGR": sb.cagr(t), "test_Sharpe": sb.sharpe(t),
                   "full_NAV": nav.iloc[-1], "full_Sharpe": sb.sharpe(f), "full_MaxDD": (nav / nav.cummax() - 1).min(),
                   "avg_daily_turnover": res.turnover.mean()}
    g = pd.DataFrame(rows).T
    g.to_csv("analysis/weighting_variants.csv")
    pd.set_option("display.width", 200)
    print(g.round(3).to_string())
    print(f"\n-> chosen on validation Sharpe: {g.val_Sharpe.idxmax()}")


if __name__ == "__main__":
    main()
