import glob, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.getcwd())
import strategy_backtest as sb

fr = pd.read_csv("prepared/final_ratios.csv", parse_dates=["date", "reportdate"])
fr["y_rel"] = fr.y_return - fr.groupby(["gsector", "date"]).y_return.transform("mean")


def ic_table(results_dir):
    preds = []
    for f in glob.glob(f"{results_dir}/sector*/df_predict_best.csv"):
        p = pd.read_csv(f, index_col=0, parse_dates=True).stack().rename("pred").reset_index()
        p.columns = ["reportdate", "tic", "pred"]
        preds.append(p)
    p = pd.concat(preds).merge(fr[["date", "reportdate", "tic", "gsector", "y_return", "y_rel"]], on=["reportdate", "tic"])
    out = {}
    for lab in ["y_return", "y_rel"]:
        pooled = p.groupby("date").apply(lambda g: g.pred.corr(g[lab], method="spearman"), include_groups=False)
        within = p.groupby(["date", "gsector"]).apply(lambda g: g.pred.corr(g[lab], method="spearman"),
                                                      include_groups=False).groupby("date").mean()
        for name, s in [("pooled", pooled), ("within-sector", within)]:
            out[f"IC {name} vs {lab}"] = {"mean": s.mean(), "t": s.mean() / s.std() * np.sqrt(len(s)), "share>0": (s > 0).mean()}
    return pd.DataFrame(out).T


def main():
    pd.set_option("display.width", 200)
    t1, t2 = ic_table("results"), ic_table("model_v2/results")
    ic = pd.concat({"v1": t1, "v2": t2}, axis=1)
    print("Out-of-sample rank IC per quarter\n", ic.round(3).to_string(), "\n")
    ic.to_csv("analysis/model_v2_ic.csv")

    daily, close, c2c, pred1, sector_of, _ = sb.load_inputs()
    sb.RESULTS_DIR = "model_v2/results"
    pred2, _ = sb.load_predictions()
    rows = {}
    for tag, pred in [("v1", pred1), ("v2", pred2)]:
        for rule, w in [("top 25% rank-weighted", sb.top_quantile_rank_weighted(pred, 0.75)),
                        ("top 25% EW", sb.top_quantile_equal(pred, 0.75)),
                        ("sector-neutral top 25%", sb.sector_neutral_top(pred, sector_of, 0.75))]:
            res = sb.drift_backtest(w, c2c, "M")
            r = res.net_ret
            v, t, f = (r.loc[a:b] for a, b in [sb.VALIDATION, sb.TEST, ("2020-05-01", sb.BACKTEST_END)])
            nav = (1 + f).cumprod()
            rows[f"{tag}: {rule}"] = {"val_CAGR": sb.cagr(v), "val_Sharpe": sb.sharpe(v), "test_CAGR": sb.cagr(t),
                                      "test_Sharpe": sb.sharpe(t), "full_NAV": nav.iloc[-1], "full_Sharpe": sb.sharpe(f),
                                      "full_MaxDD": (nav / nav.cummax() - 1).min()}
    g = pd.DataFrame(rows).T
    g.to_csv("analysis/model_v2_backtest.csv")
    print(g.round(3).to_string())
    print(f"\n-> best on validation Sharpe: {g.val_Sharpe.idxmax()}")


if __name__ == "__main__":
    main()
