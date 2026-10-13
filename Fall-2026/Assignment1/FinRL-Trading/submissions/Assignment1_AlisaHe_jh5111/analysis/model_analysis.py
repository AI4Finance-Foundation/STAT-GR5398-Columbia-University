import glob, os
import numpy as np, pandas as pd
from xgboost import XGBRegressor

wins = []
for f in sorted(glob.glob("results/sector*/df_best_model_name.csv")):
    s = pd.read_csv(f, index_col=0)["model_name"]
    wins.append(s.value_counts().rename(os.path.basename(os.path.dirname(f))))
wins = pd.DataFrame(wins).fillna(0).astype(int)
wins.loc["TOTAL"] = wins.sum()
print("Best-model wins per rolling window:\n", wins.to_string(), "\n")

fr = pd.read_csv("prepared/final_ratios.csv", parse_dates=["date", "reportdate"])
def oos_ic(models):
    parts = []
    for sec_dir in sorted(glob.glob("results/sector*")):
        frames = [pd.read_csv(f"{sec_dir}/df_predict_{m}.csv", index_col=0, parse_dates=True) for m in models]
        p = (sum(frames) / len(frames)).stack().rename("pred").reset_index()
        p.columns = ["reportdate", "tic", "pred"]
        parts.append(p)
    p = pd.concat(parts).merge(fr[["date", "reportdate", "tic", "y_return"]], on=["reportdate", "tic"]).dropna()
    ic = p.groupby("date").apply(lambda g: g.pred.corr(g.y_return, method="spearman"), include_groups=False)
    top = p.groupby("date").apply(lambda g: g[g.pred >= g.pred.quantile(.75)].y_return.mean() - g.y_return.mean(),
                                  include_groups=False)
    return p, ic, top


for name, models in [("course model (selected per window)", ["best"]), ("signal used: mean of RF / XGB / LightGBM", ["rf", "xgb", "gbm"])]:
    p, ic, top = oos_ic(models)
    print(f"{name}: matched {len(p)} predictions | quarterly rank IC mean {ic.mean():.3f}, std {ic.std():.3f}, "
          f"t-stat {ic.mean()/ic.std()*np.sqrt(len(ic)):.2f}, positive in {(ic>0).mean():.0%} of {len(ic)} quarters | "
          f"top quartile minus universe {top.mean():.2%}/quarter, positive in {(top>0).mean():.0%}")
    if models == ["rf", "xgb", "gbm"]:
        pd.DataFrame({"rank_ic": ic, "top_q_excess": top}).to_csv("analysis/ic_by_quarter.csv")

feat = [c for c in fr.columns if c not in ["tic", "date", "gsector", "gsector_name", "reportdate", "y_return", "adj_close_q"]]
train = fr[fr.date < "2020-06-30"].dropna(subset=["y_return"])
m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42).fit(train[feat], train.y_return)
imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(feat).fillna(0)
imp = (imp / imp.sum()).sort_values(ascending=False)
print("\nTop-10 features by XGBoost gain (pooled, trained on quarters before 2020-06-30):")
print(imp.head(10).map("{:.1%}".format).to_string())
imp.to_csv("analysis/feature_importance.csv", header=["gain_share"])
