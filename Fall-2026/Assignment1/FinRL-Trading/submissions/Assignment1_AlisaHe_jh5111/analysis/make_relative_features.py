import glob, os, shutil
import pandas as pd

ID_COLS = ["tic", "date", "gsector", "gsector_name", "reportdate", "y_return"]
os.makedirs("model_v2/prepared", exist_ok=True)
shutil.copy("prepared/final_ratios.csv", "model_v2/prepared/final_ratios.csv")
for f in sorted(glob.glob("prepared/sector*.xlsx")):
    d = pd.read_excel(f)
    feats = [c for c in d.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(d[c])]
    d[feats] = d.groupby("date")[feats].rank(pct=True).fillna(0.5)
    d["adj_close_q"] = 0.5
    d["y_return"] = d.y_return - d.groupby("date").y_return.transform("mean")
    d.to_excel(os.path.join("model_v2/prepared", os.path.basename(f)), index=False)
    print(os.path.basename(f), d.shape)
