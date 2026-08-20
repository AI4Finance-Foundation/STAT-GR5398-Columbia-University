"""Plot NAV and drawdown comparisons for deterministic vs ML backtests."""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "outputs")


def load_nav(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df["nav"]


def drawdown(nav):
    peak = nav.cummax()
    return (nav - peak) / peak


def main():
    det_path = os.path.join(OUT, "portfolio_nav.csv")
    ml_path = os.path.join(OUT, "portfolio_nav_ml.csv")

    det = load_nav(det_path)
    ml = load_nav(ml_path)

    idx = det.index.union(ml.index).sort_values()
    det = det.reindex(idx).ffill()
    ml = ml.reindex(idx).ffill()

    plt.figure(figsize=(10, 6))
    plt.plot(det.index, det.values, label="Deterministic")
    plt.plot(ml.index, ml.values, label="ML")
    plt.title("NAV Comparison")
    plt.ylabel("NAV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nav_compare.png"))
    plt.close()

    dd_det = drawdown(det)
    dd_ml = drawdown(ml)
    plt.figure(figsize=(10, 6))
    plt.plot(dd_det.index, dd_det.values, label="Deterministic")
    plt.plot(dd_ml.index, dd_ml.values, label="ML")
    plt.title("Drawdown Comparison")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "drawdown_compare.png"))
    plt.close()

    print("Saved plots to outputs/nav_compare.png and outputs/drawdown_compare.png")


if __name__ == "__main__":
    main()
