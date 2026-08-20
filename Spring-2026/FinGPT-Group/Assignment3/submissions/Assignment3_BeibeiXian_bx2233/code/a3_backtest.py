import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Strategy weights
W_FINGPT  = 0.40   # β — FinGPT contrarian
W_TECH    = 0.35   # α — technical momentum
W_QUALITY = 0.25   # γ — quality filter (set to 0 if fundamental_scores not available)

TOP_N = 6          # long positions per week
BOT_N = 6          # short positions per week
TC    = 0.001      # transaction cost 0.10% one-way

def load_data():
    print("Loading data ...")

    # signals — aggregate daily rows to weekly mean contrarian_score
    fg = pd.read_csv(DATA_DIR / "fingpt_signals.csv")
    fg["week_start"] = pd.to_datetime(fg["week_start"])
    fg_weekly = (fg.groupby(["week_start", "ticker"])["contrarian_score"]
                   .mean().reset_index()
                   .rename(columns={"week_start": "date", "contrarian_score": "fg_score"}))

    # prices
    px = pd.read_csv(DATA_DIR / "prices_2026q1.csv", index_col=0, parse_dates=True)
    px.index = pd.to_datetime(px.index).tz_localize(None)

    # features — compute tech_score from raw columns if not pre-computed
    feat = pd.read_csv(DATA_DIR / "features_2026q1.csv", parse_dates=["date"])
    if "tech_score" not in feat.columns:
        def zscore(s):
            return (s - s.mean()) / (s.std() + 1e-9)
        for col in ["ret4w", "ret52w", "vol4w", "bb_pos"]:
            if col in feat.columns:
                feat[col + "_z"] = feat.groupby("date")[col].transform(zscore)
        feat["tech_score"] = 0.0
        for col, w in [("ret4w_z", 0.40), ("ret52w_z", 0.25),
                       ("vol4w_z", -0.20), ("bb_pos_z", -0.15)]:
            if col in feat.columns:
                feat["tech_score"] += w * feat[col]

    # macro
    macro = None
    macro_path = DATA_DIR / "macro_regime.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)

    # merge signals + features
    merged = pd.merge(
        fg_weekly,
        feat[["date", "ticker", "tech_score"]],
        on=["date", "ticker"], how="inner",
    ).sort_values(["date", "ticker"])

    print(f"  Signals : {len(fg_weekly)} weekly rows")
    print(f"  Prices  : {px.shape}")
    print(f"  Features: {len(feat)} rows")
    print(f"  Merged  : {len(merged)} rows")
    return merged, px, macro


def get_multipliers(macro, date):
    """Return (long_mult, short_mult) from macro regime."""
    if macro is None or len(macro) == 0:
        return 0.7, 0.7
    past = macro[macro.index <= pd.Timestamp(date)]
    if len(past) == 0:
        return 0.7, 0.7
    row = past.iloc[-1]
    return float(row.get("long_mult", 0.7)), float(row.get("short_mult", 0.7))


# ── Core backtest ─────────────────────────────────────────────────────────────
def run_backtest(merged, px, macro=None,
                 use_fg=True, use_rl=True,
                 beta=W_FINGPT, alpha=W_TECH):
    """
    Weekly long-short backtest.
    Returns (portfolio_returns, spy_returns, dates).
    """
    price_dates = sorted(px.index)
    portfolio_rets, spy_rets, dates = [], [], []
    prev_longs, prev_shorts = [], []

    for i in range(1, len(price_dates)):
        curr_d = price_dates[i]
        prev_d = price_dates[i - 1]

        week_data = merged[merged["date"] == prev_d].copy()
        if week_data.empty:
            continue

        # composite score
        week_data["score"] = 0.0
        if use_fg:  week_data["score"] += beta  * week_data["fg_score"]
        if use_rl:  week_data["score"] += alpha * week_data["tech_score"]

        week_data = week_data.dropna(subset=["score"])
        if len(week_data) < TOP_N + BOT_N:
            continue

        week_data = week_data.sort_values("score", ascending=False)
        longs  = week_data.head(TOP_N)["ticker"].tolist()
        shorts = week_data.tail(BOT_N)["ticker"].tolist()

        # stock returns
        rets = {}
        for tk in set(longs + shorts):
            if tk in px.columns and prev_d in px.index and curr_d in px.index:
                p0, p1 = px.loc[prev_d, tk], px.loc[curr_d, tk]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    rets[tk] = p1 / p0 - 1

        # macro regime multipliers
        lm, sm = get_multipliers(macro, prev_d)

        long_ret  = np.mean([rets[t] for t in longs  if t in rets]) if longs  else 0.0
        short_ret = np.mean([rets[t] for t in shorts if t in rets]) if shorts else 0.0

        # transaction cost on new positions
        turnover = len(set(longs) - set(prev_longs)) + len(set(shorts) - set(prev_shorts))
        tc_cost  = turnover * TC

        port_ret = (long_ret * lm - short_ret * sm) / 2 - tc_cost
        portfolio_rets.append(port_ret)
        dates.append(curr_d)
        prev_longs, prev_shorts = longs, shorts

        if "SPY" in px.columns and prev_d in px.index and curr_d in px.index:
            p0, p1 = px.loc[prev_d, "SPY"], px.loc[curr_d, "SPY"]
            spy_rets.append(p1 / p0 - 1 if pd.notna(p0) and p0 > 0 else 0.0)
        else:
            spy_rets.append(0.0)

    return np.array(portfolio_rets), np.array(spy_rets), dates


def compute_metrics(rets, spy_rets, label="Strategy"):
    if len(rets) == 0:
        return {}
    cum      = np.cumprod(1 + rets)
    total    = cum[-1] - 1
    ann      = (1 + total) ** (52 / len(rets)) - 1
    vol      = rets.std() * np.sqrt(52)
    sharpe   = ann / vol if vol > 0 else 0.0
    mdd      = float(np.min(cum - np.maximum.accumulate(cum)))
    spy_tot  = float(np.prod(1 + spy_rets) - 1) if len(spy_rets) > 0 else None
    return {
        "label":         label,
        "total_return":  total,
        "ann_return":    ann,
        "sharpe_ratio":  sharpe,
        "max_drawdown":  mdd,
        "n_weeks":       len(rets),
        "spy_return":    spy_tot,
        "alpha_vs_spy":  total - spy_tot if spy_tot is not None else None,
    }

def run_ablation(merged, px, macro):
    print("\n" + "=" * 62)
    print("ContrarianFusion — Ablation Study (Out-of-Sample 2026 Q1)")
    print("=" * 62)

    variants = [
        ("FinGPT Contrarian only",   True,  False),
        ("Technical Momentum only",  False, True),
        ("ContrarianFusion (All 3)", True,  True),
    ]
    all_rets = {}
    spy_rets_ref = None
    rows = []

    for label, fg, rl in variants:
        p_rets, s_rets, _ = run_backtest(merged, px, macro, use_fg=fg, use_rl=rl)
        m = compute_metrics(p_rets, s_rets, label)
        all_rets[label] = p_rets
        if spy_rets_ref is None: spy_rets_ref = s_rets
        rows.append(m)
        print(f"\n  {label}")
        print(f"    Total Return : {m['total_return']:+.2%}")
        print(f"    Sharpe Ratio : {m['sharpe_ratio']:.3f}")
        print(f"    Max Drawdown : {m['max_drawdown']:.2%}")

    if spy_rets_ref is not None:
        spy_tot = float(np.prod(1 + spy_rets_ref) - 1)
        print(f"\n  SPY Benchmark")
        print(f"    Total Return : {spy_tot:+.2%}")

    print("=" * 62)

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"\n✓ Saved → outputs/ablation_results.csv")

    return all_rets, spy_rets_ref

def plot_equity_curves(all_rets, spy_rets):
    colors = {
        "FinGPT Contrarian only":   "#C00000",
        "Technical Momentum only":  "#6BAED6",
        "ContrarianFusion (All 3)": "#08519C",
    }
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, rets in all_rets.items():
        if len(rets) > 0:
            ax.plot(np.cumprod(1 + rets), label=label,
                    color=colors.get(label, "gray"), lw=2.2)

    if len(spy_rets) > 0:
        ax.plot(np.cumprod(1 + spy_rets), label="SPY Benchmark",
                color="gray", lw=2, linestyle="--", alpha=0.8)

    ax.axhline(1, color="black", lw=0.6, linestyle=":")
    ax.set_title("ContrarianFusion vs SPY — 2026 Q1 Out-of-Sample Backtest",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Return (1 = start)", fontsize=11)
    ax.set_xlabel("Week", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    path = OUTPUT_DIR / "equity_curve_real.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved → outputs/equity_curve_real.png")

if __name__ == "__main__":
    merged, px, macro = load_data()
    all_rets, spy_rets = run_ablation(merged, px, macro)
    plot_equity_curves(all_rets, spy_rets)
    print("\n✓ Backtest complete.")
