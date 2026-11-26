"""How much of B's edge over A is noise, and how much is selection? Uses saved equity curves only (no new runs).

1. Circular block bootstrap (21-session blocks, 5,000 draws) of the daily return difference B - A:
   95% interval for the annualized return difference and the Sharpe difference, and P(B beats A).
2. Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of the chosen B, and of its edge over A (daily B - A),
   given every configuration whose
   result we saw in that period (output/<run>/equity.csv; for val also the stand-in round kept in
   Assignment 2/test/source_code/output/exp_val_*).
    .venv/bin/python experiments/bootstrap_dsr.py
Writes output/experiments/bootstrap_dsr.json.
"""
from __future__ import annotations

import json
from math import e, sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

OUT = Path("output")
STANDIN = Path(__file__).resolve().parents[5] / "test" / "source_code" / "output"
BLOCK, DRAWS, SEED = 21, 5000, 0
N01 = NormalDist()
EULER = 0.5772156649


def returns(run_dir: Path) -> pd.Series:
    eq = pd.read_csv(run_dir / "equity.csv", index_col=0, parse_dates=True)["equity"]
    return eq.pct_change().dropna()


def sharpe(r) -> float:
    r = np.asarray(r)
    return float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0


def bootstrap(ra: pd.Series, rb: pd.Series) -> dict:
    df = pd.concat([ra, rb], axis=1, keys=["a", "b"]).dropna()
    a, b = df["a"].to_numpy(), df["b"].to_numpy()
    n = len(a)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(DRAWS, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(DRAWS, -1)[:, :n] % n
    da = (b[idx] - a[idx]).mean(axis=1) * 252
    sa = np.array([sharpe(a[i]) for i in idx]) * sqrt(252)
    sb = np.array([sharpe(b[i]) for i in idx]) * sqrt(252)
    ds = sb - sa
    pct = lambda x: [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]
    return {"sessions": n, "ann_return_diff": round(float((b - a).mean() * 252), 4), "ann_return_diff_95ci": pct(da),
            "sharpe_diff": round((sharpe(b) - sharpe(a)) * sqrt(252), 4), "sharpe_diff_95ci": pct(ds),
            "p_b_beats_a_sharpe": round(float((ds > 0).mean()), 3)}


def deflated_sharpe(r_chosen: pd.Series, trial_srs: list[float]) -> dict:
    """Per-session Sharpe; SR0 = expected max of N trial Sharpes under no skill."""
    sr = sharpe(r_chosen)
    t = len(r_chosen)
    skew = float(pd.Series(r_chosen).skew())
    kurt = float(pd.Series(r_chosen).kurt()) + 3.0
    n = len(trial_srs)
    v = float(np.var(trial_srs, ddof=1))
    sr0 = sqrt(v) * ((1 - EULER) * N01.inv_cdf(1 - 1 / n) + EULER * N01.inv_cdf(1 - 1 / (n * e)))
    denom = sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    psr0 = N01.cdf(sr * sqrt(t - 1) / denom)
    dsr = N01.cdf((sr - sr0) * sqrt(t - 1) / denom)
    return {"trials": n, "chosen_sharpe_ann": round(sr * sqrt(252), 3),
            "trial_sharpes_ann": [round(s * sqrt(252), 3) for s in trial_srs],
            "sr0_ann": round(sr0 * sqrt(252), 3), "psr_vs_zero": round(psr0, 3), "deflated_sharpe": round(dsr, 3)}


def trials(period: str) -> dict[str, Path]:
    found = {}
    if period == "val":
        for p in sorted(STANDIN.glob("exp_val_*")):
            if (p / "equity.csv").exists():
                found[f"standin:{p.name}"] = p
        for name in ("rules_val_none", "rules_val_abs", "rules_val_agree", "rules_val_worst_k", "rules_val_worst_k_rates",
                     "variant_val_F3", "valchoice_cheap_k"):
            found[name] = OUT / name
    else:
        for p in sorted(OUT.glob("rules_dev_*")) + sorted(OUT.glob("variant_dev_*")) + sorted(OUT.glob("sens_dev_*")):
            found[p.name] = p
        # 11/24-25 alternatives (method mix, constants, reversed filter); placebo and random controls are not candidates
        for pat in ("diag_dev_without_*", "diag_dev_only_*", "diag_dev_ERP*", "diag_dev_TERMINAL*", "diag_dev_GROWTH*",
                    "diag_dev_MAX_PEER*", "diag_dev_flip*"):
            for p in sorted(OUT.glob(pat)):
                found[p.name] = p
    return {k: v for k, v in found.items() if (v / "equity.csv").exists()}


def main():
    res = {}
    for period in ("dev", "val"):
        ra, rb = returns(OUT / f"{period}_A_selection"), returns(OUT / f"{period}_B_valuation")
        tr = trials(period)
        rets = {k: returns(p) for k, p in tr.items()}
        srs = {k: sharpe(r) for k, r in rets.items()}
        # identical configurations (same Sharpe to 6 decimals) count once
        keep = list({round(v, 6): k for k, v in srs.items()}.values())
        edge = lambda r: pd.concat([r, ra], axis=1).dropna().pipe(lambda d: d.iloc[:, 0] - d.iloc[:, 1])
        edge_srs = [sharpe(edge(rets[k])) for k in keep if np.std(edge(rets[k])) > 0]  # "none" has no edge
        res[period] = {"bootstrap_B_minus_A": bootstrap(ra, rb),
                       "deflated_sharpe_B": deflated_sharpe(rb, [srs[k] for k in keep]),
                       "deflated_sharpe_edge_B_minus_A": deflated_sharpe(edge(rb), edge_srs),
                       "trial_runs": sorted(tr)}
        print(period, json.dumps({k: v for k, v in res[period].items() if k != "trial_runs"}, indent=1))
    (OUT / "experiments" / "bootstrap_dsr.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
