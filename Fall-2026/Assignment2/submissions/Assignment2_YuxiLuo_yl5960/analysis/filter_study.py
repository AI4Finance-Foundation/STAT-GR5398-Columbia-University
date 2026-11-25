"""Which filter rule, and which names did the filter swap out? Dev and val only.

Run from source_code/ with PYTHONHASHSEED=0 (never --period test):
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/filter_study.py --period dev --variant committed

Runs portfolio B (selection + valuation filter, timing none, no stops; the same call
as tune_filter.py) with the module's overvalued() rule replaced by one variant:

  committed   the module as committed: >= 2 methods and price above every one
  min1        >= 1 method and price above every available method
  dcf, ev_ebitda, peer_multiple
              that method alone: it values the name and price is above it
  dcf+ev_ebitda, dcf+peer_multiple, ev_ebitda+peer_multiple
              both methods value the name and price is above both

The committed module file is not edited; the rule is swapped on the imported module.

At every weekly rebalance it also records the swap the filter caused: the
backtester's target_weights() is called a second time on the unfiltered candidates,
with the same tradeable set and B's actual positions. Names in the unfiltered
targets but not in B's targets are "removed holdings"; names in B's targets but not
in the unfiltered ones are their "replacements". Forward returns are open-to-open from
the rebalance session (the backtester fills at the open), over 5, 21, 63 and 126
sessions, read from the same adjusted price file. Returns that would end after the
period's last session are left empty, so nothing past the period end is read.

Writes output/tuning/<period>_B_study_<variant>/ (equity, trades, stats), a swaps CSV
and one JSON line to analysis/study/portfolio.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

import portfolio.backtester as bt
import portfolio.hooks as hooks
import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

OUT = Path(__file__).resolve().parent / "study"
HORIZONS = (5, 21, 63, 126)
COMMITTED = sv.overvalued


def nw_t(x, lags):
    """t-stat of the mean with Newey-West (Bartlett) errors: weekly windows of h sessions overlap."""
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return round(float(x.mean() / np.sqrt(var / n)), 2) if var > 0 else None


def rule(variant):
    if variant == "committed":
        return COMMITTED
    if variant == "min1":
        def f(detail):
            sv.FILTER_MIN_METHODS = 1
            try:
                return COMMITTED(detail)
            finally:
                sv.FILTER_MIN_METHODS = 2
        return f
    methods = variant.split("+")
    assert all(m in sv.METHODS for m in methods), variant

    def f(detail):
        if detail.empty:
            return pd.Series(dtype=bool)
        fairs = detail[[f"fair_{m}" for m in methods]]
        return fairs.notna().all(axis=1) & (detail["price"] > (1 + sv.FILTER_PREMIUM) * fairs.max(axis=1))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--version", default="v1", help="v1 = committed module; v2_0/v2a/v2b/v2c = valuation_v2.py")
    a = ap.parse_args()
    sv.overvalued = rule(a.variant)
    suffix = "" if a.version == "v1" else f"_{a.version}"
    if a.version != "v1":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import valuation_v2
        valuation_v2.install(a.version)

    start, end = resolve(a.period, None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    end = min(end, prices.sessions[-1].date())
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}

    # Record each week's unfiltered candidates, then the targets with and without the filter.
    state, swaps = {}, []
    base_get = hooks.FilteredPredictions.get

    def get(self, day):
        state["day"], state["raw"] = day, tuple(self._base.get(day))
        kept = base_get(self, day)
        state["removed"] = {r.symbol for r in state["raw"]} - {r.symbol for r in kept}
        return kept

    base_tw = bt.target_weights

    def target_weights(preds, params_, eligible, positions, **kw):
        w = base_tw(preds, params_, eligible, positions, **kw)
        if "raw" in state:
            raw = base_tw(state.pop("raw"), params_, eligible, dict(positions), **kw)
            out, rep = sorted(set(raw) - set(w)), sorted(set(w) - set(raw))
            swaps.append({"date": state["day"], "n_removed": len(state["removed"]),
                          "out": " ".join(out), "in": " ".join(rep),
                          "held_out": " ".join(s for s in out if positions.get(s, 0) > 0)})
        return w

    hooks.FilteredPredictions.get = get
    bt.target_weights = target_weights

    t0 = time.time()
    name = f"tuning/{a.period}_B_study_{a.variant}{suffix}"
    stats = rp.run_one(name, start, end, valuation=True, timing="none", stops=False, ctx=ctx)

    # Forward returns of the removed holdings and their replacements.
    sessions = pd.DatetimeIndex([d for d in prices.sessions if start <= d.date() <= end])
    px = prices.open.reindex(sessions).fillna(prices.close.ffill().reindex(sessions))
    rows = []
    for s in swaps:
        i = sessions.searchsorted(pd.Timestamp(s["date"]))
        rec = {k: s[k] for k in ("date", "n_removed", "out", "in", "held_out")}
        for h in HORIZONS:
            for side in ("out", "in"):
                syms = s[side].split()
                if not syms or i + h >= len(sessions):
                    rec[f"{side}_{h}"] = np.nan
                    continue
                r = px.iloc[i + h][syms].to_numpy() / px.iloc[i][syms].to_numpy() - 1
                rec[f"{side}_{h}"] = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
        rows.append(rec)
    sw = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    sw.to_csv(OUT / f"swaps_{a.period}_{a.variant}{suffix}.csv", index=False)

    active = sw[sw["out"] != ""]
    swap_stats = {"weeks": len(sw), "weeks_with_swap": len(active),
                  "names_swapped_per_week": round(active["out"].str.split().str.len().sum() / len(sw), 2)}
    for h in HORIZONS:
        d = (active[f"in_{h}"] - active[f"out_{h}"]).dropna()
        swap_stats[f"h{h}"] = {
            "n": int(len(d)),
            "out_mean": round(float(active[f"out_{h}"].mean()), 4),
            "in_mean": round(float(active[f"in_{h}"].mean()), 4),
            "in_minus_out": round(float(d.mean()), 4),
            "t_nw": nw_t(d.to_numpy(), lags=int(np.ceil(h / 5))),
            "in_beats_out": round(float((d > 0).mean()), 3),
        }
    stats.update({"variant": a.variant, "version": a.version, "period": a.period, "swaps": swap_stats,
                  "seconds": round(time.time() - t0)})
    with (OUT / "portfolio.jsonl").open("a") as fh:
        fh.write(json.dumps(stats, default=str) + "\n")
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
