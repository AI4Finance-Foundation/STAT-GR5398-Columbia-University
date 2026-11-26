"""Dev-only diagnostics of the frozen configuration (11/24). None of them changes the module.

    PYTHONHASHSEED=0 .venv/bin/python experiments/diagnostics_dev.py placebo
    PYTHONHASHSEED=0 .venv/bin/python experiments/diagnostics_dev.py methods
    PYTHONHASHSEED=0 .venv/bin/python experiments/diagnostics_dev.py constants

placebo    the rates gate vs placebo gates: always on, inverted, and circular shifts of the real weekly
           gate series (same share of weeks on, different weeks). Fair values do not depend on the gate,
           so the value cache is shared across these runs.
methods    B with each method left out, and with each method alone (the filter rule unchanged).
constants  B with one valuation constant moved at a time (ERP, terminal growth, growth cap, peer P/E cap).
flip       registered 11/25 before running: drop the 2 most UNDERvalued instead (IC is negative), with the
           real gate (R1) and always on (R2). Diagnostic only; the submitted rule does not change.
flipcheck  R2 at k = 1 / 3, and 2 random drops every week (10 seeds), for comparison with R2.
controls   same gate and k, but the 2 names dropped are chosen without valuation: the 2 lowest-ranked of
           the top 10, or 2 at random (several seeds). Tests whether valuation picks better than chance.
Writes output/experiments/diag_<part>_dev.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import run_portfolio as rp
from selection import MonthlyScores
from selection.config import load
from shared.data_view import DataBundle
from shared.periods import resolve
from valuation import student_valuation as sv

KEEP = ["total_return_pct", "cagr_pct", "volatility_pct", "sharpe", "max_drawdown_pct", "calmar",
        "turnover_per_year", "avg_removed_per_week"]
N_SHIFTS = 12


def runner():
    start, end = resolve("dev", None, None)
    _, options, params, prices, _, universe = load(rp.CONFIG)
    ctx = {"options": options, "params": params, "prices": prices, "universe": universe,
           "scores": MonthlyScores(), "bundle": DataBundle.load(prices=prices)}

    def run(label, **info):
        print("running", label, flush=True)
        s = rp.run_one(f"diag_dev_{label}", start, end, valuation=True, timing="none", stops=False, ctx=ctx)
        return {"run": label, **info, **{c: s.get(c) for c in KEEP}}
    return run


def placebo(run):
    real, gate = sv.rates_rising, {}

    def recording(as_of, data):
        gate[pd.Timestamp(as_of)] = real(as_of, data)
        return gate[pd.Timestamp(as_of)]

    sv.rates_rising = recording
    rows = [run("real_gate")]
    weeks = sorted(gate)
    on = np.array([gate[w] for w in weeks])
    rows[0]["share_on"] = on.mean()

    def fixed(values, label):
        table = dict(zip(weeks, values))
        sv.rates_rising = lambda as_of, data: bool(table.get(pd.Timestamp(as_of), real(as_of, data)))
        rows.append(run(label, share_on=float(np.mean(values))))

    fixed(np.ones_like(on), "always_on")
    fixed(~on, "inverted")
    for s in np.linspace(len(on) // (N_SHIFTS + 1), len(on) - len(on) // (N_SHIFTS + 1), N_SHIFTS).astype(int):
        fixed(np.roll(on, s), f"shift_{s}w")
    sv.rates_rising = real
    return rows


def methods(run):
    full = sv.VALUE_METHODS
    rows = []
    for name in [m[0] for m in full]:
        for label, subset in ((f"without_{name}", [m for m in full if m[0] != name]),
                              (f"only_{name}", [m for m in full if m[0] == name])):
            sv.VALUE_METHODS = tuple(subset)
            sv._VALUE_CACHE.clear()
            rows.append(run(label))
    sv.VALUE_METHODS = full
    return rows


def constants(run):
    moves = [("ERP", 0.04), ("ERP", 0.06), ("TERMINAL_GROWTH", 0.02), ("TERMINAL_GROWTH", 0.03),
             ("GROWTH_CLIP", (-0.10, 0.30)), ("GROWTH_CLIP", (-0.10, 0.50)),
             ("MAX_PEER_MULTIPLE", 100.0), ("MAX_PEER_MULTIPLE", 400.0)]
    rows = []
    for name, value in moves:
        saved = getattr(sv, name)
        setattr(sv, name, value)
        sv._VALUE_CACHE.clear()
        rows.append(run(f"{name}={value}", constant=name, value=str(value), default=str(saved)))
        setattr(sv, name, saved)
    return rows


N_SEEDS = 10


def controls(run):
    real = sv._monthly_values

    def fake(score):
        def values(as_of, tickers, data):
            m = pd.DataFrame({"mispricing": score(as_of, list(tickers)), "method_spread": 1.0, "n_methods": 1.0},
                             index=pd.Index(list(tickers), name="ticker"))
            return m.astype(float)
        return values

    rows = []
    sv._monthly_values = fake(lambda as_of, t: -1.0 - np.arange(len(t)) / 100)  # lowest rank = most negative
    rows.append(run("lowest_rank"))
    for seed in range(N_SEEDS):
        sv._monthly_values = fake(lambda as_of, t, seed=seed: -np.random.default_rng(
            [seed, pd.Timestamp(as_of).toordinal()]).uniform(0.01, 1.0, len(t)))
        rows.append(run(f"random_seed{seed}", seed=seed))
    sv._monthly_values = real
    return rows


def flip(run):
    real_values, real_gate = sv._monthly_values, sv.rates_rising

    def flipped(as_of, tickers, data):
        m = real_values(as_of, tickers, data)
        return m.assign(mispricing=-m["mispricing"])  # most undervalued now sorts first

    sv._monthly_values = flipped
    rows = [run("flip_real_gate")]
    sv.rates_rising = lambda as_of, data: True
    rows.append(run("flip_always_on"))
    sv._monthly_values, sv.rates_rising = real_values, real_gate
    return rows


def flipcheck(run):
    """Checks of R2 (flip, always on), k fixed at 2 for the decision: k = 1 / 3, and 2 random drops every week."""
    real_values, real_gate, k0 = sv._monthly_values, sv.rates_rising, sv.DROP_WORST_K
    sv.rates_rising = lambda as_of, data: True
    rows = []
    sv._monthly_values = lambda as_of, t, data: real_values(as_of, t, data).pipe(lambda m: m.assign(mispricing=-m["mispricing"]))
    for k in (1, 3):
        sv.DROP_WORST_K = k
        rows.append(run(f"flip_always_on_k{k}", k=k))
    sv.DROP_WORST_K = k0
    for seed in range(N_SEEDS):
        def values(as_of, tickers, data, seed=seed):
            t = list(tickers)
            mis = -np.random.default_rng([seed, pd.Timestamp(as_of).toordinal()]).uniform(0.01, 1.0, len(t))
            return pd.DataFrame({"mispricing": mis, "method_spread": 1.0, "n_methods": 1.0},
                                index=pd.Index(t, name="ticker")).astype(float)
        sv._monthly_values = values
        rows.append(run(f"random_always_on_seed{seed}", seed=seed))
    sv._monthly_values, sv.rates_rising = real_values, real_gate
    return rows


def main():
    part = sys.argv[1]
    rows = {"placebo": placebo, "methods": methods, "constants": constants, "controls": controls, "flip": flip, "flipcheck": flipcheck}[part](runner())
    out = Path("output") / "experiments" / f"diag_{part}_dev.csv"
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
