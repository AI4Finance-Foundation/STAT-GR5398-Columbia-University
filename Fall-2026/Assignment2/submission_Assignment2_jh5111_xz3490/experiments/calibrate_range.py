"""Calibrate value()'s [low, high] range on dev: share of prices inside it 252 sessions later.

Target fixed before running: 50%. Only range widths change; fair values (and so the rank IC) do not.
    .venv/bin/python experiments/calibrate_range.py
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import evaluate_valuation
from valuation import student_valuation as sv

TARGET = 0.50
CONFIGS = [  # (DCF WACC step, DCF growth step, multiples' band quantiles, combine rule)
    (0.01, 0.005, (0.25, 0.75), "median"),
    (0.02, 0.010, (0.25, 0.75), "median"),
    (0.02, 0.010, (0.10, 0.90), "median"),
    (0.01, 0.005, (0.25, 0.75), "envelope"),
    (0.02, 0.010, (0.25, 0.75), "envelope"),
    (0.02, 0.010, (0.10, 0.90), "envelope"),
]


def main():
    rows = []
    for w, g, q, combine in CONFIGS:
        sv.WACC_STEP, sv.TG_STEP, sv.EV_BAND_Q, sv.PE_BAND_Q, sv.RANGE_COMBINE = w, g, q, q, combine
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluate_valuation.main(["--period", "dev"])
        out = buf.getvalue()
        inside = re.search(r"inside \[low, high\]: (\d+)%", out)
        res = pd.read_csv("output/valuation_eval_dev.csv")
        width = ((res["high"] - res["low"]) / res["fair_value"]).median()
        rows.append({"dcf_grid": f"WACC +/-{w:.0%}, g +/-{g:.1%}", "multiples_band": f"P{q[0]*100:.0f}-P{q[1]*100:.0f}",
                     "combine": combine, "coverage": int(inside.group(1)) / 100 if inside else None,
                     "median_width_vs_fair": round(width, 2)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table["gap_to_target"] = (table["coverage"] - TARGET).abs()
    out = Path("output") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "range_calibration_dev.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
