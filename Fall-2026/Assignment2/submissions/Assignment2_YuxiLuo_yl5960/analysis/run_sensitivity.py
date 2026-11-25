"""One-at-a-time sensitivity of the valuation assumptions, on dev and val only.

For every variant below this runs, with PYTHONHASHSEED=0 and the course venv:
  * method_diagnostics.py  -> sens/diag/<period>_<tag>.json (+ .csv of every candidate-month)
  * tune_filter.py         -> sens/portfolio.jsonl (portfolio B with the override)
Variants marked range_only change only [low, high], not any fair value, so the filter
(and portfolio B) cannot change; they get the diagnostics run only.

    python run_sensitivity.py [--workers 6] [--only tag1,tag2] [--skip-portfolio]

Never touches the test period. The committed module file is not edited; overrides are
set on the imported module inside each child process.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_source_code():
    """Assignment2/source_code/, found by walking up from this script: the submission layout
    (Assignment2/submissions/<team>/analysis[/...]) or the working layout (<root>/a2_work beside course-repo/)."""
    for parent in HERE.parents:
        for cand in (parent / "source_code", parent / "course-repo" / "Fall-2026" / "Assignment2" / "source_code"):
            if (cand / "run_portfolio.py").is_file():
                return cand
    sys.exit("cannot find Assignment2/source_code/ above " + str(HERE))


SRC = _find_source_code()
PY = Path(os.environ.get("A2_PYTHON", sys.executable))   # the interpreter running this script, unless overridden
OUT = HERE / "sens"

# tag: (overrides, range_only)
VARIANTS = {
    "base":         ({}, False),
    # DCF
    "erp_040":      ({"EQUITY_RISK_PREMIUM": "0.04"}, False),
    "erp_060":      ({"EQUITY_RISK_PREMIUM": "0.06"}, False),
    "tg_020":       ({"TERMINAL_GROWTH": "0.02"}, False),
    "tg_030":       ({"TERMINAL_GROWTH": "0.03"}, False),
    "gw_000":       ({"GROWTH_WEIGHT_YOY": "0.0"}, False),
    "gw_100":       ({"GROWTH_WEIGHT_YOY": "1.0"}, False),
    "gcap_025":     ({"GROWTH_CAP": "0.25"}, False),
    "gcap_050":     ({"GROWTH_CAP": "0.50"}, False),
    "nwc_000":      ({"NWC_TO_REVENUE": "0.0"}, False),
    "nwc_020":      ({"NWC_TO_REVENUE": "0.20"}, False),
    "yrs_05":       ({"PROJECTION_YEARS": "5"}, False),
    "yrs_15":       ({"PROJECTION_YEARS": "15"}, False),
    "beta_3y":      ({"BETA_YEARS": "3"}, False),
    # EV/EBITDA history
    "hist_3y":      ({"HISTORY_SESSIONS": "756"}, False),
    # peer P/E
    "peer_cap100":  ({"MAX_PEER_MULTIPLE": "100.0"}, False),
    "peers_10":     ({"MIN_PEERS": "10"}, False),
    # data handling
    "no_align":     ({"ALIGN_BASIS": "False"}, False),
    # ranges only
    "bands_10_90":  ({"HIST_BAND": "(0.10, 0.90)", "PEER_BAND": "(0.10, 0.90)"}, True),
    "dcf_step_x2":  ({"WACC_STEP": "0.02", "TG_STEP": "0.01"}, True),
}
PERIODS = ("dev", "val")


def sets(overrides):
    out = []
    for k, v in overrides.items():
        out += ["--set", f"{k}={v}"]
    return out


def run(cmd, log):
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return rc, round(time.time() - t0), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only")
    ap.add_argument("--skip-portfolio", action="store_true")
    ap.add_argument("--skip-diag", action="store_true")
    a = ap.parse_args()
    tags = a.only.split(",") if a.only else list(VARIANTS)
    for d in ("diag", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    jobs = []
    for tag in tags:
        ov, range_only = VARIANTS[tag]
        for period in PERIODS:
            if not a.skip_portfolio and not range_only:
                jobs.append([str(PY), str(HERE / "tune_filter.py"), "--period", period, "--tag", f"sens_{tag}",
                             "--log", str(OUT / "portfolio.jsonl"), *sets(ov)])
                jobs[-1] = (jobs[-1], OUT / "logs" / f"portfolio_{period}_{tag}.log")
            if not a.skip_diag:
                stem = OUT / "diag" / f"{period}_{tag}"
                jobs.append(([str(PY), str(HERE / "method_diagnostics.py"), "--period", period,
                              "--json", f"{stem}.json", "--save", f"{stem}.csv", *sets(ov)],
                             OUT / "logs" / f"diag_{period}_{tag}.log"))
    # longest first: dev portfolio runs take ~4 min, val ~2 min, diagnostics ~1 min
    jobs.sort(key=lambda j: ("tune_filter" not in j[0][1], j[0][3] != "dev"))
    print(f"{len(jobs)} jobs on {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        for fut in as_completed([pool.submit(run, *j) for j in jobs]):
            rc, secs, log = fut.result()
            print(f"{'ok ' if rc == 0 else 'ERR'} {secs:4d}s {log.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
