"""Compare reproduced results with the shipped ones (used by reproduce.sh; dev and val only).

    python compare_results.py --core SUBMISSION_OUTPUT SOURCE_CODE_OUTPUT
    python compare_results.py --all  SHIPPED_ANALYSIS  REPRODUCED_ANALYSIS

--core compares the evaluation and ablation CSVs that reproduce.sh regenerates in source_code/output/ with the
copies shipped in the submission's output/. --all compares every result file of the analysis scripts.
Numbers must agree to a relative or absolute tolerance of 1e-9 (platforms differ in the last float bits);
run-time fields (seconds, slowest_s) are ignored. Prints MATCH / DIFF / MISSING per file; exit code 1 on any DIFF.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

TOL = 1e-9
IGNORE = {"seconds", "slowest_s"}
CORE = ["valuation_eval_dev.csv", "valuation_eval_val.csv", "valuation_eval_dev_example.csv",
        "valuation_eval_val_example.csv", "ablation_dev.csv", "ablation_val.csv"]
JSONL_KEYS = {"study/portfolio.jsonl": ("period", "variant", "version"), "study/seeds.jsonl": ("period", "seed"),
              "sens/portfolio.jsonl": ("period", "tag"), "tuning_seeded.jsonl": ("period", "tag")}
ALL_GLOBS = ["sens/table_*.csv", "sens/diag/*.json", "sens/diag/dev_base.csv", "sens/diag/val_base.csv", "sens/diag/*_bands_10_90.csv",
             "study/v2/*.json", "study/v2/*.csv", "study/*.csv"]


def close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= max(TOL, TOL * abs(a))
    return a == b


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if k in IGNORE:
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def cmp_dicts(a, b):
    a, b = flatten(a), flatten(b)
    bad = [k for k in a if k not in b or not close(a[k], b[k])]
    return bad


def cmp_csv(p, q):
    x, y = pd.read_csv(p), pd.read_csv(q)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return f"shape {x.shape} vs {y.shape}"
    worst = 0.0
    for c in x.columns:
        if x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi":
            d = (x[c] - y[c]).abs()
            lim = (x[c].abs() * TOL).clip(lower=TOL)
            both_nan = x[c].isna() & y[c].isna()
            bad = ~((d <= lim) | both_nan)
            if bad.any():
                return f"column {c}: {int(bad.sum())} values differ (max {float(d[bad].max()):.3g})"
            worst = max(worst, float(d.fillna(0).max()))
        elif not x[c].fillna("").astype(str).equals(y[c].fillna("").astype(str)):
            return f"column {c}: text differs"
    return None if worst == 0 else None


def cmp_jsonl(p, q, key):
    def load(f):
        recs = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        return {tuple(str(r.get(k, "v1" if k == "version" else "")) for k in key): r for r in recs}
    a, b = load(p), load(q)
    missing = [k for k in a if k not in b]
    bad = {k: cmp_dicts(a[k], b[k]) for k in a if k in b}
    bad = {k: v for k, v in bad.items() if v}
    if missing or bad:
        return f"{len(missing)} records missing, {len(bad)} differ" + (f", e.g. {next(iter(bad))}: {next(iter(bad.values()))[:3]}" if bad else "")
    return None


def report(name, problem, results):
    status = "MATCH" if problem is None else ("MISSING" if problem == "missing" else "DIFF")
    results.append(status)
    print(f"{status:8s} {name}" + ("" if problem in (None, "missing") else f"   ({problem})"))


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--core", action="store_true")
    mode.add_argument("--all", action="store_true")
    ap.add_argument("shipped", type=Path)
    ap.add_argument("reproduced", type=Path)
    a = ap.parse_args()
    results = []
    if a.core:
        for f in CORE:
            p, q = a.shipped / f, a.reproduced / f
            report(f, "missing" if not q.exists() else cmp_csv(p, q), results)
    else:
        for rel, key in JSONL_KEYS.items():
            p, q = a.shipped / rel, a.reproduced / rel
            report(rel, "missing" if not q.exists() else cmp_jsonl(p, q, key), results)
        for pattern in ALL_GLOBS:
            for p in sorted(a.shipped.glob(pattern)):
                rel = p.relative_to(a.shipped)
                q = a.reproduced / rel
                if not q.exists():
                    report(str(rel), "missing", results)
                elif p.suffix == ".json":
                    bad = cmp_dicts(json.loads(p.read_text()), json.loads(q.read_text()))
                    report(str(rel), None if not bad else f"fields {bad[:3]}", results)
                else:
                    report(str(rel), cmp_csv(p, q), results)
    n = {s: results.count(s) for s in ("MATCH", "DIFF", "MISSING")}
    print(f"\n{n['MATCH']} match, {n['DIFF']} differ, {n['MISSING']} missing")
    sys.exit(1 if n["DIFF"] else 0)


if __name__ == "__main__":
    main()
