"""Evaluation diagnostics for a valuation_v2 variant (dev and val only).

Run from source_code/ with PYTHONHASHSEED=0:
    python ../submissions/Assignment2_YuxiLuo_yl5960/analysis/v2_eval.py --period dev --variant v2c

Patches student_valuation.value_detail with valuation_v2 in this process, then runs method_diagnostics.py
(the evaluate_valuation sample: top 20 growth_tech on the first session of each month). Writes
study/v2/diag_<period>_<variant>.csv and .json.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(HERE))

import valuation_v2 as v2  # noqa: E402
import method_diagnostics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "val"], required=True)
    ap.add_argument("--variant", choices=list(v2.VARIANTS), required=True)
    a = ap.parse_args()
    v2.install(a.variant)
    out = HERE / "study" / "v2"
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["method_diagnostics.py", "--period", a.period,
                "--save", str(out / f"diag_{a.period}_{a.variant}.csv"),
                "--json", str(out / f"diag_{a.period}_{a.variant}.json")]
    method_diagnostics.main()
    print(f"peer groups: correlation {v2.STATS['corr_used']}, sector fallback {v2.STATS['corr_fallback']}")


if __name__ == "__main__":
    main()
