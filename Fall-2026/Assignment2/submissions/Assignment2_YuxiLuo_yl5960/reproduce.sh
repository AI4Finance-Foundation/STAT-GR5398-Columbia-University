#!/usr/bin/env bash
# Reproduce every dev/val number in the report with one command. NEVER runs the test period:
# test was run twice at commit 8288c74 and its outputs are shipped in output/ (see README).
#
#   ./reproduce.sh          core: evaluation (module and course example), A/B/C/D ablation,
#                           check_submission, then compares with the shipped output/  (~5 min)
#   ./reproduce.sh --all    core + every analysis in the report, compared file by file with
#                           the shipped analysis/ results                              (~30-40 min)
#
# Needs the course data bundle in Assignment2/source_code/data/ and the course requirements
# (scikit-learn 1.9.0, xgboost 3.4.0, lightgbm 4.7.0, numpy 2.5.2, pandas 2.3.3).
# Environment: PYTHON (default: python), JOBS (parallel jobs for --all, default 6).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../../source_code" && pwd)"
PY="${PYTHON:-python}"
JOBS="${JOBS:-6}"
R="$HERE/reproduced"
mkdir -p "$R/logs"
export PYTHONHASHSEED=0          # the course backtester depends on the hash seed (portfolio/backtester.py:148)
export PY R SRC

cd "$SRC"
[ -f data/fundamentals.db ] || { echo "course data bundle not found in $SRC/data/"; exit 1; }
if ! cmp -s "$HERE/student_valuation.py" valuation/student_valuation.py; then
  echo "copying the submitted module into source_code/valuation/ (as the grader does)"
  cp "$HERE/student_valuation.py" valuation/student_valuation.py
fi

echo "== core: evaluation, ablation, check_submission (dev and val)"
for p in dev val; do
  "$PY" evaluate_valuation.py --period "$p" --example > "$R/logs/eval_${p}_example.log" 2>&1
  mv "output/valuation_eval_$p.csv" "output/valuation_eval_${p}_example.csv"   # --example writes the same name
  "$PY" evaluate_valuation.py --period "$p" > "$R/logs/eval_$p.log" 2>&1
  "$PY" run_portfolio.py --period "$p" --ablation --timing full > "$R/logs/ablation_$p.log" 2>&1
  echo "   $p done"
done
"$PY" check_submission.py --valuation > "$R/check_submission.txt" 2>&1 || true
tail -1 "$R/check_submission.txt"
"$PY" "$HERE/analysis/compare_results.py" --core "$HERE/output" "$SRC/output" || echo "!! core results differ from output/"

[ "${1:-}" = "--all" ] || { echo "core done (add --all for every analysis in the report)"; exit 0; }

echo "== all: every analysis, written to reproduced/analysis/ (the shipped analysis/ is not touched)"
A="$R/analysis"
mkdir -p "$A/study/v2" "$A/logs" "$A/tmp"
cp "$HERE"/analysis/*.py "$A/"
export A

"$PY" "$A/audit_fundamentals.py" > "$A/logs/audit_fundamentals.log" 2>&1

echo "   filter tuning grid and B under seeds 1-4"
{ for m in 2 3; do for p in 0.0 0.5 1.0 2.0; do echo "dev m${m}_p$p $m $p"; done; done
  for p in 0.0 0.5 1.0; do echo "val m2_p$p 2 $p"; done; } |
  xargs -P "$JOBS" -n 4 sh -c '"$PY" "$A/tune_filter.py" --period "$0" --tag "$1" --log "$A/tuning_seeded.jsonl" \
    --set FILTER_MIN_METHODS="$2" --set FILTER_PREMIUM="$3" > "$A/logs/tune_$0_$1.log" 2>&1'
for p in dev val; do for s in 1 2 3 4; do echo "$p $s"; done; done |
  xargs -P "$JOBS" -n 2 sh -c 'PYTHONHASHSEED="$1" "$PY" "$A/tune_filter.py" --period "$0" --tag "final_seed$1" \
    --log "$A/tuning_seeded.jsonl" > "$A/logs/tune_$0_final_seed$1.log" 2>&1'

echo "   sensitivity of the valuation constants (76 jobs)"
"$PY" "$A/run_sensitivity.py" --workers "$JOBS" > "$A/logs/run_sensitivity.log" 2>&1
(cd "$A" && "$PY" summarize_sensitivity.py > "$A/logs/summarize_sensitivity.log" 2>&1)

echo "   filter-rule ablation and swaps (v1), v2 variants"
{ for p in dev val; do
    for v in committed min1 dcf ev_ebitda peer_multiple dcf+ev_ebitda dcf+peer_multiple ev_ebitda+peer_multiple; do
      echo "$p $v v1"; done
    for ver in v2_0 v2a v2b v2c; do echo "$p committed $ver"; done; done; } |
  xargs -P "$JOBS" -n 3 sh -c '"$PY" "$A/filter_study.py" --period "$0" --variant "$1" --version "$2" \
    > "$A/logs/study_$0_$1_$2.log" 2>&1'
for p in dev val; do for v in v2_0 v2a v2b v2c; do echo "$p $v"; done; done |
  xargs -P "$JOBS" -n 2 sh -c '"$PY" "$A/v2_eval.py" --period "$0" --variant "$1" > "$A/logs/v2_eval_$0_$1.log" 2>&1'
"$PY" "$A/method_diagnostics.py" --period dev --save "$A/study/v2/diag_dev_v1_mac.csv" > "$A/logs/diag_dev_v1.log" 2>&1

echo "   hash seeds, spot check, look-ahead tests, constants, DCF diagnostic"
for p in dev val; do for s in 0 1 2 3 4; do echo "$p $s"; done; done |
  xargs -P "$JOBS" -n 2 sh -c 'PYTHONHASHSEED="$1" "$PY" "$A/seed_check.py" --period "$0" > "$A/logs/seed_$0_$1.log" 2>&1'
"$PY" "$A/spot_check_basis.py" > "$A/logs/spot_check_basis.log" 2>&1
"$PY" "$A/lookahead_test.py" --tmp "$A/tmp" > "$A/logs/lookahead_test.log" 2>&1
"$PY" "$A/lookahead_test.py" --version v2c --tmp "$A/tmp" > "$A/logs/lookahead_test_v2c.log" 2>&1
"$PY" "$A/constant_binding.py" > "$A/logs/constant_binding.log" 2>&1
"$PY" "$A/dcf_diagnostic.py" > "$A/logs/dcf_diagnostic.log" 2>&1

echo "== comparing reproduced/analysis with the shipped analysis/"
"$PY" "$A/compare_results.py" --all "$HERE/analysis" "$A"
