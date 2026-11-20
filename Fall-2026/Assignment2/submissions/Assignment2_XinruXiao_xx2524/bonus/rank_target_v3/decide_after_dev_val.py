"""Compare pre-test results and apply the pre-registered admission rule."""
from __future__ import annotations

import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
old = root / "work/retraining/evaluation"
v2 = root / "outputs/retraining/feature_ablation_v2/comparison_metrics.json"
new = root / "work/retraining_v3/model/output"
v2_results = json.loads(v2.read_text())


def saved(path: Path) -> dict:
    return json.loads((path / "stats.json").read_text())


results = {}
for period in ("dev", "val"):
    results[period] = {
        "same_environment_unchanged_A": saved(
            old / "retrained_baseline/output" / f"{period}_baseline_scores_A_selection"
        ),
        "v2_A": v2_results[period]["A"]["v2"],
        "v2_B": v2_results[period]["B"]["v2"],
        "v3_A": saved(new / f"{period}_scores_v3_selection"),
        "v3_B": saved(new / f"{period}_scores_v3_valuation"),
    }

checks = {
    "val_A_return_at_least_v2_11_85": results["val"]["v3_A"]["total_return_pct"] >= 11.85,
    "val_A_sharpe_at_least_v2_0_342": results["val"]["v3_A"]["sharpe"] >= 0.342,
    "dev_A_return_at_least_177_92": results["dev"]["v3_A"]["total_return_pct"] >= 177.92,
    "dev_A_turnover_no_more_than_125pct_v2": (
        results["dev"]["v3_A"]["turnover_per_year"]
        <= 1.25 * results["dev"]["v2_A"]["turnover_per_year"]
    ),
    "val_A_turnover_no_more_than_125pct_v2": (
        results["val"]["v3_A"]["turnover_per_year"]
        <= 1.25 * results["val"]["v2_A"]["turnover_per_year"]
    ),
}
decision = {
    "protocol": "PRE_REGISTERED_PROTOCOL.md",
    "results": results,
    "admission_checks": checks,
    "admit_to_one_final_test": all(checks.values()),
    "test_or_forward_viewed_by_this_script": False,
}
(here / "dev_val_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
lines = [
    "# v3开发期与验证期决定",
    "",
    "此文件由保存的确定性回测结果生成，未读取v3的test或forward结果。",
    "",
    "| 时期 | 版本 | A累计收益% | A Sharpe | A最大回撤% | A年换手 | B累计收益% |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for period in ("dev", "val"):
    for label in ("same_environment_unchanged", "v2", "v3"):
        a = results[period][f"{label}_A"]
        b = results[period].get(f"{label}_B")
        lines.append(
            f"| {period} | {label} | {a['total_return_pct']} | {a['sharpe']} | "
            f"{a['max_drawdown_pct']} | {a['turnover_per_year']} | "
            f"{b['total_return_pct'] if b else '未在本表重算'} |"
        )
lines += ["", "## 预登记准入检查", ""]
for name, passed in checks.items():
    lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
lines += ["", f"最终决定：{'进入一次最终test' if all(checks.values()) else '不进入新的test；保留v2'}。"]
(here / "DEV_VAL_DECISION.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"admit_to_one_final_test": all(checks.values()), "checks": checks}, indent=2))

