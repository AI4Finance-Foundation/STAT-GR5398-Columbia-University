"""Check documented strategy, attribution and gate results using archived summaries and NAVs only."""
import json
from math import expm1, isclose, log1p

from build_portfolio_comparison import OUTPUT, read_rows


def same(a, b, atol=1e-7):
    if not isclose(a, b, rel_tol=0, abs_tol=atol):
        raise ValueError(f"Mismatch: {a} versus {b}")


def cagr(values):
    return ((values[-1] / values[0]) ** (252 / (len(values) - 1)) - 1) * 100


def account_for_official():
    runs = [json.loads(line) for line in (OUTPUT / "test_runs.jsonl").read_text().splitlines()]
    assert len(runs) == 1 and runs[0]["student_files_uncommitted"] is False
    assert runs[0]["git_head"] == "0c8eeec6683633120f2ed63a375463356c90ee80"
    annual = read_rows(OUTPUT / "formal/annual_returns.csv")
    pnl = read_rows(OUTPUT / "formal/stock_attribution.csv")
    for period in ("dev", "val"):
        obs = {}
        for arm, col in (("A_selection", "baseline"), ("B_valuation", "valuation")):
            series = read_rows(OUTPUT / f"formal/{period}_{arm}_equity.csv")
            pos = {row[""][:4]: float(row["equity"]) for row in series}
            start = 100_000
            for year, end in pos.items():
                row = next(r for r in annual if r["period"] == period and r["year"] == year)
                same((end / start - 1) * 100, float(row[f"{col}_return_pct"]))
                start = end
            obs[col] = start
        delta = sum(float(r["incremental_net_pnl_usd"]) for r in pnl if r["period"] == period)
        same(delta, obs["valuation"] - obs["baseline"])
        print(f"{period} formal A/B yearly NAV and stock P&L reconcile; B-A ending P&L {delta:+,.2f} USD")


def check_expansion():
    root = OUTPUT / "extension"
    fixed = read_rows(root / "gate/summary.csv")
    random = read_rows(root / "placebo/summary.csv")
    for split in ("dev", "val"):
        nav = read_rows(root / f"gate/{split}_equity.csv")
        for row in fixed:
            if row["split"] == split:
                values = [float(r[row["rule"]]) for r in nav]
                same(cagr(values), float(row["cagr_pct"]))
        placebo = next(r for r in random if r["period"] == split)
        weekly = read_rows(root / f"placebo/{split}_weeks.csv")
        for row_key, col in (("base_proxy_cagr", "baseline_gross"), ("gate_proxy_cagr", "observed_gross")):
            total_log = sum(log1p(float(r[col])) for r in weekly)
            same(expm1(total_log * 252 / (len(nav) - 1)) * 100, float(placebo[row_key]))
        print(f"{split} gate NAVs and gross target-weight benchmark match archived output")
    steps = read_rows(root / "threshold/dev_full_steps.csv")
    high = max(float(r["delta_cagr_pp"]) for r in steps)
    peaks = [r for r in steps if isclose(float(r["delta_cagr_pp"]), high, abs_tol=1e-10)]
    edges = [r for r in read_rows(root / "threshold/dev_peak_triggers.csv") if r["arm"] == "all"]
    lower, upper = (float(next(r["span"] for r in edges if r["edge"] == k)) for k in ("entry", "exit"))
    assert all(lower <= float(r["K"]) < upper for r in peaks)
    same(min(float(r["K"]) for r in peaks), lower)
    assert lower <= 1.983 < upper
    print(f"Development peak {high:+.4f} pp on [{lower:.9f}, {upper:.9f})")
    weeks = read_rows(root / "anet/weekly_activity.csv")
    same_names = [r for r in weeks if r["same_names"] == "True"]
    assert len(weeks) == 104 and len(same_names) == 92 and all(r["same_shares"] == "False" for r in same_names)
    pp = sum(float(r["pp"]) for r in weeks)
    course = next(r for r in read_rows(root / "anet/val_comparison.csv") if r["policy"] == "course")
    same(pp, float(course["gap_pp"]))
    print(f"Validation attribution {pp:+.4f} pp across {len(weeks)} weeks, {len(same_names)} with the same names")


if __name__ == "__main__":
    account_for_official()
    check_expansion()
