"""Assemble the main results table from published aggregate outputs (standard library)."""
import csv
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "output"
PERIODS = ("dev", "val", "test")
METRICS = ("total_return_pct", "cagr_pct", "volatility_pct", "sharpe",
           "max_drawdown_pct", "turnover_per_year")
LABELS = ("A_selection", "B_valuation", "C_timing_full", "D_both_full")


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_portfolio_table():
    reference = read_rows(OUTPUT / "formal/passive_benchmarks.csv")
    passive = {(r["period"], r["benchmark"]): r for r in reference}
    table = []
    for period in PERIODS:
        official = read_rows(OUTPUT / f"ablation_{period}.csv")
        assert [r["run"] for r in official] == [f"{period}_{name}" for name in LABELS]
        for key in METRICS:
            assert official[0][key] == official[2][key], (period, "A/C", key)
            assert official[1][key] == official[3][key], (period, "B/D", key)
        for name, row in zip(LABELS, official):
            table.append({"period": period, "portfolio": name, "start": row["start"],
                          "end": row["end"], **{k: row[k] for k in METRICS}})
        for ticker in ("SPY", "QQQ"):
            row = passive[period, ticker]
            for course in official:
                assert (row["start"], row["end"]) == (course["start"], course["end"])
                for key in ("total_return_pct", "sharpe"):
                    assert float(row[key]) == float(course[f"{ticker}_{key}"]), (period, ticker, key)
            assert row["turnover_per_year"] == ""
            table.append({"period": period, "portfolio": ticker, "start": row["start"],
                          "end": row["end"], **{k: row[k] for k in METRICS}})
    write_rows(OUTPUT / "portfolio_comparison.csv", list(table[0]), table)


if __name__ == "__main__":
    build_portfolio_table()
