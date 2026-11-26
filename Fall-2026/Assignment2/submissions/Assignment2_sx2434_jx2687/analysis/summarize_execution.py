"""Calculate DISCK cash constraints and starting-value sensitivity from archived validation ledgers."""
from build_portfolio_comparison import OUTPUT, read_rows, write_rows

EXECUTION = OUTPUT / "extension/anet"
DATES = {"2022-04-11", "2022-04-18", "2022-06-06", "2023-12-26"}


def stale_positions():
    traces = read_rows(EXECUTION / "weekly_anet.csv")
    positions = {r["day"]: r for r in read_rows(EXECUTION / "val_positions.csv")
                 if r["ticker"] == "DISCK"}
    out = []
    for trace in traces:
        day, arm = trace["day"], trace["arm"]
        if day not in DATES:
            continue
        position = positions[day]
        assert trace["untradeable_names"] == "DISCK" and position["has_open"] == "False"
        column = "baseline_shares" if arm == "baseline" else "strategy_shares"
        unavailable = float(trace["stale_cash_unavailable"])
        nav = float(trace["equity"])
        out.append({"day": day, "portfolio": {"baseline": "A", "filtered": "K1.983"}[arm],
                    "shares": int(float(position[column])), "no_tradable_open": True,
                    "unavailable_value_usd": round(unavailable, 2), "opening_nav_usd": round(nav, 2),
                    "unavailable_pct_of_opening_nav": round(100 * unavailable / nav, 2)})
    assert len(out) == 2 * len(DATES)
    write_rows(EXECUTION / "disck_unpriced.csv", list(out[0]), sorted(out, key=lambda r: (r["day"], r["portfolio"])))


def execution_gaps():
    equities = read_rows(EXECUTION / "val_equity.csv")
    assert len(equities) == 501
    official = {r["policy"]: r for r in read_rows(EXECUTION / "val_comparison.csv")}
    out = []
    for policy in ("course", "pro_rata_floor", "pro_rata_remainders"):
        results = {}
        for convention in ("first_close", "initial_capital"):
            pair = []
            for side in ("baseline", "K1.983"):
                col = f"{policy}_{side}"
                end = float(equities[-1][col])
                if convention == "first_close":
                    start = float(equities[0][col])
                    intervals = len(equities) - 1
                else:
                    start = 100_000
                    intervals = len(equities)
                pair.append(100 * ((end / start) ** (252 / intervals) - 1))
            results[f"{convention}_gap_pp"] = pair[1] - pair[0]
        assert abs(results["first_close_gap_pp"] - float(official[policy]["gap_pp"])) < 1e-9
        out.append({"policy": policy, **results,
                    "ending_gap_usd": round(float(official[policy]["portfolio_delta_usd"]), 2)})
    write_rows(EXECUTION / "execution_gap_by_start.csv", list(out[0]), out)


if __name__ == "__main__":
    stale_positions()
    execution_gaps()
