"""Validation execution sensitivity. Run with the course venv and PYTHONHASHSEED=0."""
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
LOCK = HERE.parents[1]
ROOT = LOCK.parents[1]
STUDY = ROOT / "weighted_valuation_research"
sys.path.insert(0, str(STUDY))
from study import CACHE, MonthlyScores, RemovalScores, evaluate, np, pd, period, rebalance_days
from portfolio import backtester as engine
from selection.config import load

START, END = period("val")
PROTOCOL = json.loads((LOCK / "protocol.json").read_text())
assert (PROTOCOL["start"], PROTOCOL["end"], PROTOCOL["K"]) == (str(START), str(END), 1.983)
assert hashlib.sha256((STUDY / "study.py").read_bytes()).hexdigest() == PROTOCOL["implementation_sha256"]
assert os.environ.get("PYTHONHASHSEED") == "0"
CFG = engine.BacktestConfig(start=START, end=END)
COST = CFG.commission + CFG.slippage
POLICIES = ("course", "pro_rata_floor", "pro_rata_remainders")


def inputs():
    # Retain the inception decision's prior close. Neither market data source
    # returns any records beyond the validation boundary.
    read = pd.read_parquet
    bounds = {
        "daily_adjusted.parquet": ("date", pd.Timestamp("2021-12-31")),
        "monthly_scores.parquet": ("month", pd.Timestamp("2022-01-01")),
    }

    def read_window(path):
        key, first = bounds[Path(path).name]
        return read(path, filters=[(key, ">=", first), (key, "<=", pd.Timestamp(END))])

    with patch.object(pd, "read_parquet", read_window):
        _, options, params, prices, _, universe = load("selection/configs/growth.json")
        scores = MonthlyScores()
    assert prices.sessions[-1].date() == END and len(scores.months) == 24

    # Prime the course membership cache with historical metadata only. Its
    # original members_at and effective-date ticker mapping still decide eligibility.
    store = universe._store
    with store._connect() as conn:
        for name in options.universes:
            frame = pd.read_sql_query(
                "SELECT snapshot_date AS date, tickers FROM universe_membership "
                "WHERE universe = ? AND snapshot_date <= ? ORDER BY snapshot_date",
                conn, params=(name, str(END)), parse_dates=["date"])
            assert not frame.empty
            store._universe_cache[name] = frame
        rows = conn.execute(
            "SELECT change_date, old_symbol, new_symbol FROM symbol_changes "
            "WHERE change_date <= ? ORDER BY change_date, old_symbol", (str(END),)).fetchall()
    store._symbol_changes = [(pd.Timestamp(d), old, new) for d, old, new in rows]
    store._symbol_change_dates = [d for d, _, _ in store._symbol_changes]

    weeks = pd.DatetimeIndex(rebalance_days(prices, START, END))
    quotes = pd.read_parquet(CACHE / "valuations_val.parquet")
    assert set(quotes.day) == set(weeks.date) and not quotes.duplicated(["day", "ticker"]).any()
    drops = {}
    for day, group in quotes.groupby("day"):
        prior = prices.last_session_on_or_before(day - pd.Timedelta(days=1))
        assert pd.to_datetime(group.as_of).eq(pd.Timestamp(prior)).all()
        scores.get(day)
        assert scores.cutoff <= prior
        data = evaluate(group)
        centers = data[["dcf", "peer", "hist_mid"]]
        veto = data.adaptive_veto & centers.max(axis=1).le(PROTOCOL["K"] * centers.min(axis=1))
        drops[day] = set(data.loc[veto, "ticker"])
    return prices, scores, params, universe, weeks, drops


def buy_plan(orders, starting_cash, day, policy, weekly_log):
    ordered = sorted(orders, key=lambda order: order[1])
    buys = [(s, q, p) for s, q, p in ordered if q > 0]
    sells = [(s, q, p) for s, q, p in ordered if q < 0]
    available = starting_cash
    for _, q, p in sells:
        available -= q * p + abs(q * p) * COST
    need = math.fsum(q * p * (1 + COST) for _, q, p in buys)
    scale = min(1., max(0., available) / need) if need else 1.
    quantities = {s: q for s, q, _ in buys}
    if policy != "course" and scale < 1.:
        ideal = {s: q * scale for s, q, _ in buys}
        quantities = {s: math.floor(q) for s, q in ideal.items()}
        if policy == "pro_rata_remainders":
            left = available - math.fsum(quantities[s] * p * (1 + COST) for s, _, p in buys)
            priority = sorted(buys, key=lambda o: (-(ideal[o[0]] - quantities[o[0]]), o[0]))
            for s, q, p in priority:
                unit_cost = p * (1 + COST)
                if quantities[s] < q and unit_cost <= left:
                    quantities[s] += 1
                    left -= unit_cost
        spent = math.fsum(quantities[s] * p * (1 + COST) for s, _, p in buys)
        assert spent <= available + 1e-8 and all(0 <= quantities[s] <= q for s, q, _ in buys)
    requested = next((q for s, q, _ in ordered if s == "ANET"), 0)
    weekly_log.append(dict(day=day, buy_cash=available, buy_need=need, cash_ratio=scale,
                           anet_order=requested))
    # Preserve the original order and tie handling. The course affordability
    # check also runs for every planned fill. It binds only under course policy.
    adjusted = [(s, quantities[s] if q > 0 else q, p) for s, q, p in orders]
    if policy != "course":
        weekly_log[-1]["planned"] = {s: q for s, q, _ in adjusted if q}
    return [item for item in adjusted if item[1] != 0]


def replay(prices, scores, params, universe, drops, policy):
    # Reuse the course function verbatim except for one call after orders have
    # been generated and before execution. No mutation of the source module.
    rows = []
    source = inspect.getsource(engine.run_backtest)
    before = "            for sym, delta, px in sorted(orders, key=lambda o: o[1]):"
    assert source.count(before) == 1
    source = source.replace(before, "            orders = plan(orders, cash, day)\n" + before)
    bindings = dict(vars(engine), plan=lambda o, c, d: buy_plan(o, c, d, policy, rows))
    exec(compile(source, "<validation execution sensitivity>", "exec"), bindings)
    run = bindings["run_backtest"](prices, RemovalScores(scores, drops), params, universe, CFG)
    if policy != "course":
        executed = {(t.date, t.symbol): t.qty for t in run.trades.itertuples()}
        planned = {(row["day"].date(), sym): qty for row in rows for sym, qty in row.pop("planned").items()}
        assert len(executed) == len(run.trades) and executed == planned
    return run, pd.DataFrame(rows).set_index("day")


def ledger(run, prices, weeks):
    values = run.equity
    trades = run.trades.assign(date=pd.to_datetime(run.trades.date))
    qty = trades.pivot(index="date", columns="symbol", values="qty").reindex(values.index).fillna(0)
    opens = prices.open.reindex_like(qty)
    closes = prices.close[qty.columns].ffill().reindex_like(qty)
    flows = ((qty + COST * qty.abs()) * opens).fillna(0).cumsum()
    shares = qty.cumsum()
    pnl = (shares * closes).fillna(0) - flows
    np.testing.assert_allclose(CFG.initial_cash + pnl.sum(axis=1), values, atol=1e-6, rtol=0)
    at_open = ((shares.shift(fill_value=0) * opens.fillna(closes)).fillna(0)
               - flows.shift(fill_value=0)).loc[weeks]
    next_open = at_open.shift(-1)
    next_open.iloc[-1] = pnl.iloc[-1]
    weekly = next_open - at_open
    np.testing.assert_allclose(weekly.sum(), pnl.iloc[-1], atol=1e-6, rtol=0)
    normalized = weekly.copy()
    normalized.iloc[0] -= pnl.iloc[0]
    return weekly, normalized / values.iloc[0], qty.ANET.loc[weeks], shares.ANET.loc[weeks]


def cagr(values):
    return 100 * ((values.iloc[-1] / values.iloc[0]) ** (252 / (len(values) - 1)) - 1)


def interval(equity):
    returns = np.log(equity).diff().dropna().to_numpy()
    n = len(returns)
    rng = np.random.default_rng(20261124)
    draws = []
    for _ in range(5000):
        starts = rng.integers(n, size=(n + 19) // 20)
        sample = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
        a, b = 100 * np.expm1(252 * returns[sample].mean(axis=0))
        draws.append(b - a)
    return np.quantile(draws, [.05, .95])


def main():
    prices, scores, params, universe, weeks, drops = inputs()
    original = pd.read_csv(LOCK / "val_equity.csv", index_col=0, parse_dates=True)
    original_stock = pd.read_csv(LOCK / "attribution/val_stock_attribution.csv", index_col=0)
    summaries, by_stock, by_week, curves = [], [], [], []
    for policy in POLICIES:
        arms, pnl, units, week_logs = {}, {}, {}, {}
        for name, exclusions in (("baseline", {}), ("K1.983", drops)):
            run, orders = replay(prices, scores, params, universe, exclusions, policy)
            assert run.equity.index.equals(original.index) and orders.index.equals(weeks)
            if policy == "course":
                np.testing.assert_allclose(run.equity, original[name], atol=1e-6, rtol=0)
            arms[name] = run.equity
            pnl[name], units[name], fills, shares = ledger(run, prices, weeks)
            weights = [log["weights"].get("ANET", 0) for log in run.weekly_targets]
            week_logs[name] = orders.assign(anet_weight=weights, anet_fill=fills, anet_shares=shares)
        equity = pd.DataFrame(arms)
        gap = cagr(equity["K1.983"]) - cagr(equity.baseline)
        ratio = equity.iloc[-1] / equity.iloc[0]
        weekly_pp = units["K1.983"].subtract(units["baseline"], fill_value=0) * (gap / (ratio["K1.983"] - ratio.baseline))
        weekly_usd = pnl["K1.983"].subtract(pnl["baseline"], fill_value=0)
        stock = pd.DataFrame(dict(pp=weekly_pp.sum(), usd=weekly_usd.sum()))
        np.testing.assert_allclose(stock.pp.sum(), gap, atol=1e-8, rtol=0)
        np.testing.assert_allclose(stock.usd.sum(), equity.iloc[-1].diff().iloc[-1], atol=1e-6, rtol=0)
        if policy == "course":
            np.testing.assert_allclose(stock.pp, original_stock.loc[stock.index, "all_pp"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(stock.usd, original_stock.loc[stock.index, "all_usd"], atol=1e-6, rtol=0)
        weekly = pd.concat(week_logs, axis=1)
        weekly.columns = ["_".join(col) for col in weekly.columns]
        weekly = weekly.assign(anet_veto=["ANET" in drops[day.date()] for day in weeks],
                               anet_usd=weekly_usd.ANET, anet_pp=weekly_pp.ANET,
                               portfolio_usd=weekly_usd.sum(axis=1))
        lo, hi = interval(equity)
        summaries.append(dict(policy=policy, baseline_cagr=cagr(equity.baseline),
                              strategy_cagr=cagr(equity["K1.983"]), gap_pp=gap,
                              lo90_pp=lo, hi90_pp=hi, portfolio_delta_usd=stock.usd.sum(),
                              anet_pp=stock.at["ANET", "pp"], anet_usd=stock.at["ANET", "usd"],
                              others_pp=gap-stock.at["ANET", "pp"],
                              constrained_weeks_baseline=weekly.baseline_cash_ratio.lt(1).sum(),
                              constrained_weeks_strategy=weekly["K1.983_cash_ratio"].lt(1).sum()))
        by_stock.append(stock.assign(policy=policy).rename_axis("symbol").reset_index())
        by_week.append(weekly.assign(policy=policy).rename_axis("day").reset_index())
        curves.append(equity.add_prefix(policy + "_"))
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(HERE / "val_comparison.csv", index=False)
    pd.concat(by_stock).to_csv(HERE / "val_attribution.csv", index=False)
    pd.concat(by_week).to_csv(HERE / "val_weekly.csv", index=False)
    pd.concat(curves, axis=1).to_csv(HERE / "val_equity.csv")
    print(comparison.round(5).to_string(index=False))
    print(pd.concat(by_week).loc[lambda d: d.day.eq("2022-06-06")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

