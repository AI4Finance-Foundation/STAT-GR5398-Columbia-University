"""Replay the two selection paths with the Discovery common-share conversion."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import replay as base

PREVIOUS = ("DISCA", "DISCK")
SUCCESSOR = "WBD"
EFFECTIVE = pd.Timestamp("2022-04-11")
SEC_8K = "https://www.sec.gov/Archives/edgar/data/1437107/000119312522103051/d328161d8k.htm"
ENGINE_SHA = "11e46d554fde566e1e5b0244586ed534cd52f2faaca04c6a863dc15d92e140e9"


def convert_shares(pos, entry, bought_at):
    for symbol in PREVIOUS:
        shares = pos.pop(symbol, 0)
        if not shares:
            continue
        previous_shares = pos.get(SUCCESSOR, 0)
        previous_basis = entry.get(SUCCESSOR, 0) * previous_shares
        opening_session = bought_at.pop(symbol)
        basis = previous_basis + entry.pop(symbol) * shares
        pos[SUCCESSOR] = previous_shares + shares
        entry[SUCCESSOR] = basis / pos[SUCCESSOR]
        bought_at[SUCCESSOR] = min(bought_at.get(SUCCESSOR, opening_session), opening_session)


def sensitivity_engine():
    # Bind two event hooks in a private copy of the verified course function.
    course = base.run_backtest
    assert hashlib.sha256(Path(inspect.getfile(course)).read_bytes()).hexdigest() == ENGINE_SHA
    code = inspect.getsource(course)
    hooks = {
        "        opens = prices.open.loc[day]": """
        if day == EFFECTIVE:
            assert not cfg.stops and not pending_stops
            convert_shares(pos, entry, bought_at)""",
        "            tradeable -= {s for s, until in cooldown.items() if until > i}": """
            if day >= EFFECTIVE:
                tradeable.difference_update(PREVIOUS)""",
    }
    for line, hook in hooks.items():
        assert code.count(line) == 1
        code = code.replace(line, line + hook)
    environment = dict(course.__globals__, EFFECTIVE=EFFECTIVE, PREVIOUS=PREVIOUS, convert_shares=convert_shares)
    exec(compile(code, "<course engine with Discovery conversion>", "exec"), environment)
    return environment["run_backtest"]


def compare_history(primary, out):
    checks = {}
    for name, dates, order in [
        ("daily_nav", "date", ["period", "date"]),
        ("trades", "date", ["period", "arm", "date", "symbol"]),
        ("weekly_stock_targets", "rebalance", ["period", "arm", "rebalance", "symbol"]),
    ]:
        frames = []
        for folder in [primary, out]:
            frame = pd.read_csv(folder / f"{name}.csv", parse_dates=[dates])
            frames.append(frame.loc[frame[dates] < EFFECTIVE].sort_values(order).reset_index(drop=True))
        pd.testing.assert_frame_equal(*frames, check_exact=True)
        checks[name + "_identical_rows_before_conversion"] = len(frames[0])
    return checks


def compare_periods(primary_dir, out):
    measures = ["return_pp", "cagr_pp", "drawdown_depth_pp", "turnover_x", "sharpe"]
    cols = ["feature_minus_control_" + key for key in measures]
    primary = pd.read_csv(primary_dir / "period_differences.csv").set_index("period")
    mapped = pd.read_csv(out / "period_differences.csv").set_index("period")
    pd.testing.assert_series_equal(primary.loc["dev"], mapped.loc["dev"], check_exact=True)
    rows = []
    for period in mapped.index:
        for key, col in zip(measures, cols):
            a, b = primary.at[period, col], mapped.at[period, col]
            rows.append(dict(period=period, measure=key, course_feature_minus_control=a,
                             converted_feature_minus_control=b, difference_in_advantage=b-a,
                             direction_changed=bool(np.sign(a) != np.sign(b))))
    pd.DataFrame(rows).to_csv(out / "primary_comparison.csv", index=False)


def reconcile(inputs, out):
    prices = inputs[0]
    raw = prices.close
    marks = raw.ffill()
    assert prices.open.at[EFFECTIVE, SUCCESSOR] > 0 and raw.at[EFFECTIVE, SUCCESSOR] > 0
    reported = pd.read_csv(out / "daily_nav.csv", parse_dates=["date"]).set_index("date")
    orders = pd.read_csv(out / "trades.csv", parse_dates=["date"])
    targets = pd.read_csv(out / "weekly_stock_targets.csv", parse_dates=["rebalance"])
    assert not orders.loc[orders.date.ge(EFFECTIVE)].symbol.isin(PREVIOUS).any()
    assert not targets.loc[targets.rebalance.ge(EFFECTIVE)].symbol.isin(PREVIOUS).any()
    opens = prices.open.stack()
    results, events, ending = {}, [], []
    for (period, arm), trades in orders.groupby(["period", "arm"]):
        nav = reported.loc[reported.period.eq(period), arm]
        open_index = pd.MultiIndex.from_frame(trades[["date", "symbol"]])
        fills = opens.reindex(open_index).to_numpy() * trades.qty.to_numpy()
        assert np.isfinite(fills).all()
        np.testing.assert_allclose(fills, trades.notional, atol=0.0050001, rtol=0)
        cash = 100000 - trades.assign(spent=fills + abs(fills) * 0.0006).groupby("date").spent.sum().reindex(nav.index, fill_value=0).cumsum()
        changes = trades.pivot_table(index="date", columns="symbol", values="qty", aggfunc="sum", fill_value=0)
        symbols = sorted(set(changes.columns) | set(PREVIOUS) | {SUCCESSOR})
        changes = changes.reindex(index=nav.index, columns=symbols, fill_value=0)
        if period == "val":
            previous_holdings = changes.loc[changes.index < EFFECTIVE].sum()
            successor_before = int(previous_holdings[SUCCESSOR])
            event_targets = targets.loc[targets.period.eq(period) & targets.arm.eq(arm) & targets.rebalance.eq(EFFECTIVE)]
            actual_order = trades.loc[trades.date.eq(EFFECTIVE) & trades.symbol.eq(SUCCESSOR)]
            for symbol in PREVIOUS:
                units = int(previous_holdings[symbol])
                changes.loc[EFFECTIVE, symbol] -= units
                changes.loc[EFFECTIVE, SUCCESSOR] += units
                events.append(dict(arm=arm, date=EFFECTIVE.date(), predecessor=symbol, old_shares=units,
                                   successor=SUCCESSOR, converted_shares=units, exchange_ratio=1,
                                   successor_preexisting_shares=successor_before, conversion_fee_usd=0,
                                   conversion_traded_usd=0, successor_open=prices.open.at[EFFECTIVE, SUCCESSOR],
                                   successor_close=raw.at[EFFECTIVE, SUCCESSOR], successor_target=SUCCESSOR in set(event_targets.symbol),
                                   successor_market_order_qty=int(actual_order.qty.sum())))
        held = changes.cumsum()
        assert (held >= 0).all().all()
        if period == "val":
            assert not held.loc[EFFECTIVE:, list(PREVIOUS)].to_numpy().any()
        price = marks.reindex(index=nav.index, columns=symbols)
        assert not (price.isna() & held.ne(0)).any().any()
        value = held * price.fillna(0)
        rebuilt = cash + value.sum(axis=1)
        np.testing.assert_allclose(rebuilt, nav, atol=1e-6, rtol=0)
        stale = (held > 0) & raw.reindex(index=nav.index, columns=symbols).isna()
        stale_money = value.where(stale, 0).sum(axis=1)
        end_pos = held.iloc[-1]
        for symbol in end_pos.index[end_pos > 0]:
            ending.append(dict(period=period, arm=arm, date=nav.index[-1].date(), symbol=symbol,
                               shares=int(end_pos[symbol]), mark=price.iloc[-1][symbol], market_value=value.iloc[-1][symbol],
                               raw_close_available=not stale.iloc[-1][symbol]))
        results[f"{period}_{arm}"] = dict(sessions=len(nav), fills=len(trades), end_cash_usd=float(cash.iloc[-1]),
                                          end_marked_holdings_usd=float(value.iloc[-1].sum()), end_nav_usd=float(nav.iloc[-1]),
                                          max_nav_reconstruction_error_usd=float((nav - rebuilt).abs().max()),
                                          stale_sessions=int(stale.any(axis=1).sum()),
                                          stale_symbols=stale.columns[stale.any()].tolist(),
                                          max_stale_nav_pct=float((100 * stale_money / nav).max()),
                                          ending_stale_value_usd=float(stale_money.iloc[-1]))
        if period == "val":
            results[f"{period}_{arm}"]["T_shares_at_2022_04_05_close"] = int(held.loc["2022-04-05"].get("T", 0))
    pd.DataFrame(events).to_csv(out / "conversions.csv", index=False)
    pd.DataFrame(ending).to_csv(out / "ending_positions.csv", index=False)
    return results


def run(inputs, scores, primary, out):
    base.run(inputs, scores, out, engine=sensitivity_engine())
    compare_periods(primary, out)
    audit = dict(source=SEC_8K, course_engine_sha256=ENGINE_SHA, effective_before_open=str(EFFECTIVE.date()),
                 prior_history=compare_history(primary, out), portfolio=reconcile(inputs, out))
    (out / "conversion_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
