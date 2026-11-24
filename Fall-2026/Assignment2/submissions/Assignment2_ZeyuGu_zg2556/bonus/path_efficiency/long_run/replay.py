"""Selection-only stock replay for matched monthly control and feature scores."""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BONUS = HERE.parents[1]
from portfolio.backtester import BacktestConfig, run_backtest
from selection import MonthlyScores, Parameters
from shared.metrics import performance
from shared.periods import PERIODS
from shared.prices import PriceStore
from shared.universe import TradeableUniverse

ARMS = ("control", "path_efficiency")
REFERENCE = {"control": BONUS / "results/control",
             "path_efficiency": BONUS / "path_efficiency/results/path_efficiency"}
REFERENCE_WINDOW = ("2018-01-02", "2018-06-29")


def make_inputs(source, scratch):
    data = source / "data"
    cutoff = pd.Timestamp(PERIODS["val"][1])
    price_file = scratch / "prices.parquet"
    prices = pd.read_parquet(data / "prices/daily_adjusted.parquet", columns=["date", "symbol", "open", "close"],
                             filters=[("date", "<=", cutoff)])
    prices.to_parquet(price_file, index=False)
    members = scratch / "membership.db"
    with sqlite3.connect((data / "fundamentals.db").as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(members) as dst:
        for table, key in (("universe_membership", "snapshot_date"), ("symbol_changes", "change_date")):
            rows = pd.read_sql(f"SELECT * FROM {table} WHERE {key} <= ?", src, params=[str(cutoff.date())])
            rows.to_sql(table, dst, index=False)
    config = json.loads((source / "selection/configs/growth.json").read_text())
    assert config["model"]["inference_mode"] == "mixed-alpha"
    universe = TradeableUniverse.from_db_path(str(members), config["model"]["universes"])
    return PriceStore(price_file), Parameters(**config["parameters"]), universe


def replay_window(inputs, scores, start, end, engine=run_backtest):
    prices, params, universe = inputs
    config = BacktestConfig(start=pd.Timestamp(start).date(), end=pd.Timestamp(end).date())
    assert not config.stops and not config.use_exposure_engine
    assert config.initial_cash == 100_000 and config.cash_buffer == 0.05
    assert np.isclose(config.commission + config.slippage, 0.0006)
    result = engine(prices, scores, params, universe, config)
    sessions = prices.sessions[(prices.sessions >= pd.Timestamp(start)) & (prices.sessions <= pd.Timestamp(end))]
    assert result.equity.index.equals(sessions) and result.equity.gt(0).all()
    weekly = sessions.to_series().groupby(sessions.to_period("W-SUN")).first()
    assert [row["date"] for row in result.weekly_targets] == list(weekly.dt.date)
    for row in result.weekly_targets:
        assert row["model_cutoff"] == prices.last_session_before_month(row["date"])
        assert row["budget"] == 0.95 and len(row["weights"]) <= 10
    assert result.trades.reason.eq("rebalance").all()
    return result, config.commission + config.slippage


def metrics(nav, trades, start, end, cost):
    # Annual/monthly measurements include the preceding close within this split.
    # Turnover and costs count only dates inside the measurement interval.
    sessions = nav.loc[pd.Timestamp(start):pd.Timestamp(end)]
    location = nav.index.get_loc(sessions.index[0])
    anchor = max(0, location - 1)
    curve = nav.iloc[anchor:nav.index.get_loc(sessions.index[-1]) + 1]
    ledger = trades.loc[trades.date.between(sessions.index[0], sessions.index[-1])]
    returns = curve.pct_change().dropna()
    growth = curve.iloc[-1] / curve.iloc[0]
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    volatility = returns.std() * np.sqrt(252)
    traded = ledger.notional.abs().sum()
    return dict(sessions=len(sessions), start=sessions.index[0].date(), end=sessions.index[-1].date(),
                start_close=curve.index[0].date(), start_nav=curve.iloc[0], end_nav=curve.iloc[-1],
                total_return_pct=100 * (growth - 1), cagr_pct=100 * (growth ** (252 / len(returns)) - 1),
                drawdown_depth_pct=-100 * drawdown.min(), drawdown_peak=curve.loc[:trough].idxmax().date(),
                drawdown_trough=trough.date(), volatility_pct=100 * volatility,
                sharpe=returns.mean() * 252 / volatility if volatility > 0 else np.nan,
                turnover_per_year=traded / sessions.mean() * 252 / len(sessions),
                trades=len(ledger), gross_traded_usd=traded, estimated_cost_usd=traded * cost)


def growth_scores(path, keys):
    frame = pd.read_parquet(path, filters=[("month", ">=", pd.Timestamp("2018-01-01")),
                                          ("month", "<=", pd.Timestamp("2018-06-01")),
                                          ("bucket", "==", "growth_tech")])
    return frame.sort_values(keys).reset_index(drop=True)


def baseline_check(inputs, score_files):
    start, end = REFERENCE_WINDOW
    verification = {}
    for arm in ARMS:
        archive = REFERENCE[arm]
        keys = ["month", "cutoff", "symbol", "bucket"]
        actual_g = growth_scores(score_files[arm], keys)
        expected_g = growth_scores(archive / "scores.parquet", keys)
        pd.testing.assert_frame_equal(actual_g[keys], expected_g[keys], check_dtype=False)
        score_cols = ["rank_score", "weight_score"]
        np.testing.assert_allclose(actual_g[score_cols], expected_g[score_cols],
                                   rtol=1e-10, atol=1e-12)
        result, _ = replay_window(inputs, MonthlyScores(score_files[arm]), start, end)
        archived_nav = pd.read_csv(archive / "equity.csv", index_col=0, parse_dates=True, converters={"equity": float}).equity
        assert result.equity.index.equals(archived_nav.index)
        np.testing.assert_array_equal(result.equity, archived_nav)
        orders = result.trades.assign(date=result.trades.date.astype(str))
        sorts = ["date", "symbol", "qty"]
        pd.testing.assert_frame_equal(orders.sort_values(sorts).reset_index(drop=True),
                                      pd.read_csv(archive / "trades.csv").sort_values(sorts).reset_index(drop=True))
        verification[arm] = dict(eligible_growth_stock_months=len(actual_g),
                                 max_abs_score_difference=float((actual_g[score_cols] - expected_g[score_cols]).abs().max().max()),
                                 exact_daily_equity_closes=len(archived_nav), exact_trade_rows=len(orders))
    return verification


def paired_metrics(table, keys):
    base = table.loc[table.arm.eq("control")].set_index(keys)
    change = table.loc[table.arm.eq("path_efficiency")].set_index(keys)
    assert base.index.equals(change.index)
    columns = {"total_return_pct": "return_pp", "cagr_pct": "cagr_pp",
               "drawdown_depth_pct": "drawdown_depth_pp", "turnover_per_year": "turnover_x", "sharpe": "sharpe"}
    differences = change[columns.keys()] - base[columns.keys()]
    return differences.rename(columns=columns).add_prefix("feature_minus_control_").reset_index()


def check_time_slices(parts, totals):
    for (period, arm), rows in parts.groupby(["period", "arm"]):
        full = totals.loc[(period, arm)]
        np.testing.assert_allclose((1 + rows.total_return_pct / 100).prod(),
                                   1 + full.total_return_pct / 100, rtol=1e-12, atol=0)
        np.testing.assert_allclose(rows.gross_traded_usd.sum(), full.gross_traded_usd)
        assert rows.trades.sum() == full.trades and rows.sessions.sum() == full.sessions


def run(inputs, score_files, out, engine=run_backtest):
    periods, years, months, navs, trade_ledgers, targets = [], [], [], [], [], []
    verification = {"2018H1_reference": baseline_check(inputs, score_files), "scores_sha256": {}}
    for arm, path in score_files.items():
        with path.open("rb") as file:
            verification["scores_sha256"][arm] = hashlib.file_digest(file, "sha256").hexdigest()
        scores = MonthlyScores(path)
        assert scores.months == list(pd.date_range("2017-01-01", "2023-12-01", freq="MS").date)
        for period in ("dev", "val"):
            start, end = PERIODS[period]
            result, cost = replay_window(inputs, scores, start, end, engine)
            curve = result.equity.rename("equity").rename_axis("date")
            trades = result.trades.assign(date=pd.to_datetime(result.trades.date))
            overall = metrics(curve, trades, start, end, cost)
            official = performance(curve)
            assert round(overall["total_return_pct"], 2) == official["total_return_pct"]
            assert round(-overall["drawdown_depth_pct"], 2) == official["max_drawdown_pct"]
            assert round(overall["sharpe"], 3) == official["sharpe"]
            periods.append(dict(period=period, arm=arm, **overall))
            for year, block in curve.groupby(curve.index.year):
                summary = metrics(curve, trades, block.index[0], block.index[-1], cost)
                years.append(dict(period=period, year=year, arm=arm, **summary))
            for month, block in curve.groupby(curve.index.to_period("M")):
                summary = metrics(curve, trades, block.index[0], block.index[-1], cost)
                months.append(dict(period=period, month=str(month), arm=arm, **summary))
            navs.append(curve.reset_index().assign(period=period, arm=arm))
            trade_ledgers.append(trades.assign(period=period, arm=arm))
            targets.extend(dict(period=period, arm=arm, rebalance=r["date"], model_cutoff=r["model_cutoff"],
                                symbol=symbol, weight_before_cash_buffer=weight)
                           for r in result.weekly_targets for symbol, weight in sorted(r["weights"].items()))
    out.mkdir(parents=True, exist_ok=True)
    for name, records, keys in (("period", periods, ["period"]), ("year", years, ["period", "year"]),
                                ("month", months, ["period", "month"])):
        frame = pd.DataFrame(records).sort_values(keys + ["arm"])
        if name != "period":
            check_time_slices(frame, pd.DataFrame(periods).set_index(["period", "arm"]))
        frame.to_csv(out / f"{name}_metrics.csv", index=False)
        paired_metrics(frame, keys).to_csv(out / f"{name}_differences.csv", index=False)
    wide = pd.concat(navs).pivot(index=["period", "date"], columns="arm", values="equity").reset_index()
    assert not wide.isna().any().any()
    wide.to_csv(out / "daily_nav.csv", index=False)
    pd.concat(trade_ledgers).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(targets).to_csv(out / "weekly_stock_targets.csv", index=False)
    (out / "portfolio_verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def stability(results):
    months = pd.read_csv(results / "month_metrics.csv").pivot(
        index=["period", "month"], columns="arm", values="total_return_pct").reset_index()
    records = []
    for name, frame in (("dev", months.loc[months.period == "dev"]),
                        ("val", months.loc[months.period == "val"]),
                        ("dev_ex_2018h1", months.loc[(months.period == "dev") &
                                                     ~months.month.between("2018-01", "2018-06")]),
                        ("dev_after_2017", months.loc[(months.period == "dev") &
                                                      (months.month >= "2018-01")])):
        spread = frame.path_efficiency - frame.control
        log_relative = np.log1p(frame.path_efficiency / 100) - np.log1p(frame.control / 100)
        records.append(dict(sample=name, months=len(frame), positive_months=int((spread > 1e-9).sum()),
                            tied_months=int((spread.abs() <= 1e-9).sum()), median_spread_pp=spread.median(),
                            control_return_pct=100*np.expm1(np.log1p(frame.control/100).sum()),
                            feature_return_pct=100*np.expm1(np.log1p(frame.path_efficiency/100).sum()),
                            cumulative_relative_wealth_pct=100*np.expm1(log_relative.sum()),
                            best_month=frame.loc[spread.idxmax(),"month"], best_spread_pp=spread.max(),
                            worst_month=frame.loc[spread.idxmin(),"month"], worst_spread_pp=spread.min()))
    return pd.DataFrame(records)
