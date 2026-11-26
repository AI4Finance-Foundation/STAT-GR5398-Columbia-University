"""Reconcile per-stock after-cost dollar contributions to both official NAV paths."""
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.backtester import BacktestConfig
from shared.data_view import DataBundle
from shared.periods import PERIODS

OUTPUT = Path("output")
REPORT = Path("../results/pretest")


def contributions(folder, prices, cfg):
    trades = pd.read_csv(folder / "trades.csv", parse_dates=["date"])
    equity = pd.read_csv(folder / "equity.csv", index_col=0, parse_dates=True).equity
    quantities = trades.pivot_table(index="date", columns="symbol", values="qty", aggfunc="sum").reindex(equity.index).fillna(0)
    gross = trades.assign(gross_qty=trades.qty.abs()).pivot_table(index="date", columns="symbol", values="gross_qty", aggfunc="sum").reindex_like(quantities).fillna(0)
    opens = prices.open.reindex_like(quantities)
    assert opens[quantities.ne(0)].count().sum() == quantities.ne(0).sum().sum()
    closes = prices.close.loc[:equity.index[-1], quantities.columns].ffill().reindex(equity.index)
    held = quantities.cumsum()
    assert closes[held.ne(0)].count().sum() == held.ne(0).sum().sum()
    costs = (gross * opens).fillna(0) * (cfg.commission + cfg.slippage)
    cumulative = (held * closes).fillna(0) - ((quantities * opens).fillna(0) + costs).cumsum()
    np.testing.assert_allclose(cumulative.sum(axis=1) + cfg.initial_cash, equity, rtol=0, atol=1e-6)
    return cumulative, equity


def main():
    prices = DataBundle.load().prices
    rows, annual = [], []
    for period in ("dev", "val"):
        cfg = BacktestConfig(*PERIODS[period])
        base, a = contributions(OUTPUT / f"{period}_A_selection", prices, cfg)
        team, b = contributions(OUTPUT / f"{period}_B_valuation", prices, cfg)
        diff = team.subtract(base, fill_value=0)
        np.testing.assert_allclose(diff.sum(axis=1), b-a, rtol=0, atol=1e-6)
        for ticker, pnl in diff.iloc[-1].items():
            rows.append(dict(period=period, ticker=ticker, incremental_net_pnl_usd=pnl))
        for year, start in a.groupby(a.index.year):
            end = start.index[-1]
            before = a.index[a.index < start.index[0]]
            a0, b0 = (a.loc[before[-1]], b.loc[before[-1]]) if len(before) else (cfg.initial_cash, cfg.initial_cash)
            annual.append(dict(period=period,year=year,baseline_return_pct=100*(a.at[end]/a0-1),
                               valuation_return_pct=100*(b.at[end]/b0-1)))
    pd.DataFrame(rows).sort_values(["period","incremental_net_pnl_usd"], ascending=[True,False]).to_csv(REPORT/"stock_attribution.csv",index=False)
    pd.DataFrame(annual).to_csv(REPORT/"annual_returns.csv",index=False)


if __name__ == "__main__":
    main()
