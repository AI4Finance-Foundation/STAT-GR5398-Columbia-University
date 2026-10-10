"""Reconstruct global portfolios from source prices and saved close holdings.

This does not invoke the backtest engine. It checks two windows, top 10/25,
source-price returns, ranking, drift, financed turnover and benchmark NAV.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from market_data import prices
from audit_and_train import membership

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--cache', type=Path, required=True)
a = ap.parse_args()
r = a.root
px = prices(a.cache, root=r, canonical=True).pivot(index='datadate', columns='tic', values='adj_close').sort_index()
at = membership(a.cache)
pred = pd.read_csv(r/'audited_run/predictions.csv', parse_dates=['decision_date'])
checked = []
for window in ['full', 'common']:
    folder = r/('comparison_'+window)
    nav = pd.read_csv(folder/'nav_comparison.csv', index_col=0, parse_dates=True)
    holdings = pd.read_csv(folder/'rebalance_holdings.csv', parse_dates=['date'])
    for name, frac in [('audited_top10', .10), ('audited_top25', .25)]:
        daily = pd.read_csv(folder/(name+'_daily.csv'), index_col=0, parse_dates=True)
        h = holdings.loc[holdings.strategy.eq(name)]
        weight = pd.Series(0., index=px.columns)
        value = 1.
        for date, rec in daily.iterrows():
            held = weight[weight > 1e-12]
            ix = px.index.get_loc(date)
            if len(held):
                assert ix > 0
                before = px.iloc[ix-1].reindex(held.index)
                after = px.iloc[ix].reindex(held.index)
                assert (before > 0).all() and (after > 0).all(), 'missing/nonpositive held quote'
                move = after/before-1
                gross = float((held*move).sum())
                drift = (held*(1+move)/(1+gross)).reindex(px.columns, fill_value=0.)
            else:
                gross = 0.
                drift = weight.copy()
            assert abs(gross-rec.gross_ret) < 1e-10, (window, name, date, 'source return')
            selected = h.loc[h.date.eq(date)]
            if len(selected):
                eligible = pred.loc[pred.decision_date.eq(date)].copy()
                eligible = eligible.loc[eligible.ticker.isin(at(date)) & eligible.ticker.isin(px.columns)]
                eligible = eligible.loc[px.loc[date].reindex(eligible.ticker).notna().to_numpy()]
                expected = eligible.sort_values(['best', 'ticker'], ascending=[False, True]).head(int(np.ceil(len(eligible)*frac)))
                assert set(selected.ticker) == set(expected.ticker)
                assert np.allclose(selected.weight, 1/len(selected), atol=1e-10)
                close = selected.set_index('ticker').weight.reindex(px.columns, fill_value=0.)
                traded = float(((1-rec.cost)*close-drift).abs().sum())
                assert abs(traded-rec.turnover) < 1e-10
                assert abs(rec.cost-.001*traded) < 1e-10
                weight = close
            else:
                assert abs(rec.cost) < 1e-12 and abs(rec.turnover) < 1e-12
                weight = drift
            assert abs((1-weight.sum())-rec.cash_weight) < 1e-10
            value *= (1+gross)*(1-rec.cost)
            assert abs(value-rec.nav) < 1e-9
        checked.append(dict(window=window, strategy=name, days=len(daily), rebalances=h.date.nunique()))
    for ticker in ['SPY', 'QQQ']:
        source = pd.read_csv(r/'reference/overnight'/(ticker+'_adjusted_close.csv'), index_col=0, parse_dates=True).iloc[:, 0]
        source = source.reindex(nav.index)
        assert source.notna().all()
        assert np.allclose(source/source.iloc[0], nav[ticker], atol=1e-10)
record = dict(passed=True, portfolios=checked, benchmark_windows_checked=4,
              checks=['source returns', 'prediction ranking and membership', 'equal weights',
                      'drift', 'turnover from holdings', 'financed costs', 'cash', 'daily NAV', 'benchmark price ratios'],
              compare_code_sha256=hashlib.sha256((r/'code/compare_backtests.py').read_bytes()).hexdigest(),
              limits='Reconstruction shares the documented market-source loader and membership mapping. It does not certify vendor histories or financial vintages.')
(r/'evidence/market_reconstruction_verification.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
